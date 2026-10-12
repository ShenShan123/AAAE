"""Canonical MatMul scope for weight-stationary Transformer injection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Mapping

import onnx


INJECTION_SCOPE_ID = "sram_cim_weight_stationary_linear_only_v1"
INCLUDED_ROLES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")

_VIT_BLOCK_ROLE = re.compile(
    r"^/blocks/blocks\.(?P<block>\d+)/"
    r"(?P<operation>attn/qkv|attn/proj|mlp/fc1|mlp/fc2)/MatMul$"
)
_SWIN_BLOCK_ROLE = re.compile(
    r"^/layers\.(?P<stage>\d+)/blocks\.(?P<block>\d+)/"
    r"(?P<operation>attn/qkv|attn/proj|mlp/fc1|mlp/fc2)/MatMul$"
)
_ROLE_BY_OPERATION = {
    "attn/qkv": "attn.qkv",
    "attn/proj": "attn.proj",
    "mlp/fc1": "mlp.fc1",
    "mlp/fc2": "mlp.fc2",
}

# Only the data input is followed. Shape, scale, zero-point, axes, and other
# metadata inputs must not make a runtime activation look like a model weight.
_IMMUTABLE_PASSTHROUGH_INPUT = {
    "Cast": 0,
    "DequantizeLinear": 0,
    "Flatten": 0,
    "Identity": 0,
    "QuantizeLinear": 0,
    "Reshape": 0,
    "Squeeze": 0,
    "Transpose": 0,
    "Unsqueeze": 0,
}


@dataclass(frozen=True)
class EncoderBlockRole:
    stage_id: int
    block_id: int
    role: str

    @property
    def block_key(self) -> tuple[int, int]:
        return self.stage_id, self.block_id


@dataclass(frozen=True)
class GraphContext:
    graph: onnx.GraphProto
    path: str
    initializer_names: frozenset[str]
    producer_scopes: tuple[Mapping[str, onnx.NodeProto], ...]


@dataclass(frozen=True)
class GraphNodeRef:
    context: GraphContext
    node: onnx.NodeProto


@dataclass(frozen=True)
class LinearMatMulPartition:
    all_matmuls: tuple[GraphNodeRef, ...]
    selected: tuple[GraphNodeRef, ...]
    excluded: tuple[GraphNodeRef, ...]
    role_by_name: Mapping[str, EncoderBlockRole]

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(str(ref.node.name) for ref in self.selected)

    @property
    def excluded_names(self) -> tuple[str, ...]:
        return tuple(str(ref.node.name) for ref in self.excluded)


def parse_encoder_block_role(node_name: str) -> EncoderBlockRole | None:
    """Return the canonical block Linear role encoded by an ONNX node name."""

    name = str(node_name)
    match = _SWIN_BLOCK_ROLE.match(name)
    if match:
        return EncoderBlockRole(
            stage_id=int(match.group("stage")),
            block_id=int(match.group("block")),
            role=_ROLE_BY_OPERATION[match.group("operation")],
        )
    match = _VIT_BLOCK_ROLE.match(name)
    if match:
        return EncoderBlockRole(
            stage_id=0,
            block_id=int(match.group("block")),
            role=_ROLE_BY_OPERATION[match.group("operation")],
        )
    return None


def _graph_producers(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output_name in node.output:
            name = str(output_name)
            if not name:
                continue
            if name in producers:
                raise ValueError(f"graph {graph.name!r} has duplicate producer for tensor {name!r}")
            producers[name] = node
    return producers


def _child_graphs(node: onnx.NodeProto) -> Iterator[tuple[str, onnx.GraphProto]]:
    parent = str(node.name or node.op_type)
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            yield f"{parent}:{attribute.name}", attribute.g
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            for index, graph in enumerate(attribute.graphs):
                yield f"{parent}:{attribute.name}[{index}]", graph


def iter_graph_contexts(model_or_graph: onnx.ModelProto | onnx.GraphProto) -> Iterator[GraphContext]:
    """Yield a graph and every nested control-flow graph with lexical scope."""

    root = model_or_graph.graph if isinstance(model_or_graph, onnx.ModelProto) else model_or_graph

    def visit(
        graph: onnx.GraphProto,
        *,
        path: str,
        inherited_initializers: frozenset[str],
        inherited_producers: tuple[Mapping[str, onnx.NodeProto], ...],
    ) -> Iterator[GraphContext]:
        initializer_names = inherited_initializers.union(str(value.name) for value in graph.initializer)
        local_producers = _graph_producers(graph)
        context = GraphContext(
            graph=graph,
            path=path,
            initializer_names=frozenset(initializer_names),
            producer_scopes=(local_producers, *inherited_producers),
        )
        yield context
        for node_index, node in enumerate(graph.node):
            for child_name, child_graph in _child_graphs(node):
                child_path = f"{path}/{node_index}:{child_name}"
                yield from visit(
                    child_graph,
                    path=child_path,
                    inherited_initializers=context.initializer_names,
                    inherited_producers=context.producer_scopes,
                )

    yield from visit(
        root,
        path=str(root.name or "main"),
        inherited_initializers=frozenset(),
        inherited_producers=(),
    )


def iter_graph_nodes(model_or_graph: onnx.ModelProto | onnx.GraphProto) -> Iterator[GraphNodeRef]:
    for context in iter_graph_contexts(model_or_graph):
        for node in context.graph.node:
            yield GraphNodeRef(context=context, node=node)


def collect_named_matmuls(model_or_graph: onnx.ModelProto | onnx.GraphProto) -> tuple[GraphNodeRef, ...]:
    refs = tuple(ref for ref in iter_graph_nodes(model_or_graph) if ref.node.op_type == "MatMul")
    if not refs:
        raise ValueError("model has no MatMul nodes")
    unnamed = [ref.context.path for ref in refs if not ref.node.name]
    if unnamed:
        raise ValueError(f"model has {len(unnamed)} unnamed MatMul nodes in {sorted(set(unnamed))}")
    names = [str(ref.node.name) for ref in refs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"model has duplicate named MatMul nodes: {duplicates}")
    return refs


def operand_traces_to_initializer(tensor_name: str, context: GraphContext) -> bool:
    """Trace wrapper data inputs until an immutable initializer is reached."""

    current = str(tensor_name)
    visited: set[str] = set()
    while current and current not in visited:
        if current in context.initializer_names:
            return True
        visited.add(current)
        producer = next((scope[current] for scope in context.producer_scopes if current in scope), None)
        if producer is None:
            return False
        input_index = _IMMUTABLE_PASSTHROUGH_INPUT.get(str(producer.op_type))
        if input_index is None or len(producer.input) <= input_index:
            return False
        current = str(producer.input[input_index])
    return False


def select_weight_stationary_block_linears(
    model_or_graph: onnx.ModelProto | onnx.GraphProto,
    *,
    expected_block_count: int | None = 12,
) -> LinearMatMulPartition:
    """Select qkv/proj/fc1/fc2 MatMuls with exactly one immutable operand."""

    all_matmuls = collect_named_matmuls(model_or_graph)
    selected: list[GraphNodeRef] = []
    role_by_name: dict[str, EncoderBlockRole] = {}
    roles_by_block: dict[tuple[int, int], dict[str, str]] = {}

    for ref in all_matmuls:
        node_name = str(ref.node.name)
        block_role = parse_encoder_block_role(node_name)
        if block_role is None:
            continue
        if len(ref.node.input) < 2:
            raise ValueError(f"canonical MatMul {node_name!r} has fewer than two inputs")
        immutable_operands = tuple(
            index
            for index, tensor_name in enumerate(ref.node.input[:2])
            if operand_traces_to_initializer(str(tensor_name), ref.context)
        )
        if len(immutable_operands) != 1:
            raise ValueError(
                f"canonical role {node_name!r} must have exactly one immutable weight operand; "
                f"found operand indices {immutable_operands}"
            )
        block_roles = roles_by_block.setdefault(block_role.block_key, {})
        if block_role.role in block_roles:
            raise ValueError(
                f"encoder block {block_role.block_key} has duplicate role {block_role.role!r}: "
                f"{block_roles[block_role.role]!r}, {node_name!r}"
            )
        block_roles[block_role.role] = node_name
        role_by_name[node_name] = block_role
        selected.append(ref)

    if expected_block_count is not None and len(roles_by_block) != int(expected_block_count):
        raise ValueError(
            f"canonical scope requires {int(expected_block_count)} encoder blocks, found {len(roles_by_block)}"
        )
    required_roles = set(INCLUDED_ROLES)
    invalid_blocks = {
        block_key: {
            "missing": sorted(required_roles.difference(roles)),
            "extra": sorted(set(roles).difference(required_roles)),
        }
        for block_key, roles in roles_by_block.items()
        if set(roles) != required_roles
    }
    if invalid_blocks:
        raise ValueError(f"encoder blocks do not satisfy the four-role canonical scope: {invalid_blocks}")
    expected_selected = len(roles_by_block) * len(INCLUDED_ROLES)
    if len(selected) != expected_selected:
        raise ValueError(f"canonical scope expected {expected_selected} MatMuls, found {len(selected)}")

    selected_names = {str(ref.node.name) for ref in selected}
    excluded = tuple(ref for ref in all_matmuls if str(ref.node.name) not in selected_names)
    return LinearMatMulPartition(
        all_matmuls=all_matmuls,
        selected=tuple(selected),
        excluded=excluded,
        role_by_name=role_by_name,
    )
