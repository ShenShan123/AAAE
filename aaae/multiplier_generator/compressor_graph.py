from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_STAGE_MODULE_RE = re.compile(r"(?ms)^\s*module\s+(STAGE\d+_\d+)\s*\((.*?)^endmodule\s*$")
_INSTANCE_RE = re.compile(r"(?ms)^\s*([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\);")
_ASSIGN_RE = re.compile(r"(?m)^\s*assign\s+([^=;]+?)\s*=\s*(.+?);\s*$")
_PORT_CONN_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\((.*?)\)", re.S)
_PP_SIGNAL_RE = re.compile(r"^(pp\d+_\d+(?:\[\d+\])?)$")
_STAGE_NAME_RE = re.compile(r"^STAGE(\d+)_(\d+)$")

_EXACT_TYPES = {"E_4_2", "Full_adder", "Half_adder"}
_OUTPUT_PORTS = ("S", "C", "COUT")
_INPUT_PORTS = ("a", "b", "c", "d", "CIN")


def _split_concat(expr: str) -> list[str]:
    expr = expr.strip()
    if not (expr.startswith("{") and expr.endswith("}")):
        return [expr]
    inner = expr[1:-1]
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in inner:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _stage_module_name(stage_idx: int, n_bits: int) -> str:
    return f"STAGE{int(stage_idx)}_{int(n_bits)}"


def _parse_stage_context(module_name: str) -> tuple[int, int]:
    m = _STAGE_NAME_RE.match(module_name)
    if not m:
        raise ValueError(f"unsupported stage module name: {module_name}")
    return int(m.group(1)), int(m.group(2))


def _previous_stage_for_term(module_name: str, term: str, stage_modules: dict[str, str]) -> str | None:
    if not _PP_SIGNAL_RE.match(str(term).strip()):
        return None
    stage_idx, n_bits = _parse_stage_context(module_name)
    if stage_idx <= 1:
        return None
    prev = _stage_module_name(stage_idx - 1, n_bits)
    if prev in stage_modules:
        return prev
    return None


def _parse_assign_aliases(module_name: str, body: str) -> dict[str, str]:
    alias: dict[str, str] = {}
    for m in _ASSIGN_RE.finditer(body):
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        if lhs in {"A", "B"}:
            continue
        parts = _split_concat(rhs)
        if len(parts) == 1:
            alias[lhs] = parts[0]
            alias[f"{lhs}[0]"] = parts[0]
            continue
        # In Verilog concatenation the right-most item is bit 0.
        for idx, part in enumerate(reversed(parts)):
            alias[f"{lhs}[{idx}]"] = part
    return alias


def _infer_column_and_local_idx(stage_idx: int, inst_name: str, outputs: dict[str, str]) -> tuple[int | None, int]:
    for net in outputs.values():
        m = re.search(rf"pp{stage_idx}_(\d+)_comp_?(\d+)", net)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(rf"pp{stage_idx}_(\d+)_ha(\d+)", net)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(rf"pp{stage_idx}_(\d+)_fa(\d+)", net)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(rf"pp{stage_idx}_(\d+)_", net)
        if m:
            return int(m.group(1)), 0
        m = re.search(r"(?:^|_)(?:comp|e42|fa|ha)_(\d+)(?:_|$)", net)
        if m:
            return int(m.group(1)), 0

    m = re.search(r"_(\d+)_(\d+)$", inst_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"_(\d+)$", inst_name)
    if m:
        return int(m.group(1)), 0
    return None, 0


def _classify_node_kind(cell_type: str) -> str:
    if cell_type == "Half_adder":
        return "half_adder"
    if cell_type == "Full_adder":
        return "full_adder"
    if cell_type == "E_4_2":
        return "exact_4_2"
    if cell_type.startswith("AE_4_2_"):
        return "approx_4_2_wrapper"
    return "approx_4_2"


def _classify_input_source(term: str) -> str:
    t = str(term).strip()
    if t in {"cin_net", "1'b0", "1'b1"}:
        return "control_or_literal"
    if any(op in t for op in ("&", "|", "^", "~")):
        return "primary_expr"
    if re.match(r"^[AB]\[\d+\]$", t):
        return "primary_bit"
    if _PP_SIGNAL_RE.match(t):
        return "packed_stage_signal"
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?$", t):
        return "named_signal"
    return "unknown"


def _edge_kind(*, src_port: str, dst_port: str, resolved_signal: str) -> str:
    sig = str(resolved_signal).strip()
    if dst_port == "CIN" or src_port == "COUT" or sig.endswith("_cout"):
        return "cout"
    if src_port == "C" or sig.endswith("_c"):
        return "carry"
    if src_port == "S" or sig.endswith("_s"):
        return "sum"
    return "signal"


def _parse_stage_instances(
    *,
    module_name: str,
    body: str,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    stage_idx, _ = _parse_stage_context(module_name)
    nodes: list[dict[str, Any]] = []
    output_net_to_node: dict[str, tuple[str, str]] = {}
    ordinal = 0

    for m in _INSTANCE_RE.finditer(body):
        cell_type = m.group(1)
        inst_name = m.group(2)
        ports_blob = m.group(3)
        ports = {k: v.strip() for k, v in _PORT_CONN_RE.findall(ports_blob)}
        outputs = {k: v for k, v in ports.items() if k in _OUTPUT_PORTS}
        if not outputs:
            continue

        column, local_idx = _infer_column_and_local_idx(stage_idx, inst_name, outputs)
        node_id = f"stage{stage_idx}:{inst_name}"
        node = {
            "id": node_id,
            "name": inst_name,
            "cell_type": cell_type,
            "kind": _classify_node_kind(cell_type),
            "is_approx": cell_type not in _EXACT_TYPES,
            "stage": int(stage_idx),
            "column": column,
            "local_index": int(local_idx),
            "module": module_name,
            "ordinal": ordinal,
            "inputs": {k: v for k, v in ports.items() if k in _INPUT_PORTS},
            "outputs": outputs,
        }
        nodes.append(node)
        ordinal += 1
        for port_name, net_name in outputs.items():
            output_net_to_node[net_name] = (node_id, port_name)

    return nodes, output_net_to_node


def _resolve_term(
    *,
    module_name: str,
    term: str,
    alias_by_module: dict[str, dict[str, str]],
    stage_modules: dict[str, str],
    depth: int = 0,
) -> str:
    cur = str(term).strip()
    if depth > 32:
        return cur

    local_alias = alias_by_module.get(module_name, {})
    mapped = local_alias.get(cur)
    if mapped is not None:
        return _resolve_term(
            module_name=module_name,
            term=mapped,
            alias_by_module=alias_by_module,
            stage_modules=stage_modules,
            depth=depth + 1,
        )

    prev_mod = _previous_stage_for_term(module_name, cur, stage_modules)
    if prev_mod is not None:
        prev_alias = alias_by_module.get(prev_mod, {})
        mapped_prev = prev_alias.get(cur)
        if mapped_prev is not None:
            return _resolve_term(
                module_name=prev_mod,
                term=mapped_prev,
                alias_by_module=alias_by_module,
                stage_modules=stage_modules,
                depth=depth + 1,
            )

    return cur


def extract_compressor_graph_from_rtl_text(
    rtl_text: str,
    *,
    design_id: str,
    top: str,
    n_bits: int,
) -> dict[str, Any]:
    stage_modules = {m.group(1): m.group(2) for m in _STAGE_MODULE_RE.finditer(rtl_text)}
    if not stage_modules:
        raise ValueError("no STAGE modules found in RTL")

    alias_by_module = {
        module_name: _parse_assign_aliases(module_name, body)
        for module_name, body in stage_modules.items()
    }

    nodes: list[dict[str, Any]] = []
    output_net_to_node: dict[tuple[str, str], tuple[str, str]] = {}
    stage_summaries: list[dict[str, Any]] = []

    for module_name in sorted(stage_modules, key=lambda x: _parse_stage_context(x)[0]):
        body = stage_modules[module_name]
        stage_nodes, stage_outputs = _parse_stage_instances(module_name=module_name, body=body)
        nodes.extend(stage_nodes)
        output_net_to_node.update({(module_name, net): ref for net, ref in stage_outputs.items()})
        stage_idx, _ = _parse_stage_context(module_name)
        stage_summaries.append(
            {
                "stage": int(stage_idx),
                "module": module_name,
                "node_count": len(stage_nodes),
                "approx_node_count": sum(1 for node in stage_nodes if bool(node["is_approx"])),
            }
        )

    edges: list[dict[str, Any]] = []
    edge_pairs: set[tuple[str, str, str, str]] = set()

    for node in nodes:
        module_name = str(node["module"])
        inputs = dict(node["inputs"])
        input_sources: list[dict[str, Any]] = []
        for pin_name, raw_signal in inputs.items():
            resolved_signal = _resolve_term(
                module_name=module_name,
                term=raw_signal,
                alias_by_module=alias_by_module,
                stage_modules=stage_modules,
            )

            src_ref = output_net_to_node.get((module_name, resolved_signal))
            if src_ref is None:
                prev_mod = _previous_stage_for_term(module_name, raw_signal, stage_modules)
                if prev_mod is not None:
                    src_ref = output_net_to_node.get((prev_mod, resolved_signal))

            if src_ref is None:
                input_sources.append(
                    {
                        "port": pin_name,
                        "signal": raw_signal,
                        "resolved_signal": resolved_signal,
                        "source_kind": _classify_input_source(resolved_signal),
                    }
                )
                continue

            src_node_id, src_port = src_ref
            dedupe_key = (src_node_id, str(node["id"]), src_port, pin_name)
            if dedupe_key not in edge_pairs:
                edge_pairs.add(dedupe_key)
                edges.append(
                    {
                        "src": src_node_id,
                        "dst": str(node["id"]),
                        "src_port": src_port,
                        "dst_port": pin_name,
                        "signal": raw_signal,
                        "resolved_signal": resolved_signal,
                        "kind": _edge_kind(src_port=src_port, dst_port=pin_name, resolved_signal=resolved_signal),
                    }
                )
            input_sources.append(
                {
                    "port": pin_name,
                    "signal": raw_signal,
                    "resolved_signal": resolved_signal,
                    "source_kind": "node",
                    "source_node": src_node_id,
                    "source_port": src_port,
                }
            )
        node["input_sources"] = input_sources

    nodes.sort(key=lambda node: (int(node["stage"]), int(node["column"] if node["column"] is not None else -1), int(node["local_index"]), int(node["ordinal"])))
    node_id_to_index = {str(node["id"]): idx for idx, node in enumerate(nodes)}
    edges.sort(key=lambda edge: (node_id_to_index.get(str(edge["src"]), -1), node_id_to_index.get(str(edge["dst"]), -1), str(edge["dst_port"])))

    type_hist: dict[str, int] = {}
    for node in nodes:
        t = str(node["cell_type"])
        type_hist[t] = int(type_hist.get(t, 0) + 1)

    return {
        "schema": "compressor_instance_graph_v1",
        "design_id": design_id,
        "top": top,
        "n_bits": int(n_bits),
        "stage_count": len(stage_summaries),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "approx_node_count": sum(1 for node in nodes if bool(node["is_approx"])),
        "exact_node_count": sum(1 for node in nodes if not bool(node["is_approx"])),
        "cell_type_histogram": type_hist,
        "stages": stage_summaries,
        "nodes": nodes,
        "edges": edges,
    }


def extract_compressor_graph_from_rtl_path(
    rtl_path: Path,
    *,
    design_id: str,
    top: str,
    n_bits: int,
) -> dict[str, Any]:
    return extract_compressor_graph_from_rtl_text(
        Path(rtl_path).read_text(encoding="utf-8"),
        design_id=design_id,
        top=top,
        n_bits=n_bits,
    )


def write_compressor_graph_json(path: Path, graph: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
