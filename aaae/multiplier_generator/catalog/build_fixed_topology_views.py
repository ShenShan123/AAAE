from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


from aaae.multiplier_generator.catalog.schema import (  # noqa: E402
    COMPRESSOR_NAME_TO_ID,
    FULL24_COUNT,
    HIGH_COMPRESSOR_COUNT,
    LOW_COMPRESSOR_COUNT,
    NODE_KIND_TO_ID,
    TOPOLOGY_KEY,
)


from aaae.paths import EA2_ROOT, LEGACY_WORKFLOW_ROOT

PROJECT_ROOT = LEGACY_WORKFLOW_ROOT
APPROX_LIB_DIR = EA2_ROOT / "approx_lib"


def _ensure_approx_lib_import() -> None:
    approx_lib_path = str(APPROX_LIB_DIR.resolve())
    if approx_lib_path not in sys.path:
        pass  # Local package imports are resolved by the entry point.


def _reverse_compressor_mapping() -> dict[str, int]:
    return {
        "ACCI1": 1,
        "kong2": 2,
        "antonio": 3,
        "momeni": 4,
        "ha_approx": 5,
        "akbar1": 6,
        "akbar2": 7,
        "sabetz": 8,
    }


def _compressor_family_id(cell_type: str) -> int:
    if cell_type.startswith("AE_4_2_"):
        base_name = cell_type.removeprefix("AE_4_2_")
        return _reverse_compressor_mapping()[base_name]
    try:
        return int(COMPRESSOR_NAME_TO_ID[cell_type])
    except KeyError as exc:
        raise KeyError(f"Unsupported cell type for family id: {cell_type}") from exc


def build_shared_topology() -> dict[str, Any]:
    _ensure_approx_lib_import()
    from aaae.multiplier_generator.compressor_graph import extract_compressor_graph_from_rtl_text  # type: ignore
    from aaae.multiplier_generator.nbit_approx import MultiplierGeneratorArbitrary  # type: ignore

    gen = MultiplierGeneratorArbitrary(
        n_bits=8,
        compressor_types=[1] * LOW_COMPRESSOR_COUNT,
        high_compressor_types=[1] * HIGH_COMPRESSOR_COUNT,
        approx_high_part=True,
    )
    rtl = gen.generate_verilog_code()
    graph = extract_compressor_graph_from_rtl_text(
        rtl,
        design_id="topology_reference",
        top="Appro4_2_8bit",
        n_bits=8,
    )
    nodes = list(graph["nodes"])
    edges = list(graph["edges"])
    if len(nodes) != FULL24_COUNT:
        raise RuntimeError(f"Expected {FULL24_COUNT} fixed nodes, got {len(nodes)}")

    node_index = {str(node["id"]): idx for idx, node in enumerate(nodes)}
    low_var_indices: list[int] = []
    high_var_indices: list[int] = []
    node_kind_template = np.empty((FULL24_COUNT,), dtype=np.uint8)
    default_family_template = np.zeros((FULL24_COUNT,), dtype=np.uint8)
    high_part_flag = np.zeros((FULL24_COUNT,), dtype=np.uint8)

    topology_nodes: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        kind = str(node["kind"])
        cell_type = str(node["cell_type"])
        if kind == "approx_4_2":
            low_var_indices.append(idx)
        elif kind == "approx_4_2_wrapper":
            high_var_indices.append(idx)
            high_part_flag[idx] = 1
        node_kind_template[idx] = NODE_KIND_TO_ID[kind]
        if kind in {"approx_4_2", "approx_4_2_wrapper"}:
            default_family_template[idx] = 0
        else:
            default_family_template[idx] = _compressor_family_id(cell_type)
        topology_nodes.append(
            {
                "position_id": idx,
                "id": str(node["id"]),
                "name": str(node["name"]),
                "kind": kind,
                "cell_type_reference": cell_type,
                "stage": int(node["stage"]),
                "column": int(node["column"]),
                "local_index": int(node["local_index"]),
                "ordinal": int(node["ordinal"]),
                "is_variable_slot": bool(kind in {"approx_4_2", "approx_4_2_wrapper"}),
                "high_part_flag": int(high_part_flag[idx]),
            }
        )

    if len(low_var_indices) != LOW_COMPRESSOR_COUNT:
        raise RuntimeError(f"Expected {LOW_COMPRESSOR_COUNT} low variable slots, got {len(low_var_indices)}")
    if len(high_var_indices) != HIGH_COMPRESSOR_COUNT:
        raise RuntimeError(f"Expected {HIGH_COMPRESSOR_COUNT} high variable slots, got {len(high_var_indices)}")

    edge_index = np.empty((2, len(edges)), dtype=np.int16)
    topology_edges: list[dict[str, Any]] = []
    for edge_idx, edge in enumerate(edges):
        src = node_index[str(edge["src"])]
        dst = node_index[str(edge["dst"])]
        edge_index[0, edge_idx] = src
        edge_index[1, edge_idx] = dst
        topology_edges.append(
            {
                "src_position_id": src,
                "dst_position_id": dst,
                "src_port": str(edge["src_port"]),
                "dst_port": str(edge["dst_port"]),
                "signal": str(edge["signal"]),
                "resolved_signal": str(edge["resolved_signal"]),
                "kind": str(edge["kind"]),
            }
        )

    return {
        "topology_key": TOPOLOGY_KEY,
        "n_bits": 8,
        "node_count": FULL24_COUNT,
        "edge_count": int(len(edges)),
        "low_var_indices": low_var_indices,
        "high_var_indices": high_var_indices,
        "var17_indices": low_var_indices + high_var_indices,
        "node_kind_template": node_kind_template,
        "default_family_template": default_family_template,
        "high_part_flag": high_part_flag,
        "topology_json": {
            "topology_key": TOPOLOGY_KEY,
            "n_bits": 8,
            "node_count": FULL24_COUNT,
            "edge_count": int(len(edges)),
            "nodes": topology_nodes,
            "edges": topology_edges,
        },
        "node_static_meta_json": {
            "topology_key": TOPOLOGY_KEY,
            "node_kind_to_id": NODE_KIND_TO_ID,
            "compressor_name_to_id": COMPRESSOR_NAME_TO_ID,
            "low_var_indices": low_var_indices,
            "high_var_indices": high_var_indices,
            "var17_indices": low_var_indices + high_var_indices,
            "nodes": topology_nodes,
        },
        "edge_index": edge_index,
    }


def write_shared_topology(output_dir: Path) -> dict[str, Any]:
    shared = build_shared_topology()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "topology_full24.json").write_text(
        json.dumps(shared["topology_json"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "node_static_meta.json").write_text(
        json.dumps(shared["node_static_meta_json"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.save(output_dir / "edge_index.npy", shared["edge_index"])
    return shared


def main() -> int:
    shared_dir = PROJECT_ROOT / "datasets" / "D_mult" / "shared"
    write_shared_topology(shared_dir)
    print(shared_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
