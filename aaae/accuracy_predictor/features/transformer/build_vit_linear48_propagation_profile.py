#!/usr/bin/env python
"""Build the A100-canonical 48-node ViT-Base propagation profile.

The numerical probe implementation is shared with the audited Swin profile
builder.  This entry point binds it to the promoted ViT-Base linear48 dataset,
model, node registry, and frozen input contracts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.features.transformer import (
    build_swin_propagation_profile as profile,
)
from aaae.accuracy_predictor.features.transformer.common import FORMAL_CACHE_ROOT

DATASET_ROOT = (
    PROJECT_ROOT / "datasets" / "vit" / "s8s8_linear48_strict50_true1000_raw500_v1"
)
SHARED_ROOT = DATASET_ROOT / "shared"
DEFAULT_MANIFEST = DATASET_ROOT / "data" / "manifest.json"
DEFAULT_MODEL = (
    EA2_ROOT
    / "ai_models"
    / "onnx_models"
    / "classification"
    / "vit_base_patch16_224_imagenet1k_int8_dynamic.onnx"
)
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "resources"
    / "data"
    / "vit_base_patch16_224_s8s8_linear48"
    / "node_registry"
    / "node_registry.json"
)
DEFAULT_OUTPUT = FORMAL_CACHE_ROOT / "vit_linear48_a100_propagation_profile256.npz"
EXPECTED_DATASET_ID = "vit_base_patch16_224_s8s8_linear48_strict50_true1000_raw500_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must contain an object: {path}")
    return value


def _bind_vit_contract(args: argparse.Namespace) -> None:
    manifest = _read_object(args.dataset_manifest.expanduser().resolve())
    registry_path = args.node_registry.expanduser().resolve()
    registry = _read_object(registry_path)
    exact_contract = _read_object(args.exact_contract.expanduser().resolve())
    input_contract = _read_object(args.input_contract.expanduser().resolve())

    injection = manifest.get("injection_mode")
    if not isinstance(injection, dict):
        raise TypeError("ViT manifest has no injection_mode contract")
    node_count = int(injection.get("injected_matmul_count", -1))
    total_matmul_count = int(injection.get("total_matmul_count", -1))
    excluded = tuple(str(value) for value in injection.get("excluded_matmuls", []))
    if (
        manifest.get("dataset_id") != EXPECTED_DATASET_ID
        or manifest.get("model_key") != "vit_base_patch16_224"
        or node_count != 48
        or total_matmul_count != 72
        or len(excluded) != total_matmul_count - node_count
        or int(registry.get("node_count", -1)) != node_count
    ):
        raise ValueError("dataset manifest is not the promoted 48-node ViT contract")

    paths = {
        "model": args.model.expanduser().resolve(),
        "registry": registry_path,
        "inputs": args.inputs.expanduser().resolve(),
        "exact_baseline": args.exact_baseline.expanduser().resolve(),
        "exact_meta": args.exact_meta.expanduser().resolve(),
        "exact_contract": args.exact_contract.expanduser().resolve(),
        "input_contract": args.input_contract.expanduser().resolve(),
    }
    hashes = {name: profile._file_hashes(path) for name, path in paths.items()}
    if hashes["model"]["sha1"] != manifest.get("model_sha1"):
        raise ValueError("ViT manifest and ONNX model hashes differ")
    if hashes["registry"]["sha1"] != manifest.get("node_registry_sha1"):
        raise ValueError("ViT manifest and node registry hashes differ")
    if exact_contract.get("model_sha1") != hashes["model"]["sha1"]:
        raise ValueError("canonical exact contract is bound to another model")
    if (
        exact_contract.get("frozen_input_contract_sha1")
        != hashes["input_contract"]["sha1"]
    ):
        raise ValueError("canonical exact and input contracts are not bound together")

    profile.EXPECTED_NODE_COUNT = node_count
    profile.EXPECTED_MATMUL_COUNT = total_matmul_count
    profile.EXPECTED_EXCLUDED_MATMULS = excluded
    profile.EXPECTED_ASSET_HASHES = hashes
    profile.EXPECTED_EXACT_LOGITS_SHA1 = str(exact_contract["exact_logits_sha1"])
    profile.EXPECTED_TRUE1000_POSITIONS_SHA1 = str(
        input_contract["true1000_positions_sha1"]
    )


def parser() -> argparse.ArgumentParser:
    result = profile.parser()
    result.set_defaults(
        model=DEFAULT_MODEL,
        node_registry=DEFAULT_REGISTRY,
        inputs=SHARED_ROOT / "union_inputs.npy",
        exact_baseline=SHARED_ROOT / "canonical_exact_baseline.npz",
        exact_meta=SHARED_ROOT / "true1000_exact_meta.npz",
        exact_contract=SHARED_ROOT / "canonical_exact_contract.json",
        input_contract=SHARED_ROOT / "input_contract.json",
        output=DEFAULT_OUTPUT,
        image_count=256,
        device_id=3,
    )
    result.add_argument("--dataset-manifest", type=Path, default=DEFAULT_MANIFEST)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        print(json.dumps(profile._self_test(), indent=2, sort_keys=True))
        return 0
    _bind_vit_contract(args)
    if not 1 <= int(args.image_count) <= profile.EXPECTED_TRUE1000_COUNT:
        raise SystemExit("--image-count must be in [1, 1000]")
    if int(args.batch_size) != 16:
        raise SystemExit("--batch-size must remain 16")
    if int(args.device_id) < 0:
        raise SystemExit("--device-id must be non-negative")
    if int(args.probe_magnitude) != 1:
        raise SystemExit("--probe-magnitude must remain 1")
    print(
        json.dumps(profile.execute(args), ensure_ascii=False, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
