#!/usr/bin/env python
"""Pack standalone neural CNN ensemble members into one checked composite file."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.cnn import correctness_conv, delta_conv  # noqa: E402
from aaae.accuracy_predictor.cnn.artifacts import (  # noqa: E402
    COMPOSITE_BUNDLE_SCHEMA_VERSION,
    HYBRID_BUNDLE_SCHEMA_VERSION,
    POSTPROCESSED_BUNDLE_SCHEMA_VERSION,
    _sha256,
    _write_checksums,
    _write_json,
    ensure_output_available,
    load_artifact,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("composite bundle build manifest must be a JSON object")
    return payload


def _resolve(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _relative_model_path(value: object, *, target: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("composite_checkpoint must be a nonempty path string")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:2] != ("models", target)
        or path.suffix != ".pt"
    ):
        raise ValueError(
            f"composite_checkpoint must be a safe models/{target}/*.pt path"
        )
    return path


def build(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    raw = _read_json(manifest_path)
    if raw.get("schema_version") != 1:
        raise ValueError("composite bundle build manifest schema_version must be 1")
    base_dir = manifest_path.parent
    base_bundle = _resolve(base_dir, raw.get("base_bundle"), field="base_bundle")
    output = _resolve(base_dir, raw.get("output"), field="output")
    target = raw.get("target", "top1_drop")
    if not isinstance(target, str) or target not in {"top1_drop", "top5_drop"}:
        raise ValueError("target must be top1_drop or top5_drop")
    composite_relative = _relative_model_path(
        raw.get("composite_checkpoint"),
        target=str(target),
    )
    ensure_output_available(output)

    artifact = load_artifact(base_bundle)
    if artifact.manifest.get("schema_version") not in {
        HYBRID_BUNDLE_SCHEMA_VERSION,
        POSTPROCESSED_BUNDLE_SCHEMA_VERSION,
    }:
        raise ValueError("base_bundle must be a checked schema-v3 or schema-v4 bundle")
    manifest = _read_json(base_bundle / "manifest.json")
    target_raw = manifest.get("model_ensembles", {}).get(target)
    if not isinstance(target_raw, dict):
        raise TypeError(f"base bundle does not define target {target!r}")
    members = target_raw.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("composite target must contain at least two members")
    if any(
        not isinstance(member, dict)
        or member.get("kind") not in {"delta_conv", "correctness_conv"}
        or not isinstance(member.get("name"), str)
        or not member.get("name")
        or not isinstance(member.get("path"), str)
        or member.get("checkpoint_member") is not None
        for member in members
    ):
        raise ValueError(
            "every composite target member must be a standalone neural checkpoint"
        )
    member_kinds = {str(member["kind"]) for member in members}
    if len(member_kinds) != 1:
        raise ValueError("composite target members must all use the same neural kind")
    member_kind = member_kinds.pop()
    if member_kind == "delta_conv":
        composite_schema = delta_conv.COMPOSITE_CHECKPOINT_SCHEMA_VERSION
        composite_exporter = delta_conv.export_delta_conv_composite_checkpoint
    else:
        composite_schema = correctness_conv.COMPOSITE_CHECKPOINT_SCHEMA_VERSION
        composite_exporter = (
            correctness_conv.export_correctness_conv_composite_checkpoint
        )
    names = [str(member["name"]) for member in members]
    if len(set(names)) != len(names):
        raise ValueError("composite target member names must be unique")
    source_paths = {
        str(member["name"]): (base_bundle / str(member["path"])).resolve()
        for member in members
    }
    if len(set(source_paths.values())) != len(source_paths):
        raise ValueError("base neural members must use separate files")
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shutil.copytree(base_bundle, temporary, dirs_exist_ok=True)
        composite_path = temporary / composite_relative
        composite_exporter(
            source_paths,
            composite_path,
            expected_feature_count=len(artifact.feature_names),
            expected_feature_schema=str(artifact.manifest["feature_schema"]),
        )
        composite_sha256 = _sha256(composite_path)
        for member in members:
            standalone = temporary / str(member["path"])
            standalone.unlink()
            name = str(member["name"])
            member["path"] = composite_relative.as_posix()
            member["checkpoint_member"] = name
            member["checkpoint_schema"] = composite_schema
            member["standalone_checkpoint_sha256"] = source_hashes[name]
            member["composite_checkpoint_sha256"] = composite_sha256
        target_raw["checkpoint_packaging"] = {
            "kind": composite_schema,
            "path": composite_relative.as_posix(),
            "sha256": composite_sha256,
            "members": names,
            "prediction_semantics_changed": False,
        }
        manifest["schema_version"] = COMPOSITE_BUNDLE_SCHEMA_VERSION
        manifest["created_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        training = manifest.setdefault("training", {}).setdefault(str(target), {})
        training.update(
            {
                "checkpoint_packaging": composite_schema,
                "checkpoint_packaging_member_count": len(names),
                "prediction_parameters_changed": False,
            }
        )
        _write_json(temporary / "manifest.json", manifest)

        metrics = _read_json(temporary / "metrics.json")
        target_metrics = metrics.get(target)
        if not isinstance(target_metrics, dict):
            raise TypeError(f"metrics.json does not define target {target!r}")
        target_metrics["checkpoint_packaging"] = {
            "kind": composite_schema,
            "member_count": len(names),
            "prediction_semantics_changed": False,
        }
        _write_json(temporary / "metrics.json", metrics)

        provenance = _read_json(temporary / "provenance.json")
        provenance["composite_checkpoint_packaging"] = {
            "schema": composite_schema,
            "member_kind": member_kind,
            "target": target,
            "members": names,
            "source_checkpoint_sha256": source_hashes,
            "output_checkpoint_sha256": composite_sha256,
            "labels_used": False,
            "parameters_changed": False,
        }
        provenance["composite_base_bundle"] = str(base_bundle)
        provenance["composite_base_bundle_manifest_sha256"] = _sha256(
            base_bundle / "manifest.json"
        )
        _write_json(temporary / "provenance.json", provenance)
        _write_json(temporary / "training_manifest.snapshot.json", raw)
        _write_checksums(temporary)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    loaded = load_artifact(output)
    return {
        "bundle": str(output),
        "bundle_manifest_sha256": _sha256(output / "manifest.json"),
        "bundle_schema_version": loaded.manifest["schema_version"],
        "composite_checkpoint": composite_relative.as_posix(),
        "composite_checkpoint_sha256": composite_sha256,
        "member_kind": member_kind,
        "logical_members": names,
        "physical_checkpoint_count": 1,
        "prediction_semantics_changed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
