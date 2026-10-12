"""Small entry points for the retained accuracy and optimization implementations."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def _invoke(module_name: str, arguments: list[str]) -> int:
    module = importlib.import_module(module_name)
    previous = sys.argv
    sys.argv = [module_name, *arguments]
    try:
        return int(module.main() or 0)
    finally:
        sys.argv = previous


def train_accuracy(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the retained model-specific AAAE accuracy predictor. "
        "Use --model MODEL -- --help for the implementation's training options."
    )
    parser.add_argument("--model", required=True,
                        choices=["swin_tiny", "vit_base", "deit_base", "resnet50", "mobilenetv3"])
    options, remaining = parser.parse_known_args(arguments)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if options.model == "swin_tiny":
        module = "aaae.accuracy_predictor.transformer.training.train_swin_linear48_causal_prefix_transfer"
    elif options.model in {"vit_base", "deit_base"}:
        module = "aaae.accuracy_predictor.training.train_structure_aware_linear48"
        model_key = {"vit_base": "vit_base_patch16_224", "deit_base": "deit_base_patch16_224"}[options.model]
        if "--model-key" not in remaining:
            remaining = ["--model-key", model_key, *remaining]
    else:
        head = "delta" if options.model == "resnet50" else "correctness"
        module = f"aaae.accuracy_predictor.training.search_cnn_{head}_conv"
    return _invoke(module, remaining)


def prepare_features(arguments: list[str] | None = None) -> int:
    modules = {
        "cnn": "aaae.accuracy_predictor.training.cache_cnn_features",
        "conditioned": "aaae.accuracy_predictor.features.transformer.prepare_swin_linear48_causal_prefix_features",
        "operand": "aaae.accuracy_predictor.features.transformer.build_swin_per_image_operand_profile",
        "propagation-swin": "aaae.accuracy_predictor.features.transformer.build_swin_linear48_propagation_profile",
        "propagation-vit": "aaae.accuracy_predictor.features.transformer.build_vit_linear48_propagation_profile",
        "propagation-deit": "aaae.accuracy_predictor.features.transformer.build_deit_linear48_propagation_profile",
        "transformer-observations": "aaae.inference_runtime.transformer.build_swin_s8s8_strict50_dataset",
        "multiplier": "aaae.multiplier_generator.materialize_asset",
    }
    parser = argparse.ArgumentParser(description="Prepare features, clean profiles or response observations.")
    parser.add_argument("--kind", required=True, choices=sorted(modules))
    options, remaining = parser.parse_known_args(arguments)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    return _invoke(modules[options.kind], remaining)


def build_accuracy_bundle(arguments: list[str] | None = None) -> int:
    modules = {
        "cnn-hybrid": "aaae.accuracy_predictor.training.build_cnn_hybrid_bundle",
        "cnn-composite": "aaae.accuracy_predictor.training.build_cnn_composite_bundle",
        "cnn-disagreement": "aaae.accuracy_predictor.training.build_cnn_disagreement_bundle",
        "swin": "aaae.accuracy_predictor.training.promote_swin_linear48_joint4497_package",
    }
    parser = argparse.ArgumentParser(description="Package trained members using the retained combination implementation.")
    parser.add_argument("--kind", required=True, choices=sorted(modules))
    options, remaining = parser.parse_known_args(arguments)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    return _invoke(modules[options.kind], remaining)


def evaluate_accuracy(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate paired Top-1 drop predictions in percentage points.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--actual-column", default="actual")
    parser.add_argument("--prediction-column", default="pred")
    parser.add_argument("--units", choices=["proportion", "pp"], default="proportion")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(arguments)
    import pandas as pd
    from .evaluation.metrics import accuracy_drop_metrics

    frame = pd.read_csv(options.predictions)
    if "target" in frame:
        frame = frame.loc[frame["target"].eq("top1_drop")]
    values = accuracy_drop_metrics(frame[options.actual_column], frame[options.prediction_column],
                                   units=options.units)
    payload = json.dumps(values, indent=2, allow_nan=False)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(payload + "\n")
    print(payload)
    return 0


def predict_accuracy(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict Top-1 drop with a retained AAAE model bundle.")
    parser.add_argument("--family", choices=["transformer", "cnn"], required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["train", "refit", "deployment"], default="deployment")
    options = parser.parse_args(arguments)
    if options.family == "transformer":
        from .accuracy_predictor.transformer.inference.strict50_prediction import predict_top1
        result = predict_top1(package_dir=options.package, output_csv=options.output,
                              features_path=options.features, manifest_path=options.manifest, scope=options.scope)
    else:
        if options.manifest is None:
            parser.error("CNN prediction requires --manifest")
        from .accuracy_predictor.cnn.manifests import load_prediction_designs
        from .accuracy_predictor.cnn.workflow import load_bundle
        designs, _ = load_prediction_designs(options.manifest)
        table = load_bundle(options.package).predict(designs)
        options.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_frame().to_csv(options.output, index=False)
        result = {"family": "cnn", "package": str(options.package),
                  "row_count": len(table.design_ids), "output_csv": str(options.output)}
    print(json.dumps(result, default=str, indent=2))
    return 0


def run_module(module_name: str, arguments: list[str] | None = None) -> int:
    return _invoke(module_name, list(sys.argv[1:] if arguments is None else arguments))
