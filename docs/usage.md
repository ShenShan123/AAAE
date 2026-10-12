# Usage

Run examples from the repository root. Each entry point exposes `--help`; model/family dispatchers forward implementation-specific options after `--`.

## Resource Configuration

`configs/paths.json` controls the external resource tree and new output directory. Relative values are resolved against the AAAE root:

```json
{
  "external_root": "../EA_2",
  "output_root": "output",
  "ppa_root": "../ppa"
}
```

| Environment variable | Purpose |
| --- | --- |
| `AAAE_PATH_CONFIG` | Alternative path configuration file |
| `AAAE_EXTERNAL_ROOT` | External experiment assets retaining their original subdirectory layout |
| `AAAE_OUTPUT_ROOT` | Generated features, models, search results, and verification outputs |
| `AAAE_AI_MODELS_ROOT` | ONNX and inference image resources |
| `AAAE_GNN_ASSET_ROOT` | Hardware-GNN weights, `Graph.csv`, and `FEATURE.csv` |
| `AAAE_PPA_ROOT` | External no-macro technology resources |
| `AAAE_ORT_INCLUDE_ROOT` | ONNX Runtime headers for the Transformer operator build |
| `AAAE_CNN_ORT_INCLUDE_ROOT` | ONNX Runtime headers for the CNN operator build |

Inputs are referenced externally, and default generated outputs stay under the configured output root. No dataset or checkpoint is bundled. Each model package supplies its feature schema, member weights, and prediction parameters.

## Feature Preparation

`examples/prepare_features.py --kind KIND -- ...` supports:

| Kind | Operation |
| --- | --- |
| `multiplier` | Materialize a signed low-only multiplier LUT asset |
| `transformer-observations` | Construct the formal Linear48 response and full-label dataset |
| `operand` | Build the per-image operand profile |
| `propagation-swin`, `propagation-vit`, `propagation-deit` | Build the clean sensitivity profile |
| `conditioned` | Build the Swin Linear48 conditioned features |
| `cnn` | Cache CNN engineered features |

```bash
python examples/prepare_features.py --kind propagation-vit -- --help
python examples/prepare_features.py --kind transformer-observations -- --help
```

The source modules under `aaae/accuracy_predictor/features/` also expose the shared functions needed for candidate-time conditioning.

## Accuracy Prediction

### Training

```bash
python examples/train_accuracy.py --model swin_tiny -- --help
python examples/train_accuracy.py --model vit_base -- --help
python examples/train_accuracy.py --model deit_base -- --help
python examples/train_accuracy.py --model resnet50 -- --help
python examples/train_accuracy.py --model mobilenetv3 -- --help
```

The Transformer base response/prior stage is available through `examples/train_response_prior.py`. The final Transformer path adds conditioned features to the base representation. Its fixed response images, prior predictions, refit protocol, and feature order must match the package's schema.

CNN training operates on member predictors. Build a final bundle with the appropriate combination/postprocessing step:

```bash
python examples/build_accuracy_bundle.py --kind cnn-hybrid -- --help
python examples/build_accuracy_bundle.py --kind cnn-composite -- --help
python examples/build_accuracy_bundle.py --kind cnn-disagreement -- --help
```

### Inference

Transformer inference accepts either a package-ready feature matrix or a runtime-observation manifest:

```bash
python examples/predict_accuracy.py \
  --family transformer \
  --package /path/to/strict50_package \
  --manifest /path/to/prediction_manifest.json \
  --output output/accuracy/predictions.csv
```

CNN inference takes a design manifest and the complete trained bundle:

```bash
python examples/predict_accuracy.py \
  --family cnn \
  --package /path/to/cnn_bundle \
  --manifest /path/to/prediction_manifest.json \
  --output output/accuracy/predictions.csv
```

### Metrics

```bash
python examples/evaluate_accuracy.py \
  --predictions /path/to/paired_predictions.csv \
  --actual-column actual \
  --prediction-column pred \
  --units proportion \
  --output output/accuracy/metrics.json
```

The CSV must pair measured and predicted drops; a prediction-only CSV needs to be joined with the measured labels first. If `target` is present, only `top1_drop` rows are evaluated. Reported MAE, P95, and maximum error use percentage points. Negative measured drops are retained; MAPE excludes measured zero drops only.

## Iterative Optimization

```bash
python examples/run_bo.py --help
python examples/run_nsga2.py --help
```

The retained iterative paths use Swin-Tiny Linear48 accuracy prediction and the hardware GNN. Their objectives are Top-1 accuracy drop, PDP, and area. Budget, seed, device assignment, candidate pool, and initialization options are exposed by each implementation; configure them explicitly when reproducing an experimental run.

These optimizers and fixed-pool A1 selection remain separate implementations. The latter has its own independent calibration inputs and query budget.

## Calibration and Selection

### A1 Correction

`examples/calibrate.py` expects:

| File | Required columns | Units |
| --- | --- | --- |
| Calibration CSV | `y`, `pred`, `bin` | Accuracy drops in percentage points |
| Target-pool CSV | `bin` | Same frozen stratum encoding |

`examples/score_calibration_bins.py` can compute the original frozen proxy strata from external model and feature assets.

```bash
python examples/calibrate.py \
  --model swin_tiny \
  --calibration /path/to/calibration.csv \
  --pool /path/to/pool.csv \
  --alpha 0.10 \
  --output output/calibration/a1.json
```

The correction uses independent calibration labels and target-pool covariates. Refit it when the pool or its stratum distribution changes.

### Query Planning and Selection

The candidate CSV contains `config_hash`, `yhat_ours` in percentage points, and positive `pdp_fj`:

```bash
python examples/select_design.py \
  --pool /path/to/pool.csv \
  --correction output/calibration/a1.json \
  --tau 2.0 \
  --budget 10 \
  --output-dir output/selection/run
```

Without `--labels`, the script writes the admitted query plan. An explicit `--labels` file with `config_hash,y` enables replay using measured labels. To query a live evaluator from Python, use:

```python
from aaae.design_selection.core import select_with_budget

result = select_with_budget(
    predictions_pp, predicted_pdp_fj, config_hashes,
    evaluate_drop_pp,
    correction_pp=q,
    tau_pp=2.0,
    budget=10,
)
```

The callback receives the chosen candidate position and returns its full-set accuracy drop in percentage points. Selection does not use unqueried labels.

## Verification

```bash
python examples/verify_designs.py --family search -- --help
python examples/verify_designs.py --family transformer -- --help
python examples/verify_designs.py --family cnn -- --help
python examples/run_ppa.py --help
```

Accuracy verification accepts explicit candidates and their matching dataset/measurement contracts. The no-macro hardware flow uses the local RTL and flow templates plus external technology resources:

```bash
python examples/run_ppa.py \
  --source-csv /path/to/selected_candidates.csv \
  --output-csv output/hardware/verified_candidates.csv \
  --jobs 1
```

The source candidate table must follow the retained configuration/hash and compressor-vector schema exposed by the hardware evaluator. The flow performs synthesis, timing, gate-level simulation, and power extraction. Verification records the selected design's result without revising the prior single-run choice.
