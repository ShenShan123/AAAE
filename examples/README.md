# Workflow Examples

These Python entry points use the implementation in `../aaae/`. Run them from the repository root; `_bootstrap.py` makes the checkout available to each process and its workers.

| Entry point | Operation |
| --- | --- |
| `generate_multiplier.py` | Generate an 8-bit low-only approximate multiplier RTL |
| `prepare_features.py` | Prepare features, clean profiles, response observations, and multiplier assets |
| `train_response_prior.py` | Train Transformer static priors and the base 50-image response model |
| `train_accuracy.py` | Dispatch model-specific accuracy-predictor training |
| `build_accuracy_bundle.py` | Combine trained CNN members and package predictors |
| `predict_accuracy.py` | Predict drops from a Transformer/CNN model package |
| `evaluate_accuracy.py` | Compute accuracy-drop prediction metrics |
| `run_bo.py` | Run the three-objective Swin Bayesian optimizer |
| `run_nsga2.py` | Run the three-objective Swin NSGA-II optimizer |
| `score_calibration_bins.py` | Compute frozen calibration proxy strata |
| `calibrate.py` | Fit the A1 pool-weighted correction |
| `select_design.py` | Plan bounded queries or replay fixed-pool selection |
| `verify_designs.py` | Rerun selected designs through full accuracy evaluation |
| `run_ppa.py` | Verify hardware cost using the no-macro flow |

```bash
python examples/generate_multiplier.py --help
python examples/train_accuracy.py --model vit_base -- --help
python examples/run_bo.py --help
```

Examples reference external resources configured in `configs/paths.json`; no dataset or checkpoint is included here. See the [quick start](../README.md#-quick-start) and [detailed usage](../docs/usage.md).
