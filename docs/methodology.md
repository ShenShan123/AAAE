# Methodology and Paper Correspondence

This document connects the implementation to *AAAE: Application Accuracy-Aware Evaluation Framework for Approximate Computing*.

## Multiplier Generation and Error Characterization — Section 2.1

`aaae/multiplier_generator/` contains the fixed-topology RTL generator, compressor graph extraction, tensor truth-table executor, and signed-domain conversion.

A design selects one of eight approximate compressor cells at each of nine low-order positions. Eight high-order positions retain exact compression. The unsigned core is expanded into a signed S8×S8 sign-magnitude table, and the residual is measured against the exact signed product. Runtime LUT axes use signed codes plus 128; zero-point masking follows the quantized operator contract.

## Accuracy Prediction — Section 2.2

| Representation | Implementation | Dimension |
| --- | --- | --- |
| Static error features `s(a)` | `aaae/accuracy_predictor/features/static/` | 3103 |
| Transformer-conditioned features `cT(m,a)` | `aaae/accuracy_predictor/features/transformer/` | 1034 |
| Transformer hybrid response `r50(m,a)` | `aaae/accuracy_predictor/transformer/proxy_training/strict50_calibration.py` | 360 |
| Fused Transformer input | `aaae/accuracy_predictor/transformer/` | 4497 |

The static representation combines four error-surface channels, row/column/grid projections, global statistics, and generator descriptors. Transformer conditioning contracts error surfaces with precomputed input-pair distributions and propagates their effects through clean-model sensitivity and margin summaries.

The fixed response set contains 50 labeled images. Ordered correctness transitions and margin changes supply the observed response; two residuals against static priors extend the 358-dimensional observed response to a 360-dimensional hybrid response.

Each Transformer has its own LightGBM regressor. ViT and DeiT use `accuracy_predictor/training/train_structure_aware_linear48.py`; Swin retains the training and packaging source that produced its joint4497 predictor. The 3463-dimensional base response stage and the final 4497-dimensional conditioned predictor are separate stages.

### CNN Extension — Appendix B.4

The CNN code is in `aaae/accuracy_predictor/cnn/`, with clean-model profiling in `accuracy_predictor/features/cnn/`.

| Model | Engineered features | Residual encoding | Prediction head |
| --- | --- | --- | --- |
| ResNet-50 | 3185 | `2×129×129` | Two-member Delta-CNN |
| MobileNetV3-Large | 3233 | `2×129×129` | Two-member Correctness-CNN with disagreement postprocessing |

CNNs do not run a candidate-level response set. Member training, combination, and postprocessing remain distinct operations, and the complete bundle loader preserves their parameters.

### Full Accuracy Measurement — Section 2.2.5

`aaae/inference_runtime/` contains the Python graph rewrites, CUDA custom operators, session handling, and dataset/label construction code. A candidate LUT is applied to all 48 weight-bearing Linear MatMuls in a Transformer, 53 Conv nodes in ResNet-50, or 62 Conv nodes in MobileNetV3-Large.

The fixed evaluation protocol compares approximate inference with the exact quantized baseline on the same 1000 labeled images. Stored drops use proportion units; the paper reports percentage points.

## Hardware-Cost Prediction and Verification — Section 2.3

`aaae/multiplier_gnn/` encodes the compressor configuration and signal dependencies, applies stage-wise GraphSAGE, and predicts power, delay, and area. PDP is computed as power multiplied by delay.

`aaae/hardware_evaluation/` contains the no-macro RTL templates, Yosys synthesis script, OpenSTA timing/power scripts, VCS testbench flow, and result extraction. Technology libraries and linked runtime objects are external resources; generated work files use configured paths.

## Accuracy-Constrained Design Optimization — Section 2.4 and Appendix D

Two workflows are retained:

1. `aaae/optimization_algorithm/` implements the Swin three-objective BO and NSGA-II search paths, using the accuracy and hardware predictors.
2. `aaae/calibration/` and `aaae/design_selection/` implement independent A1 calibration and fixed-pool selection.

For A1, positive calibration residuals are weighted by target-pool versus calibration stratum shares. Their empirical weighted quantile gives `q`, and admission uses `U = predicted_drop_pp + q ≤ τ`. The implementation retains the original smoothing and empirical quantile rule. This is an empirical correction without a finite-sample coverage guarantee.

Admitted candidates are ordered by predicted PDP with stable configuration-hash ties. At most `K` full accuracy labels are queried. The feasible queried design with the smallest predicted PDP is selected; an empty admission returns abstention, and no feasible queried design returns failed selection. Subsequent verification does not change the single-run choice.

## Experiments and Protocols

- **Table 1 / Section 3.2:** five model-specific accuracy predictors, evaluated on 3000 structural hold-out candidates per model.
- **Section 3.1 / Appendix B.6:** 20000 designs split into 14000 Train, 3000 Validation, and 3000 Test; 3000 independently generated Calibration candidates.
- **Section 3.6 / Figures 4–5:** three-objective BO and NSGA-II, with full accuracy and hardware verification.

The retained implementation excludes baseline experiments, feature-removal experiments, response-budget sweeps, and the old 30k bi-objective reference search.

See [usage.md](usage.md) for model-specific training, model-package loading, and external resource configuration.
