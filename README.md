# AAAE: Application Accuracy-Aware Evaluation Framework for Approximate Computing

This repository provides the implementation of **AAAE**, a framework for evaluating and optimizing approximate multipliers under application-level accuracy constraints.

AAAE predicts the accuracy impact of a multiplier on a target neural network, estimates its hardware cost, and supports the search and verification of efficient designs. The implementation covers three vision Transformers and two CNNs, with model-specific prediction paths and a shared multiplier representation.

The current repository contains source code, command-line examples, and configuration files. Datasets, pretrained models, checkpoints, and technology libraries are supplied separately through configurable external paths.

---

## 🔍 Overview

Approximate multiplication can reduce hardware cost, but the resulting inference accuracy depends on both the multiplier's error behavior and the target model. Circuit-level error statistics alone do not capture these interactions.

AAAE connects four components:

1. **Multiplier generation and error characterization** — generate configurable 8-bit multipliers and construct their complete signed error surfaces.
2. **Application accuracy prediction** — combine structured error features with model-conditioned information; Transformers additionally use a fixed 50-image inference response.
3. **Hardware-cost prediction** — encode multiplier structures as graphs and estimate power, delay, area, and power–delay product (PDP).
4. **Design optimization and verification** — explore multiplier configurations, apply calibrated admission to fixed candidate pools, and verify selected designs through inference and EDA.

---

## ✨ Highlights

### Model-Specific Accuracy Predictors

- Independent LightGBM predictors for Swin-Tiny, ViT-Base, and DeiT-Base.
- Multi-scale static features, model-conditioned features, and lightweight labeled inference responses.
- Delta-CNN and Correctness-CNN prediction paths for ResNet-50 and MobileNetV3-Large.

### Configurable Approximate Multipliers

- Eight approximate compressor choices at nine low-order positions.
- Exact high-order compressors within a fixed reduction topology.
- Complete truth-table evaluation and deterministic S8×S8 residual LUTs.

### Graph-Based Hardware Estimation

- Fixed-topology graph encoding with compressor behavior and signal dependencies.
- Stage-wise GraphSAGE prediction of power, delay, and area.
- PDP-based ranking and a no-macro hardware verification flow.

### Accuracy-Aware Design Exploration

- Three-objective Bayesian optimization and NSGA-II for the implemented Swin search path.
- Independent A1 empirical calibration and fixed-pool budgeted selection for the supported models.
- Explicit query budgets, abstention, and post-selection verification.

---

## 📁 Repository Structure

```text
AAAE/
├── aaae/
│   ├── multiplier_generator/       # RTL generation, truth tables, signed LUTs
│   ├── accuracy_predictor/         # Transformer/CNN predictors and training
│   │   ├── features/               # Static features and model conditioning
│   │   ├── transformer/            # LightGBM, response features, model packages
│   │   ├── cnn/                    # Delta-CNN and Correctness-CNN bundles
│   │   └── training/               # Model training and member combination
│   ├── inference_runtime/          # LUT injection, CUDA operators, accuracy measurement
│   ├── multiplier_gnn/             # Graph encoding and hardware-cost prediction
│   ├── hardware_evaluation/        # RTL, synthesis, timing, and power verification
│   ├── optimization_algorithm/     # Bayesian optimization, NSGA-II, shared utilities
│   ├── calibration/                # A1 empirical upper corrections
│   ├── design_selection/           # Accuracy admission and budgeted selection
│   └── evaluation/                 # Prediction metrics and accuracy verification
├── examples/                       # Executable workflow entry points
├── docs/                           # Methodology, usage, implementation correspondence
├── configs/                        # External resource and output paths
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── README.md
```

The main implementation lives directly under `aaae/`. Executable examples are collected under `examples/`, while detailed schemas and paper correspondence are documented under `docs/`.

---

## ⚙️ Installation

Use a Python environment with Python 3.10 or newer. From the repository root:

```bash
conda activate pytorch
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If the required packages are already installed, the example scripts can also run directly from the checkout without installing the package.

External tools are needed for the corresponding stages:

| Stage | Requirements |
| --- | --- |
| Approximate inference | CUDA, nvcc, and ONNX Runtime GPU with matching SDK headers |
| Transformer CUDA operator build | CMake |
| Hardware verification | Yosys, OpenSTA, VCS, and Nangate standard-cell resources |

Configure datasets, model packages, and tool resources in [`configs/paths.json`](configs/paths.json). Environment variables such as `AAAE_EXTERNAL_ROOT`, `AAAE_OUTPUT_ROOT`, and `AAAE_PPA_ROOT` can override these locations. See [resource configuration](docs/usage.md#resource-configuration) for details.

---

## 🚀 Quick Start

Run the following commands from the repository root. Multiplier RTL generation uses only the source code; training, prediction, search, and verification require their configured external inputs.

### 1. Generate an Approximate Multiplier

Select one compressor ID from `1..8` for each of the nine low-order positions:

```bash
python examples/generate_multiplier.py \
    --compressors 1 2 3 4 5 6 7 8 1 \
    --output output/multiplier/multiplier.v
```

The generated multiplier keeps the high-order compression exact.

### 2. Prepare Features and Train an Accuracy Predictor

Inspect the feature preparation and model-specific training options:

```bash
python examples/prepare_features.py --help
python examples/train_accuracy.py --model vit_base -- --help
```

With the ViT dataset, base response package, and conditioned features configured, train the structure-aware predictor:

```bash
python examples/train_accuracy.py \
    --model vit_base -- \
    --lgbm-device cpu \
    --jobs 12
```

Supported model keys are `swin_tiny`, `vit_base`, `deit_base`, `resnet50`, and `mobilenetv3`. CNN training produces members; the corresponding bundle-building step combines them into the final predictor. See [accuracy prediction](docs/usage.md#accuracy-prediction).

### 3. Predict and Evaluate Accuracy Drop

Predict with a Transformer package and its matching 4497-dimensional feature matrix:

```bash
python examples/predict_accuracy.py \
    --family transformer \
    --package /path/to/strict50_package \
    --features /path/to/features.npy \
    --output output/accuracy/predictions.csv
```

Evaluate paired predictions and measured drops:

```bash
python examples/evaluate_accuracy.py \
    --predictions /path/to/holdout_predictions.csv \
    --units proportion
```

The evaluation input contains measured and predicted values. MAE, P95, and maximum absolute error are reported in percentage points; use `--units pp` when the input already uses those units.

### 4. Explore Multiplier Designs

The iterative search evaluates Top-1 drop, PDP, and area. Inspect the supported search options:

```bash
python examples/run_bo.py --help
python examples/run_nsga2.py --help
```

For the retained three-objective BO path, a 20k unique-evaluation budget can be requested with:

```bash
python examples/run_bo.py \
    --run-name swin_bo_20k \
    --device cuda:0 \
    --bo-iterations 20000 \
    --max-new-evals 20000
```

Both iterative optimizers currently use the Swin-Tiny Linear48 predictor. The independent calibrated fixed-pool workflow is described below.

### 5. Calibrate and Select Under an Accuracy Budget

Fit an A1 correction from independent calibration labels and target-pool bins:

```bash
python examples/calibrate.py \
    --model swin_tiny \
    --calibration /path/to/calibration.csv \
    --pool /path/to/candidate_pool.csv \
    --alpha 0.10 \
    --output output/calibration/swin_a1.json
```

Generate a query plan for an accuracy-drop budget of two percentage points:

```bash
python examples/select_design.py \
    --pool /path/to/candidate_pool.csv \
    --correction output/calibration/swin_a1.json \
    --tau 2.0 \
    --budget 10 \
    --output-dir output/selection/swin
```

This admits candidates whose calibrated upper drop is within the budget, orders them by predicted PDP, and requests at most the specified number of full accuracy evaluations. [Detailed usage](docs/usage.md#calibration-and-selection) explains the input columns and selection API.

### 6. Verify Selected Designs

```bash
python examples/verify_designs.py --family transformer -- --help
python examples/verify_designs.py --family cnn -- --help
python examples/run_ppa.py --help
```

Verification reruns selected configurations under the fixed inference protocol and evaluates their hardware cost with the no-macro EDA flow.

---

## 🧠 Supported Models

| Model | Accuracy prediction | Candidate-level response |
| --- | --- | --- |
| Swin-Tiny | Model-specific LightGBM with static, conditioned, and response features | 50 labeled images |
| ViT-Base | Model-specific LightGBM with static, conditioned, and response features | 50 labeled images |
| DeiT-Base | Model-specific LightGBM with static, conditioned, and response features | 50 labeled images |
| ResNet-50 | Two-member Delta-CNN bundle | None |
| MobileNetV3-Large | Two-member Correctness-CNN bundle with disagreement postprocessing | None |

Transformers inject one residual LUT into 48 weight-bearing Linear MatMuls. The CNN paths inject it into 53 Conv nodes for ResNet-50 and 62 for MobileNetV3-Large.

---

## 📚 Documentation

- [Methodology and paper correspondence](docs/methodology.md)
- [Usage and external resource configuration](docs/usage.md)
- [Example entry points](examples/README.md)
The methodology document connects the implementation to the paper, while the usage guide explains input formats, model training, and external resource configuration.

---

## 🙏 Acknowledgements

The repository organization follows the module, examples, and documentation layout used by [OpenACM](https://github.com/ShenShan123/OpenACM). AAAE uses PyTorch, LightGBM, PyTorch Geometric, and ONNX Runtime for prediction and inference, BoTorch and GPyTorch for Bayesian optimization, and Yosys and OpenSTA in the hardware verification flow.
