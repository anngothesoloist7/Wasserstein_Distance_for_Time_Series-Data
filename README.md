# UROP Research Codebase — ECG / Time-Series / Optimal Transport

This repository brings together three related research codebases for time-series analysis, optimal transport, and ECG arrhythmia classification. The primary experimental focus is **ECG200 classification**, comparing scikit-learn baselines, TiOT/ETIOT (optimal-transport-based kNN), and the ECGTransForm deep-learning model.

---

## Repository Structure

```
project_root/
├── TiOT/                                        # Time-integrated Optimal Transport experiments
│   ├── TiOT_lib.py                              # Core TiOT/eTiOT/eTAOT solver
│   ├── kfold_kNN_Exp.py                         # k-fold kNN experiments (Table 1 in paper)
│   ├── sklearn_ecg200_classification.py         # scikit-learn ECG200 baselines
│   ├── ecgtransform_ecg200.py                   # ECGTransForm wrapper for ECG200
│   ├── evaluate_ecg200_all_methods.py           # Combined evaluation script (entry point)
│   ├── alignment_Exp.py                         # Section 4.1 alignment experiment
│   ├── lag_series_Exp.py                        # Section 4.2 lag-series experiment
│   ├── runtime_Exp.py                           # Section 4.3 runtime experiment
│   ├── robust_kNN_Exp.py                        # Section 4.4 robustness experiment
│   ├── time_series_kNN/                         # UCR datasets (15 datasets + MITBIH)
│   │   └── ECG200/                              # ECG200 train/test txt + pt files
│   ├── Experimental_outputs/                    # Auto-generated experiment outputs
│   └── results/                                 # ECG200 comparison outputs (CSV + PNG)
│
├── ECGTransForm/                                # Deep learning ECG model (external repo)
│   ├── main.py                                  # Training entry point (MIT-BIH / PTB)
│   ├── models.py                                # Model architecture
│   ├── trainer.py                               # Training loop
│   ├── dataloader.py                            # Data loading
│   ├── utils.py                                 # Utilities
│   ├── configs/
│   │   ├── data_configs.py                      # Dataset configs (incl. ecg200 added here)
│   │   └── hparams.py                           # Training hyperparameters
│   ├── data/
│   │   ├── mit/                                 # MIT-BIH .pt files (included)
│   │   └── ecg200/                              # ECG200 .pt files (auto-generated)
│   ├── experiments_logs/
│   │   └── ECGTransForm_ECG200/
│   │       └── checkpoint_best.pt               # Best ECG200 checkpoint (included)
│   └── requirements.txt
│
├── Wasserstein_Distance_for_Time_Series-Data/   # Wasserstein K-Means market regime clustering
│   ├── wasserstein_kmeans.py                    # Core WK-means / MK-means / MMD algorithms
│   ├── synthetic_data.py                        # GBM and Merton jump data generators
│   ├── visualization.py                         # Plotting utilities
│   ├── main_real_data.py                        # SPY analysis
│   ├── main_synthetic.py                        # Synthetic experiments
│   ├── run_all.py                               # Master script
│   ├── view_figures.py                          # Figure viewer
│   ├── spy_hourly_data.csv                      # SPY hourly data (included)
│   ├── figures/                                 # Generated plots
│   └── requirements.txt
│
├── data/                                        # Shared dataset mirror / download guide
│   ├── ECGTransForm/mit/                        # MIT-BIH .pt files (mirror)
│   ├── TiOT/MITBIH/                             # MITBIH text files
│   └── README.md                                # Download instructions
│
├── Time-integrated Optimal Transport-A Robust Minimax Framework.pdf
├── requirements.txt                             # Combined root requirements (with conflict notes)
└── README.md                                    # This file
```

---

## Main Components

### TiOT — Time-integrated Optimal Transport

Experimental code for the paper **"Time-integrated Optimal Transport: A Robust Minimax Framework"**.
Cloned from [github.com/Thai-npd/TiOT-code](https://github.com/Thai-npd/TiOT-code).

**Core library:** `TiOT/TiOT_lib.py` — implements TiOT, eTiOT, and eTAOT solvers.

**Original paper experiments (run from inside `TiOT/`):**

| Script | Purpose |
|--------|---------|
| `alignment_Exp.py` | Figure 1 — alignment (Section 4.1) |
| `lag_series_Exp.py` | Figure 2 — lag series (Section 4.2) |
| `runtime_Exp.py` | Figure 3 — runtime (Section 4.3) |
| `kfold_kNN_Exp.py` | Table 1 — k-fold kNN classification (Section 4.4) |
| `robust_kNN_Exp.py` | Figure 4 & 5 — robustness (Section 4.4 + Appendix D) |

**ECG200 evaluation scripts (added for this UROP project):**

| Script | Purpose |
|--------|---------|
| `sklearn_ecg200_classification.py` | scikit-learn baselines on ECG200 |
| `ecgtransform_ecg200.py` | Wrapper: train/evaluate ECGTransForm on ECG200 |
| `evaluate_ecg200_all_methods.py` | **Main combined evaluation** — all 3 method groups |

**Dataset path for ECG200:**
```
TiOT/time_series_kNN/ECG200/
├── ECG200_TRAIN.txt    # 100 samples, space-delimited, col 0 = label (−1/+1)
├── ECG200_TEST.txt     # 100 samples
├── ECG200.txt          # full dataset
├── train.pt / val.pt / test.pt   # auto-generated pt format for ECGTransForm
```

**TiOT results cache:**
```
TiOT/Experimental_outputs/kfold_kNN_data/saved_results/Results on ECG200 (0.01 to 0.1).csv
```

---

### ECGTransForm — Deep Learning ECG Classifier

**"ECGTransForm: Empowering Adaptive ECG Arrhythmia Classification Framework with Bidirectional Transformer"**
(El-Ghaish & Eldele, 2024. *Biomedical Signal Processing and Control.*)

This is a third-party repo used as a deep learning baseline in the ECG200 comparison. It features multi-scale CNN + squeeze-excitation residual blocks + bidirectional Transformer.

**Do not run directly for ECG200** — use the wrapper `TiOT/ecgtransform_ecg200.py` or the combined script `TiOT/evaluate_ecg200_all_methods.py`.

To run ECGTransForm on its original MIT-BIH dataset:
```bash
cd ECGTransForm
python main.py --dataset mit --data_path data
```

**ECG200 config added at:** `ECGTransForm/configs/data_configs.py`
- `trans_dim=14` (derived from T=96 after 3 pooling layers: 96→48→24→12→... adapted to 14)
- `num_heads=2`

**ECG200 checkpoint:** `ECGTransForm/experiments_logs/ECGTransForm_ECG200/checkpoint_best.pt` (included — training not required by default)

---

### Wasserstein\_Distance\_for\_Time\_Series-Data — Market Regime Clustering

Implementation of **"Clustering Market Regimes Using the Wasserstein Distance"** (Horvath, Issa, Muguruza, 2021. [arXiv:2110.11848](https://arxiv.org/abs/2110.11848)).
Code by **An Ngo Thanh**.

> Previously referenced internally as `friend_code/` during codebase restructuring. Renamed back to the original project name.

This component is **independent** from the TiOT/ECGTransForm ECG200 experiments. It implements unsupervised market regime detection using Wasserstein distance between empirical return distributions.

**Key scripts:**
- `wasserstein_kmeans.py` — WK-means, MK-means, MMD algorithms
- `synthetic_data.py` — GBM and Merton jump diffusion data generators
- `run_all.py` — master script (full or `--quick` demo)
- `main_real_data.py` — SPY analysis
- `main_synthetic.py` — synthetic GBM/Merton experiments

**Data included:** `spy_hourly_data.csv`, `spy_hourly_data1.csv`

---

## Dataset Summary

| Dataset | Location | Format | Status |
|---------|----------|--------|--------|
| ECG200 (TiOT format) | `TiOT/time_series_kNN/ECG200/` | `.txt` (space-delimited) | ✅ included |
| ECG200 (ECGTransForm format) | `ECGTransForm/data/ecg200/` | `.pt` (PyTorch) | ✅ auto-generated |
| MIT-BIH (ECGTransForm) | `ECGTransForm/data/mit/` | `.pt` | ✅ included |
| TiOT UCR datasets (15) | `TiOT/time_series_kNN/` | `.txt` | ✅ included |
| TiOT MITBIH | `TiOT/time_series_kNN/MITBIH/` | `.txt` | ✅ included |
| Delhi climate | `TiOT/DailyDelhiClimateTrain.csv` | `.csv` | ✅ included |
| SPY hourly | `Wasserstein_Distance_for_Time_Series-Data/spy_hourly_data*.csv` | `.csv` | ✅ included |
| ECGTransForm PTB | `ECGTransForm/data/ptb/` | `.pt` | ❌ download needed |

**ECG200 format detail:**
- Space-delimited text, first column = class label (−1 = abnormal, +1 = normal), remaining 96 columns = time-series features
- 100 train samples, 100 test samples

**Download PTB (if needed):**
See `data/README.md` or download from the [ECGTransForm Google Drive](https://drive.google.com/drive/folders/1hnzoYfipi9xqDJfc2R0hfLAcon6k71XZ).

---

## Environment Setup

### Local setup

**Option A — single environment (may have version tensions, works in practice):**
```bash
cd <project_root>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B — per-subproject environments (recommended if conflicts arise):**
```bash
# For TiOT + ECG200 evaluation (needs torch for ECGTransForm)
python3 -m venv .venv_tiot
source .venv_tiot/bin/activate
pip install numpy scipy scikit-learn matplotlib pandas torch tqdm pyyaml

# For ECGTransForm standalone
python3 -m venv .venv_ecgtf
source .venv_ecgtf/bin/activate
cd ECGTransForm && pip install -r requirements.txt

# For Wasserstein K-Means (no PyTorch needed)
python3 -m venv .venv_wass
source .venv_wass/bin/activate
cd Wasserstein_Distance_for_Time_Series-Data && pip install -r requirements.txt
```

**Minimum dependencies to run ECG200 evaluation:**
```
numpy scipy scikit-learn matplotlib pandas torch tqdm pyyaml
```

### Google Colab setup

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project root
import os
os.chdir('/content/drive/MyDrive/<path-to-project-root>')

# Install dependencies
!pip install numpy scipy scikit-learn matplotlib pandas torch tqdm pyyaml

# Run evaluation
os.chdir('TiOT')
!python evaluate_ecg200_all_methods.py --skip-etiot
```

**Important:** Do not use hard-coded local paths like `/Users/bichphuong/...` in scripts. All paths in the evaluation scripts use relative paths via `pathlib` or `os.path` — they resolve correctly when run from inside `TiOT/`.

---

## How to Run — ECG200 Comparison

All commands below must be run from inside the `TiOT/` directory.

```bash
cd TiOT
```

### Quick comparison (sklearn + ECGTransForm, skip slow TiOT)

```bash
python evaluate_ecg200_all_methods.py --skip-etiot
```

Runs:
- scikit-learn baselines (kNN-1/3/5, SVM-RBF/Linear, Logistic Regression, Ridge, Random Forest)
- ECGTransForm (loads from checkpoint — no training needed if `checkpoint_best.pt` exists)

Estimated time: ~1–3 minutes.

### Full comparison (all 3 method groups)

```bash
python evaluate_ecg200_all_methods.py --run-etiot
```

Runs all three groups:
- scikit-learn baselines
- TiOT/ETIOT (k-fold ε tuning with custom OT distance — **slow, multiprocessing**)
- ECGTransForm

Estimated time: 10–30+ minutes depending on hardware.

**Load TiOT from cache (if previously run):**
```bash
python evaluate_ecg200_all_methods.py
```
(Without `--run-etiot`, TiOT results are loaded from the cached CSV if it exists.)

### Force retrain ECGTransForm

```bash
python evaluate_ecg200_all_methods.py --skip-etiot --retrain-ecgtf
```

### Skip ECGTransForm entirely

```bash
python evaluate_ecg200_all_methods.py --skip-etiot --skip-ecgtf
```

### Run scikit-learn baselines only

```bash
python sklearn_ecg200_classification.py                        # train/test split
python sklearn_ecg200_classification.py --mode both            # + 5-fold CV
python sklearn_ecg200_classification.py --mode kfold --kfold 5
```

### Run ECGTransForm wrapper only

```bash
python ecgtransform_ecg200.py   # trains if no checkpoint, else evaluates
```

### Run TiOT standalone on ECG200

```python
from kfold_kNN_Exp import experiment_kNN
experiment_kNN('ECG200', w_TAOT=0.1, RUN=True)    # train and cache
experiment_kNN('ECG200', w_TAOT=0.1, RUN=False)   # load cache and plot
```

### All CLI options for the combined script

```
python evaluate_ecg200_all_methods.py --help

  --dataset DATASET     Dataset name (default: ECG200)
  --data-dir DATA_DIR   Root data directory for sklearn/TiOT
  --w-taot W_TAOT       Window parameter w for eTAOT (default: 0.1)
  --skip-etiot          Skip TiOT/ETIOT evaluation
  --run-etiot           Force re-run TiOT (slow)
  --skip-ecgtf          Skip ECGTransForm evaluation
  --retrain-ecgtf       Force retrain ECGTransForm
  --device DEVICE       Device for ECGTransForm (cpu / cuda:0)
  --out-dir OUT_DIR     Output directory for CSV and charts
```

---

## Output Files

After running `evaluate_ecg200_all_methods.py`, results are saved in:

```
TiOT/results/
├── ecg200_classification_summary.csv       # All methods: method, method_group, accuracy, f1_macro, notes
├── ecg200_classification_comparison.png    # Bar chart — accuracy by method
├── ecg200_f1_macro_comparison.png          # Bar chart — macro F1 by method
└── ecg200_sklearn_traintest.csv            # sklearn train/test results only
```

**CSV columns:** `method`, `method_group`, `accuracy`, `f1_macro`, `notes`

**Chart color convention:**

| Group | Color | Methods |
|-------|-------|---------|
| scikit-learn | steelblue | kNN-1/3/5, SVM-RBF/Linear, Logistic Regression, Ridge, Random Forest |
| TiOT/ETIOT | goldenrod | eTiOT (1-NN), eTAOT (1-NN) with k-fold ε tuning |
| ECGTransForm | crimson | Multi-scale CNN + SE residual + Bi-directional Transformer |

**Fairness note:** All groups are evaluated on the same held-out 100-sample test set (`ECG200_TEST.txt`).
- sklearn: trained on full 100-sample train set
- TiOT/ETIOT: k-fold on train set to select best ε, then trained on full train set
- ECGTransForm: trained on 80 samples, validated on 20, best checkpoint tested

---

## Original TiOT Paper Experiments

Run from inside `TiOT/`:

```bash
python alignment_Exp.py          # Figure 1
python lag_series_Exp.py         # Figure 2
python runtime_Exp.py            # Figure 3
python kfold_kNN_Exp.py          # Table 1 (all UCR datasets — slow)
python robust_kNN_Exp.py         # Figure 4 & 5
```

Outputs saved in `TiOT/Experimental_outputs/`.

---

## Wasserstein K-Means Experiments

Run from inside `Wasserstein_Distance_for_Time_Series-Data/`:

```bash
cd Wasserstein_Distance_for_Time_Series-Data
pip install -r requirements.txt

python run_all.py --quick        # quick demo (~2 min)
python run_all.py                # full run (~5–10 min)
python main_real_data.py         # SPY regime analysis only
python main_synthetic.py         # synthetic GBM/Merton only
python view_figures.py           # display generated figures
```

Figures saved in `Wasserstein_Distance_for_Time_Series-Data/figures/`.

---

## Reproducibility Notes

- **Always run evaluation scripts from inside `TiOT/`**, not from the project root.
- ECGTransForm is imported via `sys.path.insert(0, '../ECGTransForm')` in the wrapper — this requires the standard project layout (`TiOT/` and `ECGTransForm/` as siblings).
- ECG200 data must be at `TiOT/time_series_kNN/ECG200/ECG200_TRAIN.txt` and `ECG200_TEST.txt`.
- The ECG200 `.pt` files in `ECGTransForm/data/ecg200/` are auto-generated from the `.txt` files on first run.
- If using Google Colab, maintain the project structure:
  ```
  project_root/
  ├── TiOT/
  └── ECGTransForm/
  ```
  and `cd` into `TiOT/` before running any evaluation script.

---

## Known Issues / Notes

- **TiOT/ETIOT is slow** — uses multiprocessing k-fold cross-validation with a custom optimal transport distance. Use `--skip-etiot` for fast iteration; TiOT results are cached after the first run.
- **ECGTransForm checkpoint is included** (`experiments_logs/ECGTransForm_ECG200/checkpoint_best.pt`) — retraining is not needed unless you use `--retrain-ecgtf`.
- **Potential dependency conflict** between subprojects on numpy/scipy/scikit-learn minor versions. If you see import errors, use per-subproject virtualenvs (see Environment Setup).
- **PTB dataset** for ECGTransForm is not included (too large). See `data/README.md` for download link.

---

## Recent Changes

- Codebase restructured from `With_Code/` flat layout into `TiOT/`, `ECGTransForm/`, `Wasserstein_Distance_for_Time_Series-Data/` top-level folders
- `TiOT/` cloned fresh from GitHub (original `TiOT-code-main/` in `With_Code/` had only MITBIH data, no Python code)
- `Wasserstein_Distance_for_Time_Series-Data/` briefly named `friend_code/` during restructuring, then renamed back
- `ECGTransForm/` verified identical to `With_Code/ECGTransForm/` via `diff -rq`
- Added ECG200 dataset config to `ECGTransForm/configs/data_configs.py` (`trans_dim=14`, `num_heads=2`)
- Added `TiOT/sklearn_ecg200_classification.py` — scikit-learn baselines (kNN, SVM, RF, LR, Ridge)
- Added `TiOT/ecgtransform_ecg200.py` — ECGTransForm wrapper for ECG200
- Added `TiOT/evaluate_ecg200_all_methods.py` — combined evaluation across all 3 method groups
- Added `TiOT/results/` — ECG200 comparison outputs (CSV + PNG charts)
- ECGTransForm ECG200 checkpoint trained and saved at `ECGTransForm/experiments_logs/ECGTransForm_ECG200/checkpoint_best.pt`

---

## Citation

**TiOT:**
> Thai Nguyen-Phu Duc et al., "Time-integrated Optimal Transport: A Robust Minimax Framework"

**ECGTransForm:**
```bibtex
@ARTICLE{ecgTransForm,
    title = {ECGTransForm: Empowering adaptive ECG arrhythmia classification framework with bidirectional transformer},
    journal = {Biomedical Signal Processing and Control},
    volume = {89}, pages = {105714}, year = {2024},
    doi = {https://doi.org/10.1016/j.bspc.2023.105714},
    author = {Hany El-Ghaish and Emadeldeen Eldele},
}
```

**Wasserstein K-Means:**
```bibtex
@article{horvath2021clustering,
    title={Clustering Market Regimes Using the Wasserstein Distance},
    author={Horvath, Blanka and Issa, Zacharia and Muguruza, Aitor},
    journal={arXiv preprint arXiv:2110.11848}, year={2021}
}
```
