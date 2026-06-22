# Time-integrated Optimal Transport — Experimental Code

> **For the full project overview (ECG200 evaluation, environment setup, all components), see the [root README](../README.md).**

This folder provides the experimental code and data used in the paper “Time-integrated Optimal Transport: A Robust Minimax Framework.”
It includes all scripts necessary to reproduce the results and figures presented in the experimental section, plus additional ECG200 evaluation scripts added for the UROP project.

### Datasets:
 - Data for experiments in Sections 4.1 and 4.3 is synthetically generated.
 - Data supporting Section 4.2 is provided in the file DailyDelhiClimateTrain.csv.
 - Data for Section 4.4 is stored in the folder time_series_kNN.

### Code Structure:
The main solver for the TiOT and eTiOT problems, along with related components, is implemented in TiOT_lib.py.
The repository also includes five experiment scripts corresponding to the paper’s sections:
 - alignment_Exp.py – Experiment for Section 4.1 (Figure 1)
 - lag_series_Exp.py – Experiment for Section 4.2
    - run the function 'dist_lag_exp()' to reproduce Figure 2 (left)
    - run the function 'dist_w_exp()' to reproduce Figure 2 (right)
 - runtime_Exp.py – Experiment for Section 4.3
    - run the function 'deviation_experiment()' to reproduce Figure 3 (left)
    - run the function 'runtime_experiment()' to reproduce Figure 3 (right)
 - kfold_kNN_Exp.py – Generates Table 1 (Section 4.4)
 - robust_kNN_Exp.py – Generates Figure 4 (Section 4.4) and Figure 5 (Appendix D)

Running Experiments:
To reproduce results, run the desired script using:
python <script_name>.py

For example:
python lag_series_Exp.py

Experimental outputs will be saved automatically in the Experimental_outputs directory.

---

## ECG200 Classification — sklearn + TiOT/ETIOT + ECGTransForm comparison

### Dataset location
```
time_series_kNN/ECG200/ECG200_TRAIN.txt   # 100 samples
time_series_kNN/ECG200/ECG200_TEST.txt    # 100 samples
```
Format: space-delimited, first column = class label (−1 or +1), remaining 96 columns = time series features.

### Method groups and colour convention

| Group | Colour (bar charts) | Methods |
|-------|---------------------|---------|
| **scikit-learn** | steelblue | kNN-1/3/5 (Euclidean), SVM-RBF/Linear, Logistic Regression, Ridge, Random Forest |
| **TiOT/ETIOT** | goldenrod | eTiOT (1-NN), eTAOT (1-NN) with k-fold ε tuning |
| **ECGTransForm** | crimson | Multi-scale CNN + SE residual + Bi-directional Transformer |

### Files added / modified

| File | Purpose |
|------|---------|
| `sklearn_ecg200_classification.py` | scikit-learn baselines on ECG200 |
| `ecgtransform_ecg200.py` | Wrapper to train/evaluate ECGTransForm on ECG200 |
| `evaluate_ecg200_all_methods.py` | Combined evaluation across all 3 groups |
| `results/` | Output directory for CSV and PNG charts |
| `../ECGTransForm/configs/data_configs.py` | Added `ecg200` dataset config (additive only) |
| `../ECGTransForm/data/ecg200/` | ECG200 data in pt format (auto-generated on first run) |

### Run scikit-learn baselines only

```bash
python sklearn_ecg200_classification.py                        # train/test split
python sklearn_ecg200_classification.py --mode both            # + 5-fold CV
python sklearn_ecg200_classification.py --mode kfold --kfold 5
```

Results → `results/ecg200_sklearn_traintest.csv` and `results/ecg200_sklearn_kfold.csv`.

### Run combined evaluation (sklearn + TiOT + ECGTransForm)

```bash
# Fast (sklearn + ECGTransForm, skip slow TiOT):
python evaluate_ecg200_all_methods.py --skip-etiot

# All 3 groups — loads TiOT from cache if available:
python evaluate_ecg200_all_methods.py

# Force re-run TiOT (slow — multiprocessing k-fold with custom OT distance):
python evaluate_ecg200_all_methods.py --run-etiot --w-taot 0.1

# Force retrain ECGTransForm:
python evaluate_ecg200_all_methods.py --retrain-ecgtf
```

### Run TiOT standalone on ECG200

```python
from kfold_kNN_Exp import experiment_kNN
experiment_kNN('ECG200', w_TAOT=0.1, RUN=True)   # trains and saves results
experiment_kNN('ECG200', w_TAOT=0.1, RUN=False)  # load cached results and plot
```

TiOT results cached at `Experimental_outputs/kfold_kNN_data/saved_results/Results on ECG200 (0.01 to 0.1).csv`.

### Run ECGTransForm standalone on ECG200

```bash
python ecgtransform_ecg200.py   # trains if no checkpoint, else evaluates
```

ECGTransForm checkpoint: `../ECGTransForm/experiments_logs/ECGTransForm_ECG200/checkpoint_best.pt`

ECGTransForm on ECG200 details:
- Data: ECG200 txt → pt format (auto-converted to `../ECGTransForm/data/ecg200/`)
- Split: 80 train / 20 val / 100 test
- Labels: −1 → 0 (abnormal), +1 → 1 (normal)
- Config: `trans_dim=14` (T′ after 3 pooling layers for T=96), `num_heads=2`
- Training: 60 epochs, batch_size=32, Adam lr=1e-3

### Output files

| File | Description |
|------|-------------|
| `results/ecg200_classification_summary.csv` | All methods: method, method_group, accuracy, f1_macro, notes |
| `results/ecg200_classification_comparison.png` | Bar chart — accuracy by method (coloured by group) |
| `results/ecg200_f1_macro_comparison.png` | Bar chart — macro F1 by method |
| `results/ecg200_sklearn_traintest.csv` | sklearn train/test results |

### Fairness note

All groups are evaluated on the same held-out 100-sample test set (`ECG200_TEST.txt`).
- **sklearn**: trained on full 100-sample train set, tested on test set.
- **TiOT/ETIOT**: k-fold on train set to select best ε, then trained on full train set and tested.
- **ECGTransForm**: trained on 80 samples, validated on 20, best checkpoint tested on test set.

  
