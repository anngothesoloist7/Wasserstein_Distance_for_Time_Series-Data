# Data Directory

This folder contains datasets used across the three sub-projects.

---

## ECGTransForm Data (`data/ECGTransForm/`)

Pre-processed MIT-BIH and PTB ECG datasets in PyTorch `.pt` format.

- `data/ECGTransForm/mit/` — MIT-BIH Arrhythmia dataset (train.pt, val.pt, test.pt) ✅ included
- `data/ECGTransForm/ptb/` — PTB Diagnostic ECG dataset — **not included** (too large)

### Download (if needed)
Download the preprocessed datasets from the official ECGTransForm Google Drive:
> https://drive.google.com/drive/folders/1hnzoYfipi9xqDJfc2R0hfLAcon6k71XZ

Place the files as:
```
ECGTransForm/data/
├── mit/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
└── ptb/
    ├── train.pt
    ├── val.pt
    └── test.pt
```

Original sources:
- MIT-BIH: https://www.physionet.org/content/mitdb/1.0.0/
- PTB: https://physionet.org/content/ptbdb/1.0.0/

---

## TiOT Data (`data/TiOT/`)

MITBIH text data used with the TiOT kNN experiments.
- `data/TiOT/MITBIH/` — MITBIH_TRAIN.txt, MITBIH_TEST.txt, and random split versions ✅ included

The main TiOT datasets (`time_series_kNN/`) are already bundled inside the `TiOT/` folder from the GitHub repo.

Additional datasets (UCR Archive) can be downloaded from:
> https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

Place new datasets as:
```
TiOT/time_series_kNN/<DatasetName>/
├── <DatasetName>_TRAIN.txt
└── <DatasetName>_TEST.txt
```

---

## Wasserstein_Distance_for_Time_Series-Data Data

The Wasserstein_Distance_for_Time_Series-Data (Wasserstein K-Means) uses:
- `Wasserstein_Distance_for_Time_Series-Data/spy_hourly_data.csv` — SPY hourly data ✅ included
- `Wasserstein_Distance_for_Time_Series-Data/spy_hourly_data1.csv` — SPY hourly data (alternative) ✅ included

Additional SPY data can be downloaded via `yfinance`:
```python
import yfinance as yf
spy = yf.download("SPY", interval="1h", period="2y")
```
