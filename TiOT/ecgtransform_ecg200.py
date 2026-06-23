"""
Wrapper to train/evaluate ECGTransForm on ECG200 dataset.

ECGTransForm lives at ../ECGTransForm relative to this file.
This module handles:
  1. Data conversion: ECG200 txt → pt format required by ECGTransForm
  2. Training: simplified trainer (no file-copy / logging overhead)
  3. Evaluation: load best checkpoint, evaluate on test set
  4. Entry point: run_ecgtransform_ecg200()

Checkpoint is stored at:
  ECGTransForm/experiments_logs/ECGTransForm_ECG200/checkpoint_best.pt

ECG200 pt data is stored at:
  ECGTransForm/data/ecg200/{train,val,test}.pt

Dataset details:
  - 100 train (split 80/20 train/val), 100 test
  - T = 96 time steps, binary labels: -1 → 0 (abnormal), +1 → 1 (normal)
  - trans_dim = 14 (T' after 3 pooling layers), num_heads = 2
"""

import os
import sys
import warnings
import numpy as np
import torch
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ECGTRANSFORM_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'ECGTransForm'))

# Insert at module load time so internal ECGTransForm imports (utils, dataloader, models)
# resolve correctly regardless of working directory or PYTHONPATH (e.g. on Colab).
for _p in [ECGTRANSFORM_DIR, _THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ECG200_DATA_OUT = os.path.join(ECGTRANSFORM_DIR, 'data', 'ecg200')
ECG200_CHECKPOINT_DIR = os.path.join(ECGTRANSFORM_DIR, 'experiments_logs', 'ECGTransForm_ECG200')
ECG200_CHECKPOINT = os.path.join(ECG200_CHECKPOINT_DIR, 'checkpoint_best.pt')
ECG200_TXT_DIR = os.path.join(_THIS_DIR, 'time_series_kNN', 'ECG200')

HPARAMS = {
    'num_epochs': 60,
    'batch_size': 32,  # original default 128 is larger than dataset; use 32
    'weight_decay': 1e-4,
    'learning_rate': 1e-3,
    'feature_dim': 128,
}


def _add_ecgtransform_to_path():
    if ECGTRANSFORM_DIR not in sys.path:
        sys.path.insert(0, ECGTRANSFORM_DIR)


def prepare_ecg200_data(force=False):
    """Convert ECG200 txt files to .pt format expected by ECGTransForm dataloader."""
    targets = [os.path.join(ECG200_DATA_OUT, f) for f in ('train.pt', 'val.pt', 'test.pt')]
    if not force and all(os.path.exists(t) for t in targets):
        print(f"[ECGTransForm] ECG200 pt data already exists at {ECG200_DATA_OUT}")
        return False

    os.makedirs(ECG200_DATA_OUT, exist_ok=True)

    train_raw = np.loadtxt(os.path.join(ECG200_TXT_DIR, 'ECG200_TRAIN.txt'))
    test_raw = np.loadtxt(os.path.join(ECG200_TXT_DIR, 'ECG200_TEST.txt'))

    def parse(raw):
        labels_raw = raw[:, 0]
        X = raw[:, 1:].astype(np.float32)[:, np.newaxis, :]  # (N, 1, 96)
        y = np.where(labels_raw < 0, 0, 1).astype(np.float64)  # -1→0, +1→1
        return X, y

    X_all, y_all = parse(train_raw)
    X_test, y_test = parse(test_raw)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_all))
    n_val = 20
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def save_pt(X, y, path):
        torch.save({'samples': torch.from_numpy(X), 'labels': torch.from_numpy(y)}, path)

    save_pt(X_all[tr_idx], y_all[tr_idx], targets[0])
    save_pt(X_all[val_idx], y_all[val_idx], targets[1])
    save_pt(X_test, y_test, targets[2])

    print(f"[ECGTransForm] ECG200 pt data saved to {ECG200_DATA_OUT}")
    print(f"  train={len(tr_idx)}, val={n_val}, test={len(X_test)}")
    return True


def _make_dataloader(pt_path, batch_size, shuffle, drop_last):
    """Create a DataLoader from a .pt file. drop_last=False for val/test with small datasets."""
    _add_ecgtransform_to_path()
    from dataloader import Load_Dataset
    import math

    dataset_dict = torch.load(pt_path, weights_only=False)
    dataset = Load_Dataset(dataset_dict)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=0,
    )


def _class_weights(train_pt_path):
    """Compute class weights from training labels (mirrors dataloader.get_class_weight)."""
    import math
    d = torch.load(train_pt_path, weights_only=False)
    labels = d['labels'].numpy().astype(int).tolist()
    labels_dict = {i: labels.count(i) for i in sorted(set(labels))}
    total = sum(labels_dict.values())
    max_num = max(labels_dict.values())
    mu = 1.0 / (total / max_num)
    cw = {}
    for k, v in labels_dict.items():
        score = math.log(mu * total / float(v))
        cw[k] = score if score > 1.0 else 1.0
    return cw


def _build_model_and_loaders(device):
    """Import ECGTransForm modules and return (model, train_dl, val_dl, test_dl, class_weights, configs)."""
    _add_ecgtransform_to_path()
    from models import ecgTransForm
    from configs.data_configs import get_dataset_class

    configs = get_dataset_class('ecg200')()

    bs = HPARAMS['batch_size']
    train_pt = os.path.join(ECG200_DATA_OUT, 'train.pt')
    val_pt   = os.path.join(ECG200_DATA_OUT, 'val.pt')
    test_pt  = os.path.join(ECG200_DATA_OUT, 'test.pt')

    train_dl = _make_dataloader(train_pt, bs, shuffle=True,  drop_last=True)
    val_dl   = _make_dataloader(val_pt,   bs, shuffle=False, drop_last=False)  # no drop_last for small val
    test_dl  = _make_dataloader(test_pt,  bs, shuffle=False, drop_last=False)

    cw_dict = _class_weights(train_pt)
    model = ecgTransForm(configs=configs, hparams=HPARAMS).to(device)
    weights = torch.tensor([float(cw_dict[k]) for k in sorted(cw_dict)], dtype=torch.float32).to(device)
    return model, train_dl, val_dl, test_dl, weights, configs


def _evaluate_loader(model, loader, device):
    """Run inference on a DataLoader. Returns (pred_labels, true_labels) as numpy arrays."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            data = batch['samples'].float().to(device)
            labels = batch['labels'].long().to(device)
            out = model(data).argmax(dim=1)
            preds.extend(out.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return np.array(preds, dtype=int), np.array(trues, dtype=int)


def train_ecgtransform_ecg200(seed=0, device_str='cpu', verbose=True):
    """
    Train ECGTransForm on ECG200 and save best checkpoint.
    Returns (test_accuracy, test_f1_macro) in [0, 1] range.
    """
    _add_ecgtransform_to_path()
    from utils import fix_randomness, _calc_metrics

    device = torch.device(device_str)
    fix_randomness(seed)

    prepare_ecg200_data()
    model, train_dl, val_dl, test_dl, weights, configs = _build_model_and_loaders(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS['learning_rate'],
        weight_decay=HPARAMS['weight_decay'],
        betas=(0.9, 0.99),
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_f1, best_acc = 0.0, 0.0
    os.makedirs(ECG200_CHECKPOINT_DIR, exist_ok=True)

    num_epochs = HPARAMS['num_epochs']
    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_dl:
            data = batch['samples'].float().to(device)
            labels = batch['labels'].long().to(device)
            optimizer.zero_grad()
            criterion(model(data), labels).backward()
            optimizer.step()

        preds, trues = _evaluate_loader(model, val_dl, device)
        val_acc, val_f1 = _calc_metrics(preds, trues, configs.class_names)  # returns %

        if val_f1 > best_f1:
            best_f1, best_acc = val_f1, val_acc
            torch.save({'dataset': 'ecg200', 'configs': configs.__dict__,
                        'hparams': HPARAMS, 'model': model.state_dict()}, ECG200_CHECKPOINT)

        if verbose and epoch % 10 == 0:
            print(f"  [ECGTransForm] Epoch {epoch}/{num_epochs}: "
                  f"val_acc={val_acc:.1f}%  val_f1={val_f1:.4f}%  (best_f1={best_f1:.4f}%)")

    print(f"[ECGTransForm] Training done. Best val: acc={best_acc:.1f}%  f1={best_f1:.4f}%")

    # Evaluate best checkpoint on test set
    ckpt = torch.load(ECG200_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    preds, trues = _evaluate_loader(model, test_dl, device)
    test_acc, test_f1 = _calc_metrics(preds, trues, configs.class_names)
    print(f"[ECGTransForm] Test: acc={test_acc:.1f}%  f1={test_f1:.4f}%")
    return test_acc / 100.0, test_f1 / 100.0


def evaluate_ecgtransform_ecg200(device_str='cpu'):
    """
    Load best checkpoint and evaluate ECGTransForm on ECG200 test set.
    Returns (test_accuracy, test_f1_macro) in [0, 1] range.
    Raises FileNotFoundError if checkpoint doesn't exist.
    """
    if not os.path.exists(ECG200_CHECKPOINT):
        raise FileNotFoundError(
            f"No checkpoint found at {ECG200_CHECKPOINT}. "
            "Run train_ecgtransform_ecg200() first, or use run_ecgtransform_ecg200()."
        )

    _add_ecgtransform_to_path()
    from utils import _calc_metrics

    device = torch.device(device_str)
    prepare_ecg200_data()
    _, _, _, test_dl, _, configs = _build_model_and_loaders(device)

    from models import ecgTransForm
    ckpt = torch.load(ECG200_CHECKPOINT, map_location=device, weights_only=False)
    model = ecgTransForm(configs=configs, hparams=HPARAMS).to(device)
    model.load_state_dict(ckpt['model'])

    preds, trues = _evaluate_loader(model, test_dl, device)
    test_acc, test_f1 = _calc_metrics(preds, trues, configs.class_names)
    print(f"[ECGTransForm] (from checkpoint) Test: acc={test_acc:.1f}%  f1={test_f1:.4f}%")
    return test_acc / 100.0, test_f1 / 100.0


def run_ecgtransform_ecg200(force_retrain=False, device_str='cpu'):
    """
    Main entry point: run ECGTransForm evaluation on ECG200.
    - If checkpoint exists and force_retrain=False: load and evaluate.
    - Otherwise: train from scratch and evaluate.
    Returns dict: {method, method_group, accuracy, f1_macro, notes}
    """
    try:
        if force_retrain or not os.path.exists(ECG200_CHECKPOINT):
            print("\n[ECGTransForm] Training on ECG200 (60 epochs, batch_size=32, ~1-2 min on CPU)...")
            acc, f1 = train_ecgtransform_ecg200(device_str=device_str)
            notes = "trained from scratch; original train/test split (80 train, 20 val, 100 test)"
        else:
            print(f"\n[ECGTransForm] Loading checkpoint: {ECG200_CHECKPOINT}")
            acc, f1 = evaluate_ecgtransform_ecg200(device_str=device_str)
            notes = "loaded from checkpoint; original train/test split (80 train, 20 val, 100 test)"

        return {
            'method': 'ECGTransForm',
            'method_group': 'ECGTransForm',
            'accuracy': acc,
            'f1_macro': f1,
            'notes': notes,
        }
    except Exception as e:
        print(f"[ECGTransForm] ERROR: {e}")
        return {
            'method': 'ECGTransForm',
            'method_group': 'ECGTransForm',
            'accuracy': float('nan'),
            'f1_macro': float('nan'),
            'notes': f'ERROR: {e}',
        }


if __name__ == '__main__':
    result = run_ecgtransform_ecg200()
    print(f"\nResult: accuracy={result['accuracy']:.4f}  f1_macro={result['f1_macro']:.4f}")
