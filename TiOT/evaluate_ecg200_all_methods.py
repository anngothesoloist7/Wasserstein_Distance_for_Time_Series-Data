"""
Evaluate and compare all classification methods on ECG200:
  Group 1 — scikit-learn baselines (kNN-Euclidean, SVM, RF, Logistic Regression, Ridge)
  Group 2 — TiOT/ETIOT (1-NN with custom OT distance, k-fold eps tuning) from kfold_kNN_Exp.py
  Group 3 — ECGTransForm (Transformer + multi-scale CNN) from ../ECGTransForm

Usage:
    python evaluate_ecg200_all_methods.py                    # sklearn + ECGTransForm only
    python evaluate_ecg200_all_methods.py --skip-etiot       # skip slow TiOT
    python evaluate_ecg200_all_methods.py --run-etiot        # force-run TiOT (slow)
    python evaluate_ecg200_all_methods.py --retrain-ecgtf    # force retrain ECGTransForm
    python evaluate_ecg200_all_methods.py --skip-ecgtf       # skip ECGTransForm

Fairness note:
  - sklearn and ECGTransForm use original train/test split (100 train / 100 test).
    ECGTransForm further splits train into 80 train + 20 val for model selection.
  - TiOT/ETIOT uses k-fold on train set to pick best eps, then evaluates on test set.
  - All groups are compared on the same held-out 100-sample test set.

Color convention in bar charts:
  - scikit-learn  →  steelblue
  - TiOT/ETIOT   →  goldenrod
  - ECGTransForm →  crimson
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent

# Ensure TiOT modules (kfold_kNN_Exp, sklearn_ecg200_classification, etc.) are importable
# regardless of CWD — important on Colab where script is invoked from repo root.
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

ETIOT_RESULT_FILE = str(
    _THIS_DIR / 'Experimental_outputs' / 'kfold_kNN_data' / 'saved_results'
    / 'Results on {dataset} (0.01 to 0.1).csv'
)

GROUP_COLORS = {
    'sklearn': 'steelblue',
    'TiOT': 'goldenrod',
    'ECGTransForm': 'crimson',
}


# ---------------------------------------------------------------------------
# Group 1: sklearn baselines
# ---------------------------------------------------------------------------

def run_sklearn(dataset, data_dir):
    from sklearn_ecg200_classification import load_dataset, run_train_test
    X_train, y_train, X_test, y_test = load_dataset(dataset, data_dir)
    print(f"\n[sklearn] Running baselines on {dataset} ...")
    rows = run_train_test(X_train, y_train, X_test, y_test)
    for r in rows:
        r['method_group'] = 'sklearn'
    return rows


# ---------------------------------------------------------------------------
# Group 2: TiOT/ETIOT (via kfold_kNN_Exp.py)
# ---------------------------------------------------------------------------

def run_etiot(dataset, w_taot, force_run=False):
    from kfold_kNN_Exp import experiment_kNN
    result_path = ETIOT_RESULT_FILE.format(dataset=dataset)

    if force_run or not os.path.exists(result_path):
        print(f"\n[TiOT] Running experiment_kNN('{dataset}', w_TAOT={w_taot}) ...")
        print("  (1-NN k-fold eps tuning + multiprocessing — may take several minutes)")
        experiment_kNN(dataset, w_taot, RUN=True)
    else:
        print(f"\n[TiOT] Loading cached results from:\n  {result_path}")
        experiment_kNN(dataset, w_taot, RUN=False)

    return _parse_etiot_csv(result_path)


def _parse_etiot_csv(result_path):
    df = pd.read_csv(result_path, dtype=str)
    final_row = df[df['eps'] == 'Final error']
    if final_row.empty:
        print(f"  WARNING: 'Final error' row not found in {result_path}")
        return []
    rows = []
    for col in df.columns:
        if col == 'eps':
            continue
        try:
            error = float(final_row[col].values[0])
        except (ValueError, IndexError):
            continue
        acc = 1.0 - error
        print(f"  TiOT/{col:10s}  acc={acc:.4f}  (f1_macro: N/A — not tracked by kfold_kNN_Exp)")
        rows.append({
            'method': f'TiOT/{col} (1-NN)',
            'method_group': 'TiOT',
            'accuracy': acc,
            'f1_macro': float('nan'),
            'notes': '1-NN with k-fold eps tuning on train set; evaluated on test set',
        })
    return rows


# ---------------------------------------------------------------------------
# Group 3: ECGTransForm
# ---------------------------------------------------------------------------

def run_ecgtransform(force_retrain=False, device_str='cpu'):
    from ecgtransform_ecg200 import run_ecgtransform_ecg200
    row = run_ecgtransform_ecg200(force_retrain=force_retrain, device_str=device_str)
    return [row]


# ---------------------------------------------------------------------------
# Output: table + charts
# ---------------------------------------------------------------------------

def print_table(df):
    cols = [c for c in ['method', 'method_group', 'accuracy', 'f1_macro', 'notes'] if c in df.columns]
    print('\n' + '=' * 110)
    print(f"  {'method':<42} {'group':<14} {'accuracy':>10} {'f1_macro':>10}  notes")
    print('-' * 110)
    for _, row in df[cols].iterrows():
        acc = f"{row['accuracy']:.4f}" if not pd.isna(row['accuracy']) else '   N/A  '
        f1  = f"{row['f1_macro']:.4f}"  if not pd.isna(row['f1_macro'])  else '   N/A  '
        print(f"  {row['method']:<42} {row['method_group']:<14} {acc:>10} {f1:>10}  {row.get('notes','')}")
    print('=' * 110)


def _bar_chart(df, metric, title, out_path):
    """Draw a grouped bar chart coloured by method_group."""
    plot_df = df[df[metric].notna()].copy()
    if plot_df.empty:
        print(f"  No data for metric '{metric}' — skipping chart.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.9), 5))
    bars = ax.bar(
        range(len(plot_df)),
        plot_df[metric],
        color=[GROUP_COLORS.get(g, 'grey') for g in plot_df['method_group']],
        edgecolor='white',
        linewidth=0.8,
    )

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['method'], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {out_path}")


def save_charts(df, out_dir, dataset):
    print('\n[Charts]')
    _bar_chart(
        df, 'accuracy',
        f'{dataset} Classification — Accuracy Comparison',
        os.path.join(out_dir, f'{dataset.lower()}_classification_comparison.png'),
    )
    if df['f1_macro'].notna().any():
        _bar_chart(
            df, 'f1_macro',
            f'{dataset} Classification — Macro F1 Comparison',
            os.path.join(out_dir, f'{dataset.lower()}_f1_macro_comparison.png'),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare sklearn / TiOT / ECGTransForm on ECG200')
    parser.add_argument('--dataset', '--data', default='ECG200', help='Dataset name (default: ECG200)')
    parser.add_argument('--data-dir', default=str(_THIS_DIR / 'time_series_kNN'),
                        help='Root data directory for sklearn/TiOT')
    parser.add_argument('--w-taot', type=float, default=0.1,
                        help='Window parameter w for eTAOT in kfold_kNN_Exp (default: 0.1)')
    parser.add_argument('--skip-etiot',  action='store_true', help='Skip TiOT/ETIOT evaluation')
    parser.add_argument('--run-etiot',   action='store_true', help='Force re-run TiOT (slow)')
    parser.add_argument('--skip-ecgtf',  action='store_true', help='Skip ECGTransForm evaluation')
    parser.add_argument('--retrain-ecgtf', action='store_true', help='Force retrain ECGTransForm')
    parser.add_argument('--device', default='cpu', help='Device for ECGTransForm (cpu/cuda:0)')
    parser.add_argument('--out-dir', default=str(_THIS_DIR / 'results'),
                        help='Output directory for CSV and charts')
    args = parser.parse_args()

    all_rows = []

    # --- Group 1: sklearn ---
    try:
        all_rows.extend(run_sklearn(args.dataset, args.data_dir))
    except Exception as e:
        print(f'\n[sklearn] ERROR: {e}')
        all_rows.append({'method': 'sklearn (error)', 'method_group': 'sklearn',
                         'accuracy': float('nan'), 'f1_macro': float('nan'), 'notes': str(e)})

    # --- Group 2: TiOT/ETIOT ---
    if args.skip_etiot:
        print('\n[TiOT] Skipped (--skip-etiot)')
    else:
        try:
            all_rows.extend(run_etiot(args.dataset, args.w_taot, force_run=args.run_etiot))
        except Exception as e:
            print(f'\n[TiOT] ERROR: {e}')
            print('  Tip: run kfold_kNN_Exp.py first, or pass --skip-etiot / --run-etiot')
            all_rows.append({'method': 'TiOT/ETIOT (error)', 'method_group': 'TiOT',
                             'accuracy': float('nan'), 'f1_macro': float('nan'), 'notes': str(e)})

    # --- Group 3: ECGTransForm ---
    if args.skip_ecgtf:
        print('\n[ECGTransForm] Skipped (--skip-ecgtf)')
    else:
        try:
            all_rows.extend(run_ecgtransform(force_retrain=args.retrain_ecgtf, device_str=args.device))
        except Exception as e:
            print(f'\n[ECGTransForm] ERROR: {e}')
            all_rows.append({'method': 'ECGTransForm (error)', 'method_group': 'ECGTransForm',
                             'accuracy': float('nan'), 'f1_macro': float('nan'), 'notes': str(e)})

    df = pd.DataFrame(all_rows)
    print_table(df)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'ecg200_classification_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSummary CSV saved: {csv_path}')

    save_charts(df, args.out_dir, args.dataset)


if __name__ == '__main__':
    main()
