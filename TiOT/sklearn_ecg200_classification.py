"""
scikit-learn baseline classifiers on ECG200 dataset.

Usage:
    python sklearn_ecg200_classification.py
    python sklearn_ecg200_classification.py --data ECG200 --data-dir time_series_kNN
    python sklearn_ecg200_classification.py --kfold 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_dataset(dataset_name, data_dir="time_series_kNN"):
    """Load train/test split from UEA/UCR txt format (col 0 = label, rest = features)."""
    train_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_TRAIN.txt")
    test_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_TEST.txt")

    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]

    return X_train, y_train, X_test, y_test


def _make_classifiers():
    """Return list of (name, pipeline) pairs for sklearn baselines."""
    return [
        ("kNN-1 (Euclidean)", KNeighborsClassifier(n_neighbors=1, metric="euclidean")),
        ("kNN-3 (Euclidean)", KNeighborsClassifier(n_neighbors=3, metric="euclidean")),
        ("kNN-5 (Euclidean)", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        (
            "SVM (RBF)",
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, random_state=42))]),
        ),
        (
            "SVM (Linear)",
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear", C=1.0, random_state=42))]),
        ),
        (
            "Logistic Regression",
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        ),
        (
            "Ridge Classifier",
            Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier())]),
        ),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]


def run_train_test(X_train, y_train, X_test, y_test):
    """Evaluate each classifier on original train/test split."""
    rows = []
    for name, clf in _make_classifiers():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        rows.append({"method": name, "accuracy": acc, "f1_macro": f1, "notes": "original train/test split"})
        print(f"  {name:30s}  acc={acc:.4f}  f1_macro={f1:.4f}")
    return rows


def run_kfold(X_train, y_train, n_splits=5, random_state=54):
    """Evaluate each classifier with k-fold on train set (mirrors kfold_kNN_Exp.py)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, clf in _make_classifiers():
        accs, f1s = [], []
        for tr_idx, val_idx in kf.split(X_train):
            clf_copy = _clone_clf(clf)
            clf_copy.fit(X_train[tr_idx], y_train[tr_idx])
            y_pred = clf_copy.predict(X_train[val_idx])
            accs.append(accuracy_score(y_train[val_idx], y_pred))
            f1s.append(f1_score(y_train[val_idx], y_pred, average="macro"))
        acc_mean, acc_std = np.mean(accs), np.std(accs)
        f1_mean, f1_std = np.mean(f1s), np.std(f1s)
        rows.append(
            {
                "method": name,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "f1_macro_mean": f1_mean,
                "f1_macro_std": f1_std,
                "notes": f"{n_splits}-fold CV on train set",
            }
        )
        print(f"  {name:30s}  acc={acc_mean:.4f}±{acc_std:.4f}  f1={f1_mean:.4f}±{f1_std:.4f}")
    return rows


def _clone_clf(clf):
    from sklearn.base import clone
    return clone(clf)


def save_results(rows, out_path):
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="scikit-learn baselines for ECG200 classification")
    parser.add_argument("--data", default="ECG200", help="Dataset name (default: ECG200)")
    parser.add_argument("--data-dir", default="time_series_kNN", help="Root data directory")
    parser.add_argument(
        "--mode",
        choices=["train-test", "kfold", "both"],
        default="train-test",
        help="Evaluation mode (default: train-test, matching ETIOT evaluation)",
    )
    parser.add_argument("--kfold", type=int, default=5, help="Number of folds for k-fold mode (default: 5)")
    parser.add_argument("--out-dir", default="results", help="Output directory for CSV results")
    args = parser.parse_args()

    print(f"\nLoading dataset: {args.data} from {args.data_dir}/")
    X_train, y_train, X_test, y_test = load_dataset(args.data, args.data_dir)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Classes: {np.unique(y_train)}")

    all_rows = []

    if args.mode in ("train-test", "both"):
        print(f"\n--- Train/Test Split Evaluation ---")
        rows = run_train_test(X_train, y_train, X_test, y_test)
        out_path = os.path.join(args.out_dir, f"{args.data.lower()}_sklearn_traintest.csv")
        save_results(rows, out_path)
        all_rows.extend(rows)

    if args.mode in ("kfold", "both"):
        print(f"\n--- {args.kfold}-Fold CV Evaluation (on train set) ---")
        rows = run_kfold(X_train, y_train, n_splits=args.kfold, random_state=54)
        out_path = os.path.join(args.out_dir, f"{args.data.lower()}_sklearn_kfold.csv")
        save_results(rows, out_path)
        all_rows.extend(rows)

    return all_rows


if __name__ == "__main__":
    main()
