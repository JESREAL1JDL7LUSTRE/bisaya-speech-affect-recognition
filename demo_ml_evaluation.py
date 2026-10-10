"""
Part C: Machine Learning Evaluation and Benchmarking Script
CS412 - Bisaya Speech Affect Recognition (Group 7)

Demonstrates:
1. Leakage-free Cross-Validation Pipeline (median imputation + z-score standardization inside CV folds)
2. Benchmarking 4 Machine Learning Models vs. Majority Baseline:
   - Majority Baseline (DummyClassifier)
   - Logistic Regression (L2 / Ridge, C=0.1)
   - Support Vector Machine (Linear kernel, C=0.1)
   - Support Vector Machine (RBF kernel, C=1.0)
   - Random Forest (100 trees, max_depth=3)
3. 3-Class Valence & 3-Class Arousal evaluation metrics (Accuracy, Precision, Recall, Macro F1, Fold SD)
4. Out-of-fold Confusion Matrices and summary interpretation
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.multimodal_dataset import predictor_columns

def run_ml_demonstration():
    print("=" * 80)
    print(" PART C: MACHINE LEARNING EVALUATION & BENCHMARKING (CS412 GROUP 7)")
    print("=" * 80)

    dataset_path = "datasets/multimodal/cleaned_multimodal_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    audio_features = [
        c for c in predictor_columns(df, modality="audio") 
        if df[c].notna().any() and df[c].nunique(dropna=True) > 1
    ]
    
    print(f"\n[1] Dataset & Feature Specifications:")
    print(f"    - Participant Count: {len(df)} observations")
    print(f"    - Modality: Acoustic & Prosodic speech features only (Audio SER)")
    print(f"    - Acoustic Feature Count: {len(audio_features)} validated predictors")
    print(f"    - Leakage Control: Reference ratings & feelings strictly excluded from predictors")

    X = df[audio_features].values
    y_val = df["valence_group"].values
    y_aro = df["arousal_group"].values

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"    - Cross-Validation Scheme: Stratified 5-Fold CV (Random Seed: 42)")

    models = {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (L2, C=0.1)": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
        "SVM (Linear, C=0.1)": SVC(kernel="linear", C=0.1, random_state=42),
        "SVM (RBF, C=1.0)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "Random Forest (100 trees, depth 3)": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    }

    # ---------------------------------------------------------
    # Evaluation 1: Valence (3-Class)
    # ---------------------------------------------------------
    print("\n[2] Evaluating 3-Class Valence Classification (Neg: 13, Neu: 13, Pos: 14)...")
    val_table = []
    val_preds = {}
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    for name, clf in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
        cv_res = cross_validate(pipe, X, y_val, cv=cv5, scoring=scoring)
        y_pred = cross_val_predict(pipe, X, y_val, cv=cv5)
        val_preds[name] = y_pred

        acc_mean = cv_res["test_accuracy"].mean() * 100
        acc_std = cv_res["test_accuracy"].std() * 100
        prec_mean = cv_res["test_precision_macro"].mean()
        rec_mean = cv_res["test_recall_macro"].mean()
        f1_mean = cv_res["test_f1_macro"].mean()
        f1_std = cv_res["test_f1_macro"].std()

        val_table.append([
            name,
            f"{acc_mean:.1f}% (+/- {acc_std:.1f}%)",
            f"{prec_mean:.3f}",
            f"{rec_mean:.3f}",
            f"{f1_mean:.3f} (+/- {f1_std:.3f})"
        ])

    print("\n--- 3-Class Valence Classification Results (Stratified 5-Fold CV) ---")
    print(tabulate(val_table, headers=["Model", "Accuracy (+/- SD)", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"], tablefmt="github"))
    print("    * Theoretical Chance Level for 3 balanced classes = 33.3%")

    # Confusion Matrix for Best Valence Model
    best_val_model = "Random Forest (100 trees, depth 3)"
    cm_val = confusion_matrix(y_val, val_preds[best_val_model], labels=["Negative", "Neutral", "Positive"])
    print(f"\n[3] Out-of-Fold Confusion Matrix for Best Valence Model: [{best_val_model}]")
    print(tabulate(
        [
            ["True Negative (13)", cm_val[0, 0], cm_val[0, 1], cm_val[0, 2]],
            ["True Neutral (13)", cm_val[1, 0], cm_val[1, 1], cm_val[1, 2]],
            ["True Positive (14)", cm_val[2, 0], cm_val[2, 1], cm_val[2, 2]],
        ],
        headers=["Ground Truth \\ Predicted", "Pred Negative", "Pred Neutral", "Pred Positive"],
        tablefmt="github"
    ))

    # ---------------------------------------------------------
    # Evaluation 2: Arousal (3-Class)
    # ---------------------------------------------------------
    print("\n[4] Evaluating 3-Class Arousal Classification (Low: 13, Mod: 16, High: 11)...")
    aro_table = []
    aro_preds = {}

    for name, clf in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
        cv_res = cross_validate(pipe, X, y_aro, cv=cv5, scoring=scoring)
        y_pred = cross_val_predict(pipe, X, y_aro, cv=cv5)
        aro_preds[name] = y_pred

        acc_mean = cv_res["test_accuracy"].mean() * 100
        acc_std = cv_res["test_accuracy"].std() * 100
        prec_mean = cv_res["test_precision_macro"].mean()
        rec_mean = cv_res["test_recall_macro"].mean()
        f1_mean = cv_res["test_f1_macro"].mean()
        f1_std = cv_res["test_f1_macro"].std()

        aro_table.append([
            name,
            f"{acc_mean:.1f}% (+/- {acc_std:.1f}%)",
            f"{prec_mean:.3f}",
            f"{rec_mean:.3f}",
            f"{f1_mean:.3f} (+/- {f1_std:.3f})"
        ])

    print("\n--- 3-Class Arousal Classification Results (Stratified 5-Fold CV) ---")
    print(tabulate(aro_table, headers=["Model", "Accuracy (+/- SD)", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"], tablefmt="github"))
    print("    * Majority Baseline = 40.0% (Moderate class = 16/40)")

    best_aro_model = "SVM (Linear, C=0.1)"
    cm_aro = confusion_matrix(y_aro, aro_preds[best_aro_model], labels=["Low", "Moderate", "High"])
    print(f"\n[5] Out-of-Fold Confusion Matrix for Best Arousal Model: [{best_aro_model}]")
    print(tabulate(
        [
            ["True Low (13)", cm_aro[0, 0], cm_aro[0, 1], cm_aro[0, 2]],
            ["True Moderate (16)", cm_aro[1, 0], cm_aro[1, 1], cm_aro[1, 2]],
            ["True High (11)", cm_aro[2, 0], cm_aro[2, 1], cm_aro[2, 2]],
        ],
        headers=["Ground Truth \\ Predicted", "Pred Low", "Pred Moderate", "Pred High"],
        tablefmt="github"
    ))

    # ---------------------------------------------------------
    # Save Confusion Matrix Plots
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"], ax=axes[0])
    axes[0].set_title(f"Valence 3-Class Confusion Matrix\n({best_val_model})", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Ground Truth")

    sns.heatmap(cm_aro, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=["Low", "Moderate", "High"],
                yticklabels=["Low", "Moderate", "High"], ax=axes[1])
    axes[1].set_title(f"Arousal 3-Class Confusion Matrix\n({best_aro_model})", fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Ground Truth")

    fig.tight_layout()
    out_cm_path = Path("outputs/audio/11_confusion_matrices_ml.png")
    out_cm_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_cm_path, dpi=300)
    plt.close(fig)
    print(f"\n[6] Confusion matrix comparison figure saved to: {out_cm_path}")

    print("\n" + "=" * 80)
    print(" [OK] MACHINE LEARNING DEMONSTRATION COMPLETE.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_ml_demonstration()
