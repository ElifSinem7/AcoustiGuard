# =============================================================================
# AcoustiGuard — Step 4: Isolation Forest Training & Evaluation
# =============================================================================
# Input  : data/processed/acoustiguard_dataset.csv
# Output : models/acoustiguard_if_model.pkl
#          models/acoustiguard_scaler.pkl
#          models/feature_cols.pkl
#          outputs/confusion_matrix.png
#          outputs/score_distribution.png
#          outputs/roc_curve.png
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)

DATASET_PATH = "data/processed/acoustiguard_dataset.csv"
MODEL_DIR    = "models"
OUTPUT_DIR   = "outputs"
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset loaded : {df.shape}")
    print(f"Labels         : {df['label'].value_counts().to_dict()}")
    feature_cols = [c for c in df.columns if c.startswith(("audio_", "vib_"))]
    X = df[feature_cols].values
    y = df["label"].values
    print(f"Features       : {len(feature_cols)}")
    return X, y, feature_cols


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.title("Confusion Matrix — AcoustiGuard")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved -> {path}")


def plot_score_distribution(y_true, scores, path):
    plt.figure(figsize=(8, 4))
    colors = {0: "steelblue", 1: "tomato"}
    names  = {0: "Normal", 1: "Anomaly"}
    for label in [0, 1]:
        plt.hist(scores[y_true == label], bins=40, alpha=0.6,
                 label=names[label], color=colors[label])
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5,
                label="Threshold")
    plt.title("Anomaly Score Distribution (Test Set)")
    plt.xlabel("Decision Score  (lower = more anomalous)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved -> {path}")


def plot_roc(y_true, scores, path):
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(y_true, -scores, ax=ax,
                                         name="Isolation Forest")
        ax.set_title("ROC Curve — AcoustiGuard")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved -> {path}")
    except Exception as e:
        print(f"ROC curve skipped: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load
    print("\n" + "=" * 60)
    print("STEP 1 — Load Data")
    print("=" * 60)
    X, y, feature_cols = load_data(DATASET_PATH)

    # 2. Split
    print("\n" + "=" * 60)
    print("STEP 2 — Train/Test Split  (80/20, stratified)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train_normal = X_train[y_train == 0]
    print(f"Train total        : {len(X_train)}")
    print(f"Train normal only  : {len(X_train_normal)}  (used for IF fitting)")
    print(f"Test total         : {len(X_test)}")

    # 3. Normalize
    print("\n" + "=" * 60)
    print("STEP 3 — StandardScaler Normalization")
    print("=" * 60)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_normal_sc = scaler.transform(X_train_normal)
    X_test_sc         = scaler.transform(X_test)
    print("Scaler fitted on full train set.")

    # 4. Train Isolation Forest
    print("\n" + "=" * 60)
    print("STEP 4 — Isolation Forest Training")
    print("=" * 60)
    contamination = float(np.clip((y_train == 1).mean(), 0.01, 0.5))
    print(f"Contamination (from labels) : {contamination:.4f}")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_normal_sc)
    print("Training complete.")

    # 5. Evaluate
    print("\n" + "=" * 60)
    print("STEP 5 — Evaluation")
    print("=" * 60)

    raw_pred = model.predict(X_test_sc)            # 1=normal, -1=anomaly
    y_pred   = np.where(raw_pred == -1, 1, 0)      # convert: anomaly=1
    scores   = model.decision_function(X_test_sc)  # lower = more anomalous

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    auc = roc_auc_score(y_test, -scores)
    print(f"ROC-AUC Score : {auc:.4f}")

    plot_confusion_matrix(y_test, y_pred,
                          os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_score_distribution(y_test, scores,
                            os.path.join(OUTPUT_DIR, "score_distribution.png"))
    plot_roc(y_test, scores,
             os.path.join(OUTPUT_DIR, "roc_curve.png"))

    # 6. Save
    print("\n" + "=" * 60)
    print("STEP 6 — Save Model & Scaler")
    print("=" * 60)

    joblib.dump(model,        os.path.join(MODEL_DIR, "acoustiguard_if_model.pkl"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "acoustiguard_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    print(f"Model    -> {MODEL_DIR}/acoustiguard_if_model.pkl")
    print(f"Scaler   -> {MODEL_DIR}/acoustiguard_scaler.pkl")
    print(f"Features -> {MODEL_DIR}/feature_cols.pkl")
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
