# =============================================================================
# AcoustiGuard — Step 3: Merge MIMII (audio) + CWRU (vibration)
# =============================================================================
# Input  : data/processed/mimii_features.csv
#          data/processed/cwru_features.csv
# Output : data/processed/acoustiguard_dataset.csv
#
# Merge strategy: LABEL-BASED PAIRING
#   Normal  audio rows paired with Normal  vibration rows (random sample)
#   Anomaly audio rows paired with Anomaly vibration rows (random sample)
#   Shorter side determines the pair count (no data leakage).
# =============================================================================

import os
import numpy as np
import pandas as pd

MIMII_CSV   = "data/processed/mimii_features.csv"
CWRU_CSV    = "data/processed/cwru_features.csv"
OUTPUT_CSV  = "data/processed/acoustiguard_dataset.csv"
RANDOM_SEED = 42


def load_and_validate(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found at '{path}'. Run extraction script first."
        )
    df = pd.read_csv(path)
    assert "label" in df.columns, f"'label' column missing in {name}"
    print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")
    return df


def get_feature_cols(df, prefix):
    meta = {"label", "source", "machine_id", "condition", "folder", "filename"}
    return [c for c in df.columns if c.startswith(prefix) and c not in meta]


def merge_by_label(mimii, cwru, label, rng):
    audio_rows = mimii[mimii["label"] == label].reset_index(drop=True)
    vib_rows   = cwru[cwru["label"] == label].reset_index(drop=True)

    n = min(len(audio_rows), len(vib_rows))
    if n == 0:
        print(f"  WARNING: no samples for label={label}")
        return pd.DataFrame()

    a_idx = rng.choice(len(audio_rows), size=n, replace=False)
    v_idx = rng.choice(len(vib_rows),   size=n, replace=False)

    a_cols = get_feature_cols(audio_rows, "audio_")
    v_cols = get_feature_cols(vib_rows,   "vib_")

    a_sub = audio_rows.iloc[a_idx][a_cols].reset_index(drop=True)
    v_sub = vib_rows.iloc[v_idx][v_cols].reset_index(drop=True)

    merged = pd.concat([a_sub, v_sub], axis=1)
    merged["label"] = label
    return merged


def main():
    os.makedirs("data/processed", exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("Merging MIMII + CWRU")
    print("=" * 60)

    mimii = load_and_validate(MIMII_CSV, "MIMII")
    cwru  = load_and_validate(CWRU_CSV,  "CWRU")

    normal_df  = merge_by_label(mimii, cwru, label=0, rng=rng)
    anomaly_df = merge_by_label(mimii, cwru, label=1, rng=rng)

    dataset = pd.concat([normal_df, anomaly_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    audio_n = len([c for c in dataset.columns if c.startswith("audio_")])
    vib_n   = len([c for c in dataset.columns if c.startswith("vib_")])

    print(f"\nFinal dataset shape : {dataset.shape}")
    print(f"Labels              : {dataset['label'].value_counts().to_dict()}")
    print(f"Audio features      : {audio_n}")
    print(f"Vibration features  : {vib_n}")
    print(f"Missing values      : {dataset.isnull().sum().sum()}")

    dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
