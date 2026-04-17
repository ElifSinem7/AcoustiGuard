# =============================================================================
# AcoustiGuard — Step 1: MIMII Audio Feature Extraction
# =============================================================================
# Input  : data/raw/mimii/fan/{id_XX}/{normal|abnormal}/*.wav
# Output : data/processed/mimii_features.csv
#
# Features extracted per WAV file (30 total):
#   - MFCC 1-13 mean & std  (26 features)
#   - RMS energy mean & std
#   - Zero Crossing Rate mean & std
#   - Spectral Centroid mean & std
#   - Spectral Bandwidth mean & std
#   - Spectral Rolloff mean & std
# =============================================================================

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────────
MIMII_ROOT  = "data/raw/mimii/fan"
OUTPUT_CSV  = "data/processed/mimii_features.csv"
SAMPLE_RATE = 16000
DURATION    = 10       # seconds
N_MFCC      = 13
HOP_LENGTH  = 512
# ─────────────────────────────────────────────────────────────────────────────


def extract_audio_features(wav_path: str) -> dict:
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features[f"audio_mfcc{i+1:02d}_mean"] = float(np.mean(mfcc[i]))
        features[f"audio_mfcc{i+1:02d}_std"]  = float(np.std(mfcc[i]))

    # RMS
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features["audio_rms_mean"] = float(np.mean(rms))
    features["audio_rms_std"]  = float(np.std(rms))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
    features["audio_zcr_mean"] = float(np.mean(zcr))
    features["audio_zcr_std"]  = float(np.std(zcr))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    features["audio_centroid_mean"] = float(np.mean(centroid))
    features["audio_centroid_std"]  = float(np.std(centroid))

    # Spectral Bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    features["audio_bandwidth_mean"] = float(np.mean(bw))
    features["audio_bandwidth_std"]  = float(np.std(bw))

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    features["audio_rolloff_mean"] = float(np.mean(rolloff))
    features["audio_rolloff_std"]  = float(np.std(rolloff))

    return features


def process_mimii() -> pd.DataFrame:
    records = []

    for machine_id in sorted(os.listdir(MIMII_ROOT)):
        machine_path = os.path.join(MIMII_ROOT, machine_id)
        if not os.path.isdir(machine_path):
            continue

        for condition in ["normal", "abnormal"]:
            cond_path = os.path.join(machine_path, condition)
            if not os.path.isdir(cond_path):
                continue

            wav_files = [f for f in os.listdir(cond_path) if f.endswith(".wav")]
            label = 0 if condition == "normal" else 1
            print(f"  {machine_id}/{condition}: {len(wav_files)} files")

            for wav_file in tqdm(wav_files, desc=f"{machine_id}/{condition}", ncols=80):
                try:
                    feats = extract_audio_features(os.path.join(cond_path, wav_file))
                    feats["machine_id"] = machine_id
                    feats["condition"]  = condition
                    feats["label"]      = label
                    feats["source"]     = "mimii"
                    records.append(feats)
                except Exception as e:
                    print(f"\n    ERROR {wav_file}: {e}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("=" * 60)
    print("MIMII Feature Extraction")
    print("=" * 60)

    df = process_mimii()

    print(f"\nTotal samples : {len(df)}")
    print(f"Labels        : {df['label'].value_counts().to_dict()}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")
