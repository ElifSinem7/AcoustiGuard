# =============================================================================
# AcoustiGuard — Step 2: CWRU Vibration Feature Extraction
# =============================================================================
# Input  : data/raw/cwru/{NormalBaseline|12DriveEndFault}/*.mat
# Output : data/processed/cwru_features.csv
#
# Features extracted per window segment (19 total):
#   Time-domain  : mean, std, rms, max, peak2peak, skewness, kurtosis,
#                  crest_factor, shape_factor
#   Freq-domain  : fft_mean, fft_std, fft_max, dominant_freq,
#                  spectral_centroid, band_low, band_mid, band_high
# =============================================================================

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────────
CWRU_ROOT       = "data/raw/cwru"
OUTPUT_CSV      = "data/processed/cwru_features.csv"
SAMPLE_RATE     = 12000    # Hz
WINDOW_SIZE     = 1024     # samples per segment
OVERLAP         = 512      # overlap between windows
NORMAL_FOLDERS  = ["NormalBaseline"]
FAULT_FOLDERS   = ["12DriveEndFault", "12FanEndFault", "48DriveEndFault"]
# ─────────────────────────────────────────────────────────────────────────────


def extract_vibration_features(signal: np.ndarray) -> dict:
    features = {}

    # Time domain
    features["vib_mean"]       = float(np.mean(signal))
    features["vib_std"]        = float(np.std(signal))
    features["vib_rms"]        = float(np.sqrt(np.mean(signal ** 2)))
    features["vib_max"]        = float(np.max(np.abs(signal)))
    features["vib_peak2peak"]  = float(np.ptp(signal))
    features["vib_skewness"]   = float(stats.skew(signal))
    features["vib_kurtosis"]   = float(stats.kurtosis(signal))

    rms = features["vib_rms"]
    mean_abs = float(np.mean(np.abs(signal)))
    features["vib_crest_factor"] = float(features["vib_max"] / rms) if rms > 0 else 0.0
    features["vib_shape_factor"] = float(rms / mean_abs) if mean_abs > 0 else 0.0

    # Frequency domain
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs    = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLE_RATE)

    features["vib_fft_mean"]      = float(np.mean(fft_vals))
    features["vib_fft_std"]       = float(np.std(fft_vals))
    features["vib_fft_max"]       = float(np.max(fft_vals))
    features["vib_dominant_freq"] = float(freqs[np.argmax(fft_vals)])

    total_power = np.sum(fft_vals)
    features["vib_spectral_centroid"] = (
        float(np.sum(freqs * fft_vals) / total_power) if total_power > 0 else 0.0
    )

    features["vib_band_low"]  = float(np.sum(fft_vals[freqs < 1000]))
    features["vib_band_mid"]  = float(np.sum(fft_vals[(freqs >= 1000) & (freqs < 3000)]))
    features["vib_band_high"] = float(np.sum(fft_vals[freqs >= 3000]))

    return features


def load_mat_signals(mat_path: str) -> list:
    """Load .mat file and return all accelerometer time series found."""
    try:
        mat = sio.loadmat(mat_path)
    except Exception:
        try:
            import h5py
            signals = []
            with h5py.File(mat_path, "r") as f:
                for key in f.keys():
                    if key.startswith("__"):
                        continue
                    arr = np.array(f[key]).flatten()
                    if len(arr) > WINDOW_SIZE:
                        signals.append(arr)
            return signals
        except Exception as e:
            print(f"    Cannot load {mat_path}: {e}")
            return []

    signals = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        key_lower = key.lower()
        if any(tag in key_lower for tag in ["_time", "de_time", "fe_time", "ba_time"]):
            arr = np.array(value).flatten()
            if len(arr) > WINDOW_SIZE:
                signals.append(arr)

    # Fallback: any large numeric array
    if not signals:
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            try:
                arr = np.array(value, dtype=float).flatten()
                if len(arr) > WINDOW_SIZE:
                    signals.append(arr)
            except Exception:
                pass

    return signals


def segment_signal(signal: np.ndarray, label: int,
                   folder: str, filename: str) -> list:
    records = []
    start = 0
    while start + WINDOW_SIZE <= len(signal):
        segment = signal[start: start + WINDOW_SIZE]
        feats = extract_vibration_features(segment)
        feats["folder"]   = folder
        feats["filename"] = filename
        feats["label"]    = label
        feats["source"]   = "cwru"
        records.append(feats)
        start += (WINDOW_SIZE - OVERLAP)
    return records


def process_cwru() -> pd.DataFrame:
    all_records = []
    folder_label = {f: 0 for f in NORMAL_FOLDERS}
    folder_label.update({f: 1 for f in FAULT_FOLDERS})

    for folder, label in folder_label.items():
        folder_path = os.path.join(CWRU_ROOT, folder)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: not found — {folder_path}")
            continue

        mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
        condition = "normal" if label == 0 else "fault"
        print(f"  {folder} ({condition}): {len(mat_files)} .mat files")

        for mat_file in tqdm(mat_files, desc=folder, ncols=80):
            signals = load_mat_signals(os.path.join(folder_path, mat_file))
            for sig in signals:
                all_records.extend(segment_signal(sig, label, folder, mat_file))

    return pd.DataFrame(all_records)


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("=" * 60)
    print("CWRU Vibration Feature Extraction")
    print("=" * 60)

    df = process_cwru()

    print(f"\nTotal segments : {len(df)}")
    print(f"Labels         : {df['label'].value_counts().to_dict()}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")
