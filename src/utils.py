import librosa
import numpy as np
import pickle
import os
import csv
from datetime import datetime
from config import SETTINGS

SAMPLE_RATE = SETTINGS["SAMPLE_RATE"]
N_MFCC      = SETTINGS["N_MFCC"]


# ── Ses Feature Extraction

def extract_features(y):
    """
    MFCC + Delta MFCC → 160 boyutlu vektör.
    Notebook 02'deki başarılı yöntemle birebir aynı.
    Ses modeli bu boyutla eğitildi — değiştirme!
    """
    mfcc  = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([
        np.mean(mfcc,  axis=1), np.std(mfcc,  axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
    ])

# Geriye dönük uyumluluk için alias
extract_audio_features = extract_features


# ── Titreşim Feature Extraction (MPU6050 verisi için) ────────────────────────

def extract_vibration_features(vibration_window: list) -> np.ndarray:
    """
    MPU6050 penceresi → 18 boyutlu istatistiksel vektör.

    Her okuma: {"ax": float, "ay": float, "az": float}
    Her eksen için: mean, std, min, max, rms, peak-to-peak
    Toplam: 6 istatistik × 3 eksen = 18 özellik

    Bu fonksiyon realtime_detection.py'deki runtime çağrısıyla birebir aynı.
    Notebook 04'teki feature extraction da aynı mantığı kullanır.
    """
    ax = np.array([r["ax"] for r in vibration_window])
    ay = np.array([r["ay"] for r in vibration_window])
    az = np.array([r["az"] for r in vibration_window])

    def stats(arr):
        rms = np.sqrt(np.mean(arr ** 2))
        return [
            np.mean(arr),
            np.std(arr),
            np.min(arr),
            np.max(arr),
            rms,
            np.max(arr) - np.min(arr),   # peak-to-peak
        ]

    features = stats(ax) + stats(ay) + stats(az)
    return np.array(features).reshape(1, -1)


# ── Model Yükleme ─────────────────────────────────────────────────────────────

def load_model_assets(model_id_path):
    """
    Belirli fan ID klasöründen ses modeli ve scaler yükler.
    Beyzanın orijinal fonksiyonu — dokunmadık.
    """
    with open(os.path.join(model_id_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_id_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def load_vibration_model(model_id_path):
    """
    Belirli fan ID klasöründen titreşim modeli ve scaler yükler.
    Notebook 04 çalıştırıldıktan sonra bu dosyalar oluşur:
      models/fan/<id>/vibration_model.pkl
      models/fan/<id>/vibration_scaler.pkl
    """
    model_path  = os.path.join(model_id_path, "vibration_model.pkl")
    scaler_path = os.path.join(model_id_path, "vibration_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Titreşim modeli bulunamadı: {model_path}\n"
            "→ Önce notebooks/04_train_vibration_model.ipynb'i çalıştır."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ── Loglama ───────────────────────────────────────────────────────────────────

def init_log():
    """Log dosyasını başlat (yoksa oluştur)."""
    log_path = SETTINGS["LOG_FILE"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "audio_score", "vibration_score",
                "audio_anomaly", "vibration_anomaly", "trigger", "action"
            ])


def log_event(audio_score, vibration_score, audio_anomaly, vibration_anomaly, trigger, action):
    """Runtime'da her karar döngüsünde bir satır ekler."""
    with open(SETTINGS["LOG_FILE"], "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            round(audio_score, 4),
            round(vibration_score, 4),
            int(audio_anomaly),
            int(vibration_anomaly),
            trigger,
            action,
        ])