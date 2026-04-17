# =============================================================================
# AcoustiGuard — Step 5: Real-Time Inference (Raspberry Pi 4)
# =============================================================================
# Bu script Raspberry Pi'de calismak icin tasarlanmistir.
#
# Bagimliliklari RPi'ye kur:
#   pip install numpy scipy librosa sounddevice joblib pyserial
#
# Calistir:
#   python realtime_detection.py
#
# Arduino Uno R4 — MPU-6050 Serial Protocol:
#   Arduino her dongu adiminda tek satir float gonderir: "0.1234\n"
#   Yani Serial.println(accel_z_value); seklinde
# =============================================================================

import time
import numpy as np
import joblib
import librosa
import sounddevice as sd
import serial
from scipy import stats

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/acoustiguard_if_model.pkl"
SCALER_PATH  = "models/acoustiguard_scaler.pkl"
FEAT_PATH    = "models/feature_cols.pkl"

AUDIO_SR     = 16000   # MIMII training sample rate
AUDIO_DUR    = 10      # seconds per window

VIB_SR       = 12000   # CWRU training sample rate
VIB_WINDOW   = 1024    # samples per vibration window

ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 9600

N_MFCC       = 13
HOP_LENGTH   = 512
# ─────────────────────────────────────────────────────────────────────────────


def extract_audio_features(y, sr=AUDIO_SR):
    features = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features[f"audio_mfcc{i+1:02d}_mean"] = float(np.mean(mfcc[i]))
        features[f"audio_mfcc{i+1:02d}_std"]  = float(np.std(mfcc[i]))

    for name, arr in [
        ("rms",       librosa.feature.rms(y=y, hop_length=HOP_LENGTH)),
        ("zcr",       librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)),
        ("centroid",  librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)),
        ("bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)),
        ("rolloff",   librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)),
    ]:
        features[f"audio_{name}_mean"] = float(np.mean(arr))
        features[f"audio_{name}_std"]  = float(np.std(arr))

    return features


def extract_vibration_features(signal):
    features = {}

    features["vib_mean"]      = float(np.mean(signal))
    features["vib_std"]       = float(np.std(signal))
    features["vib_rms"]       = float(np.sqrt(np.mean(signal ** 2)))
    features["vib_max"]       = float(np.max(np.abs(signal)))
    features["vib_peak2peak"] = float(np.ptp(signal))
    features["vib_skewness"]  = float(stats.skew(signal))
    features["vib_kurtosis"]  = float(stats.kurtosis(signal))

    rms      = features["vib_rms"]
    mean_abs = float(np.mean(np.abs(signal)))
    features["vib_crest_factor"] = float(features["vib_max"] / rms) if rms > 0 else 0.0
    features["vib_shape_factor"] = float(rms / mean_abs) if mean_abs > 0 else 0.0

    fft_vals = np.abs(np.fft.rfft(signal))
    freqs    = np.fft.rfftfreq(len(signal), d=1.0 / VIB_SR)

    features["vib_fft_mean"]      = float(np.mean(fft_vals))
    features["vib_fft_std"]       = float(np.std(fft_vals))
    features["vib_fft_max"]       = float(np.max(fft_vals))
    features["vib_dominant_freq"] = float(freqs[np.argmax(fft_vals)])

    total = np.sum(fft_vals)
    features["vib_spectral_centroid"] = float(np.sum(freqs * fft_vals) / total) if total > 0 else 0.0
    features["vib_band_low"]  = float(np.sum(fft_vals[freqs < 1000]))
    features["vib_band_mid"]  = float(np.sum(fft_vals[(freqs >= 1000) & (freqs < 3000)]))
    features["vib_band_high"] = float(np.sum(fft_vals[freqs >= 3000]))

    return features


def build_vector(audio_feats, vib_feats, feature_cols):
    combined = {**audio_feats, **vib_feats}
    return np.array([combined.get(col, 0.0) for col in feature_cols], dtype=float)


def read_mpu6050(arduino, n_samples=VIB_WINDOW):
    """
    Arduino'dan n_samples adet float okur.
    Arduino kodu ornegi:
      void loop() {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        Serial.println(a.acceleration.z);
        delay(1);
      }
    """
    samples = []
    arduino.reset_input_buffer()
    while len(samples) < n_samples:
        line = arduino.readline().decode("utf-8", errors="ignore").strip()
        try:
            samples.append(float(line))
        except ValueError:
            pass
    return np.array(samples[:n_samples])


def main():
    print("=" * 50)
    print("AcoustiGuard — Real-Time Anomaly Detection")
    print("=" * 50)

    print("Loading model...", end=" ")
    model        = joblib.load(MODEL_PATH)
    scaler       = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    print(f"OK  ({len(feature_cols)} features)")

    # Arduino baglantisi
    arduino = None
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=2)
        time.sleep(2)
        print(f"Arduino connected: {ARDUINO_PORT}")
    except Exception as e:
        print(f"Arduino not connected: {e}")
        print("Running in AUDIO-ONLY mode (vibration = zeros)")

    print("\nListening... Press Ctrl+C to stop.\n")

    try:
        while True:
            # 1. Ses kaydet
            print("Recording audio...", end=" ", flush=True)
            audio = sd.rec(int(AUDIO_SR * AUDIO_DUR),
                           samplerate=AUDIO_SR, channels=1, dtype="float32")
            sd.wait()
            audio_feats = extract_audio_features(audio.flatten())
            print("done")

            # 2. Titresim oku
            if arduino:
                print("Reading vibration...", end=" ", flush=True)
                vib_signal = read_mpu6050(arduino)
                vib_feats  = extract_vibration_features(vib_signal)
                print("done")
            else:
                vib_feats = {col: 0.0 for col in feature_cols
                             if col.startswith("vib_")}

            # 3. Tahmin
            x_raw    = build_vector(audio_feats, vib_feats, feature_cols)
            x_scaled = scaler.transform(x_raw.reshape(1, -1))
            score    = model.decision_function(x_scaled)[0]
            pred     = model.predict(x_scaled)[0]
            is_anom  = pred == -1

            # 4. Sonuc
            status = "ANOMALY DETECTED" if is_anom else "Normal"
            icon   = "WARNING" if is_anom else "OK"
            print(f"[{icon}] {status}  |  score: {score:+.4f}\n")

            # 5. Arduino'ya bildir
            if arduino:
                arduino.write(b"ANOMALY\n" if is_anom else b"NORMAL\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")
        if arduino:
            arduino.close()


if __name__ == "__main__":
    main()
