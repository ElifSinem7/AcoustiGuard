import numpy as np
from collections import deque
from config import SETTINGS
from utils import extract_features, extract_vibration_features

AUDIO_THRESHOLD     = SETTINGS["AUDIO_THRESHOLD"]
VIBRATION_THRESHOLD = SETTINGS["VIBRATION_THRESHOLD"]


class AnomalyDetector:
    """
    Ses + titreşim Isolation Forest modellerini saran sınıf.

    OR mantığı: ses VEYA titreşimden biri anomali → combined_anomaly = True
    Buffer mantığı: son N kararın %80'i anomali ise onaylar (evham önleyici).
    """

    def __init__(self,
                 audio_model, audio_scaler,
                 vibration_model, vibration_scaler,
                 buffer_size: int = None):

        self.audio_model      = audio_model
        self.audio_scaler     = audio_scaler
        self.vibration_model  = vibration_model
        self.vibration_scaler = vibration_scaler

        buf = buffer_size if buffer_size else SETTINGS["BUFFER_SIZE"]
        self.conf_threshold = SETTINGS["CONFIRMATION_THRESHOLD"]

        # Ayrı buffer: her kanal için son N karar tutulur
        self._audio_buffer     = deque(maxlen=buf)
        self._vibration_buffer = deque(maxlen=buf)

    # ── Ses ───────────────────────────────────────────────────────────────────

    def score_audio(self, audio_signal: np.ndarray) -> tuple:
        """
        Ham ses sinyali → (skor, anomali_mi)
        extract_features: MFCC + Delta MFCC → 160 boyut (notebook 02 ile aynı)
        """
        feat   = extract_features(audio_signal).reshape(1, -1)
        scaled = self.audio_scaler.transform(feat)
        score  = float(self.audio_model.decision_function(scaled)[0])
        return score, score < AUDIO_THRESHOLD

    # ── Titreşim ──────────────────────────────────────────────────────────────

    def score_vibration(self, vibration_window: list) -> tuple:
        """
        MPU6050 penceresi → (skor, anomali_mi)
        vibration_window: [{"ax":…, "ay":…, "az":…}, …]
        """
        feat   = extract_vibration_features(vibration_window)
        scaled = self.vibration_scaler.transform(feat)
        score  = float(self.vibration_model.decision_function(scaled)[0])
        return score, score < VIBRATION_THRESHOLD

    # ── Buffer'lı Karar (evham önleyici) ─────────────────────────────────────

    def _buffered_decision(self, raw_audio_anom: bool, raw_vib_anom: bool) -> tuple:
        """
        Ham kararları buffer'a ekler.
        Buffer dolduğunda: her kanalın anomali oranı eşiği geçerse onaylar.
        Buffer dolmadan: ham karara güvenir (başlangıç için).
        """
        self._audio_buffer.append(int(raw_audio_anom))
        self._vibration_buffer.append(int(raw_vib_anom))

        buf_full = len(self._audio_buffer) == self._audio_buffer.maxlen

        if buf_full:
            audio_confirmed = (
                sum(self._audio_buffer) / len(self._audio_buffer)
                >= self.conf_threshold
            )
            vib_confirmed = (
                sum(self._vibration_buffer) / len(self._vibration_buffer)
                >= self.conf_threshold
            )
        else:
            # Buffer henüz dolmadı — ham karara güven
            audio_confirmed = raw_audio_anom
            vib_confirmed   = raw_vib_anom

        return audio_confirmed, vib_confirmed

    # ── Ana Karar Fonksiyonu ──────────────────────────────────────────────────

    def decide(self, audio_signal: np.ndarray, vibration_window: list) -> dict:
        """
        Her iki kanalı çalıştır, buffer filtrele, OR mantığıyla birleştir.

        Döndürür:
            audio_score        : float  (IF skoru)
            audio_anomaly      : bool   (buffer onaylı)
            vibration_score    : float
            vibration_anomaly  : bool   (buffer onaylı)
            combined_anomaly   : bool   (ses VEYA titreşim anomali)
            trigger            : str    "audio" | "vibration" | "both" | "none"
        """
        a_score, a_raw = self.score_audio(audio_signal)
        v_score, v_raw = self.score_vibration(vibration_window)

        a_anom, v_anom = self._buffered_decision(a_raw, v_raw)

        combined = a_anom or v_anom

        if a_anom and v_anom:
            trigger = "both"
        elif a_anom:
            trigger = "audio"
        elif v_anom:
            trigger = "vibration"
        else:
            trigger = "none"

        return {
            "audio_score":       a_score,
            "audio_anomaly":     a_anom,
            "vibration_score":   v_score,
            "vibration_anomaly": v_anom,
            "combined_anomaly":  combined,
            "trigger":           trigger,
        }