import time
import json
import threading
import numpy as np
import pyaudio
import serial
import os

from config import SETTINGS
from utils import (
    load_model_assets,
    load_vibration_model,
    init_log,
    log_event,
)
from inference import AnomalyDetector

# ── Config kısayolları ────────────────────────────────────────────────────────
MODEL_ID     = SETTINGS["MODEL_ID"]
MACHINE_TYPE = SETTINGS["MACHINE_TYPE"]
SAMPLE_RATE  = SETTINGS["SAMPLE_RATE"]
UART_PORT    = SETTINGS["UART_PORT"]
BAUD_RATE    = SETTINGS["BAUD_RATE"]
VIB_WIN      = SETTINGS["VIBRATION_WINDOW_SIZE"]
CMD_NORMAL   = SETTINGS["CMD_NORMAL"]
CMD_WARNING  = SETTINGS["CMD_WARNING"]
CMD_SHUTDOWN = SETTINGS["CMD_SHUTDOWN"]

BASE_DIR      = SETTINGS.get(
    "BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MODEL_ID_PATH = os.path.join(BASE_DIR, "models", MACHINE_TYPE, MODEL_ID)

CHUNK = 1024 * 8   # pyaudio chunk boyutu

# ── Titreşim buffer (UART reader thread tarafından doldurulur) ────────────────
vibration_buffer: list = []
buffer_lock = threading.Lock()


# ── UART Reader Thread ────────────────────────────────────────────────────────

def uart_reader(ser: serial.Serial):
    """
    Arduino'dan sürekli JSON satırları okur ve vibration_buffer'a ekler.
    Arduino her ~10ms'de bir satır gönderir:
        {"ax": 0.12, "ay": -0.03, "az": 9.81}
    """
    global vibration_buffer
    print("[UART] Okuma thread'i başladı")
    while True:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue
            data = json.loads(raw)
            if {"ax", "ay", "az"}.issubset(data.keys()):
                with buffer_lock:
                    vibration_buffer.append(data)
                    # Buffer'ın aşırı büyümesini engelle
                    if len(vibration_buffer) > VIB_WIN * 4:
                        vibration_buffer = vibration_buffer[-VIB_WIN * 2:]
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        except Exception as e:
            print(f"[UART] Okuma hatası: {e}")
            time.sleep(0.1)


# ── Arduino Komut Gönderme ────────────────────────────────────────────────────

def send_command(ser: serial.Serial, cmd: bytes, label: str):
    ser.write(cmd)
    ser.flush()
    print(f"[CMD → Arduino] {label}")


# ── Kademeli Kapatma ──────────────────────────────────────────────────────────

def graceful_shutdown(ser: serial.Serial, trigger: str):
    """
    İki aşamalı kademeli kapatma:
      Faz 1 — WARNING (W): Arduino motoru yavaşlatır, LED turuncu
      Faz 2 — SHUTDOWN (S): Arduino motoru durdurur, servo fren devreye girer
    Hangi kanalın tetiklediği log'a yazılır.
    """
    print("\n" + "=" * 55)
    print(f"  ANOMALİ ONAYLANDI  |  Tetikleyen: {trigger.upper()}")
    print("=" * 55)

    send_command(ser, CMD_WARNING, "WARNING — motor yavaşlıyor")
    time.sleep(2.0)

    send_command(ser, CMD_SHUTDOWN, "SHUTDOWN — tam durdurma")
    print("[SYS] Kademeli kapatma komutu Arduino'ya gönderildi.")
    print("[SYS] Sistem durdu. Yeniden başlatmak için programı yeniden çalıştır.")


# ── Ana Döngü ─────────────────────────────────────────────────────────────────

def main():
    init_log()

    # Modelleri yükle
    print("[SYS] Modeller yükleniyor...")
    audio_model, audio_scaler         = load_model_assets(MODEL_ID_PATH)
    vibration_model, vibration_scaler = load_vibration_model(MODEL_ID_PATH)

    detector = AnomalyDetector(
        audio_model, audio_scaler,
        vibration_model, vibration_scaler,
        buffer_size=SETTINGS["BUFFER_SIZE"],
    )
    print(f"[SYS] Modeller yüklendi → {MODEL_ID_PATH}")

    # UART aç
    print(f"[SYS] UART açılıyor: {UART_PORT} @ {BAUD_RATE}")
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        time.sleep(2.0)   # Arduino reset beklenir
        print(f"[SYS] UART bağlantısı kuruldu: {UART_PORT}")
    except Exception as e:
        print(f"[HATA] UART açılamadı: {e}")
        print("  → Arduino bağlı mı? Port doğru mu?")
        ser = None

    # UART reader thread'i başlat
    if ser:
        reader = threading.Thread(target=uart_reader, args=(ser,), daemon=True)
        reader.start()

    # Mikrofon aç
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print(f"[SYS] AcoustiGuard çalışıyor — {MODEL_ID} izleniyor")
    print("[SYS] OR mantığı: ses VEYA titreşim anomalisi → kademeli kapatma")
    print("[SYS] Durdurmak için Ctrl+C\n")

    if ser:
        send_command(ser, CMD_NORMAL, "NORMAL — sistem hazır")

    try:
        while True:
            # 1. Ses verisi oku
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # 2. Titreşim penceresini al
            if ser:
                with buffer_lock:
                    if len(vibration_buffer) < VIB_WIN:
                        print(f"[VIB] Buffer dolmadı ({len(vibration_buffer)}/{VIB_WIN}), bekleniyor...")
                        time.sleep(0.2)
                        continue
                    vib_window = vibration_buffer[-VIB_WIN:]
            else:
                # UART yoksa sıfır titreşim simüle et (geliştirme modu)
                vib_window = [{"ax": 0.0, "ay": 0.0, "az": 1.0}] * VIB_WIN

            # 3. Anomali kararı (OR + buffer filtresi)
            result = detector.decide(audio, vib_window)

            a_score  = result["audio_score"]
            v_score  = result["vibration_score"]
            a_anom   = result["audio_anomaly"]
            v_anom   = result["vibration_anomaly"]
            combined = result["combined_anomaly"]
            trigger  = result["trigger"]

            # 4. Log
            log_event(
                audio_score=a_score,
                vibration_score=v_score,
                audio_anomaly=a_anom,
                vibration_anomaly=v_anom,
                trigger=trigger,
                action="anomaly" if combined else "normal",
            )

            # 5. Konsol çıktısı
            ts = time.strftime("%H:%M:%S")
            a_mark = "⚠" if a_anom else "✓"
            v_mark = "⚠" if v_anom else "✓"
            print(
                f"[{ts}] "
                f"Ses: {a_score:+.4f} {a_mark}  |  "
                f"Titreşim: {v_score:+.4f} {v_mark}  |  "
                f"Tetikleyen: {trigger}"
            )

            # 6. Kapatma kararı
            if combined and ser:
                graceful_shutdown(ser, trigger)
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[SYS] Kullanıcı tarafından durduruldu.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        if ser:
            ser.close()
        print("[SYS] Kaynaklar kapatıldı.")


if __name__ == "__main__":
    main()