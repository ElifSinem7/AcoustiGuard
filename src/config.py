import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETTINGS = {
    # --- AI & Model Ayarları ---
    "MACHINE_TYPE": "fan",
    "MODEL_ID": "id_00",            # Test ettiğin fan ID'si
    "SAMPLE_RATE": 16000,           # Ses örnekleme hızı
    "N_MFCC": 40,                   # MFCC sayısı (Delta ile birlikte 160 boyutlu vektör)

    # --- Karar Mekanizması ---
    # OR mantığı: ses VEYA titreşimden biri anomali → uyarı tetiklenir
    "BUFFER_SIZE": 5,               # Son 5 tahmini hafızada tutar
    "CONFIRMATION_THRESHOLD": 0.8,  # 5 tahminden 4'ü (%80) anomali ise onaylar

    # --- Ses Anomali Eşiği ---
    # Isolation Forest skoru bu değerin altına düşerse anomali sayılır
    "AUDIO_THRESHOLD": 0.0,

    # --- Titreşim Ayarları (MPU6050 → Arduino → UART → RPi) ---
    "VIBRATION_WINDOW_SIZE": 50,    # Kaç MPU6050 okuması birleştirilecek (~0.5s @ 100Hz)
    "VIBRATION_THRESHOLD": 0.0,     # IF skoru bu değerin altı → titreşim anomalisi

    # --- UART Ayarları ---
    "UART_PORT": "/dev/ttyACM0",    # Raspberry Pi portu
    "BAUD_RATE": 9600,              # Arduino'daki Serial.begin ile aynı olmalı

    # --- Arduino Komutları (tek byte) ---
    "CMD_NORMAL":   b"N",           # Normal çalışma
    "CMD_WARNING":  b"W",           # Anomali tespit edildi, hız azaltılıyor
    "CMD_SHUTDOWN": b"S",           # Tam kademeli kapatma

    # --- Loglama ---
    "LOG_FILE": os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "runtime_log.csv"
    ),
}

# ── Kısayollar (inference.py ve utils.py için doğrudan erişim) ───────────────
AUDIO_THRESHOLD     = SETTINGS["AUDIO_THRESHOLD"]
VIBRATION_THRESHOLD = SETTINGS["VIBRATION_THRESHOLD"]