# src/config.py

import os

# Projenin ana dizinini bulur
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETTINGS = {
    # --- AI & Model Ayarları ---
    "MACHINE_TYPE": "fan",
    "MODEL_ID": "id_00",            # Test ettiğin fan ID'si
    "SAMPLE_RATE": 16000,           # Ses örnekleme hızı [cite: 20]
    "N_MFCC": 40,                   # Çıkardığın MFCC sayısı
    
    # --- Karar Mekanizması (Evham Önleyici) ---
    "BUFFER_SIZE": 5,               # Son 5 tahmini hafızada tutar
    "CONFIRMATION_THRESHOLD": 0.8,  # 5 tahminden 4'ü (%80) anomali ise onaylar
    
    # --- Donanım & UART (Arkadaşının Arduino koduyla uyumlu) ---
    "UART_PORT": "/dev/ttyACM0",    # Raspberry Pi portu (Windows'ta 'COM3' gibi olabilir)
    "BAUD_RATE": 9600,              # Arduino'daki Serial.begin(9600) ile aynı olmalı [cite: 4]
    "UART_COMMAND": b"ANOMALY\n"    # Arduino'nun beklediği tam metin 
}