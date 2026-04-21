import time
import serial
import pyaudio
import numpy as np
import os
from utils import extract_features, load_model_assets
from inference import AnomalyDetector
# Yeni eklediğimiz config dosyasını içeri alıyoruz
from config import SETTINGS 

# --- KONFİGÜRASYON (Config dosyasından çekiliyor) ---
MODEL_ID = SETTINGS["MODEL_ID"]
UART_PORT = SETTINGS["UART_PORT"] 
BAUD_RATE = SETTINGS["BAUD_RATE"]
CHUNK = 1024 * 8 

# 1. Modelleri ve UART'ı Hazırla 
# Dizin yapısına göre yol belirleniyor
model_path = os.path.join(SETTINGS["BASE_DIR"], "models", "fan", MODEL_ID)
model, scaler = load_model_assets(model_path)
# Evham önleyici filtre (buffer_size) config'den geliyor
detector = AnomalyDetector(model, scaler, buffer_size=SETTINGS["BUFFER_SIZE"])

try:
    ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
    print(f"UART bağlantısı kuruldu: {UART_PORT}")
except Exception as e:
    print(f"UART hatası! Arduino bağlı mı? Hata: {e}")
    ser = None

# 2. Mikrofon Akışını Başlat 
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, 
                channels=1, 
                rate=SETTINGS["SAMPLE_RATE"],
                input=True, 
                frames_per_buffer=CHUNK)

print(f"Sistem hazır. {MODEL_ID} cihazı izleniyor...")

try:
    while True:
        # Ses verisi oku
        data = stream.read(CHUNK, exception_on_overflow=False)
        y = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        
        # AI Tahmini (MFCC + Delta MFCC) 
        features = extract_features(y)
        
        # Filtrelenmiş anomali kararı
        if detector.predict(features):
            print("!!! GERÇEK ANOMALİ TESPİT EDİLDİ !!!")
            
            # Arduino'ya (Kisi 1) tam uyumlu komut gönder 
            if ser:
                ser.write(SETTINGS["UART_COMMAND"]) 
                print(f"UART: {SETTINGS['UART_COMMAND'].decode()} gönderildi.")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Sistem kullanıcı tarafından durduruluyor...")
finally:
    # Kaynakları güvenli kapatma
    if ser:
        ser.close()
    stream.stop_stream()
    stream.close()
    p.terminate()