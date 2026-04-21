import sys
import os

# src/ klasörü import path'e eklenir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SETTINGS
from realtime_detection import main

if __name__ == "__main__":
    print("=" * 55)
    print("  AcoustiGuard — Ses + Titreşim Anomali Tespiti")
    print(f"  Model: {SETTINGS['MACHINE_TYPE']}/{SETTINGS['MODEL_ID']}")
    print(f"  OR mantığı: ses VEYA titreşim → kademeli kapatma")
    print("=" * 55)
    main()