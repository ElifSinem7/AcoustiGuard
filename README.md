# AcoustiGuard
**Acoustic Anomaly Detection & Predictive Maintenance System**

Raspberry Pi 4 + Arduino Uno R4 (MPU-6050) tabanlı,
MIMII (ses) + CWRU (titreşim) veri setleri ile eğitilmiş
Isolation Forest anomali tespit sistemi.

---

## Proje Dizin Yapısı

```
AcoustiGuard/
├── download_datasets.py       # Dataset otomatik indirici
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── mimii/fan/         # WAV dosyaları (otomatik indirilir)
│   │   └── cwru/              # .mat dosyaları (otomatik indirilir)
│   └── processed/             # CSV feature dosyaları (script üretir)
├── models/                    # Eğitilmiş model ve scaler (script üretir)
├── outputs/                   # Grafikler (script üretir)
└── src/
    ├── extract_mimii.py       # MIMII ses feature çıkarımı
    ├── extract_cwru.py        # CWRU titreşim feature çıkarımı
    ├── merge_datasets.py      # İki dataset birleştirme
    ├── train_isolation_forest.py  # Model eğitimi ve değerlendirme
    └── realtime_detection.py  # RPi gerçek zamanlı inference
```

---

## Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt

# 2. Datasetleri otomatik indir
python download_datasets.py

# 3. Ses özelliklerini çıkar (MIMII)
python src/extract_mimii.py

# 4. Titreşim özelliklerini çıkar (CWRU)
python src/extract_cwru.py

# 5. İki dataseti birleştir
python src/merge_datasets.py

# 6. Isolation Forest eğit ve değerlendir
python src/train_isolation_forest.py

# 7. Raspberry Pi'de gerçek zamanlı çalıştır
python src/realtime_detection.py
```

---

## Donanım

| Bileşen       | Görev                              |
|---------------|------------------------------------|
| Raspberry Pi 4 | Feature extraction + IF inference  |
| Arduino Uno R4 | MPU-6050 okuma + aktuatör kontrol  |
| MPU-6050      | 3 eksen titreşim verisi            |
| Mikrofon      | Ses sinyali                        |

---

## Veri Setleri

- **MIMII**: Industrial machine sound dataset (Zenodo, 16kHz WAV)
- **CWRU**: Case Western Reserve University Bearing Dataset (.mat)
