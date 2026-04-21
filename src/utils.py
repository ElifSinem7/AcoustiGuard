import librosa
import numpy as np
import pickle
import os

# PDF'deki donanım ve yazılım ayarlarına göre [cite: 4, 10]
SAMPLE_RATE = 16000
N_MFCC = 40

def extract_features(y):
    """MFCC + Delta MFCC -> 160 boyutlu vektör (Notebook'taki başarılı yöntem) """
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    
    # Ortalama ve Standart Sapma birleşimi
    return np.concatenate([
        np.mean(mfcc,  axis=1), np.std(mfcc,  axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
    ])

def load_model_assets(model_id_path):
    """Belirli fan ID klasöründen model ve scaler yükler"""
    with open(os.path.join(model_id_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_id_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler