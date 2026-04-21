import numpy as np

class AnomalyDetector:
    def __init__(self, model, scaler, buffer_size=5):
        self.model = model
        self.scaler = scaler
        self.buffer = []
        self.buffer_size = buffer_size

    def predict(self, features):
        """Ölçeklendirme ve tahmin döngüsü """
        features_scaled = self.scaler.transform([features])
        # Isolation Forest: -1 anomali, 1 normaldir
        prediction = self.model.predict(features_scaled)
        
        # Karar: 1 (Anomali), 0 (Normal)
        is_anomaly = 1 if prediction[0] == -1 else 0
        
        # Filtreleme (Evhamı önlemek için son 'buffer_size' tahmine bakar)
        self.buffer.append(is_anomaly)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # Eğer tamponun çoğu anomalisi ise (örneğin 5'te 4) gerçek anomali say
        is_confirmed = sum(self.buffer) >= (self.buffer_size * 0.8)
        return is_confirmed