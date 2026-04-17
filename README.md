# AcoustiGuard

Project Title: AcoustiGuard – Acoustic Anomaly Detection & Graceful Shutdown System

Team Members: Elif Sinem Genç, Beyza Karakaya, Gülden Akkuş

Project Overview:
AcoustiGuard is an edge AI system designed to detect machine faults in industrial settings through continuous acoustic and vibration analysis. The system uses a Raspberry Pi 4 connected to an MPU6050 accelerometer and a USB microphone to capture real-time signals from a running DC motor.

Key Technical Components:
- Signal Processing: Real-time FFT feature extraction (~50 ms window) from both vibration and audio signals
- ML Model: Isolation Forest trained exclusively on normal operating conditions (unsupervised/semi-supervised anomaly detection) — this approach avoids the need for labeled fault data, which is typically unavailable in real-world deployments
- Edge Inference: Model runs locally on the Raspberry Pi with no cloud dependency
- Graceful Shutdown: Upon anomaly detection, the Pi sends a command via UART to an Arduino Uno R4, which executes a staged shutdown sequence: (1) gradually reduce motor speed via PWM, (2) transition LED strip from green to red, (3) engage a mechanical brake via servo, (4) log the event to an SD card

Why Isolation Forest:
As you suggested, we replaced the One-Class SVM with Isolation Forest (scikit-learn), which offers a simpler implementation, fewer hyperparameters to tune, and interpretable anomaly scores — making it more suitable for our project scope and timeline.

Demonstration Plan:
For the live demo, the motor will first run under normal conditions. We will then attach a small weight to the motor shaft to introduce a mechanical imbalance, allowing the system to detect the anomaly in real time and trigger the graceful shutdown sequence.
