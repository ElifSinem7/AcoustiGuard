// =============================================================================
// AcoustiGuard — Arduino Uno R4 MPU-6050 Serial Sender
// =============================================================================
// MPU-6050'den Z-eksen ivme verisini seri porta gonderir.
// Raspberry Pi bu veriyi okuyarak titresim analizi yapar.
//
// Baglantilar:
//   MPU-6050 VCC  -> Arduino 3.3V
//   MPU-6050 GND  -> Arduino GND
//   MPU-6050 SDA  -> Arduino A4 (SDA)
//   MPU-6050 SCL  -> Arduino A5 (SCL)
//
// Kutuphane: Adafruit MPU6050 (Arduino Library Manager'dan yukle)
// =============================================================================

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

// Durum LED'i (anomali sinyali RPi'den geldiginde yanar)
const int LED_RED    = 9;
const int LED_GREEN  = 10;
const int BUZZER_PIN = 8;

String incomingMsg = "";

void setup() {
  Serial.begin(9600);
  while (!Serial) delay(10);

  pinMode(LED_RED,   OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  digitalWrite(LED_GREEN, HIGH);  // Baslangicta yesil

  if (!mpu.begin()) {
    Serial.println("MPU6050 bulunamadi!");
    while (1) delay(10);
  }

  // Hassasiyet ayarlari
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100);
}

void loop() {
  // MPU6050'den veri oku
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Z-eksen ivmeyi seri porta gonder (RPi okuyacak)
  Serial.println(a.acceleration.z, 6);

  // RPi'den mesaj geldi mi kontrol et
  if (Serial.available() > 0) {
    incomingMsg = Serial.readStringUntil('\n');
    incomingMsg.trim();

    if (incomingMsg == "ANOMALY") {
      // Kirmizi LED yak, buzzer cal
      digitalWrite(LED_GREEN, LOW);
      digitalWrite(LED_RED,   HIGH);
      tone(BUZZER_PIN, 1000, 500);
      delay(500);
    } else if (incomingMsg == "NORMAL") {
      // Yesil LED yak
      digitalWrite(LED_RED,   LOW);
      digitalWrite(LED_GREEN, HIGH);
      noTone(BUZZER_PIN);
    }
  }

  delay(1);  // ~1kHz ornekleme
}
