// =============================================================================
// AcoustiGuard — Arduino Uno R4
// =============================================================================
// Görevler:
//   1. MPU6050'den ivme verisi oku → JSON olarak UART'a gönder
//   2. RPi'den gelen komutları dinle ('N', 'W', 'S')
//   3. Komuta göre motor PWM, LED ve servo kontrolü yap
//
// Kablolama:
//   MPU6050 SDA → A4 | SCL → A5 | VCC → 3.3V | GND → GND
//   Motor (L298N ENA) → Pin 9 (PWM)
//   Servo → Pin 6
//   LED Kırmızı → Pin 4 | LED Yeşil → Pin 5
//   UART TX (Pin 1) → voltaj bölücü → RPi GPIO15 (RX)
//   UART RX (Pin 0) ← RPi GPIO14 (TX)

#include <Wire.h>
#include <MPU6050.h>
#include <Servo.h>

// ── Pin Tanımları ─────────────────────────────────────────────────────────────
#define MOTOR_PWM_PIN   9    // L298N ENA (PWM)
#define SERVO_PIN       6
#define LED_GREEN_PIN   5
#define LED_RED_PIN     4

// ── Sabitler ──────────────────────────────────────────────────────────────────
#define NORMAL_SPEED     200   // PWM 0-255
#define WARNING_SPEED    100   // azaltılmış hız
#define BRAKE_ANGLE       90   // servo fren açısı (derece)
#define RELEASE_ANGLE      0   // servo serbest açısı

#define SEND_INTERVAL_MS  10   // MPU6050 gönderim aralığı (~100 Hz)

// ── Nesneler ──────────────────────────────────────────────────────────────────
MPU6050 mpu;
Servo   brakeServo;

// ── Durum ────────────────────────────────────────────────────────────────────
char    currentState = 'N';   // 'N'=Normal, 'W'=Warning, 'S'=Shutdown
unsigned long lastSendTime = 0;

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // MPU6050
  Wire.begin();
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("{\"error\":\"MPU6050 connection failed\"}");
  }

  // Motor
  pinMode(MOTOR_PWM_PIN, OUTPUT);
  analogWrite(MOTOR_PWM_PIN, NORMAL_SPEED);

  // Servo
  brakeServo.attach(SERVO_PIN);
  brakeServo.write(RELEASE_ANGLE);

  // LEDs
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_RED_PIN,   OUTPUT);
  setLED('N');
}

// ── LED Kontrol ───────────────────────────────────────────────────────────────
void setLED(char state) {
  if (state == 'N') {
    digitalWrite(LED_GREEN_PIN, HIGH);
    digitalWrite(LED_RED_PIN,   LOW);
  } else if (state == 'W') {
    // Turuncu: ikisi birden
    digitalWrite(LED_GREEN_PIN, HIGH);
    digitalWrite(LED_RED_PIN,   HIGH);
  } else {  // 'S'
    digitalWrite(LED_GREEN_PIN, LOW);
    digitalWrite(LED_RED_PIN,   HIGH);
  }
}

// ── Komut İşleyici ────────────────────────────────────────────────────────────
void handleCommand(char cmd) {
  currentState = cmd;
  setLED(cmd);

  if (cmd == 'N') {
    // Normal: tam hız, fren serbest
    analogWrite(MOTOR_PWM_PIN, NORMAL_SPEED);
    brakeServo.write(RELEASE_ANGLE);

  } else if (cmd == 'W') {
    // Uyarı: hızı kademeli düşür
    // 200 → 100 arası 5 adımda (~500ms)
    int currentPWM = NORMAL_SPEED;
    while (currentPWM > WARNING_SPEED) {
      currentPWM -= 20;
      if (currentPWM < WARNING_SPEED) currentPWM = WARNING_SPEED;
      analogWrite(MOTOR_PWM_PIN, currentPWM);
      delay(100);
    }

  } else if (cmd == 'S') {
    // Kapatma: kademeli PWM azalt → durdur → servo fren
    int currentPWM = WARNING_SPEED;
    while (currentPWM > 0) {
      currentPWM -= 10;
      if (currentPWM < 0) currentPWM = 0;
      analogWrite(MOTOR_PWM_PIN, currentPWM);
      delay(80);
    }
    // Motor tamamen durdu, fren uygula
    delay(200);
    brakeServo.write(BRAKE_ANGLE);
    Serial.println("{\"event\":\"shutdown_complete\"}");
  }
}

// ── MPU6050 Oku ve Gönder ─────────────────────────────────────────────────────
void sendVibrationData() {
  int16_t ax_raw, ay_raw, az_raw;
  int16_t gx, gy, gz;   // gyro (şimdilik kullanılmıyor)

  mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx, &gy, &gz);

  // Ham değeri g birimine çevir (MPU6050 hassasiyet ±2g → 16384 LSB/g)
  float ax = ax_raw / 16384.0;
  float ay = ay_raw / 16384.0;
  float az = az_raw / 16384.0;

  // JSON satırı olarak gönder
  Serial.print("{\"ax\":");
  Serial.print(ax, 4);
  Serial.print(",\"ay\":");
  Serial.print(ay, 4);
  Serial.print(",\"az\":");
  Serial.print(az, 4);
  Serial.println("}");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  // RPi'den komut geldi mi?
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'N' || cmd == 'W' || cmd == 'S') {
      handleCommand(cmd);
    }
  }

  // Shutdown tamamlandıysa artık veri gönderme
  if (currentState == 'S') {
    return;
  }

  // Titreşim verisini belirli aralıklarla gönder
  unsigned long now = millis();
  if (now - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = now;
    sendVibrationData();
  }
}
