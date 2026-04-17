#include <Servo.h>
#include <Adafruit_NeoPixel.h>

#define LED_BUILTIN_PIN  13
#define SERVO_PIN         9
#define ENA_PIN           5
#define IN1_PIN           6
#define IN2_PIN           7
#define LED_STRIP_PIN     8
#define LED_COUNT        24

Servo brakeServo;
Adafruit_NeoPixel strip(LED_COUNT, LED_STRIP_PIN, NEO_GRB + NEO_KHZ800);
String incomingCommand = "";

void setup() {
  Serial.begin(9600);

  pinMode(LED_BUILTIN_PIN, OUTPUT);
  digitalWrite(LED_BUILTIN_PIN, LOW);

  pinMode(ENA_PIN, OUTPUT);
  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);

  brakeServo.attach(SERVO_PIN);
  brakeServo.write(0);

  strip.begin();
  strip.show();
  setStripColor(0, 255, 0);

  startMotor();

  Serial.println("AcoustiGuard Arduino ready.");
  Serial.println("Motor running. Waiting for commands...");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      incomingCommand.trim();
      if (incomingCommand == "ANOMALY") {
        Serial.println("ANOMALY received. Executing graceful shutdown...");
        executeShutdown();
      } else {
        Serial.println("Unknown command: " + incomingCommand);
      }
      incomingCommand = "";
    } else {
      incomingCommand += c;
    }
  }
}

void startMotor() {
  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  analogWrite(ENA_PIN, 200);
  Serial.println("Motor started at full speed.");
}

void executeShutdown() {
  Serial.println("Step 1: Slowing down motor...");
  slowDownMotor();

  Serial.println("Step 2: Transitioning LED strip to red...");
  transitionLED();

  Serial.println("Step 3: Engaging servo brake...");
  engageBrake();

  Serial.println("Step 4: Stopping motor completely...");
  stopMotor();

  Serial.println("Step 5: Logging event...");
  logEvent();

  Serial.println("Graceful shutdown complete.");
}

void slowDownMotor() {
  for (int speed = 200; speed >= 0; speed -= 10) {
    analogWrite(ENA_PIN, speed);
    delay(100);
  }
}

void stopMotor() {
  analogWrite(ENA_PIN, 0);
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
  Serial.println("Motor stopped.");
}

void transitionLED() {
  for (int i = 0; i <= 255; i += 5) {
    setStripColor(i, 255 - i, 0);
    delay(20);
  }
  setStripColor(255, 0, 0);
}

void engageBrake() {
  for (int angle = 0; angle <= 90; angle += 5) {
    brakeServo.write(angle);
    delay(50);
  }
  delay(1000);
  for (int angle = 90; angle >= 0; angle -= 5) {
    brakeServo.write(angle);
    delay(50);
  }
}

void setStripColor(int r, int g, int b) {
  for (int i = 0; i < strip.numPixels(); i++) {
    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show();
}

void logEvent() {
  Serial.println("[LOG] Anomaly detected and handled.");
  Serial.print("[LOG] Timestamp (ms): ");
  Serial.println(millis());
}