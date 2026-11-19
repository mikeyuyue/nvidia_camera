// led_pwm_giga.ino - serial-controlled hardware PWM for the LED switching-speed
// sweep, for the Arduino GIGA R1 WiFi (STM32H747, mbed core).
//
// Same serial protocol as ../led_pwm/led_pwm.ino (115200 baud, one per line):
//   F <hz> <duty_percent>   start PWM, e.g. "F 5000 50"
//   ON | OFF | ?
// Reply: OK <mode> <actual_hz> <actual_duty_percent>
//
// mbed's PwmOut sets the period in whole microseconds, so the actual frequency
// is 1e6 / round(1e6 / f). Duty is set in timer ticks (fine at any period).
// Frequencies above 500 kHz (period < 2 us) are refused.

#include <Arduino.h>
#include "mbed.h"

#ifndef LED_PIN
#define LED_PIN D9          // <-- set to the pin your LED driver is wired to (D2..D13)
#endif

mbed::PwmOut pwm(digitalPinToPinName(LED_PIN));
float g_hz = 0, g_duty = 0;
const char *g_mode = "OFF";

void report() {
  Serial.print("OK "); Serial.print(g_mode); Serial.print(' ');
  Serial.print(g_hz, 3); Serial.print(' '); Serial.println(g_duty, 4);
}

void hold(bool high) {
  pwm.period_us(1000);
  pwm.write(high ? 1.0f : 0.0f);
  g_mode = high ? "ON" : "OFF"; g_hz = 0; g_duty = high ? 100 : 0;
}

bool startPwm(float hz, float duty) {
  if (hz <= 0 || duty <= 0 || duty >= 100) return false;
  long p = lroundf(1e6f / hz);
  if (p < 2 || p > 1000000) return false;
  pwm.period_us((int)p);
  pwm.write(duty / 100.0f);
  g_hz = 1e6f / p; g_duty = duty; g_mode = "PWM";
  return true;
}

void setup() {
  Serial.begin(115200);
  hold(false);
  Serial.println("READY led_pwm_giga");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;
  char c = toupper(line.charAt(0));
  if (line.equalsIgnoreCase("ON")) { hold(true); report(); }
  else if (line.equalsIgnoreCase("OFF")) { hold(false); report(); }
  else if (c == '?') report();
  else if (c == 'F') {
    float hz = 0, duty = 50;
    int sp = line.indexOf(' ');
    if (sp > 0) {
      String rest = line.substring(sp + 1); rest.trim();
      int sp2 = rest.indexOf(' ');
      hz = (sp2 > 0 ? rest.substring(0, sp2) : rest).toFloat();
      if (sp2 > 0) duty = rest.substring(sp2 + 1).toFloat();
    }
    if (startPwm(hz, duty)) report(); else Serial.println("ERR bad frequency or duty");
  }
  else Serial.println("ERR unknown command");
}
