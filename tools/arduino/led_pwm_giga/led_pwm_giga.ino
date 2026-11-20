// led_pwm_giga.ino - serial-controlled hardware PWM for the LED switching-speed
// sweep, on the Arduino GIGA R1 (STM32H747, mbed core) driving the 24x24
// matrix-addressed LED array (same wiring as sketch_nov25b/c):
//   row pins    (rowMask) are anodes:   HIGH = row selected
//   column pins (colMask) are cathodes: LOW  = column selected
//   LED (r, c) is lit when row r is HIGH and column c is LOW.
//
// For the sweep, the selected rows are held HIGH and one or more COLUMN pins are
// driven by a hardware timer (column pins D3..D13 are PWM capable). Because
// columns are active-low the PWM is inverted here: an LED duty of D means the
// column pin is high for (1 - D) of each period. The duty in replies is the LED
// (on) duty.
//
// Serial, 115200 baud, one command per line:
//   F <hz> <duty_percent>   PWM the LEDs, e.g. "F 5000 50"
//   ON                      selected LEDs held on (DC), for ROI detection
//   OFF                     all LEDs off, for dark frames
//   ?                       report state
//   P <period_us> <width_us> pulse mode: LED on for width_us (sub-us ok, set in
//                           timer ticks) every period_us (whole us)
//   R ALL | R <i,j,...>     rows held HIGH, by index 0..23 into rowMask
//   C <pin>                 which column pin is PWM driven (D3..D13)
// Reply: OK <mode> <actual_hz> <actual_led_duty_percent>   (F/ON/OFF/?/P)
//        OK ROWS <n> | OK COL <pin>                       (R/C)
// R and C keep the current mode. Defaults: all rows, column D9.
//
// mbed's PwmOut sets the period in whole microseconds, so the actual frequency
// is 1e6 / round(1e6 / f). Duty is set in timer ticks. Max 500 kHz (2 us).

#include <Arduino.h>
#include "mbed.h"

const int rowMask[24] = {42, 41, 43, 40, 44, 39, 45, 38, 46, 37, 47, 36, 48, 35, 49, 34, 50, 33, 51, 32, 52, 31, 53, 30};
const int colMask[24] = {14, 15, 13, 16, 12, 17, 11, 18, 10, 19, 9, 20, 8, 21, 7, 22, 6, 23, 5, 24, 4, 25, 3, 26};

// ---- defaults --------------------------------------------------------------
int g_col = 9;                     // PWM-driven column pin (D3..D13)
bool g_rows[24];                   // rows held HIGH
// ---------------------------------------------------------------------------

mbed::PwmOut *pwm = nullptr;
float g_hz = 0, g_duty = 0;
int g_period = 1000;
float g_colHigh = 1.0f;            // fraction of the period the column is HIGH (LED off)
const char *g_mode = "OFF";

void apply() {
  pwm->period_us(g_period);
  pwm->write(g_colHigh);
}

void report() {
  Serial.print("OK "); Serial.print(g_mode); Serial.print(' ');
  Serial.print(g_hz, 3); Serial.print(' '); Serial.println(g_duty, 4);
}

void hold(bool on) {
  g_period = 1000; g_colHigh = on ? 0.0f : 1.0f; apply();
  g_mode = on ? "ON" : "OFF"; g_hz = 0; g_duty = on ? 100 : 0;
}

bool startPwm(float hz, float duty) {
  if (hz <= 0 || duty <= 0 || duty >= 100) return false;
  long p = lroundf(1e6f / hz);
  if (p < 2 || p > 1000000) return false;
  g_period = (int)p; g_colHigh = 1.0f - duty / 100.0f; apply();
  g_hz = 1e6f / p; g_duty = duty; g_mode = "PWM";
  return true;
}

bool startPulse(long period_us, float width_us) {
  if (period_us < 2 || period_us > 1000000 || width_us <= 0 || width_us >= period_us) return false;
  g_period = (int)period_us; g_colHigh = 1.0f - width_us / period_us; apply();
  g_hz = 1e6f / period_us; g_duty = 100.0f * width_us / period_us; g_mode = "PULSE";
  return true;
}

void applyRows() {
  for (int i = 0; i < 24; i++) digitalWrite(rowMask[i], g_rows[i] ? HIGH : LOW);
}

bool setCol(int pin) {
  if (pin < 3 || pin > 13) return false;
  if (pwm) { delete pwm; pwm = nullptr; pinMode(g_col, OUTPUT); digitalWrite(g_col, HIGH); }
  g_col = pin;
  pwm = new mbed::PwmOut(digitalPinToPinName(g_col));
  apply();
  return true;
}

String arg(const String &line) {
  int sp = line.indexOf(' ');
  if (sp < 0) return String("");
  String r = line.substring(sp + 1); r.trim(); return r;
}

void setup() {
  // every column HIGH (off), every row LOW; then the defaults
  for (int i = 0; i < 24; i++) {
    pinMode(colMask[i], OUTPUT); digitalWrite(colMask[i], HIGH);
    pinMode(rowMask[i], OUTPUT); digitalWrite(rowMask[i], LOW);
    g_rows[i] = true;
  }
  setCol(g_col);
  hold(false);
  applyRows();
  Serial.begin(115200);
  Serial.println("READY led_pwm_giga");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;
  char c = toupper(line.charAt(0));
  String a = arg(line);
  if (line.equalsIgnoreCase("ON")) { hold(true); report(); }
  else if (line.equalsIgnoreCase("OFF")) { hold(false); report(); }
  else if (c == '?') report();
  else if (c == 'F' || c == 'P') {
    int sp = a.indexOf(' ');
    float x = (sp > 0 ? a.substring(0, sp) : a).toFloat();
    float y = sp > 0 ? a.substring(sp + 1).toFloat() : 50;
    bool ok = (c == 'F') ? startPwm(x, y) : startPulse(lroundf(x), y);
    if (ok) report(); else Serial.println("ERR bad arguments");
  }
  else if (c == 'R') {
    if (a.equalsIgnoreCase("ALL")) { for (int i = 0; i < 24; i++) g_rows[i] = true; }
    else {
      for (int i = 0; i < 24; i++) g_rows[i] = false;
      int start = 0;
      while (start < (int)a.length()) {
        int comma = a.indexOf(',', start); if (comma < 0) comma = a.length();
        int idx = a.substring(start, comma).toInt();
        if (idx >= 0 && idx < 24) g_rows[idx] = true;
        start = comma + 1;
      }
    }
    applyRows();
    int n = 0; for (int i = 0; i < 24; i++) n += g_rows[i];
    Serial.print("OK ROWS "); Serial.println(n);
  }
  else if (c == 'C') {
    if (setCol(a.toInt())) { Serial.print("OK COL "); Serial.println(g_col); }
    else Serial.println("ERR column pin must be 3..13");
  }
  else Serial.println("ERR unknown command");
}
