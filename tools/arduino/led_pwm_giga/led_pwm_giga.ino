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
// Reply: OK <mode> <actual_hz> <actual_led_duty_percent>
//
// mbed's PwmOut sets the period in whole microseconds, so the actual frequency
// is 1e6 / round(1e6 / f). Duty is set in timer ticks. Max 500 kHz (2 us).

#include <Arduino.h>
#include "mbed.h"

const int rowMask[24] = {42, 41, 43, 40, 44, 39, 45, 38, 46, 37, 47, 36, 48, 35, 49, 34, 50, 33, 51, 32, 52, 31, 53, 30};
const int colMask[24] = {14, 15, 13, 16, 12, 17, 11, 18, 10, 19, 9, 20, 8, 21, 7, 22, 6, 23, 5, 24, 4, 25, 3, 26};

// ---- what to light --------------------------------------------------------
// Columns driven by PWM: must be PWM-capable pins (D3..D13 here).
const int PWM_COLS[] = {9};
// Rows held HIGH: all 24 by default, i.e. one full column of 24 LEDs.
const bool ALL_ROWS = true;
const int ROWS[] = {46};          // used only when ALL_ROWS is false
// ---------------------------------------------------------------------------

const int NPWM = sizeof(PWM_COLS) / sizeof(PWM_COLS[0]);
mbed::PwmOut *pwm[NPWM];
float g_hz = 0, g_duty = 0;
const char *g_mode = "OFF";

bool isPwmCol(int pin) {
  for (int i = 0; i < NPWM; i++) if (PWM_COLS[i] == pin) return true;
  return false;
}

void setCols(float colHighFraction, int period_us) {
  for (int i = 0; i < NPWM; i++) {
    pwm[i]->period_us(period_us);
    pwm[i]->write(colHighFraction);   // column HIGH = LED off
  }
}

void report() {
  Serial.print("OK "); Serial.print(g_mode); Serial.print(' ');
  Serial.print(g_hz, 3); Serial.print(' '); Serial.println(g_duty, 4);
}

void hold(bool on) {
  setCols(on ? 0.0f : 1.0f, 1000);
  g_mode = on ? "ON" : "OFF"; g_hz = 0; g_duty = on ? 100 : 0;
}

bool startPwm(float hz, float duty) {
  if (hz <= 0 || duty <= 0 || duty >= 100) return false;
  long p = lroundf(1e6f / hz);
  if (p < 2 || p > 1000000) return false;
  setCols(1.0f - duty / 100.0f, (int)p);
  g_hz = 1e6f / p; g_duty = duty; g_mode = "PWM";
  return true;
}

void setup() {
  // every non-PWM column HIGH (off), every row LOW, then select the rows
  for (int i = 0; i < 24; i++) {
    if (!isPwmCol(colMask[i])) { pinMode(colMask[i], OUTPUT); digitalWrite(colMask[i], HIGH); }
    pinMode(rowMask[i], OUTPUT); digitalWrite(rowMask[i], LOW);
  }
  for (int i = 0; i < NPWM; i++) pwm[i] = new mbed::PwmOut(digitalPinToPinName(PWM_COLS[i]));
  hold(false);
  if (ALL_ROWS) { for (int i = 0; i < 24; i++) digitalWrite(rowMask[i], HIGH); }
  else { for (unsigned i = 0; i < sizeof(ROWS) / sizeof(ROWS[0]); i++) digitalWrite(ROWS[i], HIGH); }
  Serial.begin(115200);
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
