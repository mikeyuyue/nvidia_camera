// led_pwm_giga.ino - serial-controlled hardware PWM for the 24x24 micro-LED
// matrix, on the Arduino GIGA R1 (STM32H747, mbed core). Talk to it from Python
// with pyserial (tools/led_control.py, tools/led_switching.py).
// Wiring (same as sketch_nov25b/c):
//   row pins    (rowMask) are anodes:   HIGH = row selected
//   column pins (colMask) are cathodes: LOW  = column selected
//   LED (r, c) is lit when row r is HIGH and column c is LOW.
//
// The selected rows are held HIGH and one COLUMN pin (the "PWM column") is
// driven by a hardware timer (column pins D3..D13 are PWM capable). Columns are
// active-low, so the PWM is inverted here: an LED duty of D means the column
// pin is high for (1 - D) of each period. Duties in replies are LED (on) duty.
//
// Serial, 115200 baud, one command per line:
//   F <hz> <duty_percent>   PWM the LEDs, e.g. "F 5000 50"
//   P <period_us> <width_us> pulse mode: LED on for width_us every period_us
//                           (both fractional; one timer tick ~4 ns resolution)
//   D <duty_percent>        change the duty, keep the frequency (PWM/PULSE mode)
//   B <n>                   burst: n periods at the current frequency/duty from a
//                           clean start, then OFF. The end is written at a period
//                           boundary (compare preload), so the count is exact
//                           unless a period is shorter than the ~2 us loop jitter.
//   ON                      PWM column held on (DC)
//   OFF                     PWM column off
//   R ALL | R NONE | R <i,j,...>   rows held HIGH, by index 0..23 into rowMask
//   C <pin>                 which column pin is PWM driven (D3..D13)
//   K ALL | K NONE | K <i,j,...>   extra columns held LOW (DC on), by index
//                           0..23 into colMask; the PWM column is not touched
//   L <row> <col>           light exactly ONE LED, DC (matrix indices 0..23). If
//                           its column pin is PWM capable it becomes the PWM
//                           column, so F/D/P/B then modulate that single LED.
//   A                       every LED on, DC (what led_all_on did)
//   ?                       short state;  S  full status;  T  timer registers
// Reply: OK <mode> <actual_hz> <actual_led_duty_percent>  (F/P/D/B/ON/OFF/?/L/A)
//        OK ROWS <n> | OK COL <pin> | OK COLS <n> | OK STATUS ... | OK TIMER ...
// R, C and K keep the current mode. Defaults at power-up: all rows HIGH, PWM
// column D9 OFF, no extra columns.
//
// Timing: mbed's PwmOut keeps period AND pulse width in whole microseconds
// (it rounds the pulse to 1 us, so e.g. 50% of a 5 us period came out as 40%).
// Here mbed only picks the pin's timer and channel; the period (PSC/ARR) and
// the compare value (CCR) are written straight into the STM32 timer registers,
// so any frequency is reachable and duty/pulse width have one-timer-tick
// resolution (~4 ns at 240 MHz). Replies carry the values the registers produce.

#include <Arduino.h>
#include "mbed.h"

const int rowMask[24] = {42, 41, 43, 40, 44, 39, 45, 38, 46, 37, 47, 36, 48, 35, 49, 34, 50, 33, 51, 32, 52, 31, 53, 30};
const int colMask[24] = {14, 15, 13, 16, 12, 17, 11, 18, 10, 19, 9, 20, 8, 21, 7, 22, 6, 23, 5, 24, 4, 25, 3, 26};

// ---- state -----------------------------------------------------------------
int g_col = 9;                     // PWM-driven column pin (D3..D13)
bool g_rows[24];                   // rows held HIGH
bool g_dccols[24];                 // extra columns held LOW (DC on)
// ---------------------------------------------------------------------------

// PwmOut keeps its pwmout_t (timer base + channel) protected; expose it.
class RegPwm : public mbed::PwmOut {
 public:
  explicit RegPwm(PinName p) : mbed::PwmOut(p) {}
  TIM_TypeDef *tim() { return (TIM_TypeDef *)_pwm.pwm; }
  int channel() { return _pwm.channel; }
};

RegPwm *pwm = nullptr;
double g_timclk = 0;               // timer input clock, Hz (measured at start)
float g_hz = 0, g_duty = 0;
float g_colHigh = 1.0f;            // fraction of the period the column is HIGH (LED off)
const char *g_mode = "OFF";

volatile uint32_t *ccr() {
  TIM_TypeDef *t = pwm->tim();
  switch (pwm->channel()) {
    case 1: return &t->CCR1;
    case 2: return &t->CCR2;
    case 3: return &t->CCR3;
    default: return &t->CCR4;
  }
}

// Let mbed set a known 1000 us period, then read back what it put in PSC/ARR.
void measureTimerClock() {
  pwm->period_us(1000);
  pwm->write(0.5f);
  TIM_TypeDef *t = pwm->tim();
  g_timclk = (double)(t->PSC + 1) * (double)(t->ARR + 1) * 1000.0;
}

// Program period and column-high fraction directly. Returns the actual
// frequency; *ledDuty gets the actual LED-on fraction (0..1).
double setTimer(double hz, double colHigh, double *ledDuty) {
  TIM_TypeDef *t = pwm->tim();
  double total = g_timclk / hz;                       // ticks per period
  uint32_t psc = (uint32_t)ceil(total / 65536.0);     // fits 16- and 32-bit timers
  if (psc < 1) psc = 1;
  uint32_t arr = (uint32_t)llround(total / psc);
  if (arr < 2) arr = 2;
  uint32_t c = (uint32_t)llround(colHigh * arr);
  if (c > arr) c = arr;
  t->CR1 &= ~TIM_CR1_CEN;
  t->PSC = psc - 1;
  t->ARR = arr - 1;
  *ccr() = c;
  t->CNT = 0;
  t->EGR = TIM_EGR_UG;                                // load PSC/ARR/CCR now
  t->CR1 |= TIM_CR1_CEN;
  *ledDuty = 1.0 - (double)c / arr;
  return g_timclk / ((double)psc * arr);
}

void apply() {
  double d;
  double f = setTimer(g_hz > 0 ? g_hz : 1000.0, g_colHigh, &d);
  if (g_hz > 0) { g_hz = f; g_duty = 100.0 * d; }
}

void report() {
  Serial.print("OK "); Serial.print(g_mode); Serial.print(' ');
  Serial.print(g_hz, 3); Serial.print(' '); Serial.println(g_duty, 4);
}

void hold(bool on) {
  g_hz = 0; g_colHigh = on ? 0.0f : 1.0f; apply();
  g_mode = on ? "ON" : "OFF"; g_duty = on ? 100 : 0;
}

bool startPwm(float hz, float duty) {
  if (hz <= 0 || hz > 20e6 || duty <= 0 || duty >= 100) return false;
  g_hz = hz; g_colHigh = 1.0f - duty / 100.0f; apply();
  g_mode = "PWM";
  return true;
}

bool startPulse(float period_us, float width_us) {
  if (period_us < 0.05f || width_us <= 0 || width_us >= period_us) return false;
  g_hz = 1e6f / period_us; g_colHigh = 1.0f - width_us / period_us; apply();
  g_mode = "PULSE";
  return true;
}

bool modulating() { return strcmp(g_mode, "PWM") == 0 || strcmp(g_mode, "PULSE") == 0; }

bool setDuty(float duty) {
  if (!modulating() || duty <= 0 || duty >= 100) return false;
  const char *m = g_mode;
  bool ok = startPwm(g_hz, duty);
  g_mode = m;
  return ok;
}

// n periods from CNT = 0, then the compare is set to "always off". CCR is
// preloaded, so the new value takes effect at the next update event, i.e.
// exactly at a period boundary.
bool burst(long n) {
  if (!modulating() || n < 1) return false;
  double period_us = 1e6 / g_hz;
  float hz = g_hz, duty = g_duty;
  apply();                                           // restarts the counter at 0
  unsigned long t0 = micros();
  double wait = (n - 0.5) * period_us;               // land mid-way in the last period
  while ((double)(micros() - t0) < wait) {}
  TIM_TypeDef *t = pwm->tim();
  *ccr() = t->ARR + 1;                               // column HIGH = LED off from next period
  delayMicroseconds((unsigned)(2 * period_us) + 2);
  hold(false);
  g_mode = "BURST_DONE"; g_hz = hz; g_duty = duty;   // report what was sent
  return true;
}

void applyRows() {
  for (int i = 0; i < 24; i++) digitalWrite(rowMask[i], g_rows[i] ? HIGH : LOW);
}

void applyCols() {
  for (int i = 0; i < 24; i++) {
    if (colMask[i] == g_col) continue;               // the timer owns this pin
    pinMode(colMask[i], OUTPUT);
    digitalWrite(colMask[i], g_dccols[i] ? LOW : HIGH);
  }
}

bool setCol(int pin) {
  if (pin < 3 || pin > 13) return false;
  if (pwm) { delete pwm; pwm = nullptr; pinMode(g_col, OUTPUT); digitalWrite(g_col, HIGH); }
  g_col = pin;
  pwm = new RegPwm(digitalPinToPinName(g_col));
  measureTimerClock();
  apply();
  applyCols();                                       // old PWM pin back to its DC state
  return true;
}

// "ALL" | "NONE" | "i,j,k" -> flags[24]
void parseList(const String &a, bool *flags) {
  bool all = a.equalsIgnoreCase("ALL");
  for (int i = 0; i < 24; i++) flags[i] = all;
  if (all || a.length() == 0 || a.equalsIgnoreCase("NONE")) return;
  int start = 0;
  while (start < (int)a.length()) {
    int comma = a.indexOf(',', start);
    if (comma < 0) comma = a.length();
    String tok = a.substring(start, comma);
    tok.trim();
    if (tok.length() && isDigit(tok.charAt(0))) {
      int idx = tok.toInt();
      if (idx >= 0 && idx < 24) flags[idx] = true;
    }
    start = comma + 1;
  }
}

int count24(const bool *f) { int n = 0; for (int i = 0; i < 24; i++) n += f[i]; return n; }

bool lightOne(int r, int c) {
  if (r < 0 || r > 23 || c < 0 || c > 23) return false;
  for (int i = 0; i < 24; i++) { g_rows[i] = (i == r); g_dccols[i] = false; }
  int pin = colMask[c];
  if (pin >= 3 && pin <= 13) {                       // PWM capable: modulatable LED
    if (pin != g_col) setCol(pin);
    applyCols();
    applyRows();
    hold(true);
  } else {                                           // DC only; PWM column stays off
    hold(false);
    g_dccols[c] = true;
    applyCols();
    applyRows();
    g_mode = "ON"; g_duty = 100;
  }
  return true;
}

void allOn() {
  for (int i = 0; i < 24; i++) { g_rows[i] = true; g_dccols[i] = true; }
  applyCols();
  applyRows();
  hold(true);
}

void printList(const bool *f) {
  int n = 0;
  for (int i = 0; i < 24; i++) if (f[i]) { if (n++) Serial.print(','); Serial.print(i); }
  if (!n) Serial.print("none");
}

void status() {
  int ci = -1;
  for (int i = 0; i < 24; i++) if (colMask[i] == g_col) ci = i;
  Serial.print("OK STATUS mode="); Serial.print(g_mode);
  Serial.print(" hz="); Serial.print(g_hz, 3);
  Serial.print(" duty="); Serial.print(g_duty, 4);
  Serial.print(" pwm_col_pin="); Serial.print(g_col);
  Serial.print(" pwm_col_idx="); Serial.print(ci);
  Serial.print(" rows="); printList(g_rows);
  Serial.print(" dc_cols="); printList(g_dccols);
  Serial.println();
}

String arg(const String &line) {
  int sp = line.indexOf(' ');
  if (sp < 0) return String("");
  String r = line.substring(sp + 1);
  r.trim();
  return r;
}

void twoNums(const String &a, float *x, float *y, float ydef) {
  int sp = a.indexOf(' ');
  *x = (sp > 0 ? a.substring(0, sp) : a).toFloat();
  *y = sp > 0 ? a.substring(sp + 1).toFloat() : ydef;
}

void setup() {
  // every column HIGH (off), every row LOW; then the defaults
  for (int i = 0; i < 24; i++) {
    pinMode(colMask[i], OUTPUT); digitalWrite(colMask[i], HIGH);
    pinMode(rowMask[i], OUTPUT); digitalWrite(rowMask[i], LOW);
    g_rows[i] = true;
    g_dccols[i] = false;
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
  String cmd = line;
  int sp = line.indexOf(' ');
  if (sp > 0) cmd = line.substring(0, sp);
  cmd.toUpperCase();
  String a = arg(line);
  float x, y;
  if (cmd == "ON") { hold(true); report(); }
  else if (cmd == "OFF") { hold(false); report(); }
  else if (cmd == "?") report();
  else if (cmd == "S") status();
  else if (cmd == "T") {                             // timer registers, for debugging
    TIM_TypeDef *t = pwm->tim();
    Serial.print("OK TIMER clk_hz="); Serial.print(g_timclk, 0);
    Serial.print(" ch="); Serial.print(pwm->channel());
    Serial.print(" psc="); Serial.print(t->PSC); Serial.print(" arr="); Serial.print(t->ARR);
    Serial.print(" ccr="); Serial.println(*ccr());
  }
  else if (cmd == "F") {
    twoNums(a, &x, &y, 50);
    if (startPwm(x, y)) report(); else Serial.println("ERR F <hz> <duty, 0<d<100>");
  }
  else if (cmd == "P") {
    twoNums(a, &x, &y, 0);
    if (startPulse(x, y)) report(); else Serial.println("ERR P <period_us> <width_us>");
  }
  else if (cmd == "D") {
    if (setDuty(a.toFloat())) report(); else Serial.println("ERR D needs PWM mode and 0<duty<100");
  }
  else if (cmd == "B") {
    if (burst(a.toInt())) report(); else Serial.println("ERR B <n> needs PWM mode");
  }
  else if (cmd == "R") {
    parseList(a, g_rows); applyRows();
    Serial.print("OK ROWS "); Serial.println(count24(g_rows));
  }
  else if (cmd == "K") {
    parseList(a, g_dccols); applyCols();
    Serial.print("OK COLS "); Serial.println(count24(g_dccols));
  }
  else if (cmd == "C") {
    if (setCol(a.toInt())) { Serial.print("OK COL "); Serial.println(g_col); }
    else Serial.println("ERR column pin must be 3..13");
  }
  else if (cmd == "L") {
    twoNums(a, &x, &y, -1);
    if (lightOne((int)x, (int)y)) report(); else Serial.println("ERR L <row 0..23> <col 0..23>");
  }
  else if (cmd == "A" || cmd == "ALL") { allOn(); report(); }
  else Serial.println("ERR unknown command");
}
