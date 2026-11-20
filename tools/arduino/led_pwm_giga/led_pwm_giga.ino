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
//   P <period_us> <width_us> pulse mode: LED on for width_us every period_us
//                           (both fractional; one timer tick ~4 ns resolution)
//   T                       timer debug info (clock, channel, PSC/ARR/CCR)
//   R ALL | R <i,j,...>     rows held HIGH, by index 0..23 into rowMask
//   C <pin>                 which column pin is PWM driven (D3..D13)
// Reply: OK <mode> <actual_hz> <actual_led_duty_percent>   (F/ON/OFF/?/P)
//        OK ROWS <n> | OK COL <pin>                       (R/C)
// R and C keep the current mode. Defaults: all rows, column D9.
//
// Timing: mbed's PwmOut keeps period AND pulse width in whole microseconds
// (it rounds the pulse to 1 us, so e.g. 50% of a 5 us period came out as 40%,
// which made odd-us periods read dim). Here mbed only picks the pin's timer and
// channel; the period (PSC/ARR) and the compare value (CCR) are then written
// straight into the STM32 timer registers, so any frequency is reachable and
// duty/pulse width have one-timer-tick resolution (~4 ns at 240 MHz). Replies
// carry the frequency and duty that the registers actually produce.

#include <Arduino.h>
#include "mbed.h"

const int rowMask[24] = {42, 41, 43, 40, 44, 39, 45, 38, 46, 37, 47, 36, 48, 35, 49, 34, 50, 33, 51, 32, 52, 31, 53, 30};
const int colMask[24] = {14, 15, 13, 16, 12, 17, 11, 18, 10, 19, 9, 20, 8, 21, 7, 22, 6, 23, 5, 24, 4, 25, 3, 26};

// ---- defaults --------------------------------------------------------------
int g_col = 9;                     // PWM-driven column pin (D3..D13)
bool g_rows[24];                   // rows held HIGH
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

void applyRows() {
  for (int i = 0; i < 24; i++) digitalWrite(rowMask[i], g_rows[i] ? HIGH : LOW);
}

bool setCol(int pin) {
  if (pin < 3 || pin > 13) return false;
  if (pwm) { delete pwm; pwm = nullptr; pinMode(g_col, OUTPUT); digitalWrite(g_col, HIGH); }
  g_col = pin;
  pwm = new RegPwm(digitalPinToPinName(g_col));
  measureTimerClock();
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
  else if (c == 'T') {                    // timer info, for debugging
    TIM_TypeDef *t = pwm->tim();
    Serial.print("OK TIMER clk_hz="); Serial.print(g_timclk, 0);
    Serial.print(" ch="); Serial.print(pwm->channel());
    Serial.print(" psc="); Serial.print(t->PSC); Serial.print(" arr="); Serial.print(t->ARR);
    Serial.print(" ccr="); Serial.println(*ccr());
  }
  else if (c == 'F' || c == 'P') {
    int sp = a.indexOf(' ');
    float x = (sp > 0 ? a.substring(0, sp) : a).toFloat();
    float y = sp > 0 ? a.substring(sp + 1).toFloat() : 50;
    bool ok = (c == 'F') ? startPwm(x, y) : startPulse(x, y);
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
