// led_pwm.ino - serial-controlled hardware PWM for the LED switching-speed sweep.
//
// Board: ATmega328P (Uno / Nano / Pro Mini, 16 MHz). Output: pin 9 (OC1A, Timer1).
// Timer1 runs in fast-PWM mode 14 (TOP = ICR1), so frequency and duty are set by
// hardware and do not jitter with the main loop.
//
// Serial, 115200 baud, one command per line:
//   F <hz> <duty_percent>   start PWM, e.g. "F 5000 50"
//   ON                      LED held on (DC), for ROI detection / reference
//   OFF                     LED held off, for dark frames
//   ?                       report the current state
// Every command answers one line:  OK <mode> <actual_hz> <actual_duty_percent>
// The actual values are what the timer really produces after integer rounding.
// At high frequency the duty resolution gets coarse (TOP is small), so the
// Python side normalizes the camera signal by the actual duty it reports.

const uint8_t PIN = 9;
const uint32_t F_CPU_HZ = 16000000UL;

float g_hz = 0, g_duty = 0;
const char *g_mode = "OFF";

void pinHold(bool high) {
  TCCR1A = 0; TCCR1B = 0;            // stop timer, disconnect OC1A
  digitalWrite(PIN, high ? HIGH : LOW);
  g_mode = high ? "ON" : "OFF";
  g_hz = 0; g_duty = high ? 100 : 0;
}

bool startPwm(float hz, float duty) {
  if (hz <= 0 || duty <= 0 || duty >= 100) return false;
  const uint16_t pres[] = {1, 8, 64, 256, 1024};
  const uint8_t  bits[] = {1, 2, 3, 4, 5};   // CS12:0
  for (uint8_t i = 0; i < 5; i++) {
    float top = (float)F_CPU_HZ / (pres[i] * hz) - 1.0f;
    if (top <= 65535.0f) {
      if (top < 2.0f) return false;          // too fast for any useful duty
      uint16_t t = (uint16_t)(top + 0.5f);
      long ocr = lround(duty / 100.0f * (t + 1)) - 1;
      if (ocr < 0) ocr = 0;
      if (ocr >= t) ocr = t - 1;              // keep a real off phase
      TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
      ICR1 = t;
      OCR1A = (uint16_t)ocr;
      TCCR1A = _BV(COM1A1) | _BV(WGM11);                  // non-inverting
      TCCR1B = _BV(WGM13) | _BV(WGM12) | bits[i];         // mode 14 + clock
      g_hz = (float)F_CPU_HZ / (pres[i] * (float)(t + 1));
      g_duty = 100.0f * (ocr + 1) / (float)(t + 1);
      g_mode = "PWM";
      return true;
    }
  }
  return false;                              // below ~0.24 Hz
}

void report() {
  Serial.print("OK "); Serial.print(g_mode); Serial.print(' ');
  Serial.print(g_hz, 3); Serial.print(' '); Serial.println(g_duty, 4);
}

void setup() {
  pinMode(PIN, OUTPUT);
  pinHold(false);
  Serial.begin(115200);
  Serial.println("READY led_pwm");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;
  char c = toupper(line.charAt(0));
  if (line.equalsIgnoreCase("ON")) { pinHold(true); report(); }
  else if (line.equalsIgnoreCase("OFF")) { pinHold(false); report(); }
  else if (c == '?') { report(); }
  else if (c == 'F') {
    float hz = 0, duty = 50;
    int sp = line.indexOf(' ');
    if (sp > 0) {
      String rest = line.substring(sp + 1); rest.trim();
      int sp2 = rest.indexOf(' ');
      hz = (sp2 > 0 ? rest.substring(0, sp2) : rest).toFloat();
      if (sp2 > 0) duty = rest.substring(sp2 + 1).toFloat();
    }
    if (startPwm(hz, duty)) report();
    else Serial.println("ERR bad frequency or duty");
  }
  else Serial.println("ERR unknown command");
}
