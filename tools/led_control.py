#!/usr/bin/env python3
"""led_control.py - drive the 24x24 micro-LED matrix through the GIGA (pyserial).

Talks to tools/arduino/led_pwm_giga/led_pwm_giga.ino. The GIGA does not reset
when the port opens, so each call changes the board's state and leaves it
there (use `off` to finish). Matrix indices are 0..23 (rowMask / colMask order).

  python3 tools/led_control.py status
  python3 tools/led_control.py pwm --freq 20k --duty 30            # PWM column, rows as set
  python3 tools/led_control.py pwm --freq 20k --duty 30 --rows 0,1,2 --col 9
  python3 tools/led_control.py duty 75                              # keep f, new duty
  python3 tools/led_control.py pulse --period-us 10 --width-us 1.5
  python3 tools/led_control.py burst 1000 --freq 50k --duty 50      # 1000 pulses, then off
  python3 tools/led_control.py led 5 10                             # one LED on (DC)
  python3 tools/led_control.py led 5 10 --freq 10k --duty 25        # one LED, PWM
  python3 tools/led_control.py rows 0,3,7      | rows all | rows none
  python3 tools/led_control.py cols 2,4        | cols all | cols none   (extra DC columns)
  python3 tools/led_control.py col 5                                # PWM column pin D3..D13
  python3 tools/led_control.py all | on | off      (off = PWM column off AND no DC columns)
  python3 tools/led_control.py scan --dwell 0.5 [--rows 0-23] [--cols 0-23]
  python3 tools/led_control.py raw "F 1000 10" "S"

Frequencies take k/M suffixes. PWM-capable column pins are D3..D13, i.e.
matrix columns 2,4,6,...,22 (colMask); other columns can only be DC on/off.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from led_switching import ArduinoDriver, parse_freq, fmt_hz  # noqa: E402

COL_MASK = [14, 15, 13, 16, 12, 17, 11, 18, 10, 19, 9, 20, 8, 21, 7, 22, 6, 23, 5, 24, 4, 25, 3, 26]
PWM_COLS = [i for i, p in enumerate(COL_MASK) if 3 <= p <= 13]


def idx_list(s: str | None):
    """'all' -> None, 'none' -> [], '0,3,5-8' -> [0,3,5,6,7,8]"""
    if s is None or s.lower() == "all":
        return None
    if s.lower() == "none":
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i <= 23]
    if bad:
        sys.exit(f"indices must be 0..23, got {bad}")
    return out


def show(drv):
    s = drv.state()
    mode, hz, duty = s.get("mode"), s.get("hz", 0), s.get("duty", 0)
    f = f"{fmt_hz(hz)} at {duty:.4g}% LED duty" if mode in ("PWM", "PULSE") else ""
    print(f"mode {mode} {f}")
    print(f"PWM column: pin D{s.get('pwm_col_pin')} (matrix col {s.get('pwm_col_idx')})")
    r, c = s.get("rows", []), s.get("dc_cols", [])
    print(f"rows HIGH : {'all' if len(r) == 24 else (r or 'none')}")
    print(f"DC columns: {'all' if len(c) == 24 else (c or 'none')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sel(p):
        p.add_argument("--rows", help="rows to hold HIGH first: all | none | 0,3,5-8")
        p.add_argument("--col", type=int, help="PWM column pin first (3..13)")
        p.add_argument("--cols", help="extra DC-on columns first: all | none | list")

    sub.add_parser("status")
    for name in ("on", "off", "all"):
        sel(sub.add_parser(name))
    p = sub.add_parser("pwm"); sel(p)
    p.add_argument("--freq", required=True); p.add_argument("--duty", type=float, default=50)
    p = sub.add_parser("duty"); p.add_argument("duty", type=float)
    p = sub.add_parser("pulse"); sel(p)
    p.add_argument("--period-us", type=float, required=True)
    p.add_argument("--width-us", type=float, required=True)
    p = sub.add_parser("burst"); sel(p)
    p.add_argument("n", type=int)
    p.add_argument("--freq", help="set PWM first (else the current PWM is used)")
    p.add_argument("--duty", type=float, default=50)
    p = sub.add_parser("led")
    p.add_argument("row", type=int); p.add_argument("col", type=int)
    p.add_argument("--freq"); p.add_argument("--duty", type=float, default=50)
    p = sub.add_parser("rows"); p.add_argument("which")
    p = sub.add_parser("cols"); p.add_argument("which")
    p = sub.add_parser("col"); p.add_argument("pin", type=int)
    p = sub.add_parser("scan", help="light every LED in turn (DC)")
    p.add_argument("--rows", default="0-23"); p.add_argument("--cols", default="0-23")
    p.add_argument("--dwell", type=float, default=0.5, help="seconds per LED")
    p.add_argument("--order", choices=["row", "col"], default="row")
    p = sub.add_parser("raw"); p.add_argument("lines", nargs="+")
    a = ap.parse_args(argv)

    drv = ArduinoDriver(a.port, boot_wait=0.3)
    try:
        if getattr(a, "rows", None) is not None and a.cmd not in ("scan",):
            drv.rows(idx_list(a.rows))
        if getattr(a, "col", None) is not None and a.cmd not in ("led",):
            drv.col(a.col)
        if getattr(a, "cols", None) is not None and a.cmd not in ("scan",):
            drv.dc_cols(idx_list(a.cols))

        if a.cmd == "status":
            pass
        elif a.cmd == "on":
            drv.on()
        elif a.cmd == "off":                      # everything dark
            drv.off()
            drv.dc_cols([])
        elif a.cmd == "all":
            drv.all_on()
        elif a.cmd == "pwm":
            hz, d = drv.set_pwm(parse_freq(a.freq), a.duty)
            print(f"PWM {fmt_hz(hz)} at {d:.4f}% (actual)")
        elif a.cmd == "duty":
            hz, d = drv.duty(a.duty)
            print(f"PWM {fmt_hz(hz)} at {d:.4f}% (actual)")
        elif a.cmd == "pulse":
            hz, d = drv.pulse(a.period_us, a.width_us)
            print(f"pulses {a.width_us:g} us every {1e6 / hz:.4g} us ({fmt_hz(hz)}, {d:.3f}%)")
        elif a.cmd == "burst":
            if a.freq:
                drv.set_pwm(parse_freq(a.freq), a.duty)
            hz, d = drv.burst(a.n)
            print(f"sent {a.n} pulses at {fmt_hz(hz)}, {d:.3f}% duty; LEDs now off")
        elif a.cmd == "led":
            drv.led(a.row, a.col)
            if a.freq:
                if a.col not in PWM_COLS:
                    sys.exit(f"column {a.col} is DC only; PWM-capable columns: {PWM_COLS}")
                hz, d = drv.set_pwm(parse_freq(a.freq), a.duty)
                print(f"LED ({a.row},{a.col}) PWM {fmt_hz(hz)} at {d:.4f}%")
            else:
                print(f"LED ({a.row},{a.col}) on")
        elif a.cmd == "rows":
            print(f"{drv.rows(idx_list(a.which))} rows HIGH")
        elif a.cmd == "cols":
            print(f"{drv.dc_cols(idx_list(a.which))} extra DC columns LOW")
        elif a.cmd == "col":
            drv.col(a.pin)
        elif a.cmd == "scan":
            rows = idx_list(a.rows) or list(range(24))
            cols = idx_list(a.cols) or list(range(24))
            pairs = [(r, c) for r in rows for c in cols] if a.order == "row" \
                else [(r, c) for c in cols for r in rows]
            print(f"scanning {len(pairs)} LEDs, {a.dwell}s each (Ctrl-C to stop)")
            for r, c in pairs:
                drv.led(r, c)
                print(f"  LED row {r:2d} col {c:2d}", flush=True)
                time.sleep(a.dwell)
            drv.off()
        elif a.cmd == "raw":
            for line in a.lines:
                drv.ser.write((line + "\n").encode())
                print(f"{line} -> {drv.ser.readline().decode(errors='replace').strip()}")
            return
        show(drv)
    finally:
        drv.close(keep_state=True)


if __name__ == "__main__":
    main()
