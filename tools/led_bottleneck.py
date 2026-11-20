#!/usr/bin/env python3
"""led_bottleneck.py - what limits how fast the LED matrix can switch?

Camera-only experiments, built on tools/led_switching.py and the
led_pwm_giga sketch (needs its R / C / P commands):

  A. load test   Same column, 1 .. 24 rows lit. The fixed-duty frequency sweep
                 gives the lost on-time per period (tau) for each load. If tau
                 grows with the number of lit LEDs, the column driver (board
                 transistor / GIGA pin current / wiring RC) is the limit. A
                 micro-LED's own response does not depend on how many of its
                 neighbours share the column, so a flat tau points at the LED
                 or a fixed driver delay instead.
  B. pulse sweep Fixed period, the on-pulse shrunk from many us down to 0.1 us
                 (set in GIGA timer ticks, ~5 ns). Light per pulse vs width:
                   - x-intercept of the straight part = net lost time per pulse
                   - a sharp threshold (no light below some width, then slope 1)
                     means a pure turn-on DELAY (typical of a transistor that
                     must come out of cut-off / saturation, or a line that must
                     charge up to the LED threshold)
                   - a soft curved knee means slow RAMPS (RC-limited edges)
                 The shortest pulse that still gives half its ideal light sets
                 the fastest usable rate.
  C. column test Repeat the sweep on a second column pin. A different result
                 means part-to-part spread in the board's column drivers.

    python3 tools/led_bottleneck.py --port /dev/ttyACM0 --exposure-us 3000
    python3 tools/led_bottleneck.py --backend sim --sim-load-ns 40   # selftest

Writes results/led_switching/<stamp>_bottleneck.{json,png} and prints a verdict.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import led_switching as L  # noqa: E402


def measure(cam, frames, discard):
    return cam.grab_n(frames, discard)


def detect(cam, drv, a):
    drv.on()
    on = cam.grab_n(a.roi_frames, a.discard).astype(np.float32).mean(0)
    drv.off()
    off = cam.grab_n(a.roi_frames, a.discard).astype(np.float32).mean(0)
    roi = L.detect_led_roi(on, off, close_px=a.close_px)
    roi.mask &= on < a.sat_level
    roi.n_led = int(roi.mask.sum())
    dc = float(L.frame_signal(on[None], roi, off)[0])
    return roi, off, dc, on


def level(cam, roi, dark, a):
    sig = L.frame_signal(cam.grab_n(a.frames, a.discard), roi, dark)
    return float(sig.mean()), float(sig.std(ddof=1) / math.sqrt(len(sig)))


def freq_sweep(cam, drv, roi, dark, a, freqs):
    f_out, y, s = [], [], []
    for f in freqs:
        fa, d = drv.set_pwm(f, a.duty)
        time.sleep(a.settle_s)
        m, e = level(cam, roi, dark, a)
        f_out.append(fa)
        y.append(m / (d / 100))
        s.append(max(e / (d / 100), 1e-6))
    return np.array(f_out), np.array(y), np.array(s)


def fit_tau(f, y, s, duty, max_drop=0.3):
    """-> L0, tau (lost on-time per period, s), tau_err, points used."""
    L0 = y[: max(2, len(y) // 3)].max()
    use = np.ones_like(f, bool)
    for _ in range(10):
        new = (y / L0) >= 1 - max_drop
        if new.sum() < 2:
            new[:2] = True
        a0, b0, cov, _ = L.wlinfit(f[new], y[new], s[new])
        done = np.array_equal(new, use)
        use, L0 = new, a0
        if done:
            break
    slope = -b0 / a0
    J = np.array([b0 / a0 ** 2, -1 / a0])
    err = math.sqrt(max(J @ cov @ J, 0))
    D = duty / 100
    return a0, slope * D, err * D, int(use.sum())


def spread_rows(n):
    """n row indices spread over 0..23 (rowMask order alternates sides)."""
    if n >= 24:
        return None
    return sorted({int(round(i * 23 / max(n - 1, 1))) for i in range(n)}) if n > 1 else [12]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["v4l2", "sim"], default="v4l2")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--pixel-format", default="RG10")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--exposure-us", type=int, default=3000)
    ap.add_argument("--gain", type=float, default=0)
    ap.add_argument("--duty", type=float, default=50)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--roi-frames", type=int, default=8)
    ap.add_argument("--discard", type=int, default=8,
                    help="frames dropped after each change (V4L2 queue lags ~4 frames)")
    ap.add_argument("--settle-s", type=float, default=0.05)
    ap.add_argument("--close-px", type=int, default=7)
    ap.add_argument("--sat-level", type=float, default=60000)
    ap.add_argument("--rows", default="1,3,6,12,24", help="lit-row counts for the load test")
    ap.add_argument("--freqs", default="2k,10k,25k,50k,100k,125k",
                    help="sweep used per load (stay in the linear region)")
    ap.add_argument("--period-us", type=int, default=20, help="pulse sweep period")
    ap.add_argument("--widths", default="0.1,0.2,0.3,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,6,8,10,12,15",
                    help="pulse widths, us")
    ap.add_argument("--col2", type=int, default=5, help="second column pin (0 = skip)")
    ap.add_argument("--sim-tr-us", type=float, default=1.5)
    ap.add_argument("--sim-tf-us", type=float, default=0.8)
    ap.add_argument("--sim-load-ns", type=float, default=0.0,
                    help="sim: extra edge time per lit row")
    a = ap.parse_args(argv)

    os.makedirs(L.OUTDIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(L.OUTDIR, stamp + "_bottleneck")
    freqs = L.parse_freqs(a.freqs)
    widths = [float(w) for w in a.widths.split(",")]

    if a.backend == "sim":
        drv = L.SimRig(t_r=a.sim_tr_us * 1e-6, t_f=a.sim_tf_us * 1e-6,
                       exposure_us=a.exposure_us)
        drv.load_s_per_row = a.sim_load_ns * 1e-9
    else:
        drv = L.ArduinoDriver(a.port)
    cam = L.Camera(a.backend, a.device, a.pixel_format, a.exposure_us, a.gain,
                   sim=drv if a.backend == "sim" else None)
    res = dict(stamp=stamp, exposure_us=a.exposure_us, duty=a.duty, load=[],
               pulse=None, col2=None)
    try:
        # ---------------- A. load test ---------------------------------
        print("[A] load test: tau vs number of lit rows (column D9)")
        drv.col(9)
        for n in [int(x) for x in a.rows.split(",")]:
            idx = spread_rows(n)
            got = drv.rows(idx)
            try:
                roi, dark, dc, _ = detect(cam, drv, a)
            except RuntimeError as e:
                print(f"    {n:2d} rows: no LED found ({e}); skipped")
                continue
            f, y, s = freq_sweep(cam, drv, roi, dark, a, freqs)
            L0, tau, err, k = fit_tau(f, y, s, a.duty)
            res["load"].append(dict(rows=got, row_idx=idx, led_px=roi.n_led,
                                    L0=L0, dc=dc, tau_s=tau, tau_err_s=err,
                                    t_sw_s=2 * tau, f=f.tolist(),
                                    y_norm=(y / L0).tolist()))
            print(f"    {got:2d} rows: lost on-time {tau * 1e9:6.0f} +- {err * 1e9:4.0f} ns"
                  f"  (t_r+t_f {2 * tau * 1e6:.2f} us, {roi.n_led} px, {k} pts)")
        drv.rows(None)

        # ---------------- B. pulse-width sweep -------------------------
        print(f"[B] pulse sweep: period {a.period_us} us, all rows")
        roi, dark, dc, on = detect(cam, drv, a)
        w_out, lv, le = [], [], []
        for w in widths + [a.period_us * 0.5]:
            if w >= a.period_us:
                continue
            _, d = drv.pulse(a.period_us, w)
            time.sleep(a.settle_s)
            m, e = level(cam, roi, dark, a)
            w_out.append(d / 100 * a.period_us)
            lv.append(m)
            le.append(e)
            print(f"    width {w_out[-1]:6.3f} us  light {m / dc:7.4f} of DC  "
                  f"(ideal {w_out[-1] / a.period_us:.4f})")
        w = np.array(w_out)
        eff = np.array(lv) / dc * a.period_us        # light per pulse, in us of DC-on
        eff_err = np.array(le) / dc * a.period_us
        lin = w >= 4
        if lin.sum() >= 2:
            b, m_, cov, _ = L.wlinfit(w[lin], eff[lin], np.maximum(eff_err[lin], 1e-4))
            w0 = -b / m_
        else:
            b, m_, w0 = 0.0, 1.0, float("nan")
        ratio = eff / w
        order = np.argsort(w)
        w50 = float("nan")
        for i in range(1, len(order)):
            i0, i1 = order[i - 1], order[i]
            if ratio[i0] < 0.5 <= ratio[i1]:
                t = (0.5 - ratio[i0]) / (ratio[i1] - ratio[i0])
                w50 = float(w[i0] + t * (w[i1] - w[i0]))
        lit = eff[order] > 0.02                     # > 20 ns worth of light
        w_dark = float(w[order][np.nonzero(lit)[0][0]]) if lit.any() else float("nan")
        res["pulse"] = dict(period_us=a.period_us, width_us=w.tolist(),
                            light_us=eff.tolist(), light_err_us=eff_err.tolist(),
                            slope=m_, intercept_us=w0, w50_us=w50,
                            first_light_us=w_dark, dc=dc)
        print(f"    straight part: slope {m_:.3f}, x-intercept {w0:.3f} us (lost per pulse)")
        print(f"    half-efficiency pulse width w50 = {w50:.3f} us; first light at "
              f"~{w_dark:.2f} us")

        # ---------------- C. second column -----------------------------
        if a.col2:
            print(f"[C] column D{a.col2} vs D9, all rows")
            drv.col(a.col2)
            try:
                roi2, dark2, dc2, _ = detect(cam, drv, a)
                f, y, s = freq_sweep(cam, drv, roi2, dark2, a, freqs)
                L0, tau, err, _ = fit_tau(f, y, s, a.duty)
                res["col2"] = dict(pin=a.col2, tau_s=tau, tau_err_s=err)
                print(f"    D{a.col2}: lost on-time {tau * 1e9:.0f} +- {err * 1e9:.0f} ns")
            except RuntimeError as e:
                print(f"    D{a.col2}: no LED found ({e})")
            drv.col(9)
        drv.off()
    finally:
        try:
            drv.close()
        finally:
            cam.close()

    verdict(res)
    with open(base + ".json", "w") as fh:
        json.dump(res, fh, indent=2)
    plot(base, res)
    print(f"saved {base}.json / .png")
    return res


def verdict(res):
    print("\n=== verdict ===")
    ld = res["load"]
    lines = []
    if len(ld) >= 2:
        n = np.array([d["rows"] for d in ld], float)
        t = np.array([d["tau_s"] for d in ld])
        e = np.array([max(d["tau_err_s"], 1e-10) for d in ld])
        b, m, cov, _ = L.wlinfit(n, t, e)
        per_row, per_row_err = m, math.sqrt(cov[1, 1])
        share = m * 23 / max(t.max(), 1e-12)
        res["load_fit"] = dict(tau0_s=b, per_row_s=m, per_row_err_s=per_row_err,
                               load_share_at_24=share)
        lines.append(f"tau = {b * 1e9:.0f} ns + {m * 1e9:.1f} +- {per_row_err * 1e9:.1f} ns per lit row "
                     f"-> load explains {share:.0%} of tau at 24 rows")
        if abs(m) < 2 * per_row_err or share < 0.2:
            lines.append("load-INDEPENDENT: not limited by line capacitance or by how much "
                         "current the column driver can sink. The limit is a fixed delay: "
                         "the board transistor's switching delay, or the LED itself.")
        elif share < 0.6:
            lines.append(f"MOSTLY a fixed delay ({b * 1e9:.0f} ns that does not depend on load: "
                         "board transistor switching delay or the LED), plus a smaller "
                         "load-dependent part (charging the column through the driver).")
        else:
            lines.append("load-DEPENDENT: more lit LEDs -> slower. The column driver "
                         "(board transistor / its drive from the GIGA pin) runs out of "
                         "current to charge the column, so it is the bottleneck, not the LED.")
    p = res.get("pulse")
    if p:
        lines.append(f"pulse sweep: lost {p['intercept_us']:.2f} us per pulse; pulses shorter "
                     f"than ~{p['w50_us']:.2f} us give under half their ideal light")
        if math.isfinite(p["w50_us"]):
            fmax = 1 / (2 * p["w50_us"] * 1e-6)
            res["f_max_50duty_hz"] = fmax
            lines.append(f"fastest usable 50%-duty PWM ~ {L.fmt_hz(fmax)} "
                         "(each on-pulse still gives >= half its ideal light)")
    if res.get("col2") and ld:
        t9 = ld[-1]["tau_s"]
        t2 = res["col2"]["tau_s"]
        lines.append(f"column D{res['col2']['pin']} vs D9: {t2 * 1e9:.0f} vs {t9 * 1e9:.0f} ns")
    lines.append("micro-LEDs themselves typically respond in ns to tens of ns, and the "
                 "GIGA's timer/pin edges are ns-scale, so a us-scale tau that does not "
                 "track load points at the board's drive transistor (e.g. saturation "
                 "storage time). A scope on D9 vs the column net confirms it in one look.")
    for s in lines:
        print("  - " + s)
    res["verdict"] = lines


def plot(base, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ld = res["load"]
    if ld:
        n = [d["rows"] for d in ld]
        ax[0].errorbar(n, [d["tau_s"] * 1e9 for d in ld],
                       yerr=[d["tau_err_s"] * 1e9 for d in ld], fmt="o-", capsize=3)
        if res.get("load_fit"):
            lf = res["load_fit"]
            xx = np.array([0, 24])
            ax[0].plot(xx, (lf["tau0_s"] + lf["per_row_s"] * xx) * 1e9, "r--", lw=1,
                       label=f"{lf['per_row_s'] * 1e9:.1f} ns per row")
            ax[0].legend(fontsize=8)
        ax[0].set_xlabel("lit rows on the column (load)")
        ax[0].set_ylabel("lost on-time per period (ns)")
        ax[0].set_title("A. does the edge slow down with load?")
        ax[0].set_ylim(bottom=0)
        ax[0].grid(alpha=0.3)
        for d in ld:
            ax[1].plot(d["f"], d["y_norm"], "o-", ms=3, label=f"{d['rows']} rows")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("PWM frequency (Hz)")
        ax[1].set_ylabel("L / L0 (per duty)")
        ax[1].set_title("A. roll-off per load")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
    p = res.get("pulse")
    if p:
        w = np.array(p["width_us"])
        e = np.array(p["light_us"])
        ax[2].errorbar(w, e, yerr=p["light_err_us"], fmt="o", ms=4, label="measured")
        ww = np.linspace(0, w.max(), 100)
        ax[2].plot(ww, ww, ":", color="gray", label="ideal (instant edges)")
        if math.isfinite(p["intercept_us"]):
            ax[2].plot(ww, p["slope"] * (ww - p["intercept_us"]), "r--", lw=1,
                       label=f"line: lost {p['intercept_us']:.2f} µs/pulse")
        if math.isfinite(p["w50_us"]):
            ax[2].axvline(p["w50_us"], color="g", ls="--", lw=1,
                          label=f"w50 = {p['w50_us']:.2f} µs")
        ax[2].set_xlim(0, min(w.max(), 6))
        ax[2].set_ylim(-0.1, min(w.max(), 6))
        ax[2].set_xlabel("commanded on-pulse width (µs)")
        ax[2].set_ylabel("light per pulse (µs of steady-on)")
        ax[2].set_title(f"B. pulse width sweep, period {p['period_us']} µs")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)
    fig.suptitle(f"LED matrix speed bottleneck, exposure {res['exposure_us']} µs", fontsize=10)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
