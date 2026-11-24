#!/usr/bin/env python3
"""led_pixel_pwm.py - per-pixel PWM tests: forward model and backward (inverse) proof.

Each LED images as 2-4 lobes (dark cross), so nothing here averages over an
LED first: every camera pixel is its own measurement.

MODEL (per pixel p). An ideal PWM column gives light in proportion to the duty D.
A real LED loses a fixed time delta per period to its rise/fall edges, so the
fraction of each period that is "on" is D - delta*f, and

    G_p(f, D) = A_p * (D - delta_p * f) + C_p          (dark-subtracted grey)

A_p   = the pixel's grey level at 100 % duty, DC (optics, LED brightness, gain)
delta = lost on-time per period, seconds (~ (t_r + t_f)/2 for linear edges)
C_p   = offset; ~0 if the dark subtraction is right (checked, not assumed)

1. forward   python3 tools/led_pixel_pwm.py forward --led-map <map.npz> \\
                 --freqs 1k,5k,10k,20k,50k,100k,150k,200k --duties 20,35,50,65,80
   Captures every (f, D) step in RANDOM order, with a DC reference step
   re-measured every --ref-every steps (LED/temperature drift) and dark frames
   at start, middle and end. Exposure, gain, frame rate and black level are set
   once and READ BACK from the sensor driver at every step; the run aborts if
   any of them moves. Saves the per-pixel grey cube and fits A, delta, C for
   every pixel (maps + per-LED summaries).

2. backward  python3 tools/led_pixel_pwm.py backward <forward.npz>
   Offline inverse proof. Calibrates A_p, delta_p on half the frequencies, then
   on the held-out half uses ONLY the camera grey + the known PWM settings to
     (a) deduce the LED speed:   delta from G(f)/G(f_ref) at known f, D
     (b) deduce the frequency:   f from G(f)/G(f_ref) at known D, calibrated delta
   and compares with the true values.

3. closedloop python3 tools/led_pixel_pwm.py closedloop <forward.npz> --freqs ...
   Live feedback loop: at each frequency the duty is servoed until the camera
   grey of the LED pixels hits a target. The model says the required duty is
   D_req(f) = G_target/A + delta*f, so its slope vs f is delta: the LED speed
   measured with the camera in the loop.

Outputs: <repo>/results/led_pixel_pwm/<stamp>_*.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import led_switching as ls  # noqa: E402

OUTDIR = os.path.join(ls.REPO, "results", "led_pixel_pwm")
CTRLS = ("exposure", "gain", "frame_rate", "black_level", "sensor_mode", "trigger_mode")


# ----------------------------------------------------------------------------
# controls
# ----------------------------------------------------------------------------
def read_controls(device: str) -> dict:
    """Read the sensor controls back from the driver (not what we asked for)."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", device, "-C", ",".join(CTRLS)],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    d = {}
    for line in out.splitlines():
        k, _, v = line.partition(":")
        try:
            d[k.strip()] = int(v.strip())
        except ValueError:
            pass
    return d


def soc_temp_c() -> float:
    best = float("nan")
    for z in range(12):
        try:
            with open(f"/sys/class/thermal/thermal_zone{z}/temp") as fh:
                t = int(fh.read()) / 1000.0
            best = t if not math.isfinite(best) else max(best, t)
        except FileNotFoundError:
            break
        except Exception:                 # some zones refuse reads on Jetson
            continue
    return best


# ----------------------------------------------------------------------------
# model fits (vectorised over pixels)
# ----------------------------------------------------------------------------
def fit_pixels(f, D, G, use):
    """G (steps, P) ~ A*D - B*f + C. -> A, delta=B/A, C, rms residual (P,)."""
    X = np.stack([D, -f, np.ones_like(f)], 1)[use]
    coef, *_ = np.linalg.lstsq(X, G[use], rcond=None)
    A, B, C = coef
    res = G[use] - X @ coef
    rms = np.sqrt((res ** 2).mean(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(A > 0, B / A, np.nan)
    return A, delta, C, rms


def global_fit(f, D, g, max_loss=0.3):
    """Fit the summed grey g(step) of all LED pixels; drop steps where the edge
    loss delta*f exceeds max_loss of D (the linear-edge model stops holding once
    the pulse is not much longer than the edges). -> A, delta, C, use mask."""
    use = np.ones(len(f), bool)
    for _ in range(4):
        X = np.stack([D, -f, np.ones_like(f)], 1)
        A, B, C = np.linalg.lstsq(X[use], g[use], rcond=None)[0]
        delta = B / A if A > 0 else 0.0
        new = delta * f <= max_loss * D
        if new.sum() < 4 or (new == use).all():
            break
        use = new
    return float(A), float(delta), float(C), use


# ----------------------------------------------------------------------------
# rig
# ----------------------------------------------------------------------------
class Rig:
    def __init__(self, a):
        from led_control import idx_list
        self.a = a
        self.drv = ls.ArduinoDriver(a.port, a.baud, boot_wait=0.3)
        self.drv.col(a.pwm_col)
        self.drv.rows(idx_list(a.rows))
        self.drv.dc_cols([])
        self.drv.off()
        self.cam = ls.Camera("v4l2", a.device, a.pixel_format, a.exposure_us, a.gain)
        self.ctrl0 = read_controls(a.device)
        print(f"[ctrl] sensor controls read back: {self.ctrl0}")
        if self.ctrl0 and (self.ctrl0.get("exposure") != a.exposure_us
                           or self.ctrl0.get("gain") != int(round(a.gain))):
            raise RuntimeError(f"driver did not take exposure/gain: {self.ctrl0}")

    def check_controls(self) -> dict:
        c = read_controls(self.a.device)
        if self.ctrl0 and c != self.ctrl0:
            raise RuntimeError(f"sensor controls changed during the run: {self.ctrl0} -> {c}")
        return c

    def grab(self, n=None):
        fr = self.cam.grab_n(n or self.a.frames, self.a.discard).astype(np.float32)
        return fr.mean(0), fr.std(0, ddof=1) if len(fr) > 1 else np.zeros(fr.shape[1:], np.float32)

    def dark(self):
        self.drv.off()
        time.sleep(self.a.settle_s)
        return self.grab(self.a.dark_frames)[0]

    def dc(self):
        self.drv.on()
        time.sleep(self.a.settle_s)
        return self.grab()[0]

    def pwm(self, f, d):
        f_act, d_act = self.drv.set_pwm(f, d)
        time.sleep(self.a.settle_s)
        return f_act, d_act

    def close(self):
        try:
            self.drv.close()
        finally:
            self.cam.close()


def region_from_map(a, shape):
    if a.led_map:
        labels, emitters, _ = ls.load_led_map(a.led_map)
        if labels.shape != shape:
            sys.exit(f"--led-map is {labels.shape}, frames are {shape}: same sensor ROI needed")
        return labels, emitters
    return None, []


# ----------------------------------------------------------------------------
# 1. forward
# ----------------------------------------------------------------------------
def cmd_forward(a):
    os.makedirs(OUTDIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUTDIR, stamp + "_forward")
    freqs = ls.parse_freqs(a.freqs)
    duties = [float(x) for x in a.duties.split(",")]
    steps = [(f, d) for f in freqs for d in duties]
    random.Random(a.seed).shuffle(steps)
    rig = Rig(a)
    log = []
    try:
        dark0 = rig.dark()
        dc0 = rig.dc() - dark0
        labels, emitters = region_from_map(a, dc0.shape)
        if labels is not None:
            region = labels > 0
        else:
            roi = ls.detect_led_roi(dc0 + dark0, dark0)
            region = roi.mask
        ys, xs = np.nonzero(ls._dilate(region, a.margin))
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = (slice(y0, y1), slice(x0, x1))
        clip = float((dc0 + dark0)[region].max())
        refs, darks = [], [(0, float(dark0[crop].mean()))]
        print(f"[forward] crop x {x0}..{x1} y {y0}..{y1}; DC max in LEDs {clip:.0f} "
              f"(full scale ~65535); {len(steps)} steps in random order")
        if clip > a.sat_level:
            print("[forward] WARNING: some LED pixels clip at DC; lower --exposure-us")
        # warm-up: the LED/driver brightness creeps up for minutes after being
        # off (measured +24% over 2 min, 2026-11-24). Hold it on until stable.
        t_w, last, stable = time.time(), None, 0
        rig.pwm(10e3, 50.0)
        while time.time() - t_w < a.warmup_s:
            g = float((rig.grab(4)[0] - dark0)[region].mean())
            if last is not None and abs(g / last - 1) < a.warmup_tol / 100:
                stable += 1
            else:
                stable = 0
            last = g
            print(f"[warmup] {time.time() - t_w:5.0f} s  grey {g:8.1f}", flush=True)
            if stable >= 3:
                break
            time.sleep(a.warmup_step_s)
        dc0 = rig.dc() - dark0
        refs.append((0, float(dc0[region].mean())))
        G = np.empty((len(steps), y1 - y0, x1 - x0), np.float32)
        S = np.empty_like(G)
        dark = dark0
        t0 = time.time()
        for i, (f, d) in enumerate(steps):
            if i and i % a.ref_every == 0:
                ref = rig.dc() - dark
                refs.append((i, float(ref[region].mean())))
            if i and i % a.dark_every == 0:
                dark = rig.dark()
                darks.append((i, float(dark[crop].mean())))
            f_act, d_act = rig.pwm(f, d)
            m, s = rig.grab()
            G[i] = (m - dark)[crop]
            S[i] = s[crop]
            c = rig.check_controls()
            log.append(dict(i=i, f_cmd=f, d_cmd=d, f_act=f_act, d_act=d_act,
                            t_s=time.time() - t0, temp_c=soc_temp_c(),
                            n_periods=a.exposure_us * 1e-6 * f_act,
                            mean_led=float(G[i][region[crop]].mean()), controls=c))
            print(f"  {i + 1:3d}/{len(steps)}  f {ls.fmt_hz(f_act):>10}  D {d_act:6.2f}%  "
                  f"LED grey {log[-1]['mean_led']:8.1f}", flush=True)
        ref = rig.dc() - dark
        refs.append((len(steps), float(ref[region].mean())))
        darkE = rig.dark()
        darks.append((len(steps), float(darkE[crop].mean())))
    finally:
        rig.close()
    meta = dict(stamp=stamp, exposure_us=a.exposure_us, gain_mdb=a.gain, pwm_col=a.pwm_col,
                rows=a.rows or "all", frames=a.frames, discard=a.discard, seed=a.seed,
                controls=rig.ctrl0, crop=[int(x0), int(y0), int(x1), int(y1)],
                led_map=a.led_map, dc_refs=refs, darks=darks, steps=log,
                emitters=emitters)
    np.savez_compressed(base + ".npz", G=G, S=S.astype(np.float16),
                        f=np.array([r["f_act"] for r in log]),
                        D=np.array([r["d_act"] for r in log]) / 100.0,
                        dc=dc0[crop], region=region[crop],
                        labels=(labels[crop] if labels is not None else region[crop].astype(np.int32)),
                        meta=json.dumps(meta))
    print(f"saved {base}.npz")
    analyze_forward(base + ".npz", max_loss=a.max_loss)


def drift_factor(meta, n):
    refs = np.array(meta["dc_refs"])
    if len(refs) < 2:
        return np.ones(n, np.float32)
    return (np.interp(np.arange(n), refs[:, 0], refs[:, 1]) / np.median(refs[:, 1])).astype(np.float32)


def register_steps(G, ref):
    """The image moves by ~1-5 px during a run (stage/focus drift, measured
    2026-11-24). With lobes only a few px wide that dominates any per-pixel
    comparison, so every step is shifted onto the DC reference image
    (sub-pixel phase correlation). -> registered G, shifts (n, 2) as (dx, dy)."""
    import cv2
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    r = ref.astype(np.float32)
    out = np.empty_like(G)
    sh = np.zeros((len(G), 2))
    for i, g in enumerate(G):
        (dx, dy), _ = cv2.phaseCorrelate(r, g.astype(np.float32), win)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out[i] = cv2.warpAffine(g, M, g.shape[::-1], flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        sh[i] = dx, dy
    return out, sh


def load_G(z, meta, drift_corr=True, register=True, verbose=True):
    G = z["G"].astype(np.float32)
    if register:
        G, sh = register_steps(G, z["dc"])
        if verbose:
            print(f"  image motion vs the DC reference: dx {sh[:, 0].min():+.2f}..{sh[:, 0].max():+.2f} px, "
                  f"dy {sh[:, 1].min():+.2f}..{sh[:, 1].max():+.2f} px (registered out)")
    if drift_corr:
        k = drift_factor(meta, len(G))
        G = G / k[:, None, None]
        if verbose:
            print(f"  brightness drift: each step scaled by the interpolated DC reference "
                  f"(factor {k.min():.3f} .. {k.max():.3f})")
    return G


def load_run(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return z, meta


def analyze_forward(path, max_loss=0.3, drift_corr=True, register=True):
    z, meta = load_run(path)
    base = path[:-4]
    G, f, D, region, labels = z["G"], z["f"], z["D"], z["region"], z["labels"]
    n, H, W = G.shape
    refs = np.array(meta["dc_refs"])
    drift = refs[:, 1].max() / refs[:, 1].min() - 1 if len(refs) > 1 else 0.0
    darks = np.array(meta["darks"])
    print("\n=== control variables ===")
    print(f"  exposure {meta['exposure_us']} us, gain {meta['gain_mdb']} mdB, read back from "
          f"the driver at every step: {meta['controls']} (unchanged, else the run aborts)")
    print(f"  DC reference re-measured {len(refs)}x: drift {drift:.2%} peak-to-peak")
    print(f"  dark level re-measured {len(darks)}x: {darks[:, 1].min():.1f} .. {darks[:, 1].max():.1f} counts")
    temps = [s["temp_c"] for s in meta["steps"] if math.isfinite(s["temp_c"])]
    if temps:
        print(f"  SoC temperature {min(temps):.1f} .. {max(temps):.1f} C")
    G = load_G(z, meta, drift_corr, register)
    g = G[:, region].mean(1)
    A, delta, C, use = global_fit(f, D, g, max_loss)
    print("\n=== all LED pixels together ===")
    print(f"  G = A*(D - delta*f) + C: A {A:.1f} counts at 100% duty, delta {delta * 1e6:.3f} us "
          f"(t_r+t_f ~ {2 * delta * 1e6:.2f} us), C {C:.2f} counts; {use.sum()}/{n} steps in fit")
    P = G.reshape(n, -1)
    Ap, dp, Cp, rms = fit_pixels(f, D, P, use)
    Ap, dp, Cp, rms = (x.reshape(H, W) for x in (Ap, dp, Cp, rms))
    good = region & (Ap > 20 * np.median(np.abs(Cp[region])) + 1e-9) & (Ap > 0.1 * np.percentile(Ap[region], 99))
    print(f"\n=== per pixel ({region.sum()} LED pixels, {good.sum()} bright enough to fit delta) ===")
    dg = dp[good] * 1e6
    q = np.percentile(dg, [5, 25, 50, 75, 95])
    print(f"  delta per pixel (us): 5% {q[0]:.3f}  25% {q[1]:.3f}  median {q[2]:.3f}  "
          f"75% {q[3]:.3f}  95% {q[4]:.3f}")
    rel = rms[good] / (Ap[good] * 0.5)
    top = Ap[good] >= np.median(Ap[good])
    print(f"  fit residual: median {np.median(rel):.2%} of the pixel's 50%-duty grey "
          f"(brighter half of pixels {np.median(rel[top]):.2%})")
    # duty linearity at the lowest frequency
    fl = f.min()
    lo = np.isclose(f, fl, rtol=0.02)
    if lo.sum() >= 3:
        Xd = np.stack([D[lo], np.ones(lo.sum())], 1)
        cd, *_ = np.linalg.lstsq(Xd, P[lo][:, good.ravel()], rcond=None)
        resd = P[lo][:, good.ravel()] - Xd @ cd
        nl = np.sqrt((resd ** 2).mean(0)) / np.abs(cd[0] * 0.5)
        print(f"  duty linearity at {ls.fmt_hz(fl)}: per-pixel rms deviation from a straight "
              f"line, median {np.median(nl):.2%}")
    per_led = []
    for e in meta.get("emitters", []):
        m = good & (labels == e["label"])
        if m.sum() < 5:
            continue
        per_led.append(dict(id=e["id"], address=e["address"], n_px=int(m.sum()),
                            delta_us=float(np.median(dp[m]) * 1e6),
                            delta_iqr_us=float(np.subtract(*np.percentile(dp[m], [75, 25])) * 1e6),
                            A_mean=float(Ap[m].mean())))
    if per_led:
        print("\n=== per LED (median of its pixels) ===")
        for r in per_led:
            print(f"  LED {r['id']:3d}: delta {r['delta_us']:.3f} us (IQR {r['delta_iqr_us']:.3f}), "
                  f"{r['n_px']} px, A {r['A_mean']:.0f}")
    res = dict(A=A, delta_s=delta, C=C, steps_in_fit=int(use.sum()), dc_drift=float(drift),
               pixel_delta_us_pct=dict(zip(["p5", "p25", "p50", "p75", "p95"], map(float, q))),
               per_led=per_led)
    with open(base + "_result.json", "w") as fh:
        json.dump(res, fh, indent=2)
    np.savez_compressed(base + "_fit.npz", A=Ap, delta=dp, C=Cp, rms=rms, good=good, use=use)
    plot_forward(base, f, D, G, region, good, Ap, dp, Cp, A, delta, C, use)
    return res


def plot_forward(base, f, D, G, region, good, Ap, dp, Cp, A, delta, C, use):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 5)
    dc = Ap
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(np.where(region, dc, np.nan), cmap="gray")
    ax.set_title("A: grey at 100% duty\n(per pixel)", fontsize=9)
    ax = fig.add_subplot(gs[:, 1])
    v = np.where(good, dp * 1e6, np.nan)
    lo, hi = np.nanpercentile(v, [2, 98])
    im = ax.imshow(v, cmap="viridis", vmin=lo, vmax=hi)
    fig.colorbar(im, ax=ax, label="delta (us)")
    ax.set_title("delta = lost on-time per period\n(per pixel)", fontsize=9)
    # pick pixels: brightest, median, dim (lobe edge / notch)
    idx = np.argwhere(good)
    vals = Ap[good]
    order = np.argsort(vals)
    picks = {"bright": idx[order[-1]], "median": idx[order[len(order) // 2]],
             "dim": idx[order[len(order) // 20]]}
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 2:4])
    duties = sorted(set(np.round(D, 3)))
    cols = plt.cm.plasma(np.linspace(0, 0.85, len(duties)))
    for name, (y, x) in picks.items():
        g = G[:, y, x] / Ap[y, x]
        for dd, cc in zip(duties, cols):
            s = np.isclose(D, dd, atol=2e-3)
            o = np.argsort(f[s])
            ax2.plot(f[s][o], g[s][o], "o-" if name == "bright" else ".:", color=cc, ms=4, lw=1,
                     label=f"D={dd:.0%}" if name == "bright" else None)
            ax2.plot(f[s][o], dd - dp[y, x] * f[s][o] + Cp[y, x] / Ap[y, x], "-",
                     color=cc, lw=0.5, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("PWM frequency (Hz)")
    ax2.set_ylabel("pixel grey / A  (= effective duty)")
    ax2.set_title("single pixels: grey vs frequency (bright o-, median/dim dotted; lines = model)",
                  fontsize=9)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3, which="both")
    freqs = sorted(set(np.round(f, -1)))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(freqs)))
    y, x = picks["bright"]
    for ff, cc in zip(freqs, cols):
        s = np.isclose(f, ff, rtol=0.02)
        o = np.argsort(D[s])
        ax3.plot(D[s][o] * 100, G[s, y, x][o], "o-", color=cc, ms=4, lw=1,
                 label=ls.fmt_hz(ff))
    ax3.set_xlabel("duty (%)")
    ax3.set_ylabel("grey (counts, dark-subtracted)")
    ax3.set_title("brightest pixel: grey vs duty at each frequency", fontsize=9)
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(alpha=0.3)
    ax4 = fig.add_subplot(gs[0, 4])
    ax4.hist(dp[good] * 1e6, bins=60, color="#1f77b4")
    ax4.axvline(delta * 1e6, color="r", ls="--", lw=1, label=f"all pixels {delta * 1e6:.3f} us")
    ax4.set_xlabel("delta (us)")
    ax4.set_title("per-pixel delta", fontsize=9)
    ax4.legend(fontsize=7)
    ax5 = fig.add_subplot(gs[1, 4])
    g = G[:, region].mean(1)
    pred = A * (D - delta * f) + C
    ax5.plot(pred, g, "o", ms=3, color="gray", label="excluded")
    ax5.plot(pred[use], g[use], "o", ms=3, color="#1f77b4", label="in fit")
    m = max(g.max(), pred.max())
    ax5.plot([0, m], [0, m], "k:", lw=0.8)
    ax5.set_xlabel("model")
    ax5.set_ylabel("measured")
    ax5.set_title("all LED pixels: measured vs model", fontsize=9)
    ax5.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=110)
    plt.close(fig)
    print(f"  plot -> {base}.png")


# ----------------------------------------------------------------------------
# 2. backward (offline inverse proof)
# ----------------------------------------------------------------------------
def cmd_backward(a):
    z, meta = load_run(a.run)
    base = a.run[:-4] + "_backward"
    G, f, D, region, labels = z["G"], z["f"], z["D"], z["region"], z["labels"]
    G = load_G(z, meta)
    fit = np.load(a.run[:-4] + "_fit.npz")
    good = fit["good"]
    freqs = np.array(sorted(set(np.round(f, -1))))
    cal_f = set(freqs[::2])                      # calibrate on every other frequency
    is_cal = np.array([round(x, -1) in cal_f for x in f])
    P = G[:, good]
    # calibration: half the frequencies, linear region only
    gsum = P.mean(1)
    Ag, dglob, Cg, use = global_fit(f[is_cal], D[is_cal], gsum[is_cal], a.max_loss)
    Ap, dp, Cp, _ = fit_pixels(f[is_cal], D[is_cal], P[is_cal], use)
    print(f"[backward] calibrated on {sorted(int(x) for x in cal_f)} Hz "
          f"({is_cal.sum()} steps); held out {sorted(int(x) for x in set(freqs) - cal_f)} Hz")
    print(f"[backward] calibration (all pixels): A {Ag:.1f}, delta {dglob * 1e6:.3f} us, C {Cg:.1f}")
    rows = []
    for i in np.nonzero(~is_cal)[0]:
        d_ = D[i]
        # effective duty seen by the camera: D_eff = (G - C)/A = D - delta*f
        eff_all = (gsum[i] - Cg) / Ag
        eff_px = (P[i] - Cp) / Ap
        # (a) LED speed from camera grey + the KNOWN f and D
        dl_all = (d_ - eff_all) / f[i]
        dl_px = (d_ - eff_px) / f[i]
        # (b) PWM frequency from camera grey + the known D + calibrated delta
        f_all = (d_ - eff_all) / dglob
        f_px = (d_ - eff_px) / dp
        inlin = dglob * f[i] <= a.max_loss * d_
        rows.append(dict(f_true=float(f[i]), duty=float(d_), linear_region=bool(inlin),
                         loss=float(1 - eff_all / d_),
                         delta_all_us=float(dl_all * 1e6),
                         delta_px_median_us=float(np.median(dl_px) * 1e6),
                         delta_px_iqr_us=float(np.subtract(*np.percentile(dl_px, [75, 25])) * 1e6),
                         f_all=float(f_all), f_err_all=float(f_all / f[i] - 1),
                         f_px_median=float(np.median(f_px)),
                         f_err_px=float(np.median(f_px) / f[i] - 1)))
    print(f"\n{'f true':>10} {'D':>5} {'loss':>6} | {'delta(all)':>10} {'delta(px med)':>13} | "
          f"{'f deduced':>10} {'err':>7}  {'f px med':>10} {'err':>7}")
    for r in sorted(rows, key=lambda r: (r["f_true"], r["duty"])):
        print(f"{ls.fmt_hz(r['f_true']):>10} {r['duty']:5.0%} {r['loss']:6.1%} | "
              f"{r['delta_all_us']:9.3f}us {r['delta_px_median_us']:12.3f}us | "
              f"{ls.fmt_hz(r['f_all']):>10} {r['f_err_all']:+7.1%}  {ls.fmt_hz(r['f_px_median']):>10} "
              f"{r['f_err_px']:+7.1%}" + ("" if r["linear_region"] else "  (beyond linear region)"))
    lin = [r for r in rows if r["linear_region"] and r["loss"] > a.min_loss]
    if lin:
        e = np.array([r["f_err_all"] for r in lin])
        dd = np.array([r["delta_all_us"] for r in lin])
        print(f"\n[backward] held-out steps in the linear region with loss > {a.min_loss:.0%} "
              f"(smaller losses drown in the ~1% step-to-step noise): {len(lin)}")
        print(f"  speed deduced from camera: delta {np.mean(dd):.3f} +- {np.std(dd, ddof=1) if len(dd) > 1 else 0:.3f} us "
              f"(calibration {dglob * 1e6:.3f} us)")
        print(f"  frequency deduced from camera: rms error {np.sqrt((e ** 2).mean()):.1%}")
    with open(base + ".json", "w") as fh:
        json.dump(dict(cal_freqs=sorted(float(x) for x in cal_f), delta_cal_us=dglob * 1e6,
                       rows=rows), fh, indent=2)
    plot_backward(base, rows, dglob)


def plot_backward(base, rows, dglob):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for r in rows:
        c = "#1f77b4" if r["linear_region"] else "gray"
        ax[0].plot(r["f_true"], r["f_all"], "o", color=c, ms=5)
        ax[1].plot(r["f_true"], r["delta_all_us"], "o", color=c, ms=5)
    lim = [min(r["f_true"] for r in rows) * 0.7, max(r["f_true"] for r in rows) * 1.4]
    ax[0].plot(lim, lim, "k:", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("true PWM frequency (Hz)")
    ax[0].set_ylabel("frequency deduced from camera (Hz)")
    ax[0].set_title("backward: frequency (held-out steps; grey = beyond linear region)", fontsize=9)
    ax[1].axhline(dglob * 1e6, color="r", ls="--", lw=1, label="forward calibration")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("PWM frequency (Hz)")
    ax[1].set_ylabel("delta deduced from camera (us)")
    ax[1].set_title("backward: LED speed from each held-out step", fontsize=9)
    ax[1].legend(fontsize=8)
    for a_ in ax:
        a_.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=110)
    plt.close(fig)
    print(f"  plot -> {base}.png")


# ----------------------------------------------------------------------------
# 3. closed loop (live)
# ----------------------------------------------------------------------------
def cmd_closedloop(a):
    z, meta = load_run(a.run)
    a.exposure_us, a.gain, a.pwm_col = meta["exposure_us"], meta["gain_mdb"], meta["pwm_col"]
    a.rows = None if meta["rows"] == "all" else meta["rows"]
    res = json.load(open(a.run[:-4] + "_result.json"))
    C, delta = res["C"], res["delta_s"]
    x0, y0, x1, y1 = meta["crop"]
    crop = (slice(y0, y1), slice(x0, x1))
    region = z["region"]
    freqs = ls.parse_freqs(a.freqs)
    tgt = a.target_duty / 100
    base = os.path.join(OUTDIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_closedloop")
    rig = Rig(a)
    out = []
    try:
        dark = rig.dark()
        for f in freqs:
            # Start from what an IDEAL LED would need (no knowledge of delta);
            # the camera feedback has to find the edge loss by itself.
            d = a.target_duty
            hist = []
            for it in range(a.iters):
                # measured effective duty = (G_pwm - C) / (G_dc - C): the DC frame
                # taken right before makes it immune to LED brightness drift
                gdc = float((rig.dc() - dark)[crop][region].mean())
                f_act, d_act = rig.pwm(f, min(d, 99.0))
                g = float((rig.grab()[0] - dark)[crop][region].mean())
                eff = (g - C) / (gdc - C)
                hist.append(dict(duty=d_act, grey=g, dc=gdc, eff=eff))
                err = tgt - eff
                print(f"     {ls.fmt_hz(f_act):>10} it {it}: duty {d_act:6.2f}%  effective "
                      f"{eff:6.2%}  (target {tgt:.0%})", flush=True)
                if abs(err) < a.tol / 100:
                    break
                d = d_act + 100 * err                  # integral-type update, loop gain 1
            out.append(dict(f=f_act, d_req=d_act, eff=eff, iters=len(hist), hist=hist,
                            converged=abs(err) < a.tol / 100))
            print(f"  {ls.fmt_hz(f_act):>10}: duty needed {d_act:6.2f}% to look like {tgt:.0%} "
                  f"after {len(hist)} iterations", flush=True)
        dark2 = rig.dark()
    finally:
        rig.close()
    fa = np.array([r["f"] for r in out])
    dr = np.array([r["d_req"] for r in out]) / 100
    X = np.stack([np.ones_like(fa), fa], 1)
    (c0, slope), *_ = np.linalg.lstsq(X, dr, rcond=None)
    print(f"\n[closedloop] required duty = {c0:.4f} + delta*f  ->  delta {slope * 1e6:.3f} us "
          f"(forward model {delta * 1e6:.3f} us); dark drift "
          f"{float((dark2 - dark)[crop].mean()):+.2f} counts")
    with open(base + ".json", "w") as fh:
        json.dump(dict(forward=a.run, target_duty=a.target_duty, delta_loop_us=slope * 1e6,
                       delta_forward_us=delta * 1e6, steps=out), fh, indent=2)
    print(f"saved {base}.json")


# ----------------------------------------------------------------------------
def rig_args(p):
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--pixel-format", default="RG10")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--frames", type=int, default=8, help="frames averaged per step")
    p.add_argument("--dark-frames", type=int, default=16)
    p.add_argument("--discard", type=int, default=8,
                   help="frames dropped after every change (V4L2 queue holds ~4 old frames)")
    p.add_argument("--settle-s", type=float, default=0.05)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fw = sub.add_parser("forward", help="per-pixel grey vs PWM frequency and duty")
    rig_args(fw)
    fw.add_argument("--led-map", help="_ledmap.npz from led_switching.py ledmap (same exposure)")
    fw.add_argument("--pwm-col", type=int, default=9)
    fw.add_argument("--rows", help="rows lit (all | 0,3,5-8)")
    fw.add_argument("--exposure-us", type=int, default=20000)
    fw.add_argument("--gain", type=float, default=0)
    fw.add_argument("--freqs", default="1k,3k,10k,20k,40k,70k,100k,150k,200k")
    fw.add_argument("--duties", default="20,35,50,65,80")
    fw.add_argument("--ref-every", type=int, default=1,
                    help="DC reference every N steps (1 = right before every step: "
                         "LED brightness wanders a few %% over minutes)")
    fw.add_argument("--warmup-s", type=float, default=240, help="max warm-up time")
    fw.add_argument("--warmup-tol", type=float, default=0.3,
                    help="warm-up ends when 3 checks in a row change less than this %%")
    fw.add_argument("--warmup-step-s", type=float, default=10)
    fw.add_argument("--dark-every", type=int, default=15, help="new dark every N steps")
    fw.add_argument("--margin", type=int, default=6)
    fw.add_argument("--sat-level", type=float, default=60000)
    fw.add_argument("--max-loss", type=float, default=0.3)
    fw.add_argument("--seed", type=int, default=1)
    an = sub.add_parser("analyze", help="re-fit a saved forward run")
    an.add_argument("run")
    an.add_argument("--max-loss", type=float, default=0.3)
    an.add_argument("--no-drift-corr", dest="drift_corr", action="store_false")
    an.add_argument("--no-register", dest="register", action="store_false")
    bw = sub.add_parser("backward", help="inverse proof on held-out steps of a forward run")
    bw.add_argument("run", help="..._forward.npz")
    bw.add_argument("--max-loss", type=float, default=0.3)
    bw.add_argument("--min-loss", type=float, default=0.05,
                    help="only score steps whose light loss exceeds this (inversion needs signal)")
    cl = sub.add_parser("closedloop", help="servo duty to a target grey at each frequency")
    rig_args(cl)
    cl.add_argument("run", help="..._forward.npz (gives A, delta, exposure, region)")
    cl.add_argument("--freqs", default="1k,20k,50k,100k,150k")
    cl.add_argument("--target-duty", type=float, default=40.0,
                    help="target grey = this %% of A (what an ideal LED gives at this duty)")
    cl.add_argument("--iters", type=int, default=6)
    cl.add_argument("--tol", type=float, default=0.2, help="stop within this %% duty")
    a = ap.parse_args(argv)
    if a.cmd == "forward":
        cmd_forward(a)
    elif a.cmd == "analyze":
        analyze_forward(a.run, a.max_loss, a.drift_corr, a.register)
    elif a.cmd == "backward":
        cmd_backward(a)
    else:
        cmd_closedloop(a)


if __name__ == "__main__":
    main()
