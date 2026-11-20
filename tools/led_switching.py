#!/usr/bin/env python3
"""led_switching.py - measure LED switching speed with a slow camera.

Idea: at a fixed PWM duty D the time-averaged light should not depend on the
PWM frequency f. Real LEDs (and their drivers) take a finite time to turn on
and off, and that edge time eats into each period. The fraction lost grows
with f, so at fixed duty

    L(f) ~= L0 * [1 - k * (t_r + t_f) * f]

A camera whose exposure spans many PWM periods integrates L(f) directly, so
the slope of grey level vs f gives (t_r + t_f) without a fast camera.

Pipeline (runs ON THE JETSON, SSH is fine, close the GUI first):
  1. ROI    LED held on vs off -> difference image -> threshold + morphology +
            connected components -> LED pixel mask, bounding box, and a
            background annulus around it. Optional sensor crop / software bin.
  2. sweep  for each frequency: Arduino sets f at duty D (or you set it by hand
            and type it in), frames are captured, and every frame is
            dark-subtracted on its own: LED-mask mean minus the annulus mean of
            the SAME frame (cancels black-level drift and row offsets), after a
            pixelwise master-dark removal of fixed pattern.
  3. fit    weighted linear fit of the low-f (linear) part, slope -> lost
            on-time per period -> t_r + t_f, plus the roll-off (knee) frequency
            and a plot of normalized grey level vs frequency.

Commands:
  python3 tools/led_switching.py sweep --port /dev/ttyACM0 --duty 50 \\
          --freqs 100,1k,10k,20k,50k,100k,200k,500k --exposure-us 20000
  python3 tools/led_switching.py sweep --manual --duty 50 --freqs 1k,10k,100k
  python3 tools/led_switching.py sweep --manual --duty 50          # type each f
  python3 tools/led_switching.py analyze results/led_switching/<stamp>
  python3 tools/led_switching.py selftest --tsw-us 3                # simulated

Outputs go to <repo>/results/led_switching/<stamp>_*.{csv,json,png,npz}.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OUTDIR = os.path.join(REPO, "results", "led_switching")


# ============================================================================
# helpers
# ============================================================================
def parse_freq(s: str) -> float:
    s = s.strip().lower().replace("hz", "")
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    return float(s) * mult


def parse_freqs(s: Optional[str]) -> Optional[list[float]]:
    if not s:
        return None
    return [parse_freq(x) for x in s.split(",") if x.strip()]


def fmt_hz(f: float) -> str:
    if f >= 1e6:
        return f"{f / 1e6:.3g} MHz"
    if f >= 1e3:
        return f"{f / 1e3:.3g} kHz"
    return f"{f:.3g} Hz"


def bin_image(img: np.ndarray, b: int) -> np.ndarray:
    """Sum-free mean binning by an integer factor (crops the ragged edge)."""
    if b <= 1:
        return img
    h, w = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    return img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


# ============================================================================
# 1. LED ROI detection
# ============================================================================
@dataclass
class LedRoi:
    mask: np.ndarray            # bool, full-frame, LED pixels
    bg: np.ndarray              # bool, full-frame, background annulus
    bbox: tuple                 # x, y, w, h of the LED mask
    n_led: int = 0
    n_bg: int = 0
    contrast: float = 0.0       # mean(diff on LED) / noise sigma

    def to_json(self) -> dict:
        return {"bbox": list(map(int, self.bbox)), "n_led_px": int(self.n_led),
                "n_bg_px": int(self.n_bg), "contrast_snr": float(self.contrast)}


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    try:
        import cv2
        return cv2.GaussianBlur(img, (0, 0), sigma)
    except ImportError:
        from scipy import ndimage
        return ndimage.gaussian_filter(img, sigma)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(mask.astype(np.uint8), k) > 0
    except ImportError:
        from scipy import ndimage
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return ndimage.binary_dilation(mask, structure=(xx ** 2 + yy ** 2) <= r * r)


def _close_open(mask: np.ndarray, close_px: int, open_px: int) -> np.ndarray:
    try:
        import cv2
        m = mask.astype(np.uint8)
        if close_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        if open_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1,) * 2)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        return m > 0
    except ImportError:
        from scipy import ndimage
        m = mask
        if close_px > 0:
            m = ndimage.binary_closing(m, iterations=close_px)
        if open_px > 0:
            m = ndimage.binary_opening(m, iterations=open_px)
        return m


def _components(mask: np.ndarray):
    """-> (labels, areas[1..n])"""
    try:
        import cv2
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        return lab, stats[1:, cv2.CC_STAT_AREA]
    except ImportError:
        from scipy import ndimage
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        return lab, np.bincount(lab.ravel())[1:]


def detect_led_roi(on_mean: np.ndarray, off_mean: np.ndarray, *,
                   blur_sigma: float = 2.0, k_sigma: float = 6.0,
                   rel_thresh: float = 0.25, close_px: int = 7, open_px: int = 2,
                   min_area: int = 20, keep_rel_area: float = 0.05,
                   largest_only: bool = False, bg_gap: int = 10,
                   bg_width: int = 25) -> LedRoi:
    """Find the LED in (LED on) - (LED off).

    Threshold = max(k_sigma * robust noise, rel_thresh * peak) of the blurred
    difference. Closing merges an LED array into one blob; components smaller
    than keep_rel_area of the largest are dropped as glints. The background
    annulus sits bg_gap px outside the mask, bg_width px wide, and excludes
    anything that also lit up (reflections).
    """
    diff = on_mean.astype(np.float32) - off_mean.astype(np.float32)
    sm = _blur(diff, blur_sigma) if blur_sigma > 0 else diff
    med = float(np.median(sm))
    sigma = 1.4826 * float(np.median(np.abs(sm - med))) + 1e-6
    peak = float(np.percentile(sm, 99.99))
    thr = max(med + k_sigma * sigma, med + rel_thresh * (peak - med))
    raw = sm > thr
    m = _close_open(raw, close_px, open_px)
    lab, areas = _components(m)
    if len(areas) == 0 or areas.max() < min_area:
        raise RuntimeError(
            f"no LED found: peak diff {peak - med:.1f} vs noise {sigma:.2f}. "
            "Is the LED in view and actually switching between the on/off captures?")
    if largest_only:
        keep = [int(np.argmax(areas)) + 1]
    else:
        keep = [i + 1 for i, a in enumerate(areas)
                if a >= max(min_area, keep_rel_area * areas.max())]
    mask = np.isin(lab, keep)
    ys, xs = np.nonzero(mask)
    bbox = (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    lit = sm > med + 3 * sigma                    # anything that lit up at all
    inner = _dilate(mask, bg_gap)
    outer = _dilate(mask, bg_gap + bg_width)
    bg = outer & ~inner & ~_dilate(lit, 2)
    contrast = float(diff[mask].mean() / max(sigma, 1e-6))
    return LedRoi(mask=mask, bg=bg, bbox=bbox, n_led=int(mask.sum()),
                  n_bg=int(bg.sum()), contrast=contrast)


def save_roi_png(path: str, on_mean, off_mean, roi: LedRoi) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    diff = on_mean.astype(np.float32) - off_mean.astype(np.float32)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    lo, hi = np.percentile(diff, (1, 99.9))
    ax[0].imshow(diff, cmap="gray", vmin=lo, vmax=hi)
    ax[0].set_title("LED on - off")
    ov = np.zeros(diff.shape + (4,), np.float32)
    ov[roi.mask] = (1, 0.2, 0.2, 0.45)
    ov[roi.bg] = (0.2, 0.6, 1, 0.35)
    ax[0].imshow(ov)
    x, y, w, h = roi.bbox
    ax[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="yellow", lw=1))
    pad = max(w, h) // 2 + 40
    ax[1].imshow(diff, cmap="gray", vmin=lo, vmax=hi)
    ax[1].imshow(ov)
    ax[1].set_xlim(max(0, x - pad), min(diff.shape[1], x + w + pad))
    ax[1].set_ylim(min(diff.shape[0], y + h + pad), max(0, y - pad))
    ax[1].set_title(f"zoom: LED {roi.n_led} px (red), background {roi.n_bg} px (blue)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================================
# LED drivers
# ============================================================================
class LedDriver:
    """set_pwm returns the (frequency, duty%) actually produced."""
    name = "base"

    def on(self) -> None: ...
    def off(self) -> None: ...
    def set_pwm(self, f_hz: float, duty: float) -> tuple[float, float]: ...
    def close(self) -> None: ...


class ArduinoDriver(LedDriver):
    """Talks to tools/arduino/led_pwm/led_pwm.ino over USB serial."""
    name = "arduino"

    def __init__(self, port: str, baud: int = 115200):
        try:
            import serial
        except ImportError:
            sys.exit("pyserial missing: pip install pyserial (or sudo apt install python3-serial)")
        self.ser = serial.Serial(port, baud, timeout=2.0)
        time.sleep(2.0)                      # the board resets when the port opens
        self.ser.reset_input_buffer()
        self._cmd("?")                       # fails loudly if the sketch isn't there

    def _cmd(self, line: str):
        self.ser.write((line + "\n").encode())
        reply = self.ser.readline().decode(errors="replace").strip()
        while reply.startswith("READY"):
            reply = self.ser.readline().decode(errors="replace").strip()
        if not reply.startswith("OK"):
            raise RuntimeError(f"Arduino said {reply!r} to {line!r}")
        _, mode, hz, duty = reply.split()
        return mode, float(hz), float(duty)

    def on(self):
        self._cmd("ON")

    def off(self):
        self._cmd("OFF")

    def set_pwm(self, f_hz, duty):
        _, hz, d = self._cmd(f"F {f_hz:.6g} {duty:.6g}")
        return hz, d

    # --- led_pwm_giga extras (row/column selection, sub-us pulses) ----------
    def _raw(self, line: str) -> str:
        self.ser.write((line + "\n").encode())
        reply = self.ser.readline().decode(errors="replace").strip()
        if not reply.startswith("OK"):
            raise RuntimeError(f"Arduino said {reply!r} to {line!r} "
                               "(is the latest led_pwm_giga sketch flashed?)")
        return reply

    def rows(self, idx) -> int:
        arg = "ALL" if idx is None else ",".join(str(int(i)) for i in idx)
        return int(self._raw(f"R {arg}").split()[-1])

    def col(self, pin: int) -> None:
        self._raw(f"C {int(pin)}")

    def pulse(self, period_us: int, width_us: float) -> tuple[float, float]:
        _, hz, d = self._cmd(f"P {int(period_us)} {width_us:.4f}")
        return hz, d

    def close(self):
        try:
            self.off()
        finally:
            self.ser.close()


class ManualDriver(LedDriver):
    """You set the generator / sketch by hand; the tool waits and asks."""
    name = "manual"

    def on(self):
        input("\n>> Turn the LED fully ON (DC), then press Enter ")

    def off(self):
        input(">> Turn the LED OFF, then press Enter ")

    def set_pwm(self, f_hz, duty):
        r = input(f">> Set PWM to {fmt_hz(f_hz)} at {duty:g}% duty. Enter to accept, "
                  f"or type the actual 'freq [duty]' you set: ").strip()
        if r:
            parts = r.split()
            f_hz = parse_freq(parts[0])
            if len(parts) > 1:
                duty = float(parts[1])
        return f_hz, duty


# ============================================================================
# simulated camera + LED (selftest, and for trying the tool without hardware)
# ============================================================================
class SimRig(LedDriver):
    """Trapezoid LED (rise t_r, fall t_f, both inside the commanded on window)
    imaged as a blob on a noisy sensor with per-frame black-level drift and
    row offsets. Each frame integrates the waveform over the exposure from a
    random PWM phase, which is exactly what the real camera does."""
    name = "sim"

    def __init__(self, t_r=2e-6, t_f=1e-6, exposure_us=20000, shape=(400, 600),
                 peak=600.0, seed=0):
        self.t_r, self.t_f = t_r, t_f
        self.exp = exposure_us * 1e-6
        self.rng = np.random.default_rng(seed)
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xx - 0.62 * w) / 22.0) ** 2 + ((yy - 0.41 * h) / 15.0) ** 2) ** 2)
        self.blob = (peak * blob).astype(np.float32)
        self.glint = 30 * np.exp(-(((xx - 80) / 4) ** 2 + ((yy - 60) / 4) ** 2))
        self.fpn = self.rng.normal(0, 3, shape).astype(np.float32)
        self.shape = shape
        self.state = ("off", 0.0, 0.0)

    # driver side
    def on(self):
        self.state = ("on", 0.0, 100.0)

    def off(self):
        self.state = ("off", 0.0, 0.0)

    def set_pwm(self, f_hz, duty):
        self.state = ("pwm", f_hz, duty)
        return f_hz, duty

    load_s_per_row = 0.0             # extra edge time per lit row (selftest)
    n_rows = 24

    def rows(self, idx):
        self.n_rows = 24 if idx is None else len(idx)
        return self.n_rows

    def col(self, pin):
        pass

    def pulse(self, period_us, width_us):
        return self.set_pwm(1e6 / period_us, 100.0 * width_us / period_us)

    def _avg_level(self) -> float:
        mode, f, d = self.state
        if mode == "on":
            return 1.0
        if mode == "off":
            return 0.0
        T = 1.0 / f
        ton = d / 100.0 * T
        extra = self.load_s_per_row * (self.n_rows - 1)
        tr, tf = min(self.t_r + extra, ton / 2), min(self.t_f + extra / 2, ton / 2)
        per_period = ton - tr / 2 - tf / 2           # area of the trapezoid
        n_full, rem = divmod(self.exp, T)
        # partial period at a random phase: integrate numerically
        ph = self.rng.uniform(0, T)
        t = (ph + np.linspace(0, rem, 200, endpoint=False)) % T
        wave = np.clip(np.minimum(t / max(tr, 1e-12), (ton - t) / max(tf, 1e-12)), 0, 1)
        wave[t > ton] = 0
        part = wave.mean() * rem if rem > 0 else 0.0
        return (n_full * per_period + part) / self.exp

    # camera side
    def grab(self, timeout=1.0):
        lvl = self._avg_level()
        black = 64 + self.rng.normal(0, 2)                 # per-frame drift
        rows = self.rng.normal(0, 1.5, (self.shape[0], 1))  # row noise
        img = black + rows + self.fpn + lvl * (self.blob + self.glint)
        img = img + self.rng.normal(0, 1.0, self.shape) + \
            self.rng.normal(0, 1, self.shape) * np.sqrt(np.maximum(lvl * self.blob, 0) / 4)
        return np.clip(img, 0, 1023).astype(np.uint16)

    def set_exposure(self, us):
        self.exp = us * 1e-6

    def close(self):
        pass


# ============================================================================
# camera wrapper
# ============================================================================
class Camera:
    def __init__(self, backend: str, device: str, pixfmt: str, exposure_us: int,
                 gain: float, sim: Optional[SimRig] = None):
        self.sim = sim
        self.exposure_us = exposure_us
        if backend == "sim":
            self.be = None
            return
        from camera_gui.config import CameraConfig
        from camera_gui.backends.v4l2_backend import V4l2Backend
        cfg = CameraConfig()
        # start at the full sensor; crop after the ROI is known
        cfg.roi.x, cfg.roi.y, cfg.roi.width, cfg.roi.height = 0, 0, 2464, 2064
        cfg.exposure_us = exposure_us
        cfg.gain = gain
        cfg.fps = min(40.0, 0.9e6 / exposure_us)
        self.be = V4l2Backend(cfg, device=device, pixel_format=pixfmt)
        try:
            self.be.open()
        except OSError as e:
            if e.errno == 16:
                sys.exit(f"{device} is busy: close the camera GUI first "
                         "(only one process can stream).")
            raise
        self.be.start_streaming()

    def grab(self) -> np.ndarray:
        if self.be is None:
            return self.sim.grab()
        for _ in range(5):
            f = self.be.grab(timeout=max(2.0, 3 * self.exposure_us * 1e-6))
            if f is not None:
                return f
        raise RuntimeError("camera timeout")

    def grab_n(self, n: int, discard: int = 0) -> np.ndarray:
        for _ in range(discard):
            self.grab()
        return np.stack([self.grab() for _ in range(n)])

    def crop_sensor(self, bbox, margin: int) -> tuple:
        """Hardware ROI around the LED (aligned to the driver's steps).
        Returns the new sensor ROI (x, y, w, h)."""
        if self.be is None:
            return None
        x, y, w, h = bbox
        x0, y0 = max(0, x - margin), max(0, y - margin)
        W = max(608, int(math.ceil((w + 2 * margin) / 32.0)) * 32)
        H = max(32, int(math.ceil((h + 2 * margin) / 8.0)) * 8)
        x0 = min(x0 - x0 % 2, 2464 - W)
        y0 = min(y0 - y0 % 2, 2064 - H)
        self.be.stop_streaming()
        self.be.set_roi(x0, y0, W, H)
        self.be.start_streaming()
        return (x0, y0, W, H)

    def close(self):
        if self.be is not None:
            self.be.stop_streaming()
            self.be.close()


# ============================================================================
# 2. sweep
# ============================================================================
def frame_signal(frames: np.ndarray, roi: LedRoi, master_dark: np.ndarray):
    """Per-frame dark subtraction: each frame -> mean(LED) - mean(annulus) of
    the same frame, after removing the pixelwise master dark (fixed pattern)."""
    out = np.empty(len(frames))
    for i, fr in enumerate(frames):
        d = fr.astype(np.float32) - master_dark
        out[i] = d[roi.mask].mean() - d[roi.bg].mean()
    return out


def cmd_sweep(a) -> str:
    os.makedirs(OUTDIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUTDIR, stamp)
    freqs = parse_freqs(a.freqs)
    if a.backend == "sim":
        rig = SimRig(t_r=a.sim_tr_us * 1e-6, t_f=a.sim_tf_us * 1e-6,
                     exposure_us=a.exposure_us, seed=a.seed)
        driver = rig
    else:
        rig = None
        driver = ManualDriver() if a.manual else ArduinoDriver(a.port, a.baud)
    cam = Camera(a.backend, a.device, a.pixel_format, a.exposure_us, a.gain, sim=rig)
    rows = []
    try:
        # --- ROI: LED held on vs off ------------------------------------
        print(f"[roi] capturing {a.roi_frames} LED-on and LED-off frames ...")
        driver.on()
        on = cam.grab_n(a.roi_frames, a.discard).astype(np.float32).mean(0)
        driver.off()
        off_stack = cam.grab_n(a.dark_frames, a.discard)
        off = off_stack.astype(np.float32).mean(0)
        roi = detect_led_roi(on, off, close_px=a.close_px, k_sigma=a.k_sigma,
                             largest_only=a.largest_only)
        print(f"[roi] LED bbox x,y,w,h = {roi.bbox}, {roi.n_led} px, background "
              f"{roi.n_bg} px, on/off contrast {roi.contrast:.0f} sigma")
        if roi.n_bg < 200:
            print("[roi] WARNING: small background annulus, per-frame dark is noisy")
        save_roi_png(base + "_roi.png", on, off, roi)
        sensor_roi = None
        if a.hw_crop:
            sensor_roi = cam.crop_sensor(roi.bbox, a.crop_margin)
            print(f"[roi] sensor cropped to {sensor_roi}; re-detecting in the crop")
            driver.on()
            on = cam.grab_n(a.roi_frames, a.discard).astype(np.float32).mean(0)
            driver.off()
            off_stack = cam.grab_n(a.dark_frames, a.discard)
            off = off_stack.astype(np.float32).mean(0)
            roi = detect_led_roi(on, off, close_px=a.close_px, k_sigma=a.k_sigma,
                                 largest_only=a.largest_only)
        master_dark = off
        sat = float(np.percentile(on[roi.mask], 99.9))
        print(f"[roi] DC-on 99.9th pct in LED = {sat:.0f} counts")
        # Clipped pixels flatten the roll-off (they stay at full scale while the
        # true level drops). Drop any pixel near full scale with the LED held on,
        # which is the brightest state it will ever see.
        clipped = roi.mask & (on >= a.sat_level)
        if clipped.any():
            frac = clipped.sum() / roi.mask.sum()
            roi.mask = roi.mask & ~clipped
            roi.n_led = int(roi.mask.sum())
            print(f"[roi] excluded {clipped.sum()} pixels ({frac:.1%}) at >= {a.sat_level} "
                  f"counts with the LED on; {roi.n_led} left. Lower --exposure-us if this is large.")
            if roi.n_led < 50:
                sys.exit("almost every LED pixel saturates: lower the exposure or gain")
        x, y, w, h = roi.bbox
        pad = a.crop_margin
        ys, xs = slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad)
        dc_ref = float(frame_signal(on[None], roi, master_dark)[0])   # LED held on
        crops = {}

        # --- frequency steps -------------------------------------------
        def steps():
            if freqs:
                yield from freqs
                return
            print("\nNo --freqs given: type one frequency per step (e.g. 2k), "
                  "blank line to finish.")
            while True:
                s = input(">> next frequency: ").strip()
                if not s:
                    return
                yield parse_freq(s)

        for f_req in steps():
            f_act, d_act = driver.set_pwm(f_req, a.duty)
            time.sleep(a.settle_s)
            n_per = a.exposure_us * 1e-6 * f_act
            if n_per < a.min_periods:
                print(f"  WARNING {fmt_hz(f_act)}: exposure spans only {n_per:.1f} "
                      f"periods (want >= {a.min_periods}); expect phase noise ~1/{n_per:.0f}")
            fr = cam.grab_n(a.frames, a.discard)
            sig = frame_signal(fr, roi, master_dark)
            dark_sig = np.nan
            if a.dark_each_step and not a.manual:
                driver.off()
                dark_sig = float(frame_signal(cam.grab_n(a.dark_frames // 2 or 1, a.discard),
                                              roi, master_dark).mean())
            corr = sig - (0 if np.isnan(dark_sig) else dark_sig)
            m, s = float(corr.mean()), float(corr.std(ddof=1)) if len(corr) > 1 else 0.0
            rows.append(dict(freq_hz=f_act, freq_req_hz=f_req, duty_pct=d_act,
                             n_frames=len(corr), signal=m, std=s,
                             sem=s / math.sqrt(len(corr)),
                             signal_per_duty=m / (d_act / 100.0),
                             step_dark=dark_sig, n_periods=n_per,
                             roi_max=int(fr[:, roi.mask].max()),
                             n_clipped=int((fr[:, roi.mask] >= a.sat_level).any(0).sum())))
            crops[f"f{len(rows):03d}"] = bin_image(fr.astype(np.float32).mean(0)[ys, xs]
                                                  - master_dark[ys, xs], a.bin)
            print(f"  {fmt_hz(f_act):>11}  duty {d_act:6.3f}%  grey {m:9.2f} +- "
                  f"{rows[-1]['sem']:.2f}  ({n_per:.0f} periods/exposure)")
        driver.off()
    finally:
        try:
            driver.close()
        finally:
            cam.close()

    if not rows:
        sys.exit("no frequency steps captured")
    with open(base + "_sweep.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    meta = dict(stamp=stamp, driver=driver.name, duty_cmd_pct=a.duty,
                exposure_us=a.exposure_us, gain=a.gain, frames=a.frames,
                dark_frames=a.dark_frames, dc_on_signal=dc_ref,
                roi=roi.to_json(), sensor_roi=sensor_roi, bin=a.bin,
                sim=dict(t_r_us=a.sim_tr_us, t_f_us=a.sim_tf_us) if a.backend == "sim" else None)
    with open(base + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    np.savez_compressed(base + "_roi.npz", mask=roi.mask, bg=roi.bg,
                        master_dark=master_dark, dc_on=on, **crops)
    print(f"\nsaved {base}_sweep.csv / _meta.json / _roi.npz / _roi.png")
    analyze(base, k=a.k, fit_max_drop=a.fit_max_drop, knee_drop=a.knee_drop)
    return base


# ============================================================================
# 3. fit + plot
# ============================================================================
def load_sweep(base: str):
    for sfx in ("_sweep.csv", "_meta.json", "_roi.npz", "_roi.png", "_result.json"):
        if base.endswith(sfx):
            base = base[: -len(sfx)]
    with open(base + "_sweep.csv") as fh:
        rows = list(csv.DictReader(fh))
    meta = {}
    if os.path.exists(base + "_meta.json"):
        with open(base + "_meta.json") as fh:
            meta = json.load(fh)
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return base, cols, meta


def wlinfit(x, y, s):
    """Weighted y = a + b x. Returns a, b, cov(a,b)."""
    w = 1.0 / np.maximum(s, 1e-12) ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, b = cov @ (A.T @ (w * y))
    r = y - (a + b * x)
    dof = max(len(x) - 2, 1)
    chi2r = float(np.sum(w * r ** 2) / dof)
    return float(a), float(b), cov * max(chi2r, 1.0), chi2r


def analyze(base: str, k: Optional[float] = None, fit_max_drop: float = 0.3,
            knee_drop: float = 0.1) -> dict:
    base, c, meta = load_sweep(base)
    order = np.argsort(c["freq_hz"])
    f = c["freq_hz"][order]
    duty = c["duty_pct"][order] / 100.0
    # normalize by the duty actually produced: at high f the Arduino's duty
    # quantizes, which would otherwise masquerade as switching loss
    y = c["signal_per_duty"][order]
    s = np.maximum(c["sem"][order] / duty, 1e-3 * np.abs(y).max())
    D = float(np.median(duty))
    if k is None:
        k = 1.0 / (2.0 * D)

    # iterative fit of the linear (low-f) region
    use = np.ones_like(f, bool)
    L0 = float(np.max(y[: max(3, len(y) // 4)]))
    for _ in range(10):
        use_new = (y / L0) >= 1.0 - fit_max_drop
        if use_new.sum() < 2:
            use_new = np.zeros_like(use)
            use_new[: min(len(f), 3)] = True
        a0, b0, cov, chi2r = wlinfit(f[use_new], y[use_new], s[use_new])
        same = np.array_equal(use_new, use)
        use, L0 = use_new, a0
        if same:
            break
    slope = -b0 / a0                                   # = k (t_r + t_f)  [1/Hz]
    # error propagation for slope = -b/a
    J = np.array([b0 / a0 ** 2, -1.0 / a0])
    slope_err = float(math.sqrt(max(J @ cov @ J, 0)))
    tsw = slope / k                                    # t_r + t_f
    tsw_err = slope_err / k
    tau_lost = slope * D                               # lost on-time per period
    yn, sn = y / a0, s / a0

    # knee: where the normalized level has dropped by knee_drop
    f_knee_fit = knee_drop / slope if slope > 0 else float("inf")
    f_knee_data = float("nan")
    below = np.nonzero(yn < 1.0 - knee_drop)[0]
    if len(below) and below[0] > 0:
        i = below[0]
        x0, x1 = math.log(f[i - 1]), math.log(f[i])
        y0, y1 = yn[i - 1], yn[i]
        t = (1.0 - knee_drop - y0) / (y1 - y0) if y1 != y0 else 0.0
        f_knee_data = float(math.exp(x0 + t * (x1 - x0)))
    f_corner = 1.0 / tsw if tsw > 0 else float("inf")  # period == switching time

    res = dict(duty=D, k=k, L0=a0, slope_per_hz=slope, slope_err=slope_err,
               t_sw_s=tsw, t_sw_err_s=tsw_err, lost_on_time_s=tau_lost,
               fit_points=int(use.sum()), chi2_reduced=chi2r,
               f_knee_fit_hz=f_knee_fit, f_knee_data_hz=f_knee_data,
               knee_drop=knee_drop, f_corner_hz=f_corner)
    print("\n=== LED switching-speed fit ===")
    print(f"  model  L(f) = L0 [1 - k (t_r+t_f) f],  k = {k:.4g}"
          f"{'  (= 1/(2D): both edges inside the on window)' if abs(k - 1 / (2 * D)) < 1e-9 else ''}")
    print(f"  L0               = {a0:.2f} grey/duty  ({use.sum()} of {len(f)} points in fit, "
          f"chi2r {chi2r:.2f})")
    print(f"  slope k(t_r+t_f) = {slope:.4g} +- {slope_err:.2g} per Hz")
    print(f"  t_r + t_f        = {tsw * 1e6:.4g} +- {tsw_err * 1e6:.2g} us")
    print(f"  lost on-time     = {tau_lost * 1e9:.4g} ns per period")
    print(f"  knee (-{knee_drop:.0%})     = {fmt_hz(f_knee_fit)} (fit),  "
          f"{fmt_hz(f_knee_data) if f_knee_data == f_knee_data else 'not reached'} (data)")
    print(f"  corner 1/(t_r+t_f) = {fmt_hz(f_corner)}")
    if meta.get("dc_on_signal"):
        r = a0 / meta["dc_on_signal"]
        res["L0_over_dc"] = r
        print(f"  L0 / DC-on level = {r:.3f}  (~1 expected; far off means the LED or "
              "driver is not linear in duty)")
    if meta.get("sim"):
        true = meta["sim"]["t_r_us"] + meta["sim"]["t_f_us"]
        print(f"  [sim] true t_r+t_f = {true:.4g} us  -> error "
              f"{(tsw * 1e6 - true) / true:+.1%}")
        res["sim_true_t_sw_us"] = true
    with open(base + "_result.json", "w") as fh:
        json.dump(res, fh, indent=2)
    plot(base, f, yn, sn, use, res, meta)
    return res


def plot(base, f, yn, sn, use, res, meta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing: no plot)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ff = np.logspace(math.log10(f.min()), math.log10(f.max()), 300)
    model = 1.0 - res["slope_per_hz"] * ff
    for i, a in enumerate(ax):
        a.errorbar(f[use], yn[use], yerr=sn[use], fmt="o", color="#1f77b4",
                   ms=5, capsize=2, label="data (in fit)")
        if (~use).any():
            a.errorbar(f[~use], yn[~use], yerr=sn[~use], fmt="o", mfc="none",
                       color="#1f77b4", ms=5, capsize=2, label="data (excluded)")
        a.plot(ff, model, "-", color="#d62728", lw=1.3,
               label=f"fit: t_r+t_f = {res['t_sw_s'] * 1e6:.3g} ± "
                     f"{res['t_sw_err_s'] * 1e6:.2g} µs")
        a.axhline(1 - res["knee_drop"], color="gray", ls=":", lw=1)
        if math.isfinite(res["f_knee_fit_hz"]) and res["f_knee_fit_hz"] < 10 * f.max():
            a.axvline(res["f_knee_fit_hz"], color="#d62728", ls="--", lw=1,
                      label=f"knee (−{res['knee_drop']:.0%}) ≈ {fmt_hz(res['f_knee_fit_hz'])}")
        a.set_xlabel("PWM frequency (Hz)")
        a.set_ylabel("normalized grey level  L/L0  (per duty)")
        a.grid(alpha=0.3, which="both")
    ax[0].set_xscale("log")
    ax[0].set_title(f"Grey level vs PWM frequency at {res['duty']:.0%} duty")
    lo = yn.min()
    ax[0].set_ylim(min(lo - 0.05, 0.85), 1.05)
    fmax = f[use].max()
    ax[1].set_xlim(0, fmax * 1.1)
    sel = f <= fmax * 1.1
    ax[1].set_ylim(min(yn[sel].min(), 1 - res["slope_per_hz"] * fmax) - 0.02,
                   max(yn[sel].max(), 1) + 0.02)
    ax[1].set_title("linear region: slope = k·(t_r+t_f)")
    ax[1].legend(fontsize=8, loc="lower left")
    sub = f"exposure {meta.get('exposure_us', '?')} µs, driver {meta.get('driver', '?')}"
    if res.get("sim_true_t_sw_us"):
        sub += f", simulated true t_r+t_f = {res['sim_true_t_sw_us']:.3g} µs"
    fig.suptitle(sub, fontsize=9)
    fig.tight_layout()
    fig.savefig(base + "_rolloff.png", dpi=120)
    plt.close(fig)
    print(f"  plot -> {base}_rolloff.png")


# ============================================================================
# CLI
# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def fit_args(p):
        p.add_argument("--k", type=float, default=None,
                       help="model k; default 1/(2*duty), i.e. both edges cut into "
                            "the on window. The fitted slope itself is k*(t_r+t_f).")
        p.add_argument("--fit-max-drop", type=float, default=0.3,
                       help="only fit points within this fractional drop of L0 "
                            "(the model is linear only while t_r+t_f << period)")
        p.add_argument("--knee-drop", type=float, default=0.1,
                       help="report the frequency where L/L0 has dropped this much")

    sw = sub.add_parser("sweep", help="capture a frequency sweep and fit it")
    sw.add_argument("--backend", choices=["v4l2", "sim"], default="v4l2")
    sw.add_argument("--device", default="/dev/video0")
    sw.add_argument("--pixel-format", default="RG10")
    sw.add_argument("--port", default="/dev/ttyACM0", help="Arduino serial port")
    sw.add_argument("--baud", type=int, default=115200)
    sw.add_argument("--manual", action="store_true",
                    help="set each frequency by hand; the tool prompts and you can "
                         "type the frequency you actually set")
    sw.add_argument("--freqs", help="comma list, k/M suffixes ok, e.g. 100,1k,10k,100k. "
                                    "Omit to type them one at a time.")
    sw.add_argument("--duty", type=float, default=50.0, help="duty cycle, percent")
    sw.add_argument("--exposure-us", type=int, default=20000,
                    help="must span many PWM periods at the lowest frequency")
    sw.add_argument("--gain", type=float, default=0)
    sw.add_argument("--frames", type=int, default=30, help="frames per frequency")
    sw.add_argument("--dark-frames", type=int, default=20)
    sw.add_argument("--roi-frames", type=int, default=10)
    sw.add_argument("--discard", type=int, default=3,
                    help="frames dropped after every change (stale buffers)")
    sw.add_argument("--settle-s", type=float, default=0.05)
    sw.add_argument("--min-periods", type=float, default=50)
    sw.add_argument("--sat-level", type=float, default=60000,
                    help="pixels at or above this with the LED held on are left out "
                         "(frames are 16-bit scaled on this rig: full scale 65535)")
    sw.add_argument("--dark-each-step", action="store_true",
                    help="also take LED-off frames after every step (Arduino only)")
    sw.add_argument("--hw-crop", action="store_true",
                    help="crop the sensor ROI to the LED after detection")
    sw.add_argument("--crop-margin", type=int, default=48)
    sw.add_argument("--bin", type=int, default=1,
                    help="software bin factor for the saved per-step ROI crops")
    sw.add_argument("--close-px", type=int, default=7,
                    help="closing radius that merges an LED array into one blob")
    sw.add_argument("--k-sigma", type=float, default=6.0)
    sw.add_argument("--largest-only", action="store_true")
    sw.add_argument("--sim-tr-us", type=float, default=2.0)
    sw.add_argument("--sim-tf-us", type=float, default=1.0)
    sw.add_argument("--seed", type=int, default=0)
    fit_args(sw)

    an = sub.add_parser("analyze", help="re-fit a saved sweep")
    an.add_argument("base", help="results/led_switching/<stamp> (any of its files)")
    fit_args(an)

    st = sub.add_parser("selftest", help="simulated rig with a known t_r+t_f")
    st.add_argument("--tsw-us", type=float, default=3.0, help="true t_r + t_f")
    st.add_argument("--duty", type=float, default=50.0)
    st.add_argument("--exposure-us", type=int, default=20000)
    st.add_argument("--seed", type=int, default=0)

    a = ap.parse_args(argv)
    if a.cmd == "analyze":
        analyze(a.base, k=a.k, fit_max_drop=a.fit_max_drop, knee_drop=a.knee_drop)
    elif a.cmd == "sweep":
        cmd_sweep(a)
    else:
        t = a.tsw_us
        freqs = "100,300,1k,3k,10k,20k,30k,50k,70k,100k,150k"
        args = ["sweep", "--backend", "sim", "--freqs", freqs, "--duty", str(a.duty),
                "--exposure-us", str(a.exposure_us), "--sim-tr-us", str(t * 2 / 3),
                "--sim-tf-us", str(t / 3), "--seed", str(a.seed), "--frames", "20"]
        main(args)


if __name__ == "__main__":
    main()
