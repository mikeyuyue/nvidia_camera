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
  python3 tools/led_switching.py ledmap --pwm-col 9 --exposure-us 500  # each LED alone
  python3 tools/led_switching.py sweep ... --led-map results/led_switching/<stamp>_D9_ledmap.npz

Each LED images with a dark cross, i.e. as 2-4 bright lobes. 'ledmap' lights one
address at a time, so all lobes of an LED are grouped as one LED (and dead or
miswired addresses show up); 'sweep --led-map' then measures per LED with it.
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

# IMX568 (VC MIPI driver): the "gain" control is in milli-dB (0..48001), so the
# linear gain is 10**(gain/20000). Measured 2026-11-20: 3/6/9/12 dB gave
# 1.419/1.993/2.807/3.968x (expected 1.413/1.995/2.818/3.981x), and the
# dark-subtracted signal is linear in exposure (within ~1.5% from 100 us to
# 1 ms, until pixels clip). Black-level pedestal is ~3866 counts (16-bit) at
# 0 dB and shifts by tens of counts with gain, so the dark must be taken at the
# same exposure AND gain. results/linearity_20261120_232833.


def gain_linear(gain_mdb: float) -> float:
    return 10.0 ** (float(gain_mdb) / 20000.0)


def norm_factor(exposure_us: float, gain_mdb: float) -> float:
    """Multiply a dark-subtracted grey level by this to get counts per us of
    exposure at 0 dB gain, comparable across exposure/gain settings."""
    return 1.0 / (float(exposure_us) * gain_linear(gain_mdb))


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
    gap = bg_gap
    while bg.sum() < 300 and gap < 300:          # glow ate the ring: move it out
        gap += 25
        bg = _dilate(mask, gap + bg_width) & ~_dilate(mask, gap) & ~_dilate(lit, 2)
    if bg.sum() < 300:                           # last resort: ignore glow map
        bg = _dilate(mask, bg_gap + 3 * bg_width) & ~_dilate(mask, bg_gap)
    contrast = float(diff[mask].mean() / max(sigma, 1e-6))
    return LedRoi(mask=mask, bg=bg, bbox=bbox, n_led=int(mask.sum()),
                  n_bg=int(bg.sum()), contrast=contrast)


def _merge_lobes(lab: np.ndarray, keep: list) -> list:
    if len(keep) < 2:
        return [[k] for k in keep]
    pts = {k: np.argwhere(lab == k)[:, ::-1].astype(float) for k in keep}   # (x, y)
    cents = np.array([pts[k].mean(0) for k in keep])
    axis = np.linalg.svd(cents - cents.mean(0), full_matrices=False)[2][0]
    perp = np.array([-axis[1], axis[0]])
    ext = {}
    for k in keep:
        t = pts[k] @ axis
        u = pts[k] @ perp
        ext[k] = (t.min(), t.max(), u.min(), u.max())
    parent = {k: k for k in keep}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            ta0, ta1, ua0, ua1 = ext[a]
            tb0, tb1, ub0, ub1 = ext[b]
            ov = min(ta1, tb1) - max(ta0, tb0)
            span = min(ta1 - ta0, tb1 - tb0) + 1
            gap_u = max(ub0 - ua1, ua0 - ub1)
            width = max(ua1 - ua0, ub1 - ub0) + 1
            if ov > 0.5 * span and gap_u < 0.5 * width:
                parent[find(b)] = find(a)
    groups = {}
    for k in keep:
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def segment_emitters(on_mean: np.ndarray, off_mean: np.ndarray, roi: "LedRoi", *,
                      blur_sigma: float = 1.5, rel_thresh: float = 0.3,
                      min_area: int = 15, close_px: int = 1, line_thresh: float = 0.4):
    """Split the LED mask into individual emitters.

    The on-off image is thresholded again inside the (dilated) LED mask, at
    rel_thresh of each region's local peak, with only a tiny closing so that
    neighbouring emitters stay apart. Components are numbered along the main
    axis of the column (PCA of the centroids), so ID 0 is one end of the line.
    Returns (labels int32 full-frame, 0 = none; list of dicts per emitter).
    """
    diff = on_mean.astype(np.float32) - off_mean.astype(np.float32)
    proj = _segment_line(diff, roi, rel_thresh=line_thresh)
    if proj is not None:
        info = proj
    else:
        info = _segment_2d(diff, roi, blur_sigma, rel_thresh, min_area, close_px)
    return _number_emitters(diff, info)


def _segment_line(diff, roi, rel_thresh=0.25, min_len=4):
    """LEDs along one line (a lit column/row): project the on-off image onto the
    line's axis and cut it at the dark gaps between emitters. Much more robust
    than 2D blobs when emitters sit a few px apart. None if the lit area is
    not line-shaped."""
    region = _dilate(roi.mask, 3)
    ys, xs = np.nonzero(region)
    if len(xs) < 50:
        return None
    pts = np.stack([xs, ys], 1).astype(float)
    c0 = pts.mean(0)
    u_, sv, vt = np.linalg.svd(pts - c0, full_matrices=False)
    if sv[0] < 5 * sv[1]:                       # not elongated: use 2D blobs
        return None
    axis = vt[0]
    if abs(axis[1]) >= abs(axis[0]):
        axis = axis * np.sign(axis[1] or 1)
    else:
        axis = axis * np.sign(axis[0] or 1)
    # The gaps between emitters follow the sensor grid, not the (slightly
    # tilted) line, so project onto the nearest image axis when the line is
    # within ~10 deg of it; otherwise onto the line itself.
    if abs(axis[1]) > 0.985:
        t = pts[:, 1]
    elif abs(axis[0]) > 0.985:
        t = pts[:, 0]
    else:
        t = (pts - c0) @ axis
    ti = np.round(t - t.min()).astype(int)
    w = np.maximum(diff[ys, xs], 0)
    prof = np.bincount(ti, weights=w)
    # threshold against a running local maximum so dim stretches still split
    win = 41
    pad = np.pad(prof, win // 2, mode="edge")
    locmax = np.array([pad[i:i + win].max() for i in range(len(prof))])
    on = prof > rel_thresh * np.maximum(locmax, 0.1 * prof.max())
    on &= prof > 0.05 * prof.max()
    segs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= min_len:
                segs.append((i, j))
            i = j
        else:
            i += 1
    if len(segs) < 2:
        return None
    info = []
    for a, b in segs:
        sel = (ti >= a) & (ti < b)
        pix = np.zeros(diff.shape, bool)
        pix[ys[sel], xs[sel]] = True
        pix &= roi.mask                          # drops clipped px
        if pix.sum() < 3:
            continue
        yy, xx = np.nonzero(pix)
        info.append(dict(cy=float(yy.mean()), cx=float(xx.mean()), area=int(pix.sum()),
                         dc=float(diff[pix].mean()), pix=pix))
    return info


def _segment_2d(diff, roi, blur_sigma, rel_thresh, min_area, close_px):
    sm = _blur(diff, blur_sigma) if blur_sigma > 0 else diff
    region = _dilate(roi.mask, 3)
    vals = sm[region]
    peak = float(np.percentile(vals, 99.5))
    base = float(np.median(sm[roi.bg])) if roi.bg.any() else 0.0
    m = region & (sm > base + rel_thresh * (peak - base))
    m = _close_open(m, close_px, 0)
    lab, areas = _components(m)
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_area]
    # One emitter can image as two lobes (dark notch in the middle). Merge
    # components that sit side by side across the line: their extents along
    # the line's main axis overlap and they are close across it.
    groups = _merge_lobes(lab, keep)
    out = np.zeros(diff.shape, np.int32)
    info = []
    for ks in groups:
        pix = np.isin(lab, ks) & roi.mask    # roi.mask already drops clipped px
        if pix.sum() < max(3, min_area // 3):
            continue
        ys, xs = np.nonzero(pix)
        info.append(dict(cy=float(ys.mean()), cx=float(xs.mean()), area=int(pix.sum()),
                         dc=float(diff[pix].mean()), pix=pix))
    return info


def _number_emitters(diff, info):
    out = np.zeros(diff.shape, np.int32)
    if not info:
        return out, []
    c = np.array([[d["cx"], d["cy"]] for d in info])
    if len(c) > 1:
        cc = c - c.mean(0)
        axis = np.linalg.svd(cc, full_matrices=False)[2][0]
        if abs(axis[1]) >= abs(axis[0]):
            axis = axis * np.sign(axis[1] or 1)   # top -> bottom
        else:
            axis = axis * np.sign(axis[0] or 1)   # left -> right
        tpos = cc @ axis
        order = np.argsort(tpos)
        st = np.diff(tpos[order])
        # pitch = typical neighbour spacing; IDs are grid slots along the line, so
        # a dark/missing LED keeps its slot (IDs then follow the physical rows)
        pitch = float(np.median(st)) if len(st) else 1.0
        inc = np.maximum(1, np.round(st / max(pitch, 1e-6))).astype(int)
        slots = np.concatenate([[0], np.cumsum(inc)])
    else:
        order, slots = [0], [0]
    emitters = []
    for lab_no, (i, slot) in enumerate(zip(order, slots)):
        d = info[i]
        out[d.pop("pix")] = lab_no + 1
        d["id"] = int(slot)
        d["label"] = lab_no + 1
        emitters.append(d)
    med = np.median([d["dc"] for d in emitters])
    for d in emitters:
        d["dc_rel"] = d["dc"] / med if med else float("nan")
    return out, emitters


# ----------------------------------------------------------------------------
# Address-based LED map. Each LED images with a dark cross through its middle
# (microscope photo, 2026-11-24), so one LED looks like 2 or 4 bright lobes and
# any segmentation of the all-on image cuts it at the notch (24 LEDs -> 48).
# Lighting one address at a time removes the ambiguity: everything that lights
# up with address r belongs to LED r, however many lobes it has.
# ----------------------------------------------------------------------------
def _spots_one_address(diff, *, sigma_noise, blur_sigma=2.0, rel=0.12, close_px=8,
                       merge_r=25.0, min_spot_rel=0.08, lobe_rel=0.6):
    """Bright regions of one address's on-off image.
    -> list of dicts (pix, cx, cy, flux, peak, lobes), brightest first.
    Lobes of one LED are joined by a closing of close_px and by merging any
    region whose centroid is within merge_r px of a brighter one."""
    sm = _blur(diff, blur_sigma) if blur_sigma > 0 else diff
    peak = float(sm.max())
    thr = max(rel * peak, 6 * sigma_noise)
    m = _close_open(sm > thr, close_px, 1)
    lab, areas = _components(m)
    regs = []
    for k in range(1, len(areas) + 1):
        pix = lab == k
        w = np.maximum(diff[pix], 0)
        flux = float(w.sum())
        if flux <= 0:
            continue
        ys, xs = np.nonzero(pix)
        regs.append(dict(pix=pix, flux=flux, cx=float((xs * w).sum() / flux),
                         cy=float((ys * w).sum() / flux), peak=float(sm[pix].max())))
    regs.sort(key=lambda r: -r["flux"])
    spots = []
    for r in regs:                                   # merge leftover lobes
        host = next((s for s in spots if math.hypot(s["cx"] - r["cx"], s["cy"] - r["cy"]) < merge_r), None)
        if host is None:
            spots.append(r)
            continue
        f = host["flux"] + r["flux"]
        host["cx"] = (host["cx"] * host["flux"] + r["cx"] * r["flux"]) / f
        host["cy"] = (host["cy"] * host["flux"] + r["cy"] * r["flux"]) / f
        host["flux"], host["pix"] = f, host["pix"] | r["pix"]
        host["peak"] = max(host["peak"], r["peak"])
    if not spots:
        return []
    spots = [s for s in spots if s["flux"] >= min_spot_rel * spots[0]["flux"]]
    for s in spots:                                  # how many lobes inside
        _, a = _components(s["pix"] & (diff > lobe_rel * float(diff[s["pix"]].max())))
        s["lobes"] = int((a >= 4).sum())
    return spots


def build_led_map(diffs: dict, *, min_snr=30.0, min_rel=0.05, pitch=None,
                  **spot_kw) -> tuple:
    """diffs: {address: on-off image with only that address lit}.
    -> (labels int32 full-frame 0 = none, emitters list, report dict).

    One emitter per physical spot. A spot's id is its grid slot along the
    line (fitted from the primary spots: slot = address for a good LED), and
    'address' says which drive line lit it. So a LED wired to the wrong line
    keeps its own slot and shows address != id."""
    addrs = sorted(diffs)
    any_d = diffs[addrs[0]]
    sigma = 1.4826 * float(np.median(np.abs(any_d - np.median(any_d)))) + 1e-6
    per = {}
    peaks = {}
    for a in addrs:
        sp = _spots_one_address(diffs[a].astype(np.float32), sigma_noise=sigma, **spot_kw)
        per[a] = sp
        peaks[a] = sp[0]["peak"] if sp else 0.0
    alive_pk = [p for p in peaks.values() if p > min_snr * sigma]
    med_pk = float(np.median(alive_pk)) if alive_pk else 0.0
    dead = [a for a in addrs if peaks[a] < max(min_snr * sigma, min_rel * med_pk)]
    live = [a for a in addrs if a not in dead]
    if not live:
        raise RuntimeError("no address lit anything")
    # grid along the line from primary spots: position t = t0 + pitch * address
    c = np.array([[per[a][0]["cx"], per[a][0]["cy"]] for a in live])
    axis = np.array([0.0, 1.0])
    if len(live) > 1:
        axis = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][0]
        if abs(axis[1]) >= abs(axis[0]):
            axis = axis * np.sign(axis[1] or 1)
        else:
            axis = axis * np.sign(axis[0] or 1)
    t = c @ axis
    if len(live) >= 3:
        A = np.stack([np.ones(len(live)), np.array(live, float)], 1)
        coef = np.linalg.lstsq(A, t, rcond=None)[0]
        for _ in range(2):                             # drop gross outliers (miswired)
            res = t - A @ coef
            ok = np.abs(res) < max(3 * 1.4826 * np.median(np.abs(res)), 0.25 * abs(coef[1]))
            if ok.sum() >= 3:
                coef = np.linalg.lstsq(A[ok], t[ok], rcond=None)[0]
        t0, p_fit = float(coef[0]), float(coef[1])
    else:
        t0, p_fit = float(t[0]), float(pitch or 40.0)
    if pitch:
        p_fit = math.copysign(pitch, p_fit)
    spots = []
    for a in live:
        for k, s in enumerate(per[a]):
            slot = int(round((np.array([s["cx"], s["cy"]]) @ axis - t0) / p_fit))
            spots.append(dict(s, address=a, primary=(k == 0), slot=slot))
    # pixels claimed by several spots go to the one brightest there (relative)
    shape = any_d.shape
    best = np.zeros(shape, np.float32)
    owner = np.full(shape, -1, np.int32)
    for i, s in enumerate(spots):
        v = np.where(s["pix"], np.maximum(diffs[s["address"]], 0) / max(s["peak"], 1e-6), -1)
        take = v > best
        owner[take], best[take] = i, v[take]
    used, emitters, labels = set(), [], np.zeros(shape, np.int32)
    # primaries first so a good LED keeps its slot
    for i in sorted(range(len(spots)), key=lambda i: (not spots[i]["primary"], spots[i]["slot"])):
        s = spots[i]
        pix = owner == i
        if pix.sum() < 3:
            continue
        sid = s["slot"] if s["slot"] not in used else 100 + s["address"]
        used.add(sid)
        emitters.append(dict(id=int(sid), address=int(s["address"]), primary=bool(s["primary"]),
                             cx=s["cx"], cy=s["cy"], area=int(pix.sum()), lobes=s["lobes"],
                             flux=s["flux"], dc=float(diffs[s["address"]][pix].mean()), _pix=pix))
    emitters.sort(key=lambda d: d["id"])
    for n, d in enumerate(emitters, 1):
        labels[d.pop("_pix")] = n
        d["label"] = n
    med = np.median([d["dc"] for d in emitters if d["primary"]])
    for d in emitters:
        d["dc_rel"] = d["dc"] / med if med else float("nan")
    miswired = [dict(id=d["id"], address=d["address"]) for d in emitters if d["id"] != d["address"]]
    lobes = [d["lobes"] for d in emitters]
    report = dict(addresses=addrs, dead=dead, n_leds=len(emitters), pitch_px=abs(p_fit),
                  miswired=miswired, noise_sigma=sigma,
                  lobes_hist={str(k): lobes.count(k) for k in sorted(set(lobes))},
                  multi_spot_addresses={str(a): len(per[a]) for a in live if len(per[a]) > 1})
    return labels, emitters, report


def save_led_map(path_base, labels, emitters, report, img=None, extra=None):
    np.savez_compressed(path_base + "_ledmap.npz", labels=labels,
                        emitters=json.dumps(emitters), report=json.dumps(report),
                        **({"img": img} if img is not None else {}), **(extra or {}))
    with open(path_base + "_ledmap.json", "w") as fh:
        json.dump(dict(report=report, emitters=emitters), fh, indent=2)
    if img is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ys, xs = np.nonzero(labels)
    pad = 50
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    fig, ax = plt.subplots(1, 2, figsize=(9, 10))
    for a_ in ax:
        a_.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=np.percentile(img[labels > 0], 99),
                  extent=(x0, x1, y1, y0))
    ax[1].contour(np.arange(x0, x1), np.arange(y0, y1), labels[y0:y1, x0:x1],
                  levels=np.arange(0.5, labels.max() + 1), colors="lime", linewidths=0.5)
    for d in emitters:
        col = "yellow" if d["id"] == d["address"] else "red"
        txt = str(d["id"]) if d["id"] == d["address"] else f"{d['id']}<-{d['address']}"
        ax[1].annotate(txt, (d["cx"] + 18, d["cy"]), color=col, fontsize=7, va="center")
    ax[0].set_title("all addresses summed (on - off)", fontsize=9)
    ax[1].set_title(f"{report['n_leds']} LEDs; dead {report['dead']}; "
                    f"lobes/LED {report['lobes_hist']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path_base + "_ledmap.png", dpi=110)
    plt.close(fig)
    print(f"  plot -> {path_base}_ledmap.png")


def load_led_map(path):
    z = np.load(path)
    return z["labels"], json.loads(str(z["emitters"])), json.loads(str(z["report"]))


def print_led_map(report, emitters):
    print(f"[ledmap] {report['n_leds']} LEDs from {len(report['addresses'])} addresses, "
          f"pitch {report['pitch_px']:.1f} px")
    print(f"[ledmap] lobes per LED (bright regions inside one LED): {report['lobes_hist']}")
    if report["dead"]:
        print(f"[ledmap] dead addresses (light nothing): {report['dead']}")
    for m in report["miswired"]:
        print(f"[ledmap] LED in slot {m['id']} is lit by address {m['address']} (miswired)")
    for d in emitters:
        print(f"   id {d['id']:3d}  addr {d['address']:2d}  ({d['cx']:7.1f},{d['cy']:7.1f})  "
              f"area {d['area']:5d}  lobes {d['lobes']}  brightness {d['dc_rel']:.2f}")


def cmd_ledmap(a):
    os.makedirs(OUTDIR, exist_ok=True)
    kw = dict(close_px=a.close_px, merge_r=a.merge_r, rel=a.rel)
    if a.from_scan:
        import glob
        d = a.from_scan
        dark = np.load(os.path.join(d, "dark_full.npy")).astype(np.float32)
        diffs = {}
        for f in sorted(glob.glob(os.path.join(d, f"D{a.pwm_col}_row[0-9][0-9].npy"))):
            r = int(f[-6:-4])
            diffs[r] = np.load(f).astype(np.float32) - dark
        if not diffs:
            sys.exit(f"no D{a.pwm_col}_rowNN.npy in {d}")
        base = os.path.join(OUTDIR, os.path.basename(os.path.normpath(d)) + f"_D{a.pwm_col}")
        extra = {}
    else:
        from led_control import idx_list
        rows = idx_list(a.rows) if a.rows else list(range(24))
        driver = ArduinoDriver(a.port, a.baud)
        cam = Camera("v4l2", a.device, a.pixel_format, a.exposure_us, a.gain)
        try:
            driver.col(a.pwm_col)
            driver.rows(None)
            driver.off()
            dark = cam.grab_n(a.frames, a.discard).astype(np.float32).mean(0)
            diffs = {}
            for r in rows:
                driver.rows([r])
                driver.on()
                time.sleep(a.settle_s)
                diffs[r] = cam.grab_n(a.frames, a.discard).astype(np.float32).mean(0) - dark
                print(f"  address {r:2d}: peak {diffs[r].max():.0f}", flush=True)
            driver.rows(None)
            driver.off()
        finally:
            try:
                driver.close()
            finally:
                cam.close()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(OUTDIR, f"{stamp}_D{a.pwm_col}")
        extra = {"dark": dark}
    labels, emitters, report = build_led_map(diffs, **kw)
    report.update(pwm_col=a.pwm_col, exposure_us=a.exposure_us, gain=a.gain,
                  source=a.from_scan or "live")
    print_led_map(report, emitters)
    img = np.maximum(sum(diffs.values()), 0)
    save_led_map(base, labels, emitters, report, img=img, extra=extra)
    print(f"saved {base}_ledmap.npz / .json  (use: sweep --led-map {base}_ledmap.npz)")
    return base


def emitter_signals(frames: np.ndarray, labels: np.ndarray, n: int,
                    roi: "LedRoi", master_dark: np.ndarray) -> np.ndarray:
    """Per-frame dark-subtracted mean of every emitter -> (frames, n)."""
    flat = labels.ravel()
    sel = flat > 0
    idx = flat[sel] - 1
    cnt = np.bincount(idx, minlength=n).astype(np.float64)
    out = np.empty((len(frames), n))
    for i, fr in enumerate(frames):
        d = fr.astype(np.float32) - master_dark
        sums = np.bincount(idx, weights=d.ravel()[sel], minlength=n)
        out[i] = sums / np.maximum(cnt, 1) - d[roi.bg].mean()
    return out


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
    """Talks to tools/arduino/led_pwm_giga (or the Uno led_pwm) over USB serial."""
    name = "arduino"

    def __init__(self, port: str, baud: int = 115200, boot_wait: float = 2.0):
        try:
            import serial
        except ImportError:
            sys.exit("pyserial missing: pip install pyserial (or sudo apt install python3-serial)")
        self.ser = serial.Serial(port, baud, timeout=2.0)
        time.sleep(boot_wait)                # an Uno resets when the port opens (GIGA doesn't)
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
        arg = "ALL" if idx is None else (",".join(str(int(i)) for i in idx) or "NONE")
        return int(self._raw(f"R {arg}").split()[-1])

    def col(self, pin: int) -> None:
        self._raw(f"C {int(pin)}")

    def pulse(self, period_us: float, width_us: float) -> tuple[float, float]:
        _, hz, d = self._cmd(f"P {period_us:.4f} {width_us:.4f}")
        return hz, d

    def duty(self, duty: float) -> tuple[float, float]:
        """Change duty, keep the frequency (PWM/PULSE mode)."""
        _, hz, d = self._cmd(f"D {duty:.6g}")
        return hz, d

    def burst(self, n: int) -> tuple[float, float]:
        """n periods at the current frequency/duty, then off. Blocks until done."""
        old = self.ser.timeout
        self.ser.timeout = max(old or 0, 2.0 + 1.5 * n / max(self._last_hz(), 1e-3))
        try:
            _, hz, d = self._cmd(f"B {int(n)}")
        finally:
            self.ser.timeout = old
        return hz, d

    def _last_hz(self) -> float:
        return self.state().get("hz", 1.0) or 1.0

    def dc_cols(self, idx) -> int:
        """Extra columns held LOW (DC on), by matrix index; None = all, [] = none."""
        arg = "ALL" if idx is None else (",".join(str(int(i)) for i in idx) or "NONE")
        return int(self._raw(f"K {arg}").split()[-1])

    def led(self, row: int, col: int) -> tuple[str, float, float]:
        """Exactly one LED on (DC), matrix indices 0..23."""
        return self._cmd(f"L {int(row)} {int(col)}")

    def all_on(self) -> None:
        self._cmd("A")

    def state(self) -> dict:
        """Parsed 'S' status: mode, hz, duty, pwm_col_pin, pwm_col_idx, rows, dc_cols."""
        reply = self._raw("S")
        out = {}
        for tok in reply.split()[2:]:
            k, _, v = tok.partition("=")
            if k in ("rows", "dc_cols"):
                out[k] = [] if v == "none" else [int(x) for x in v.split(",")]
            elif k == "mode":
                out[k] = v
            else:
                out[k] = float(v) if "." in v else int(v)
        return out

    def close(self, keep_state: bool = False):
        try:
            if not keep_state:
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
                 peak=600.0, seed=0, n_leds=0):
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
        # optional column of separate emitters with their own edge times:
        # LED 3 dim, LED 8 twice as slow, positions 5 and 12 dead (not drawn)
        self.leds = []
        if n_leds:
            sc = 1 + 0.08 * self.rng.standard_normal(n_leds)
            sc[8 % n_leds] = 2.0
            for i in range(n_leds):
                if i in (5, 12):
                    continue
                cy, cx = 30 + 18 * i, int(0.62 * w)
                b = ((abs(yy - cy) <= 6) & (abs(xx - cx) <= 9)).astype(np.float32)
                b = _blur(b, 1.0) * peak * (0.5 if i == 3 else 1.0)
                self.leds.append((b, float(sc[i])))
            self.true_scale = [s_ for _, s_ in self.leds]

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

    def _avg_level(self, scale: float = 1.0, ph: Optional[float] = None) -> float:
        mode, f, d = self.state
        if mode == "on":
            return 1.0
        if mode == "off":
            return 0.0
        T = 1.0 / f
        ton = d / 100.0 * T
        extra = self.load_s_per_row * (self.n_rows - 1)
        tr = min(scale * self.t_r + extra, ton / 2)
        tf = min(scale * self.t_f + extra / 2, ton / 2)
        per_period = ton - tr / 2 - tf / 2           # area of the trapezoid
        n_full, rem = divmod(self.exp, T)
        # partial period at a random phase: integrate numerically
        ph = self.rng.uniform(0, T) if ph is None else ph * T
        t = (ph + np.linspace(0, rem, 200, endpoint=False)) % T
        wave = np.clip(np.minimum(t / max(tr, 1e-12), (ton - t) / max(tf, 1e-12)), 0, 1)
        wave[t > ton] = 0
        part = wave.mean() * rem if rem > 0 else 0.0
        return (n_full * per_period + part) / self.exp

    # camera side
    def grab(self, timeout=1.0):
        black = 64 + self.rng.normal(0, 2)                 # per-frame drift
        rows = self.rng.normal(0, 1.5, (self.shape[0], 1))  # row noise
        if self.leds:
            ph = self.rng.uniform()
            light = sum(self._avg_level(sc, ph) * b for b, sc in self.leds)
            light = light + self._avg_level(1.0, ph) * self.glint
        else:
            light = self._avg_level() * (self.blob + self.glint)
        img = black + rows + self.fpn + light
        img = img + self.rng.normal(0, 1.0, self.shape) + \
            self.rng.normal(0, 1, self.shape) * np.sqrt(np.maximum(light, 0) / 4)
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
                     exposure_us=a.exposure_us, seed=a.seed, n_leds=a.sim_leds)
        driver = rig
    else:
        rig = None
        driver = ManualDriver() if a.manual else ArduinoDriver(a.port, a.baud)
        if isinstance(driver, ArduinoDriver) and (a.rows or a.pwm_col or a.dc_cols):
            # which LEDs take part (led_pwm_giga sketch): rows HIGH, the PWM
            # column pin, and optional extra columns held on (DC)
            from led_control import idx_list
            if a.pwm_col:
                driver.col(a.pwm_col)
            if a.rows:
                driver.rows(idx_list(a.rows))
            if a.dc_cols:
                driver.dc_cols(idx_list(a.dc_cols))
            print(f"[led] {driver.state()}")
    cam = Camera(a.backend, a.device, a.pixel_format, a.exposure_us, a.gain, sim=rig)
    nf = norm_factor(a.exposure_us, a.gain)
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
        clipped = (on >= a.sat_level) if a.led_map else roi.mask & (on >= a.sat_level)
        if clipped.any():
            frac = clipped.sum() / roi.mask.sum()
            if frac > 0.01:
                print(f"[roi] WARNING {frac:.1%} of LED pixels clip: the brightest parts of "
                      f"each LED are lost and per-LED levels are compressed. Use a shorter "
                      f"--exposure-us (about {int(a.exposure_us * a.sat_level / max(sat, 1) * 0.8)} us) "
                      f"rather than relying on clip exclusion.")
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
        if a.per_led and a.led_map:
            # address-based map (each LED lit alone): lobes of one LED stay one
            if a.hw_crop:
                sys.exit("--led-map needs the full sensor frame: drop --hw-crop")
            labels, emitters, _rep = load_led_map(a.led_map)
            if labels.shape != on.shape:
                sys.exit(f"--led-map is {labels.shape}, frames are {on.shape}")
            labels = labels.copy()
            labels[clipped] = 0
            # the column background must not overlap any LED of the map
            roi.bg &= ~_dilate(labels > 0, a.close_px)
            print(f"[roi] LED map {a.led_map}: {len(emitters)} LEDs (address-based)")
        elif a.per_led:
            labels, emitters = segment_emitters(on, off, roi)
        else:
            labels, emitters = None, []
        n_em = len(emitters)
        if n_em:
            em_dc = emitter_signals(on[None], labels, n_em, roi, master_dark)[0]
            for d, v in zip(emitters, em_dc):
                d["dc_signal"] = float(v)
            print(f"[roi] segmented {n_em} individual LEDs (IDs 0..{emitters[-1]['id']} "
                  f"along the line; {emitters[-1]['id'] + 1 - n_em} slots dark)")
        em_rows = []

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
                             signal_norm=m * nf,
                             sem=s / math.sqrt(len(corr)),
                             signal_per_duty=m / (d_act / 100.0),
                             step_dark=dark_sig, n_periods=n_per,
                             roi_max=int(fr[:, roi.mask].max()),
                             n_clipped=int((fr[:, roi.mask] >= a.sat_level).any(0).sum())))
            if n_em:
                es = emitter_signals(fr, labels, n_em, roi, master_dark)
                # Frame-to-frame noise (PWM phase within the exposure, LED
                # current ripple) is common to every LED in the column. Measure
                # each LED relative to the whole column in the SAME frame, so its
                # error bar only holds what differs between LEDs; the column's
                # own uncertainty is shared by all of them.
                n_f = len(es)
                cm_ = float(sig.mean())
                if n_f > 1 and cm_ > 10 * sig.std(ddof=1) / math.sqrt(n_f) and np.all(sig > 0.2 * cm_):
                    ratio = es / sig[:, None]
                    em_m = ratio.mean(0) * cm_
                    em_s = (ratio.std(0, ddof=1) / math.sqrt(n_f) * abs(cm_)
                            if n_f > 1 else np.zeros(n_em))
                else:                                    # LED dark: plain means
                    em_m = es.mean(0)
                    em_s = es.std(0, ddof=1) / math.sqrt(n_f) if n_f > 1 else np.zeros(n_em)
                em_raw = es.std(0, ddof=1) / math.sqrt(n_f) if n_f > 1 else np.zeros(n_em)
                for i in range(n_em):
                    em_rows.append(dict(freq_hz=f_act, duty_pct=d_act, id=emitters[i]["id"],
                                        signal=float(em_m[i]), sem=float(em_s[i]),
                                        signal_norm=float(em_m[i]) * nf,
                                        sem_raw=float(em_raw[i])))
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
                exposure_us=a.exposure_us, gain=a.gain, gain_db=a.gain / 1000.0,
                gain_linear=gain_linear(a.gain), norm_factor=nf,
                signal_norm_units="dark-subtracted counts per us exposure at 0 dB",
                frames=a.frames, dark_frames=a.dark_frames, dc_on_signal=dc_ref,
                dc_on_signal_norm=dc_ref * nf,
                roi=roi.to_json(), sensor_roi=sensor_roi, bin=a.bin,
                sim=dict(t_r_us=a.sim_tr_us, t_f_us=a.sim_tf_us) if a.backend == "sim" else None)
    meta["emitters"] = emitters
    with open(base + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    extra = {"labels": labels} if n_em else {}
    np.savez_compressed(base + "_roi.npz", mask=roi.mask, bg=roi.bg,
                        master_dark=master_dark, dc_on=on, **extra, **crops)
    if em_rows:
        with open(base + "_emitters.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(em_rows[0].keys()))
            wr.writeheader()
            wr.writerows(em_rows)
    print(f"\nsaved {base}_sweep.csv / _meta.json / _roi.npz / _roi.png"
          + (" / _emitters.csv" if em_rows else ""))
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
    if os.path.exists(base + "_emitters.csv"):
        res["per_led"] = analyze_emitters(base, meta, k, fit_max_drop, (f, yn),
                                          col=dict(f=f, y=y, use=use, t_sw_s=tsw,
                                                   t_sw_err_s=tsw_err))
    return res


def fit_rolloff(f, y, s, fit_max_drop=0.3):
    """Iterative weighted line fit of the low-f part.
    -> L0, slope (fractional drop per Hz), slope_err, used-mask"""
    L0 = float(np.max(y[: max(2, len(y) // 4)]))
    use = np.ones_like(f, bool)
    for _ in range(10):
        new = (y / L0) >= 1.0 - fit_max_drop
        if new.sum() < 2:
            new = np.zeros_like(use)
            new[: min(len(f), 2)] = True
        a0, b0, cov, _ = wlinfit(f[new], y[new], s[new])
        done = np.array_equal(new, use)
        use, L0 = new, a0
        if done:
            break
    slope = -b0 / a0
    J = np.array([b0 / a0 ** 2, -1.0 / a0])
    return a0, slope, float(math.sqrt(max(J @ cov @ J, 0))), use


def analyze_emitters(base, meta, k, fit_max_drop, whole=None, col=None):
    """Per-LED fit + defect flags + plot. Writes _per_led.csv / _per_led.png.

    With the column fit given (col), each LED is fitted RELATIVE to the column:
    r_i(f) = y_i(f) / y_col(f) ~= r_i0 * [1 - k * dt_i * f] over the same
    frequencies the column fit used, so t_i = t_col + dt_i. Frame-to-frame
    noise common to the whole column (PWM phase inside the exposure, supply
    ripple) cancels in r_i; the column's own error is shared by every LED and
    is reported once, not folded into each LED's error bar."""
    with open(base + "_emitters.csv") as fh:
        rows = list(csv.DictReader(fh))
    ids = sorted({int(r["id"]) for r in rows})
    em = {d["id"]: d for d in meta.get("emitters", [])}
    out = []
    curves = {}
    for i in ids:
        rr = sorted((r for r in rows if int(r["id"]) == i), key=lambda r: float(r["freq_hz"]))
        f = np.array([float(r["freq_hz"]) for r in rr])
        d = np.array([float(r["duty_pct"]) for r in rr]) / 100
        y = np.array([float(r["signal"]) for r in rr]) / d
        s = np.maximum(np.array([float(r["sem"]) for r in rr]) / d, 1e-3 * max(abs(y).max(), 1))
        info = em.get(i, {})
        rec = dict(id=i, cx=info.get("cx", np.nan), cy=info.get("cy", np.nan),
                   area_px=info.get("area", 0), dc_rel=info.get("dc_rel", np.nan))
        try:
            if col is not None and len(col["f"]) == len(f) and \
                    np.allclose(col["f"], f, rtol=1e-6):
                use = col["use"]
                r = y / col["y"]
                rs = np.maximum(s / np.abs(col["y"]), 1e-6)
                a0, b0, cov, _ = wlinfit(f[use], r[use], rs[use])
                dslope = -b0 / a0
                J = np.array([b0 / a0 ** 2, -1.0 / a0])
                derr = float(math.sqrt(max(J @ cov @ J, 0)))
                t_i = col["t_sw_s"] + dslope / k
                rec.update(L0=a0, t_sw_us=t_i * 1e6, t_sw_err_us=derr / k * 1e6,
                           dt_vs_column_us=dslope / k * 1e6, fit_points=int(use.sum()))
                curves[i] = (f, (r / a0) * whole[1] if whole is not None else r / a0)
            else:
                L0, slope, err, use = fit_rolloff(f, y, s, fit_max_drop)
                rec.update(L0=L0, t_sw_us=slope / k * 1e6, t_sw_err_us=err / k * 1e6,
                           fit_points=int(use.sum()))
                curves[i] = (f, y / L0)
        except (np.linalg.LinAlgError, ZeroDivisionError):
            rec.update(L0=np.nan, t_sw_us=np.nan, t_sw_err_us=np.nan, fit_points=0)
        out.append(rec)
    t = np.array([r["t_sw_us"] for r in out])
    ok = np.isfinite(t)
    med = float(np.median(t[ok])) if ok.any() else np.nan
    mad = float(1.4826 * np.median(np.abs(t[ok] - med))) if ok.any() else np.nan
    for r in out:
        flags = []
        if r["dc_rel"] < 0.7:
            flags.append("dim")
        if r["dc_rel"] > 1.3:
            flags.append("bright")
        if np.isfinite(r["t_sw_us"]) and mad > 0 and \
                abs(r["t_sw_us"] - med) > max(4 * mad, 3 * r["t_sw_err_us"]):
            flags.append("slow" if r["t_sw_us"] > med else "fast")
        r["flags"] = "+".join(flags)
    # missing emitters: gaps along the line much larger than the usual pitch
    missing = []
    for r0, r1 in zip(out[:-1], out[1:]):
        if r1["id"] - r0["id"] > 1:
            missing.append(dict(after_id=r0["id"], count=r1["id"] - r0["id"] - 1,
                                ids=list(range(r0["id"] + 1, r1["id"]))))
    n_missing = sum(m["count"] for m in missing)
    with open(base + "_per_led.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    print(f"\n=== per-LED ({len(out)} LEDs found, ~{n_missing} missing/dark in gaps) ===")
    print(f"  t_r+t_f median {med:.3g} us, spread (robust sigma) {mad:.2g} us, "
          f"range {np.nanmin(t):.3g} .. {np.nanmax(t):.3g} us")
    if col is not None:
        print(f"  (each LED fitted relative to the column; error bars are LED-to-LED only, "
              f"the shared column error is +-{col['t_sw_err_s'] * 1e6:.2g} us)")
    for r in out:
        if r["flags"]:
            print(f"  LED {r['id']:3d}: {r['flags']:12s} t_r+t_f {r['t_sw_us']:.3g} us, "
                  f"brightness {r['dc_rel']:.2f} x median")
    for m in missing:
        print(f"  LED slot(s) {m['ids']}: dark or missing")
    plot_emitters(base, out, curves, med, missing, whole)
    return dict(n=len(out), n_missing=n_missing, t_sw_median_us=med, t_sw_sigma_us=mad,
                flagged=[r["id"] for r in out if r["flags"]], gaps=missing)


def plot_emitters(base, out, curves, med, missing, whole):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
    except ImportError:
        return
    z = np.load(base + "_roi.npz")
    img = z["dc_on"] - z["master_dark"]
    t = np.array([r["t_sw_us"] for r in out])
    ok = np.isfinite(t)
    norm = colors.Normalize(np.nanpercentile(t, 2), np.nanpercentile(t, 98)) if ok.any() \
        else colors.Normalize(0, 1)
    cmap = cm.viridis
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.3, 1.5])
    ax0 = fig.add_subplot(gs[0])
    cx = np.array([r["cx"] for r in out])
    cy = np.array([r["cy"] for r in out])
    pad = 60
    x0, x1 = max(0, int(cx.min()) - pad), min(img.shape[1], int(cx.max()) + pad)
    y0, y1 = max(0, int(cy.min()) - pad), min(img.shape[0], int(cy.max()) + pad)
    ax0.imshow(img[y0:y1, x0:x1], cmap="gray", vmax=np.percentile(img, 99.9),
               extent=(x0, x1, y1, y0))
    if "labels" in z:
        lab = z["labels"][y0:y1, x0:x1]
        ax0.contour(np.arange(x0, x1), np.arange(y0, y1), lab > 0, [0.5],
                    colors="w", linewidths=0.4)
    for r in out:
        col = cmap(norm(r["t_sw_us"])) if np.isfinite(r["t_sw_us"]) else "r"
        ax0.annotate(str(r["id"]), (r["cx"], r["cy"]), color=col, fontsize=7,
                     ha="center", va="center", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.1", fc="k",
                               ec="r" if r["flags"] else "none", alpha=0.6))
    ax0.set_title("LED IDs (colour = t_r+t_f, red box = flagged)", fontsize=9)
    ax1 = fig.add_subplot(gs[1])
    id_list = [r["id"] for r in out]
    for i, (f, yn) in curves.items():
        ax1.plot(f, yn, "-", lw=0.8, alpha=0.7, color=cmap(norm(t[id_list.index(i)])))
    if whole is not None:
        ax1.plot(whole[0], whole[1], "k--", lw=1.5, label="whole column")
        ax1.legend(fontsize=8)
    ax1.set_xscale("log")
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_xlabel("PWM frequency (Hz)")
    ax1.set_ylabel("L / L0 (per duty), each LED")
    ax1.set_title("roll-off of every LED", fontsize=9)
    ax1.grid(alpha=0.3, which="both")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, label="t_r+t_f (µs)")
    ax2 = fig.add_subplot(gs[2])
    ax3 = ax2.twinx()
    ax3.bar(id_list, [r["dc_rel"] for r in out], color="orange", alpha=0.25, width=0.8)
    ax3.set_ylabel("steady-on brightness / median (bars)", color="darkorange")
    ax3.set_ylim(0, 2.0)
    ax2.errorbar(id_list, t, yerr=[r["t_sw_err_us"] for r in out], fmt="o", ms=4,
                 capsize=2, color="#1f77b4", label="t_r+t_f")
    fl = [r for r in out if r["flags"]]
    if fl:
        ax2.plot([r["id"] for r in fl], [r["t_sw_us"] for r in fl], "o", mfc="none",
                 mec="r", ms=10, label="flagged")
    ax2.axhline(med, color="gray", ls=":", lw=1)
    for m in missing:
        ax2.axvspan(m["after_id"] + 0.5, m["after_id"] + m["count"] + 0.5,
                    color="r", alpha=0.15)
    ax2.set_xlabel("LED ID along the line (red bands = dark/missing LEDs)")
    ax2.set_ylabel("t_r + t_f (µs)")
    ax2.set_zorder(ax3.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("per-LED switching time and brightness", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(base + "_per_led.png", dpi=120)
    plt.close(fig)
    print(f"  plot -> {base}_per_led.png")


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
    sw.add_argument("--rows", help="GIGA: rows lit during the sweep (all | none | 0,3,5-8)")
    sw.add_argument("--pwm-col", type=int, help="GIGA: PWM column pin, 3..13 (default D9)")
    sw.add_argument("--dc-cols", help="GIGA: extra columns held on (DC) during the sweep")
    sw.add_argument("--baud", type=int, default=115200)
    sw.add_argument("--manual", action="store_true",
                    help="set each frequency by hand; the tool prompts and you can "
                         "type the frequency you actually set")
    sw.add_argument("--freqs", help="comma list, k/M suffixes ok, e.g. 100,1k,10k,100k. "
                                    "Omit to type them one at a time.")
    sw.add_argument("--duty", type=float, default=50.0, help="duty cycle, percent")
    sw.add_argument("--exposure-us", type=int, default=20000,
                    help="must span many PWM periods at the lowest frequency")
    sw.add_argument("--gain", type=float, default=0, help="sensor gain in milli-dB (0..48001); keep fixed, outputs are also normalised to 0 dB")
    sw.add_argument("--frames", type=int, default=30, help="frames per frequency")
    sw.add_argument("--dark-frames", type=int, default=20)
    sw.add_argument("--roi-frames", type=int, default=10)
    sw.add_argument("--discard", type=int, default=8,
                    help="frames dropped after every change: the V4L2 queue still holds ~4 "
                         "frames of the previous state (measured), keep a margin")
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
    sw.add_argument("--no-per-led", dest="per_led", action="store_false",
                    help="skip segmenting individual LEDs")
    sw.add_argument("--led-map", help="per-LED pixel map from the 'ledmap' command "
                                      "(groups each LED's lobes into one LED)")
    sw.add_argument("--bin", type=int, default=1,
                    help="software bin factor for the saved per-step ROI crops")
    sw.add_argument("--close-px", type=int, default=7,
                    help="closing radius that merges an LED array into one blob")
    sw.add_argument("--k-sigma", type=float, default=6.0)
    sw.add_argument("--largest-only", action="store_true")
    sw.add_argument("--sim-tr-us", type=float, default=2.0)
    sw.add_argument("--sim-tf-us", type=float, default=1.0)
    sw.add_argument("--seed", type=int, default=0)
    sw.add_argument("--sim-leds", type=int, default=0,
                    help="sim: draw a column of this many separate LEDs")
    fit_args(sw)

    an = sub.add_parser("analyze", help="re-fit a saved sweep")
    an.add_argument("base", help="results/led_switching/<stamp> (any of its files)")
    fit_args(an)

    st = sub.add_parser("selftest", help="simulated rig with a known t_r+t_f")
    st.add_argument("--tsw-us", type=float, default=3.0, help="true t_r + t_f")
    st.add_argument("--duty", type=float, default=50.0)
    st.add_argument("--exposure-us", type=int, default=20000)
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--leds", type=int, default=0,
                    help="simulate a column of separate LEDs (per-LED test)")

    lm = sub.add_parser("ledmap", help="light each address alone and map every LED's "
                                       "pixels (all its lobes) -> _ledmap.npz")
    lm.add_argument("--from-scan", help="build from a saved led_addr_scan_* folder instead "
                                        "of capturing (needs dark_full.npy, D<col>_rowNN.npy)")
    lm.add_argument("--pwm-col", type=int, default=9, help="column pin, 3..13")
    lm.add_argument("--rows", help="addresses to light one at a time (default 0-23)")
    lm.add_argument("--device", default="/dev/video0")
    lm.add_argument("--pixel-format", default="RG10")
    lm.add_argument("--port", default="/dev/ttyACM0")
    lm.add_argument("--baud", type=int, default=115200)
    lm.add_argument("--exposure-us", type=int, default=500,
                    help="keep it unclipped: ~0.5 ms for one LED fully on at 0 dB")
    lm.add_argument("--gain", type=float, default=0)
    lm.add_argument("--frames", type=int, default=6)
    lm.add_argument("--discard", type=int, default=8)
    lm.add_argument("--settle-s", type=float, default=0.05)
    lm.add_argument("--close-px", type=int, default=8,
                    help="closing radius that joins the lobes of one LED across its dark cross")
    lm.add_argument("--merge-r", type=float, default=25.0,
                    help="regions of one address closer than this (px) are the same LED")
    lm.add_argument("--rel", type=float, default=0.12,
                    help="LED pixels: above this fraction of the address's peak")

    a = ap.parse_args(argv)
    if a.cmd == "ledmap":
        cmd_ledmap(a)
    elif a.cmd == "analyze":
        analyze(a.base, k=a.k, fit_max_drop=a.fit_max_drop, knee_drop=a.knee_drop)
    elif a.cmd == "sweep":
        cmd_sweep(a)
    else:
        t = a.tsw_us
        freqs = "100,300,1k,3k,10k,20k,30k,50k,70k,100k,150k"
        args = ["sweep", "--backend", "sim", "--freqs", freqs, "--duty", str(a.duty),
                "--exposure-us", str(a.exposure_us), "--sim-tr-us", str(t * 2 / 3),
                "--sim-tf-us", str(t / 3), "--seed", str(a.seed), "--frames", "20",
                "--sim-leds", str(a.leds)]
        main(args)


if __name__ == "__main__":
    main()
