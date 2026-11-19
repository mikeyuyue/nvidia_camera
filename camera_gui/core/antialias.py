"""Anti-aliasing for display and export.

Why this exists: imaging a periodic target (e.g. an LED microarray) whose pitch
is close to the sensor pixel pitch, rotated against the pixel grid, produces
diagonal moiré stripes wherever the image is resampled without a low-pass
filter first. In this app that happens in the live preview (an image larger
than the widget is drawn nearest-neighbour) and in any downscaled export.

Rule: low-pass *before* you resample. ``gaussian_blur`` is the low-pass,
``downscale`` does blur + area-average, and ``sigma_for_scale`` picks the
blur width for a given shrink factor.

Limits: this only removes aliasing that *we* introduce downstream. Aliasing
created at the sensor itself (scene detail above the sensor's own Nyquist
limit) is already baked into the RAW frames and can't be undone in software;
blurring then only trades the stripes for lost contrast. The real fixes there
are optical: align the camera to the LED grid, add magnification so one LED
spans several pixels, or defocus slightly (an optical low-pass filter).

RAW recordings are never filtered: they stay lossless.
"""
from __future__ import annotations

import numpy as np

try:  # OpenCV is ~10x faster, but keep a pure-numpy fallback.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depends on the machine
    cv2 = None


def sigma_for_scale(scale: float) -> float:
    """Gaussian sigma (in source pixels) for shrinking by ``scale`` (>1).

    sigma = scale / 2 puts the filter's -3 dB point near the new Nyquist
    frequency; it suppresses most aliasing without visibly over-blurring.
    """
    return 0.0 if scale <= 1.0 else 0.5 * float(scale)


def _kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian low-pass. Returns float32 (keeps 10-bit precision).

    ``sigma <= 0`` returns the input unchanged (as float32).
    """
    out = np.asarray(img, dtype=np.float32)
    if sigma <= 0:
        return out
    if cv2 is not None:
        return cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT_101)
    k = _kernel(sigma)
    r = len(k) // 2
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="reflect")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for i, w in enumerate(k):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(i, i + n)
            acc += w * p[tuple(sl)]
        out = acc
    return out


def downscale(img: np.ndarray, scale: float) -> np.ndarray:
    """Shrink by ``scale`` (>1) with proper anti-aliasing (blur, then area).

    Returns float32. ``scale <= 1`` only converts.
    """
    f = gaussian_blur(img, sigma_for_scale(scale))
    if scale <= 1.0:
        return f
    h, w = f.shape[:2]
    nh, nw = max(1, int(round(h / scale))), max(1, int(round(w / scale)))
    if cv2 is not None:
        return cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
    # numpy fallback: integer block mean on the cropped image
    s = max(1, int(round(scale)))
    h2, w2 = (h // s) * s, (w // s) * s
    return f[:h2, :w2].reshape(h2 // s, s, w2 // s, s).mean(axis=(1, 3))
