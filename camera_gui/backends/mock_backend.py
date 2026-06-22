"""Synthetic camera backend for development without hardware.

Generates a moving pattern at the configured ROI size and honours fps,
exposure (as brightness), gain, binning and the three trigger modes so the full
UI — including the trigger workflow — can be exercised on a laptop.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ..config import CameraConfig, TriggerMode
from .base import CameraBackend


class MockBackend(CameraBackend):
    name = "mock"

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._streaming = False
        self._t0 = 0.0
        self._frame_idx = 0
        self._next_due = 0.0
        # Software-trigger handshake.
        self._sw_trigger = threading.Event()

    # --- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        self._t0 = time.perf_counter()

    def close(self) -> None:
        self._streaming = False

    def start_streaming(self) -> None:
        self._streaming = True
        self._next_due = time.perf_counter()

    def stop_streaming(self) -> None:
        self._streaming = False

    # --- frames ------------------------------------------------------------
    def grab(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        if not self._streaming:
            time.sleep(min(timeout, 0.02))
            return None

        deadline = time.perf_counter() + timeout

        if self.config.trigger in (TriggerMode.EXTERNAL, TriggerMode.SOFTWARE):
            # Wait for an (emulated) trigger. EXTERNAL fires on a pseudo 10 Hz
            # external clock; SOFTWARE waits for software_trigger().
            if self.config.trigger == TriggerMode.SOFTWARE:
                if not self._sw_trigger.wait(timeout=timeout):
                    return None
                self._sw_trigger.clear()
            else:
                period = 0.1
                now = time.perf_counter()
                wait = period - ((now - self._t0) % period)
                if wait > timeout:
                    time.sleep(timeout)
                    return None
                time.sleep(wait)
        else:
            # Free-run: pace to the configured fps.
            period = 1.0 / max(self.config.fps, 1e-3)
            now = time.perf_counter()
            if self._next_due > now:
                sleep = min(self._next_due - now, timeout)
                time.sleep(max(sleep, 0))
                if time.perf_counter() >= deadline and self._next_due > time.perf_counter():
                    return None
            self._next_due = max(self._next_due + period, time.perf_counter())

        return self._render()

    def _render(self) -> np.ndarray:
        roi = self.config.roi
        b = max(self.config.binning, 1)
        h = max(roi.height // b, 1)
        w = max(roi.width // b, 1)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        phase = self._frame_idx * 0.15
        # Drifting diagonal interference pattern + a moving bright blob.
        pattern = 0.5 + 0.5 * np.sin((xx + yy) * 0.08 + phase)
        cx = (0.5 + 0.4 * np.sin(phase * 0.7)) * w
        cy = (0.5 + 0.4 * np.cos(phase * 0.5)) * h
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (max(w, h) * 0.08) ** 2)))
        img = 0.6 * pattern + 0.8 * blob

        # Exposure scales brightness; gain multiplies; add shot-like noise.
        exposure = self.config.exposure_us / 1000.0
        img = img * exposure * self.config.gain
        img += np.random.default_rng(self._frame_idx).normal(0, 0.02, img.shape)
        frame = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        self._frame_idx += 1
        return frame

    # --- controls ----------------------------------------------------------
    def set_exposure(self, exposure_us: int) -> None:
        self.config.exposure_us = int(exposure_us)

    def set_gain(self, gain: float) -> None:
        self.config.gain = float(gain)

    def set_fps(self, fps: float) -> None:
        self.config.fps = float(fps)

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        self.config.roi.x = x
        self.config.roi.y = y
        self.config.roi.width = width
        self.config.roi.height = height

    def set_binning(self, factor: int) -> None:
        self.config.binning = int(factor)

    def set_trigger_mode(self, mode: TriggerMode) -> None:
        self.config.trigger = mode
        self._sw_trigger.clear()

    def set_flash(self, enabled: bool) -> None:
        self.config.flash_enable = bool(enabled)

    def software_trigger(self) -> None:
        self._sw_trigger.set()
