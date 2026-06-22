"""The capture thread.

Runs ``backend.grab()`` in a tight loop at the full sensor rate, pushes each
frame into the ring buffer and (when recording) forwards it to the recorder.
Everything here is plain threading — no Qt — so the hot path stays lean enough
for high frame rates. The UI reads :meth:`CaptureThread.stats` and the ring
buffer on its own timer.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from ..backends.base import CameraBackend
from .recorder import Recorder
from .ring_buffer import Frame, RingBuffer


@dataclass
class CaptureStats:
    running: bool = False
    frames: int = 0          # total frames captured this session
    timeouts: int = 0        # grab() returned None (e.g. waiting on a trigger)
    capture_fps: float = 0.0  # measured, over a sliding window


class CaptureThread:
    """Owns the capture loop. Not a QThread — a bare daemon thread."""

    def __init__(
        self,
        backend: CameraBackend,
        ring: RingBuffer,
        recorder: Optional[Recorder] = None,
        grab_timeout: float = 0.5,
    ):
        self._backend = backend
        self._ring = ring
        self._recorder = recorder
        self._grab_timeout = grab_timeout

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stats = CaptureStats()

        # Sliding-window fps measurement.
        self._win_t0 = 0.0
        self._win_count = 0

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._stats = CaptureStats(running=True)
        self._win_t0 = time.perf_counter()
        self._win_count = 0
        self._backend.start_streaming()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._backend.stop_streaming()
        except Exception:
            pass
        with self._lock:
            self._stats.running = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stats(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(**self._stats.__dict__)

    # --- loop --------------------------------------------------------------
    def _run(self) -> None:
        idx = 0
        while not self._stop.is_set():
            try:
                img = self._backend.grab(timeout=self._grab_timeout)
            except Exception:
                # A transient backend error shouldn't kill the thread; count it
                # as a timeout and keep going so the UI can recover.
                with self._lock:
                    self._stats.timeouts += 1
                continue

            if img is None:
                with self._lock:
                    self._stats.timeouts += 1
                continue

            now = time.perf_counter()
            frame = Frame(index=idx, timestamp=now, data=img)
            idx += 1
            self._ring.push(frame)
            if self._recorder is not None:
                self._recorder.submit(frame)

            self._tick_fps(now)

        # final stats flush
        with self._lock:
            self._stats.frames = idx

    def _tick_fps(self, now: float) -> None:
        self._win_count += 1
        with self._lock:
            self._stats.frames += 1
        elapsed = now - self._win_t0
        if elapsed >= 0.5:                       # update twice a second
            fps = self._win_count / elapsed
            with self._lock:
                self._stats.capture_fps = fps
            self._win_t0 = now
            self._win_count = 0
