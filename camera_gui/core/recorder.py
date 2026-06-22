"""Frame recorder.

Recording runs on its own writer thread fed by a bounded queue. The capture
thread calls :meth:`Recorder.submit`, which never blocks — if the writer can't
keep up the frame is dropped and counted, so a slow disk degrades the *recording*
rather than stalling capture (and the live view stays responsive).

Two formats:

* ``RAW`` — append every frame's bytes to a single ``.raw`` file with a JSON
  sidecar describing dtype/shape/count. This is the lossless, full-rate path you
  want at 1000 fps. Reload with :func:`load_raw`.
* ``MP4`` — best-effort 8-bit preview via OpenCV (handy for sharing a clip; not
  for quantitative analysis). Silently unavailable if OpenCV isn't installed.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .ring_buffer import Frame


class RecordFormat(Enum):
    RAW = "raw"
    MP4 = "mp4"


@dataclass
class RecordStats:
    active: bool = False
    path: str = ""
    frames_written: int = 0
    frames_dropped: int = 0      # dropped because the writer queue was full
    bytes_written: int = 0
    started_at: float = 0.0


class Recorder:
    """Asynchronous frame writer."""

    def __init__(self, record_dir: str = "recordings", queue_size: int = 512):
        self._dir = record_dir
        self._queue: queue.Queue[Optional[Frame]] = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stats = RecordStats()
        # Set on start(), used by the writer thread.
        self._fmt = RecordFormat.RAW
        self._meta: dict = {}
        self._fh = None          # raw file handle
        self._writer = None      # cv2.VideoWriter
        self._base = ""

    # --- public API --------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._stats.active

    def stats(self) -> RecordStats:
        with self._lock:
            return RecordStats(**asdict(self._stats))

    def start(self, fmt: RecordFormat, meta: dict, fps: float = 30.0) -> str:
        """Begin a new recording. Returns the base path written to."""
        if self.is_recording:
            raise RuntimeError("already recording")

        os.makedirs(self._dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._base = os.path.join(self._dir, f"vc_{stamp}")
        self._fmt = fmt
        self._meta = dict(meta)
        self._meta["fps"] = fps

        with self._lock:
            self._stats = RecordStats(
                active=True,
                path=self._base,
                started_at=time.perf_counter(),
            )

        # Drain any stale items, then launch the writer.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        self._thread = threading.Thread(
            target=self._run, name="recorder", daemon=True
        )
        self._thread.start()
        return self._base

    def submit(self, frame: Frame) -> None:
        """Hand a frame to the writer. Non-blocking; drops if the queue is full."""
        if not self.is_recording:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            with self._lock:
                self._stats.frames_dropped += 1

    def stop(self) -> RecordStats:
        if not self.is_recording:
            return self.stats()
        with self._lock:
            self._stats.active = False
        self._queue.put(None)               # sentinel
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        return self.stats()

    # --- writer thread -----------------------------------------------------
    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._write(item)
        finally:
            self._finalize()

    def _write(self, frame: Frame) -> None:
        if self._fmt is RecordFormat.RAW:
            if self._fh is None:
                self._fh = open(self._base + ".raw", "wb")
                self._meta.update(
                    dtype=str(frame.data.dtype),
                    height=int(frame.data.shape[0]),
                    width=int(frame.data.shape[1]),
                )
            buf = np.ascontiguousarray(frame.data)
            self._fh.write(buf.tobytes())
            with self._lock:
                self._stats.frames_written += 1
                self._stats.bytes_written += buf.nbytes
        else:  # MP4
            if self._writer is None:
                self._writer = self._open_mp4(frame)
                if self._writer is None:      # OpenCV unavailable
                    with self._lock:
                        self._stats.active = False
                    return
            img8 = frame.data
            if img8.dtype != np.uint8:
                img8 = (img8 >> 8).astype(np.uint8) if img8.dtype == np.uint16 else img8.astype(np.uint8)
            self._writer.write(img8)          # cv2 writes grayscale as-is in newer builds; see _open_mp4
            with self._lock:
                self._stats.frames_written += 1

    def _open_mp4(self, frame: Frame):
        try:
            import cv2
        except Exception:
            return None
        h, w = frame.data.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # isColor=False -> grayscale writer; OpenCV expands internally.
        return cv2.VideoWriter(
            self._base + ".mp4", fourcc, float(self._meta.get("fps", 30.0)),
            (w, h), isColor=False,
        )

    def _finalize(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            with self._lock:
                self._meta["frames"] = self._stats.frames_written
            with open(self._base + ".json", "w") as f:
                json.dump(self._meta, f, indent=2)
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def load_raw(base_path: str) -> np.ndarray:
    """Load a ``.raw`` recording back into a ``(frames, H, W)`` array.

    ``base_path`` may be given with or without the ``.raw``/``.json`` suffix.
    """
    base = base_path
    for suffix in (".raw", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    with open(base + ".json") as f:
        meta = json.load(f)
    data = np.fromfile(base + ".raw", dtype=np.dtype(meta["dtype"]))
    return data.reshape(-1, meta["height"], meta["width"])
