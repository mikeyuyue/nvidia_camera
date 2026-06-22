"""A small, thread-safe ring buffer holding the most recent frames.

The capture thread is the sole writer; the UI display timer and any snapshot
action are readers. We keep a bounded history (for a future "scrub back" feature)
but the common case is :meth:`RingBuffer.latest`, which the display polls.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """One captured frame plus the metadata we care about downstream."""

    index: int          # monotonically increasing capture index
    timestamp: float    # time.perf_counter() at grab, in seconds
    data: np.ndarray    # 2-D image (H, W), dtype uint8 or uint16


class RingBuffer:
    """Bounded, lock-protected store of the latest ``size`` frames."""

    def __init__(self, size: int):
        self._size = max(int(size), 1)
        self._dq: deque[Frame] = deque(maxlen=self._size)
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()

    def push(self, frame: Frame) -> None:
        with self._lock:
            self._dq.append(frame)
            self._latest = frame

    def latest(self) -> Optional[Frame]:
        """Most recently pushed frame, or ``None`` before anything arrives."""
        with self._lock:
            return self._latest

    def fill(self) -> int:
        with self._lock:
            return len(self._dq)

    @property
    def capacity(self) -> int:
        return self._size

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()
            self._latest = None
