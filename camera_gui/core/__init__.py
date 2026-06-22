"""Threading, buffering and recording — the hardware-agnostic plumbing that sits
between a :class:`~camera_gui.backends.base.CameraBackend` and the Qt UI.

Design: the capture loop runs in a plain (non-Qt) thread at the full sensor
frame rate and only ever touches the backend, a lock-protected ring buffer and a
non-blocking recording queue. The UI polls the ring buffer on a timer at a
decimated *display* rate, so Qt never sits on the high-speed hot path (critical
for the 1000 fps target).
"""

from .ring_buffer import Frame, RingBuffer
from .capture import CaptureThread, CaptureStats
from .recorder import Recorder, RecordFormat

__all__ = [
    "Frame",
    "RingBuffer",
    "CaptureThread",
    "CaptureStats",
    "Recorder",
    "RecordFormat",
]
