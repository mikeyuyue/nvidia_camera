"""Abstract camera backend.

A backend is purely a *frame source plus controls*. It knows nothing about
threading, the ring buffer, or the GUI — those live in ``core`` and ``ui``.
This keeps the v4l2/Jetson specifics isolated and lets the mock backend stand
in for the hardware during development.
"""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from ..config import CameraConfig, TriggerMode


class CameraBackend(abc.ABC):
    """Interface every backend implements."""

    def __init__(self, config: CameraConfig):
        self.config = config

    # --- lifecycle ---------------------------------------------------------
    @abc.abstractmethod
    def open(self) -> None:
        """Open the device and apply the initial configuration."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the device."""

    @abc.abstractmethod
    def start_streaming(self) -> None:
        """Begin streaming frames."""

    @abc.abstractmethod
    def stop_streaming(self) -> None:
        """Stop streaming frames."""

    # --- frames ------------------------------------------------------------
    @abc.abstractmethod
    def grab(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Return the next frame as a 2-D ``np.ndarray`` (or ``None`` on timeout).

        In EXTERNAL/SOFTWARE trigger modes this may legitimately block until a
        trigger arrives; the caller passes a finite ``timeout`` to stay
        responsive.
        """

    # --- controls ----------------------------------------------------------
    @abc.abstractmethod
    def set_exposure(self, exposure_us: int) -> None: ...

    @abc.abstractmethod
    def set_gain(self, gain: float) -> None: ...

    @abc.abstractmethod
    def set_fps(self, fps: float) -> None: ...

    @abc.abstractmethod
    def set_roi(self, x: int, y: int, width: int, height: int) -> None: ...

    @abc.abstractmethod
    def set_binning(self, factor: int) -> None: ...

    @abc.abstractmethod
    def set_trigger_mode(self, mode: TriggerMode) -> None: ...

    @abc.abstractmethod
    def set_flash(self, enabled: bool) -> None: ...

    def software_trigger(self) -> None:
        """Emit a single software trigger. No-op for backends without support."""

    # --- context manager sugar --------------------------------------------
    def __enter__(self) -> "CameraBackend":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
