"""Configuration dataclasses shared across backends and UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TriggerMode(Enum):
    """Camera trigger modes.

    Maps to the VC MIPI ``trigger_mode`` v4l2 control on the Jetson. The integer
    values are the typical VC MIPI encoding; confirm against your sensor's
    register map / VC "Trigger and Flash" application note and adjust in
    ``V4l2Backend.TRIGGER_VALUES`` if your module differs.
    """

    FREERUN = "free-run"          # sensor self-times frames (default)
    EXTERNAL = "external"          # expose on each external TTL pulse
    SOFTWARE = "software"          # expose on a software trigger command


@dataclass
class ROI:
    """Region of interest, in sensor pixels (pre-binning)."""

    x: int = 0
    y: int = 0
    width: int = 608
    height: int = 104

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class CameraConfig:
    """Desired camera state. The backend applies what the hardware supports."""

    roi: ROI = field(default_factory=ROI)
    binning: int = 1               # 1 = none, 2 = 2x2
    fps: float = 97.0
    exposure_us: int = 1000        # exposure time in microseconds
    gain: float = 1.0              # analog gain (multiplier or dB per backend)
    trigger: TriggerMode = TriggerMode.FREERUN
    flash_enable: bool = False     # strobe output gated to exposure (LED sync)


@dataclass
class AppConfig:
    """Top-level application configuration."""

    backend: str = "mock"          # "mock" | "v4l2"
    device: str = "/dev/video0"    # v4l2 device node (the "cam0" path)
    pixel_format: str = "GREY"     # v4l2 fourcc; e.g. GREY (8-bit) / Y10 / RG10
    display_fps: int = 30          # decimated preview rate; capture runs full rate
    ring_size: int = 256           # frames kept in the in-memory ring buffer
    record_dir: str = "recordings"
    camera: CameraConfig = field(default_factory=CameraConfig)

    # Bounds advertised to the UI; the v4l2 backend refreshes these from the
    # driver when it opens the device.
    width_range: tuple[int, int, int] = (608, 2464, 32)     # min, max, step
    height_range: tuple[int, int, int] = (32, 2064, 8)
    fps_range: tuple[float, float] = (1.0, 1000.0)
    exposure_range: tuple[int, int] = (10, 33000)            # microseconds
    gain_range: tuple[float, float] = (1.0, 16.0)
