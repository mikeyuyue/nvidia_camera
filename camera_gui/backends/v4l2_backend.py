"""V4L2 backend for the Vision Components VC MIPI IMX568 on the Jetson.

Scope: this is the deployable *shell*. Frame grabbing goes through OpenCV's V4L2
capture, and camera controls go through ``v4l2-ctl`` (from ``v4l-utils``), which
is the reliable way to reach the VC MIPI custom controls (trigger mode, flash,
exposure, gain) without hand-rolling ioctls. Once you're running on hardware,
the highest-rate path is a direct mmap/DQBUF loop — see ``grab`` for where to
swap that in.

IMPORTANT — confirm these against your module before trusting them:

* ``CONTROL_NAMES`` — the exact v4l2 control IDs the VC MIPI driver exposes.
  Run ``v4l2-ctl -d /dev/video0 --list-ctrls-menus`` and edit the mapping.
* ``TRIGGER_VALUES`` — the integer your driver uses for each trigger mode.
* Pixel format / ROI — set via the device tree mode table + ``--set-fmt-video``.

Nothing here imports OpenCV at module load, so the module is importable (and the
mock backend usable) on a machine without it.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional

import numpy as np

from ..config import CameraConfig, TriggerMode
from .base import CameraBackend


class V4l2Backend(CameraBackend):
    name = "v4l2"

    #: Control names confirmed from `v4l2-ctl --list-ctrls-menus` on this rig
    #: (Tegra/Argus V4L2 interface + VC MIPI controls).
    CONTROL_NAMES = {
        "exposure": "exposure",          # int64, microseconds (default 10000)
        "gain": "gain",                  # int64, 0..48001 step 100 (~milli-dB)
        "fps": "frame_rate",             # int64; sets target rate (mode-capped)
        "trigger_mode": "trigger_mode",  # int 0..7 (VC MIPI)
        "flash": "io_mode",              # int 0..5 (GPIO/flash routing)
        "binning": "binning_mode",       # int 0..7
        "software_trigger": "single_trigger",  # button: write 1 to fire
    }

    #: VC MIPI trigger_mode encoding (0..7). Confirm against the VC "Trigger and
    #: Flash" application note; typical: 0=off/free-run, 1=external, 4=single.
    TRIGGER_VALUES = {
        TriggerMode.FREERUN: 0,
        TriggerMode.EXTERNAL: 1,
        TriggerMode.SOFTWARE: 4,
    }

    def __init__(self, config: CameraConfig, device: str = "/dev/video0",
                 pixel_format: str = "GREY"):
        super().__init__(config)
        self.device = device
        self.pixel_format = pixel_format
        self._cap = None
        self._v4l2ctl = shutil.which("v4l2-ctl")

    # --- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        import cv2  # imported lazily; only needed on hardware

        self._apply_format()
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open {self.device} via V4L2")
        # Ask OpenCV not to convert to RGB so we keep raw mono frames.
        try:
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        except Exception:
            pass
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Push the initial control state to the sensor.
        self.set_exposure(self.config.exposure_us)
        self.set_gain(self.config.gain)
        self.set_trigger_mode(self.config.trigger)
        self.set_flash(self.config.flash_enable)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def start_streaming(self) -> None:
        # OpenCV streams implicitly on read(); nothing to do.
        pass

    def stop_streaming(self) -> None:
        pass

    # --- frames ------------------------------------------------------------
    def grab(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Return the next frame as a 2-D ndarray, or None on timeout.

        NOTE: ``VideoCapture.read()`` blocks without honouring ``timeout``. For
        the 1000 fps target, replace this with a direct V4L2 mmap loop
        (VIDIOC_QBUF/DQBUF with select() on the fd) so you get true timeouts and
        zero-copy buffers. The OpenCV path here is correct but adds a copy and a
        little latency.
        """
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if frame.ndim == 3:
            # OpenCV gave us BGR despite CONVERT_RGB=0 (some builds ignore it);
            # collapse to mono.
            frame = frame[..., 0]
        return frame

    # --- controls ----------------------------------------------------------
    def set_exposure(self, exposure_us: int) -> None:
        self.config.exposure_us = int(exposure_us)
        self._set_ctrl(self.CONTROL_NAMES["exposure"], int(exposure_us))

    def set_gain(self, gain: float) -> None:
        self.config.gain = float(gain)
        # Most drivers take an integer gain register value; round.
        self._set_ctrl(self.CONTROL_NAMES["gain"], int(round(gain)))

    def set_fps(self, fps: float) -> None:
        self.config.fps = float(fps)
        # This Tegra interface ignores --set-parm; rate is set via the
        # 'frame_rate' control. The achievable rate is still capped by the
        # device-tree mode table (currently only the full-res mode exists).
        self._set_ctrl(self.CONTROL_NAMES["fps"], int(round(fps)))

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        self.config.roi.x, self.config.roi.y = x, y
        self.config.roi.width, self.config.roi.height = width, height
        # ROI changes the capture format and must happen while stopped.
        was_open = self._cap is not None
        if was_open:
            self.close()
        self._apply_format()
        if was_open:
            self.open()

    def set_binning(self, factor: int) -> None:
        self.config.binning = int(factor)
        # 'binning_mode' is a runtime VC MIPI control (0..7). We map the UI's
        # 1x/2x to indices 0/1 — confirm the index→factor mapping for your
        # module's mode table.
        index = 0 if factor <= 1 else 1
        self._set_ctrl(self.CONTROL_NAMES["binning"], index)

    def set_trigger_mode(self, mode: TriggerMode) -> None:
        self.config.trigger = mode
        value = self.TRIGGER_VALUES.get(mode, 0)
        self._set_ctrl(self.CONTROL_NAMES["trigger_mode"], value)

    def set_flash(self, enabled: bool) -> None:
        self.config.flash_enable = bool(enabled)
        # Flash/strobe is routed via 'io_mode' (0..5). The exact index that
        # gates the strobe to exposure is module-specific; 1 is the usual
        # "flash out" — confirm against the VC Trigger & Flash app note.
        self._set_ctrl(self.CONTROL_NAMES["flash"], 1 if enabled else 0)

    def software_trigger(self) -> None:
        # 'single_trigger' is a write-only button control: writing 1 fires one
        # frame (use with trigger_mode = SOFTWARE/single).
        self._set_ctrl(self.CONTROL_NAMES["software_trigger"], 1)

    # --- helpers -----------------------------------------------------------
    def _effective_size(self) -> tuple[int, int]:
        b = max(self.config.binning, 1)
        return (self.config.roi.width // b, self.config.roi.height // b)

    def _apply_format(self) -> None:
        w, h = self._effective_size()
        self._run_v4l2ctl([
            f"--set-fmt-video=width={w},height={h},"
            f"pixelformat={self.pixel_format}",
        ])

    def _set_ctrl(self, name: str, value: int) -> None:
        if not name:
            return
        self._run_v4l2ctl([f"--set-ctrl={name}={value}"])

    def _run_v4l2ctl(self, args: list[str]) -> None:
        if not self._v4l2ctl:
            return  # v4l-utils not installed; controls are no-ops
        cmd = [self._v4l2ctl, "-d", self.device, *args]
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=2.0)
        except Exception:
            pass
