"""Main application window.

Wires the control panel to the backend and the capture/recorder threads, and
runs the live display. The display is driven by a QTimer at ``cfg.display_fps``
that polls the ring buffer for the newest frame — the capture thread keeps
running at full rate underneath, so the preview is decimated, not throttling.
"""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QWidget,
)

from ..backends.base import CameraBackend
from ..config import AppConfig, TriggerMode
from ..core import CaptureThread, RingBuffer, Recorder
from ..core.recorder import RecordFormat
from .controls import ControlPanel


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, backend: CameraBackend):
        super().__init__()
        self.cfg = cfg
        self.backend = backend

        self.ring = RingBuffer(cfg.ring_size)
        self.recorder = Recorder(cfg.record_dir)
        self.capture = CaptureThread(backend, self.ring, self.recorder)

        self._levels_set = False
        self._disp_t0 = time.perf_counter()
        self._disp_count = 0
        self._disp_fps = 0.0
        self._last_index = -1

        self.setWindowTitle(f"VC Camera — {cfg.backend} ({cfg.device})")
        self.resize(1100, 600)
        self._build_ui()

        self.backend.open()

        # Display timer.
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / max(cfg.display_fps, 1)))
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # --- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        # Live image view (pyqtgraph). ImageView gives us a histogram + level
        # sliders for contrast control out of the box.
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.view = pg.ImageView()
        self.view.ui.roiBtn.hide()
        self.view.ui.menuBtn.hide()
        layout.addWidget(self.view, stretch=1)

        self.controls = ControlPanel(self.cfg)
        self.controls.setFixedWidth(300)
        layout.addWidget(self.controls)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self._status = QLabel("Idle")
        self.statusBar().addWidget(self._status)

        self._connect()

    def _connect(self) -> None:
        c = self.controls
        c.streamToggled.connect(self._set_streaming)
        c.exposureChanged.connect(self.backend.set_exposure)
        c.gainChanged.connect(self.backend.set_gain)
        c.fpsChanged.connect(self.backend.set_fps)
        c.binningChanged.connect(self._set_binning)
        c.roiApplied.connect(self._set_roi)
        c.triggerModeChanged.connect(self.backend.set_trigger_mode)
        c.flashToggled.connect(self.backend.set_flash)
        c.softwareTrigger.connect(self.backend.software_trigger)
        c.recordToggled.connect(self._set_recording)
        c.autoLevelsRequested.connect(self._auto_levels)

    # --- control handlers --------------------------------------------------
    def _set_streaming(self, on: bool) -> None:
        if on:
            self.capture.start()
        else:
            self.capture.stop()

    def _set_binning(self, factor: int) -> None:
        self._with_capture_paused(lambda: self.backend.set_binning(factor))
        self._levels_set = False

    def _set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self._with_capture_paused(lambda: self.backend.set_roi(x, y, w, h))
        self._levels_set = False

    def _with_capture_paused(self, fn) -> None:
        """Apply a format-changing control safely around the capture loop."""
        was_running = self.capture.is_running
        if was_running:
            self.capture.stop()
        fn()
        if was_running:
            self.capture.start()

    def _set_recording(self, on: bool, fmt: RecordFormat) -> None:
        if on:
            if not self.capture.is_running:
                # Recording without a stream is almost never intended.
                self.controls.set_recording_state(False)
                self._status.setText("Start streaming before recording")
                return
            cam = self.cfg.camera
            meta = {
                "backend": self.cfg.backend,
                "device": self.cfg.device,
                "pixel_format": self.cfg.pixel_format,
                "roi": cam.roi.as_tuple(),
                "binning": cam.binning,
                "exposure_us": cam.exposure_us,
                "gain": cam.gain,
                "trigger": cam.trigger.value,
            }
            base = self.recorder.start(fmt, meta, fps=cam.fps)
            self._status.setText(f"Recording → {base}")
        else:
            stats = self.recorder.stop()
            self._status.setText(
                f"Saved {stats.frames_written} frames "
                f"({stats.bytes_written / 1e6:.1f} MB) to {stats.path}"
            )

    def _auto_levels(self) -> None:
        frame = self.ring.latest()
        if frame is not None:
            lo, hi = np.percentile(frame.data, (1, 99))
            self.view.setLevels(float(lo), float(max(hi, lo + 1)))

    # --- display loop ------------------------------------------------------
    def _on_tick(self) -> None:
        frame = self.ring.latest()
        if frame is not None and frame.index != self._last_index:
            self._last_index = frame.index
            # transpose so row-major (H, W) shows upright with row-major order on.
            # imageAxisOrder is "row-major", so (H, W) displays upright as-is.
            self.view.setImage(
                frame.data,
                autoRange=not self._levels_set,
                autoLevels=not self._levels_set,
                autoHistogramRange=not self._levels_set,
            )
            self._levels_set = True
            self._disp_count += 1

        self._update_status()

    def _update_status(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._disp_t0
        if elapsed >= 0.5:
            self._disp_fps = self._disp_count / elapsed
            self._disp_t0 = now
            self._disp_count = 0

        cs = self.capture.stats()
        rs = self.recorder.stats()
        parts = [
            f"capture {cs.capture_fps:6.1f} fps",
            f"display {self._disp_fps:4.1f} fps",
            f"frames {cs.frames}",
            f"ring {self.ring.fill()}/{self.ring.capacity}",
        ]
        if cs.timeouts:
            parts.append(f"timeouts {cs.timeouts}")
        if rs.active or rs.frames_written:
            rec = f"REC {rs.frames_written}"
            if rs.frames_dropped:
                rec += f" (dropped {rs.frames_dropped})"
            parts.append(rec)
        self.statusBar().showMessage("   |   ".join(parts))

    # --- shutdown ----------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self.recorder.is_recording:
            self.recorder.stop()
        self.capture.stop()
        try:
            self.backend.close()
        except Exception:
            pass
        super().closeEvent(event)
