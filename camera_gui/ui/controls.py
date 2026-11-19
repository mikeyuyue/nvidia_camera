"""The control panel widget.

A self-contained column of camera controls that emits high-level Qt signals. It
holds no backend reference — :class:`~camera_gui.ui.main_window.MainWindow` wires
the signals to the backend and capture/recorder lifecycle. This keeps the widget
testable and the hardware side-effects in one place.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig, TriggerMode
from ..core.recorder import RecordFormat


class ControlPanel(QWidget):
    streamToggled = Signal(bool)
    exposureChanged = Signal(int)
    gainChanged = Signal(float)
    fpsChanged = Signal(float)
    roiApplied = Signal(int, int, int, int)
    binningChanged = Signal(int)
    triggerModeChanged = Signal(object)        # TriggerMode
    flashToggled = Signal(bool)
    softwareTrigger = Signal()
    recordToggled = Signal(bool, object)       # (on, RecordFormat)
    autoLevelsRequested = Signal()
    antialiasChanged = Signal(float)           # preview Gaussian sigma, px

    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        cam = cfg.camera

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- Stream ---------------------------------------------------------
        self.btn_stream = QPushButton("Start streaming")
        self.btn_stream.setCheckable(True)
        self.btn_stream.toggled.connect(self._on_stream_toggled)
        root.addWidget(self.btn_stream)

        # --- Exposure / gain / fps -----------------------------------------
        acq = QGroupBox("Acquisition")
        acq_form = QFormLayout(acq)

        self.sp_exposure = QSpinBox()
        self.sp_exposure.setRange(*cfg.exposure_range)
        self.sp_exposure.setSuffix(" us")
        self.sp_exposure.setValue(cam.exposure_us)
        self.sp_exposure.valueChanged.connect(self.exposureChanged)
        acq_form.addRow("Exposure", self.sp_exposure)

        self.sp_gain = QDoubleSpinBox()
        self.sp_gain.setRange(*cfg.gain_range)
        self.sp_gain.setSingleStep(0.5)
        self.sp_gain.setValue(cam.gain)
        self.sp_gain.valueChanged.connect(self.gainChanged)
        acq_form.addRow("Gain", self.sp_gain)

        self.sp_fps = QDoubleSpinBox()
        self.sp_fps.setRange(*cfg.fps_range)
        self.sp_fps.setDecimals(1)
        self.sp_fps.setValue(cam.fps)
        self.sp_fps.setSuffix(" fps")
        self.sp_fps.valueChanged.connect(self.fpsChanged)
        acq_form.addRow("Target FPS", self.sp_fps)
        root.addWidget(acq)

        # --- ROI / binning --------------------------------------------------
        roi_box = QGroupBox("ROI && binning")
        roi_form = QFormLayout(roi_box)
        wmin, wmax, wstep = cfg.width_range
        hmin, hmax, hstep = cfg.height_range

        self.sp_x = self._int_spin(0, wmax, 1, cam.roi.x)
        self.sp_y = self._int_spin(0, hmax, 1, cam.roi.y)
        self.sp_w = self._int_spin(wmin, wmax, wstep, cam.roi.width)
        self.sp_h = self._int_spin(hmin, hmax, hstep, cam.roi.height)
        roi_form.addRow("X", self.sp_x)
        roi_form.addRow("Y", self.sp_y)
        roi_form.addRow("Width", self.sp_w)
        roi_form.addRow("Height", self.sp_h)

        self.cb_binning = QComboBox()
        self.cb_binning.addItem("None (1x1)", 1)
        self.cb_binning.addItem("2x2", 2)
        self.cb_binning.setCurrentIndex(0 if cam.binning == 1 else 1)
        self.cb_binning.currentIndexChanged.connect(
            lambda _: self.binningChanged.emit(self.cb_binning.currentData())
        )
        roi_form.addRow("Binning", self.cb_binning)

        self.btn_roi = QPushButton("Apply ROI")
        self.btn_roi.clicked.connect(
            lambda: self.roiApplied.emit(
                self.sp_x.value(), self.sp_y.value(),
                self.sp_w.value(), self.sp_h.value(),
            )
        )
        roi_form.addRow(self.btn_roi)
        root.addWidget(roi_box)

        # --- Trigger / flash ------------------------------------------------
        trig_box = QGroupBox("Trigger && flash")
        trig_form = QFormLayout(trig_box)

        self.cb_trigger = QComboBox()
        for mode in TriggerMode:
            self.cb_trigger.addItem(mode.value, mode)
        self.cb_trigger.setCurrentText(cam.trigger.value)
        self.cb_trigger.currentIndexChanged.connect(self._on_trigger_changed)
        trig_form.addRow("Mode", self.cb_trigger)

        self.btn_sw_trigger = QPushButton("Software trigger")
        self.btn_sw_trigger.clicked.connect(self.softwareTrigger)
        self.btn_sw_trigger.setEnabled(cam.trigger == TriggerMode.SOFTWARE)
        trig_form.addRow(self.btn_sw_trigger)

        self.chk_flash = QCheckBox("Flash / strobe output (LED sync)")
        self.chk_flash.setChecked(cam.flash_enable)
        self.chk_flash.toggled.connect(self.flashToggled)
        trig_form.addRow(self.chk_flash)
        root.addWidget(trig_box)

        # --- Recording ------------------------------------------------------
        rec_box = QGroupBox("Recording")
        rec_form = QFormLayout(rec_box)
        self.cb_format = QComboBox()
        self.cb_format.addItem("RAW (lossless, full rate)", RecordFormat.RAW)
        self.cb_format.addItem("MP4 (8-bit preview)", RecordFormat.MP4)
        rec_form.addRow("Format", self.cb_format)

        self.btn_record = QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.toggled.connect(self._on_record_toggled)
        rec_form.addRow(self.btn_record)
        root.addWidget(rec_box)

        # --- Display --------------------------------------------------------
        disp_row = QHBoxLayout()
        self.btn_auto = QPushButton("Auto levels")
        self.btn_auto.clicked.connect(self.autoLevelsRequested)
        disp_row.addWidget(self.btn_auto)
        root.addLayout(disp_row)

        # Anti-alias low-pass for the *preview only* (RAW stays lossless).
        # 0 = off. ~0.7-1.0 px tames moire from periodic targets (LED arrays);
        # the preview also area-averages when the image is shrunk to fit.
        aa_form = QFormLayout()
        self.sp_aa = QDoubleSpinBox()
        self.sp_aa.setRange(0.0, 5.0)
        self.sp_aa.setSingleStep(0.1)
        self.sp_aa.setDecimals(1)
        self.sp_aa.setSuffix(" px")
        self.sp_aa.setValue(cfg.preview_aa_sigma)
        self.sp_aa.setToolTip(
            "Gaussian low-pass applied to the preview before display. "
            "Removes moire added by display scaling. It cannot remove "
            "aliasing already sampled by the sensor (fix that optically).")
        self.sp_aa.valueChanged.connect(self.antialiasChanged)
        aa_form.addRow("Anti-alias σ", self.sp_aa)
        root.addLayout(aa_form)

        root.addStretch(1)

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _int_spin(lo: int, hi: int, step: int, val: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    def _on_stream_toggled(self, on: bool) -> None:
        self.btn_stream.setText("Stop streaming" if on else "Start streaming")
        self.streamToggled.emit(on)

    def _on_trigger_changed(self, _idx: int) -> None:
        mode = self.cb_trigger.currentData()
        self.btn_sw_trigger.setEnabled(mode == TriggerMode.SOFTWARE)
        self.triggerModeChanged.emit(mode)

    def _on_record_toggled(self, on: bool) -> None:
        self.btn_record.setText("Stop recording" if on else "Record")
        self.recordToggled.emit(on, self.cb_format.currentData())

    def set_recording_state(self, on: bool) -> None:
        """Let the window force the button state (e.g. on a failed start)."""
        self.btn_record.blockSignals(True)
        self.btn_record.setChecked(on)
        self.btn_record.setText("Stop recording" if on else "Record")
        self.btn_record.blockSignals(False)
