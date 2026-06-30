#!/usr/bin/env python3
"""View a RAW recording (.raw + .json) interactively, or export it to MP4.

Uses pyqtgraph (already a GUI dependency) so it needs no ffmpeg. Pass the base
path with or without the .raw/.json suffix.

    python tools/view_raw.py D:\\nvidia_camera\\recordings\\recordings\\vc_20260626_165329
    python tools/view_raw.py <path> --mp4 out.mp4     # needs opencv installed
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_gui.core.recorder import load_raw


def to_8bit(frames: np.ndarray) -> np.ndarray:
    """Percentile-stretch to 8-bit so 10-bit data is actually visible."""
    lo, hi = np.percentile(frames, (1, 99))
    hi = max(hi, lo + 1)
    out = (frames.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="recording base path (.raw/.json)")
    ap.add_argument("--mp4", metavar="OUT", help="export to MP4 instead of viewing")
    ap.add_argument("--fps", type=float, default=30.0, help="MP4 playback fps")
    args = ap.parse_args()

    frames = load_raw(args.path)        # (N, H, W)
    print(f"loaded {frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]} "
          f"dtype={frames.dtype} min={frames.min()} max={frames.max()} "
          f"mean={frames.mean():.1f}")

    if args.mp4:
        import cv2
        h, w = frames.shape[1:]
        vw = cv2.VideoWriter(args.mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h), isColor=False)
        for f in to_8bit(frames):
            vw.write(f)
        vw.release()
        print(f"wrote {args.mp4}")
        return

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    n = frames.shape[0]
    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QtWidgets.QApplication([])

    win = QtWidgets.QWidget()
    win.setWindowTitle(os.path.basename(args.path))
    win.resize(900, 640)
    vbox = QtWidgets.QVBoxLayout(win)

    iv = pg.ImageView()
    iv.setImage(frames, autoLevels=True)     # drag the histogram sliders for contrast
    iv.setPredefinedGradient("grey")
    vbox.addWidget(iv, 1)

    # --- playback bar ---
    bar = QtWidgets.QHBoxLayout()
    play_btn = QtWidgets.QPushButton("▶  Play")
    play_btn.setCheckable(True)
    fps_spin = QtWidgets.QDoubleSpinBox()
    fps_spin.setRange(0.1, 240.0)
    fps_spin.setValue(args.fps)
    fps_spin.setSuffix(" fps")
    loop_chk = QtWidgets.QCheckBox("Loop")
    loop_chk.setChecked(True)
    frame_lbl = QtWidgets.QLabel()
    for w in (play_btn, QtWidgets.QLabel("Speed:"), fps_spin, loop_chk):
        bar.addWidget(w)
    bar.addStretch(1)
    bar.addWidget(frame_lbl)
    vbox.addLayout(bar)

    timer = QtCore.QTimer()

    def advance():
        i = iv.currentIndex + 1
        if i >= n:
            if loop_chk.isChecked():
                i = 0
            else:
                play_btn.setChecked(False)
                return
        iv.setCurrentIndex(i)

    def toggle(on):
        play_btn.setText("⏸  Pause" if on else "▶  Play")
        if on:
            timer.start(int(1000 / max(fps_spin.value(), 0.1)))
        else:
            timer.stop()

    def on_fps(v):
        if timer.isActive():
            timer.setInterval(int(1000 / max(v, 0.1)))

    def on_time(ind, _t):
        frame_lbl.setText(f"frame {int(ind) + 1} / {n}")

    timer.timeout.connect(advance)
    play_btn.toggled.connect(toggle)
    fps_spin.valueChanged.connect(on_fps)
    iv.sigTimeChanged.connect(on_time)
    on_time(0, 0)

    win.show()
    app.exec()


if __name__ == "__main__":
    main()
