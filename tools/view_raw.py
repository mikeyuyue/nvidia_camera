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
    from pyqtgraph.Qt import QtWidgets

    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QtWidgets.QApplication([])
    iv = pg.image(frames, title=os.path.basename(args.path))   # time slider + levels
    iv.setPredefinedGradient("grey")
    app.exec()


if __name__ == "__main__":
    main()
