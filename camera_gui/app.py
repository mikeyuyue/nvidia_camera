"""Application entry point: parse args, build the config + backend, run the GUI."""
from __future__ import annotations

import argparse
import sys

from .backends import create_backend
from .config import AppConfig


def build_config(argv=None) -> AppConfig:
    p = argparse.ArgumentParser(description="VC Camera GUI (PySide6 + pyqtgraph)")
    p.add_argument("--backend", choices=["mock", "v4l2"], default="mock",
                   help="frame source: 'mock' (no hardware) or 'v4l2' (Jetson)")
    p.add_argument("--device", default="/dev/video0",
                   help="v4l2 device node (the cam0 path)")
    p.add_argument("--pixel-format", default="GREY",
                   help="v4l2 fourcc, e.g. GREY (8-bit), Y10, RG10")
    p.add_argument("--display-fps", type=int, default=30,
                   help="decimated preview rate; capture runs at full rate")
    p.add_argument("--aa-sigma", type=float, default=1.0,
                   help="preview anti-alias Gaussian sigma in px (default 1; 0 = off)")
    p.add_argument("--ring-size", type=int, default=256,
                   help="frames kept in the in-memory ring buffer")
    p.add_argument("--record-dir", default="recordings")
    args = p.parse_args(argv)

    cfg = AppConfig(
        backend=args.backend,
        device=args.device,
        pixel_format=args.pixel_format,
        display_fps=args.display_fps,
        preview_aa_sigma=args.aa_sigma,
        ring_size=args.ring_size,
        record_dir=args.record_dir,
    )
    return cfg


def main(argv=None) -> int:
    cfg = build_config(argv)

    # Import Qt lazily so --help works without a display / PySide6 installed.
    from PySide6.QtWidgets import QApplication

    from .ui import MainWindow

    app = QApplication(sys.argv[:1])

    # Qt's event loop runs in C and swallows SIGINT, so Ctrl+C normally won't
    # stop the app. Restore the default handler so Ctrl+C terminates it, and run
    # a no-op timer periodically to give Python a chance to process the signal.
    import signal
    from PySide6.QtCore import QTimer

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _sigint_timer = QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)

    backend = create_backend(cfg)
    win = MainWindow(cfg, backend)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
