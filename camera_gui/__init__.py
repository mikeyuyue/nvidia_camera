"""VC Camera GUI — PySide6 + pyqtgraph live viewer, recorder and control panel.

Targets the Vision Components VC MIPI IMX568 on an NVIDIA Jetson, captured via
v4l2. Runs on a developer machine with a synthetic (mock) backend so the UI can
be built and tested away from the hardware.
"""

import os as _os

# pyqtgraph's ImageView imports matplotlib only to enumerate colormap names. On
# JetPack the system matplotlib is too old to initialise a Qt backend against the
# pip-installed PySide6 (it raises on the Qt KeyboardModifier enum). Force a
# non-interactive backend before anything imports matplotlib so that import can't
# touch Qt. Set before pyqtgraph is imported anywhere.
_os.environ.setdefault("MPLBACKEND", "Agg")

__version__ = "0.1.0"
