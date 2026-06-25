# VC Camera GUI

A PySide6 + pyqtgraph control GUI for the **Vision Components VC MIPI IMX568**
on an **NVIDIA Jetson Orin Nano**, captured via V4L2. Provides a live decimated
preview, full-rate recording, and camera controls including ROI/binning,
trigger modes and the flash/strobe output.

Develop and test it on a laptop with the built-in **mock** backend (synthetic
frames), then deploy to the Jetson over git and switch to the **v4l2** backend.

## Quick start (laptop, no hardware)

```bash
pip install -r requirements.txt
python -m camera_gui --backend mock
```

A window opens with a moving synthetic pattern. Click **Start streaming**, then
try Record, the trigger modes (free-run / external / software), ROI, etc.

## On the Jetson

```bash
git pull                       # deploy
# system deps: OpenCV + v4l2-ctl, and libxcb-cursor0 (Qt6/PySide6 xcb plugin)
sudo apt-get install -y python3-opencv v4l-utils libxcb-cursor0 libxcb-xinerama0
python3 -m venv --system-site-packages .venv    # so the venv sees system cv2
source .venv/bin/activate
pip install -r requirements.txt                 # PySide6, pyqtgraph, numpy<2
python -m camera_gui --backend v4l2 --device /dev/video0 --pixel-format RG10
```

Run it on the Jetson's own desktop (it's a GUI; needs a display / `$DISPLAY`),
not a plain SSH session.

CLI options: `--backend {mock,v4l2}`, `--device`, `--pixel-format` (RG10/Y10/…),
`--display-fps`, `--ring-size`, `--record-dir`.

## Architecture

```
backends/   frame source + controls only (mock_backend, v4l2_backend)
core/       ring_buffer · capture thread · recorder   (no Qt; the hot path)
ui/         controls panel · main_window              (PySide6 + pyqtgraph)
app.py      arg parsing + wiring
```

The **capture thread** runs `backend.grab()` at the full sensor rate and pushes
frames into a lock-protected **ring buffer** and a non-blocking **recorder**
queue. The UI polls the ring buffer on a `QTimer` at `--display-fps`, so Qt never
sits on the high-speed path — the preview is decimated, the recording is not.
This is what keeps the design viable at the 1000 fps target.

## Recording formats

- **RAW** — every frame appended to `vc_<timestamp>.raw` with a `vc_<timestamp>.json`
  sidecar (dtype, shape, count, camera state). Lossless, full rate. Reload with
  `camera_gui.core.recorder.load_raw(path)` → `(frames, H, W)` array.
- **MP4** — best-effort 8-bit preview via OpenCV; for sharing clips, not analysis.

If the disk can't keep up, the recorder drops frames and counts them (shown in
the status bar) rather than stalling capture.

## Hardware bring-up checklist (v4l2 backend)

The v4l2 backend is a working shell; a few values must be confirmed against your
specific module before the controls do the right thing. See the docstring in
[`backends/v4l2_backend.py`](camera_gui/backends/v4l2_backend.py).

1. `v4l2-ctl -d /dev/video0 --list-ctrls-menus` → fill in `CONTROL_NAMES`
   (exposure, gain, trigger_mode, flash) with the exact IDs your driver exposes.
2. Confirm the `trigger_mode` integer encoding against the VC **"Trigger and
   Flash"** application note → `TRIGGER_VALUES`.
3. For real 1000 fps, replace the OpenCV `read()` in `grab()` with a direct V4L2
   mmap loop (`VIDIOC_QBUF`/`DQBUF` + `select()`), and raise the frame rate via
   the **device-tree mode table** (lane count, PLL/clock, VMAX/HMAX blanking) —
   that ceiling lives in the driver, not in this app.

## Trigger & flash

- **External**: sensor exposes on each TTL pulse on the module's trigger input —
  drive from a Jetson GPIO or a master clock to sync with behavior/other cameras.
- **Software**: snapshot-on-command from the GUI button.
- **Flash/strobe**: pulse synced to exposure; gate the excitation LED so it only
  fires during the (≤1 ms at 1000 fps) exposure window — saves heat and reduces
  photobleaching.
