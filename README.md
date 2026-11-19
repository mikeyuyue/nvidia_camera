# VC Camera GUI

A PySide6 + pyqtgraph control GUI for the **Vision Components VC MIPI IMX568**
on an **NVIDIA Jetson Orin Nano**, captured via V4L2. Provides a live decimated
preview, full-rate recording, and camera controls including ROI/binning,
trigger modes and the flash/strobe output.

## Quick start (laptop, no hardware)

```bash
pip install -r requirements.txt
python -m camera_gui --backend mock
```


## On the Jetson

```bash
git pull                       # deploy
# system deps: OpenCV + v4l2-ctl, and libxcb-cursor0 (Qt6/PySide6 xcb plugin)
sudo apt-get install -y python3-opencv v4l-utils libxcb-cursor0 libxcb-xinerama0
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m camera_gui --backend v4l2 --device /dev/video0 --pixel-format RG10
```

Run it on the Jetson's own desktop, not SSH.

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
3. Currently the camera can reach around 770 fps. For higher fps, consider changing the device tree, 
   replacing the OpenCV `read()` in `grab()` with a direct V4L2
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

## LED switching speed without a fast camera

`tools/led_switching.py` measures an LED's rise + fall time from how its
time-averaged brightness drops with PWM frequency at a fixed duty:
`L(f) ≈ L0·[1 − k·(t_r+t_f)·f]`. The camera exposure spans many PWM periods,
so a 40 fps camera resolves sub-microsecond edges.

1. Flash `tools/arduino/led_pwm/led_pwm.ino` (Uno/Nano, LED driver on pin 9) and
   plug the Arduino's USB into the Jetson (`/dev/ttyACM0`). Needs `python3-serial`.
2. Close the GUI, then on the Jetson:

```bash
python3 tools/led_switching.py sweep --port /dev/ttyACM0 --duty 50 --freqs 100,1k,3k,10k,20k,50k,100k,200k,500k --exposure-us 20000
```

It finds the LED automatically (on − off image → threshold → components;
`--hw-crop` also crops the sensor to it, `--bin` bins saved crops), subtracts
each frame's own background ring as its dark level, fits the linear part and
writes `results/led_switching/<stamp>_rolloff.png` (normalized grey vs f with the
fit and the knee) plus CSV/JSON. `--manual` sets frequencies by hand (omit
`--freqs` to type them one at a time); `analyze <stamp>` re-fits; `selftest`
runs a simulated LED with a known t_r+t_f.
