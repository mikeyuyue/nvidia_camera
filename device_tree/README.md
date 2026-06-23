# Device tree — high-frame-rate IMX568 mode

`imx568_highfps_mode.dtsi` is a from-scratch Tegra/VC-MIPI mode-table entry for a
**608×104, up to ~1000 fps** capture on **cam0 / 2 CSI lanes**. Read the comment
header in that file — it has the timing math and which values to confirm with the
IMX568 datasheet / Vision Components.

## Try this first (may make flashing unnecessary)
The VC driver supports runtime ROI. Crop on the *existing* tree and measure:
```bash
v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=608,height=104,pixelformat=RG10 \
  --set-ctrl=exposure=1000 --set-ctrl=frame_rate=1000 \
  --stream-mmap --stream-count=1000 --stream-to=/dev/shm/crop.raw
```
- fps stays ~42 → the mode ceiling limits you; flash the new mode below.
- fps jumps well past 42 → the driver honors the crop; you may be done, or just
  need `max_framerate` raised.

## How to integrate the mode entry
1. Find the IMX568 sensor node in your VC MIPI device tree (the `mode0 { … }` /
   `mode_tbl` under the `cam_module@1a` style node for cam0).
2. Add this block as the next `modeN` and bump the node's `num_modes` / the
   `mode_list`. Match `tegra_sinterface`/ports to cam0 in your tree.
3. Rebuild the DTB, re-apply the overlay (extlinux `OVERLAYS=`), reboot.
4. Verify the new mode appears and runs:
   ```bash
   v4l2-ctl -d /dev/video0 --list-formats-ext      # new size should show
   v4l2-ctl -d /dev/video0 --set-ctrl=frame_rate=1000 \
     --set-fmt-video=width=608,height=104,pixelformat=RG10 \
     --stream-mmap --stream-count=1000 --stream-to=/dev/shm/crop.raw   # read fps
   ```

## Key constraint
On a VC MIPI camera the sensor's real timing is programmed by the module MCU at
runtime; the device tree configures the **Tegra capture side and the advertised
ceiling**, not the sensor's physical limit. Confirm the target rate at this ROI
on 2 lanes with Vision Components — the DT cannot push the sensor past its
firmware's limit.
