vc_568@nvidia:~$ for H in 104 64 48 40 32; do
  v4l2-ctl -d /dev/video0 --set-fmt-video=width=608,height=$H,pixelformat=RG10 >/dev/null
  echo -n "H=$H : "; v4l2-ctl -d /dev/video0 --list-ctrls | grep frame_rate
done
H=104 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
H=64 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
H=48 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
H=40 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
H=32 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
vc_568@nvidia:~$ for B in 0 1 2 3 4; do
  v4l2-ctl -d /dev/video0 --set-ctrl=binning_mode=$B >/dev/null 2>&1
  v4l2-ctl -d /dev/video0 --set-fmt-video=width=608,height=104,pixelformat=RG10 >/dev/null 2>&1
  echo -n "binning=$B : "; v4l2-ctl -d /dev/video0 --list-ctrls | grep frame_rate
done
binning=0 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
binning=1 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
binning=2 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
binning=3 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider
binning=4 :                      frame_rate 0x009a200b (int64)  : min=0 max=416840 step=1 default=0 value=416840 flags=slider




vc_568@nvidia:~$ v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=608,height=104,pixelformat=RG10 \
  --set-ctrl=exposure=500 \
  --set-ctrl=frame_rate=<the new max from above> \
  --stream-mmap --stream-count=2000 --stream-to=/dev/shm/hs.raw
-bash: the: No such file or directory
vc_568@nvidia:~$ v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=608,height=104,pixelformat=RG10 \
  --set-ctrl=exposure=500 \
  --set-ctrl=frame_rate=<the new max from above> \
  --stream-mmap --stream-count=2000 --stream-to=/dev/shm/hs.raw
-bash: the: No such file or directory
vc_568@nvidia:~$ v4l2-ctl -d /dev/video0   --set-fmt-video=width=608,height=104,pixelformat=RG10   --set-ctrl=exposure=500   --set-ctrl=frame_rate=416840   --stream-mmap --stream-count=2000 --stream-to=/dev/shm/hs.raw
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 416.67 fps
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 416.67 fps
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 416.67 fps
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 416.67 fps
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<