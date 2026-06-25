#!/usr/bin/env python3
"""Verbose V4L2 mmap capture self-test — prints every step so we can see exactly
where streaming fails. Run on the Jetson with the GUI closed:

    cd ~/nvidia_camera && source .venv/bin/activate
    python tools/v4l2_selftest.py /dev/video0 608 104 RG10
"""
import ctypes
import errno
import fcntl
import mmap
import os
import select
import sys

from camera_gui.backends import v4l2_capture as v


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 608
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 104
    pf = sys.argv[4] if len(sys.argv) > 4 else "RG10"

    print(f"open {dev}")
    fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)

    fmt = v.v4l2_format()
    fmt.type = v.V4L2_BUF_TYPE_VIDEO_CAPTURE
    fmt.fmt.pix.width = w
    fmt.fmt.pix.height = h
    fmt.fmt.pix.pixelformat = v.fourcc(pf)
    fmt.fmt.pix.field = v.V4L2_FIELD_NONE
    fcntl.ioctl(fd, v.VIDIOC_S_FMT, fmt)
    print(f"S_FMT -> {fmt.fmt.pix.width}x{fmt.fmt.pix.height} "
          f"bytesperline={fmt.fmt.pix.bytesperline} "
          f"sizeimage={fmt.fmt.pix.sizeimage} "
          f"pixfmt=0x{fmt.fmt.pix.pixelformat:08x}")

    req = v.v4l2_requestbuffers()
    req.count = 4
    req.type = v.V4L2_BUF_TYPE_VIDEO_CAPTURE
    req.memory = v.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v.VIDIOC_REQBUFS, req)
    print(f"REQBUFS -> count={req.count}")

    bufs = []
    for i in range(req.count):
        b = v.v4l2_buffer()
        b.type = v.V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = v.V4L2_MEMORY_MMAP
        b.index = i
        fcntl.ioctl(fd, v.VIDIOC_QUERYBUF, b)
        mm = mmap.mmap(fd, b.length, mmap.MAP_SHARED,
                       mmap.PROT_READ | mmap.PROT_WRITE, offset=b.m.offset)
        bufs.append(mm)
        fcntl.ioctl(fd, v.VIDIOC_QBUF, b)
    print(f"queued {len(bufs)} buffers (length={bufs and len(bufs[0])})")

    bt = ctypes.c_int(v.V4L2_BUF_TYPE_VIDEO_CAPTURE)
    fcntl.ioctl(fd, v.VIDIOC_STREAMON, bt)
    print("STREAMON ok; waiting for frames...")

    got = 0
    for i in range(20):
        r, _, _ = select.select([fd], [], [], 2.0)
        if not r:
            print(f"  [{i}] select TIMEOUT (no frame in 2s)")
            continue
        b = v.v4l2_buffer()
        b.type = v.V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = v.V4L2_MEMORY_MMAP
        try:
            fcntl.ioctl(fd, v.VIDIOC_DQBUF, b)
        except OSError as e:
            print(f"  [{i}] DQBUF error {e.errno} "
                  f"({errno.errorcode.get(e.errno, '?')})")
            continue
        got += 1
        if got <= 2:
            import numpy as np
            raw = bytes(bufs[b.index][:b.bytesused])
            a = np.frombuffer(raw, dtype=np.uint16)
            print(f"  [{i}] frame idx={b.index} bytesused={b.bytesused} "
                  f"min={a.min()} max={a.max()} mean={a.mean():.1f}")
        fcntl.ioctl(fd, v.VIDIOC_QBUF, b)

    print(f"TOTAL frames: {got}")
    fcntl.ioctl(fd, v.VIDIOC_STREAMOFF, bt)
    os.close(fd)


if __name__ == "__main__":
    main()
