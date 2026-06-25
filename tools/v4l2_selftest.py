#!/usr/bin/env python3
"""Verbose V4L2 mmap capture self-test — prints every step so we can see exactly
where streaming fails.

Examples (run on the Jetson, GUI closed):
    cd ~/nvidia_camera && source .venv/bin/activate

    # A) our S_FMT path:
    python tools/v4l2_selftest.py --width 608 --height 104

    # B) let v4l2-ctl set format+controls, skip our S_FMT (uses current device fmt):
    v4l2-ctl -d /dev/video0 --set-fmt-video=width=608,height=104,pixelformat=RG10
    python tools/v4l2_selftest.py --no-sfmt --rate 30000 --exposure 2000
"""
import argparse
import ctypes
import errno
import fcntl
import mmap
import os
import select
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_gui.backends import v4l2_capture as v


def get_fmt(fd):
    fmt = v.v4l2_format()
    fmt.type = v.V4L2_BUF_TYPE_VIDEO_CAPTURE
    fcntl.ioctl(fd, v.VIDIOC_G_FMT, fmt)
    p = fmt.fmt.pix
    return p.width, p.height, p.bytesperline, p.sizeimage, p.pixelformat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=608)
    ap.add_argument("--height", type=int, default=104)
    ap.add_argument("--format", default="RG10")
    ap.add_argument("--no-sfmt", action="store_true",
                    help="don't call S_FMT; use the device's current format")
    ap.add_argument("--rate", type=int, default=None,
                    help="set frame_rate control (fps*1000) via v4l2-ctl first")
    ap.add_argument("--exposure", type=int, default=None,
                    help="set exposure control via v4l2-ctl first")
    ap.add_argument("--nbuf", type=int, default=4)
    args = ap.parse_args()

    if args.rate is not None or args.exposure is not None:
        ctrls = []
        if args.exposure is not None:
            ctrls.append(f"exposure={args.exposure}")
        if args.rate is not None:
            ctrls.append(f"frame_rate={args.rate}")
        cmd = ["v4l2-ctl", "-d", args.device, f"--set-ctrl={','.join(ctrls)}"]
        print("pre-set:", " ".join(cmd))
        subprocess.run(cmd, check=False)

    print(f"open {args.device}")
    fd = os.open(args.device, os.O_RDWR | os.O_NONBLOCK)

    if not args.no_sfmt:
        fmt = v.v4l2_format()
        fmt.type = v.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = args.width
        fmt.fmt.pix.height = args.height
        fmt.fmt.pix.pixelformat = v.fourcc(args.format)
        fmt.fmt.pix.field = v.V4L2_FIELD_NONE
        fcntl.ioctl(fd, v.VIDIOC_S_FMT, fmt)
        print("S_FMT done")

    w, h, bpl, size, pf = get_fmt(fd)
    print(f"G_FMT -> {w}x{h} bytesperline={bpl} sizeimage={size} "
          f"pixfmt=0x{pf:08x} ('{''.join(chr((pf>>(8*i))&0xff) for i in range(4))}')")

    req = v.v4l2_requestbuffers()
    req.count = args.nbuf
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
            print(f"  [{i}] select TIMEOUT")
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
        if got <= 3:
            print(f"  [{i}] frame idx={b.index} bytesused={b.bytesused} "
                  f"seq={b.sequence}")
        fcntl.ioctl(fd, v.VIDIOC_QBUF, b)

    print(f"TOTAL frames: {got}")
    fcntl.ioctl(fd, v.VIDIOC_STREAMOFF, bt)
    os.close(fd)


if __name__ == "__main__":
    main()
