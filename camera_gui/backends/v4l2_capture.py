"""Direct V4L2 mmap capture (REQBUFS / QBUF / DQBUF).

This is the reliable frame path for the VC MIPI IMX568's ``RG10`` (10-bit Bayer
in a 16-bit container). OpenCV's ``VideoCapture`` does not decode RG10 and hands
back nothing, so we talk to the kernel directly — exactly what
``v4l2-ctl --stream-mmap`` does. Pure ``ctypes``/``fcntl``/``mmap``; no third
party deps. Linux-only (imported lazily by the v4l2 backend).
"""
from __future__ import annotations

import ctypes
import mmap
import os
import select
from typing import Optional

try:
    import fcntl  # Linux-only; required at runtime, optional just for import
except ImportError:  # allows struct/ioctl verification on non-Linux dev hosts
    fcntl = None

import numpy as np

# --- ioctl number encoding (asm-generic) -----------------------------------
_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS = 8, 8, 14
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2


def _IOC(d: int, t: str, nr: int, size: int) -> int:
    return (d << _IOC_DIRSHIFT) | (ord(t) << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)


def _IOR(t, nr, st):  return _IOC(_IOC_READ, t, nr, ctypes.sizeof(st))
def _IOW(t, nr, st):  return _IOC(_IOC_WRITE, t, nr, ctypes.sizeof(st))
def _IOWR(t, nr, st): return _IOC(_IOC_READ | _IOC_WRITE, t, nr, ctypes.sizeof(st))


_u8, _u32, _i32, _long, _ulong = (
    ctypes.c_uint8, ctypes.c_uint32, ctypes.c_int32, ctypes.c_long, ctypes.c_ulong
)

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1


def fourcc(s: str) -> int:
    s = (s + "    ")[:4]
    return ord(s[0]) | (ord(s[1]) << 8) | (ord(s[2]) << 16) | (ord(s[3]) << 24)


# --- structures (must match <linux/videodev2.h> on 64-bit) -----------------
class v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ("width", _u32), ("height", _u32), ("pixelformat", _u32), ("field", _u32),
        ("bytesperline", _u32), ("sizeimage", _u32), ("colorspace", _u32),
        ("priv", _u32), ("flags", _u32), ("ycbcr_enc", _u32),
        ("quantization", _u32), ("xfer_func", _u32),
    ]


class _fmt_union(ctypes.Union):
    _fields_ = [("pix", v4l2_pix_format), ("raw", _u8 * 204)]


class v4l2_format(ctypes.Structure):
    _fields_ = [("type", _u32), ("fmt", _fmt_union)]


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [
        ("count", _u32), ("type", _u32), ("memory", _u32),
        ("capabilities", _u32), ("flags", _u8), ("reserved", _u8 * 3),
    ]


class _timeval(ctypes.Structure):
    _fields_ = [("tv_sec", _long), ("tv_usec", _long)]


class v4l2_timecode(ctypes.Structure):
    _fields_ = [
        ("type", _u32), ("flags", _u32), ("frames", _u8), ("seconds", _u8),
        ("minutes", _u8), ("hours", _u8), ("userbits", _u8 * 4),
    ]


class _buf_m(ctypes.Union):
    _fields_ = [("offset", _u32), ("userptr", _ulong), ("fd", _i32)]


class v4l2_buffer(ctypes.Structure):
    _fields_ = [
        ("index", _u32), ("type", _u32), ("bytesused", _u32), ("flags", _u32),
        ("field", _u32), ("timestamp", _timeval), ("timecode", v4l2_timecode),
        ("sequence", _u32), ("memory", _u32), ("m", _buf_m), ("length", _u32),
        ("reserved2", _u32), ("request_fd", _i32),
    ]


VIDIOC_G_FMT = _IOWR("V", 4, v4l2_format)
VIDIOC_S_FMT = _IOWR("V", 5, v4l2_format)
VIDIOC_REQBUFS = _IOWR("V", 8, v4l2_requestbuffers)
VIDIOC_QUERYBUF = _IOWR("V", 9, v4l2_buffer)
VIDIOC_QBUF = _IOWR("V", 15, v4l2_buffer)
VIDIOC_DQBUF = _IOWR("V", 17, v4l2_buffer)
VIDIOC_STREAMON = _IOW("V", 18, ctypes.c_int)
VIDIOC_STREAMOFF = _IOW("V", 19, ctypes.c_int)

# On the 64-bit Linux target the struct layout must match the kernel ABI exactly
# (the ioctl number encodes the struct size). Verify on LP64; skip on LLP64 dev
# hosts (Windows) where `long` is 4 bytes and the sizes legitimately differ.
if fcntl is not None and ctypes.sizeof(ctypes.c_long) == 8:
    assert ctypes.sizeof(v4l2_format) == 208, ctypes.sizeof(v4l2_format)
    assert ctypes.sizeof(v4l2_buffer) == 88, ctypes.sizeof(v4l2_buffer)

# 16-bit container formats (10/12-bit raw). Everything else treated as 8-bit.
_U16_FORMATS = {"RG10", "BA10", "Y10 ", "Y10", "RG12", "Y12 ", "Y12", "RG16"}


class V4l2MmapCapture:
    """Minimal streaming-capture device wrapper."""

    def __init__(self, device: str, n_buffers: int = 4):
        self.device = device
        self.n_buffers = n_buffers
        self.fd: Optional[int] = None
        self._bufs: list[mmap.mmap] = []
        self.width = self.height = 0
        self.bytesperline = self.sizeimage = 0
        self.dtype = np.uint16
        self.streaming = False

    def open(self) -> None:
        self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)

    def set_format(self, width: int, height: int, pixfmt: str = "RG10") -> None:
        fmt = v4l2_format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = width
        fmt.fmt.pix.height = height
        fmt.fmt.pix.pixelformat = fourcc(pixfmt)
        fmt.fmt.pix.field = V4L2_FIELD_NONE
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)            # driver may adjust
        self.width = fmt.fmt.pix.width
        self.height = fmt.fmt.pix.height
        self.bytesperline = fmt.fmt.pix.bytesperline
        self.sizeimage = fmt.fmt.pix.sizeimage
        self.dtype = np.uint16 if pixfmt.strip() in _U16_FORMATS else np.uint8

    def start(self) -> None:
        req = v4l2_requestbuffers()
        req.count = self.n_buffers
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)

        self._bufs = []
        for i in range(req.count):
            b = v4l2_buffer()
            b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            b.memory = V4L2_MEMORY_MMAP
            b.index = i
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, b)
            mm = mmap.mmap(self.fd, b.length, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, offset=b.m.offset)
            self._bufs.append(mm)
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)

        bt = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, bt)
        self.streaming = True

    def grab(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        if not self.streaming:
            return None
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        b = v4l2_buffer()
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = V4L2_MEMORY_MMAP
        try:
            fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
        except OSError:
            return None

        mm = self._bufs[b.index]
        n = self.sizeimage or b.bytesused
        raw = bytes(mm[:n])                 # copy out before requeueing
        fcntl.ioctl(self.fd, VIDIOC_QBUF, b)

        arr = np.frombuffer(raw, dtype=self.dtype)
        itemsize = 2 if self.dtype == np.uint16 else 1
        stride = (self.bytesperline // itemsize) if self.bytesperline else self.width
        try:
            return arr.reshape(self.height, stride)[:, : self.width].copy()
        except ValueError:
            return arr.reshape(self.height, self.width).copy()

    def stop(self) -> None:
        if self.fd is None:
            return
        if self.streaming:
            bt = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, bt)
            except OSError:
                pass
            self.streaming = False
        for mm in self._bufs:
            mm.close()
        self._bufs = []
        # release kernel buffers so a subsequent set_format/start can re-REQBUFS
        try:
            req = v4l2_requestbuffers()
            req.count = 0
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            req.memory = V4L2_MEMORY_MMAP
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
        except OSError:
            pass

    def close(self) -> None:
        self.stop()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
