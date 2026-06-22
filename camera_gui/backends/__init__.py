"""Camera backends and a small factory to pick one by name."""
from __future__ import annotations

from ..config import AppConfig
from .base import CameraBackend
from .mock_backend import MockBackend

__all__ = ["CameraBackend", "MockBackend", "create_backend"]


def create_backend(cfg: AppConfig) -> CameraBackend:
    """Instantiate the backend named in ``cfg.backend`` ("mock" | "v4l2")."""
    if cfg.backend == "mock":
        return MockBackend(cfg.camera)
    if cfg.backend == "v4l2":
        # Imported here so the dependency (OpenCV) is only needed on hardware.
        from .v4l2_backend import V4l2Backend

        return V4l2Backend(cfg.camera, device=cfg.device,
                           pixel_format=cfg.pixel_format)
    raise ValueError(f"unknown backend: {cfg.backend!r}")
