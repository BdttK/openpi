"""Camera capture for the bimanual Piper station (3x RealSense).

Backends:
  * ``mock``      -- zeros (smoke test, no hardware)
  * ``opencv``    -- USB/UVC cameras via ``cv2.VideoCapture``
  * ``realsense`` -- Intel RealSense via ``pyrealsense2`` (by device serial)

Returns a dict of uint8 ``(H, W, 3)`` RGB frames keyed by ``cam_high`` /
``cam_left_wrist`` / ``cam_right_wrist``, in ``keys`` order. ``cv2`` and
``pyrealsense2`` are imported lazily so only the selected backend is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraGroupConfig:
    keys: list[str] = field(
        default_factory=lambda: ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    )
    width: int = 640
    height: int = 480
    fps: int = 10
    backend: str = "mock"  # "mock" | "opencv" | "realsense"
    # opencv: device index (int) or device path (e.g. /dev/video0).
    # realsense: device serial number (str).
    device_ids: dict[str, str] = field(default_factory=dict)


class CameraGroup:
    """Grabs a synchronized frame from every camera in ``keys`` order."""

    def __init__(self, cfg: CameraGroupConfig):
        self.cfg = cfg
        self._devices: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        if self.cfg.backend == "mock":
            return
        if self.cfg.backend == "opencv":
            self._connect_opencv()
            return
        if self.cfg.backend == "realsense":
            self._connect_realsense()
            return
        raise ValueError(f"unknown camera backend {self.cfg.backend!r}")

    def _connect_opencv(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("opencv backend needs `opencv-python` (`pip install opencv-python`)") from exc

        for key in self.cfg.keys:
            src = self.cfg.device_ids.get(key, key)
            index = int(src) if str(src).isdigit() else src  # int -> index, str -> device path
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            if not cap.isOpened():
                raise RuntimeError(f"could not open camera {key!r} at {src!r}")
            cap.read()  # drop the first (often dark/unsynced) frame
            self._devices[key] = cap
            logger.info("camera %s opened (opencv, src=%s)", key, src)

    def _connect_realsense(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # pragma: no cover
            raise ImportError("realsense backend needs `pyrealsense2`") from exc

        for key in self.cfg.keys:
            serial = self.cfg.device_ids.get(key)
            pipe = rs.pipeline()
            cfg = rs.config()
            if serial:
                cfg.enable_device(str(serial))
            cfg.enable_stream(
                rs.stream.color, self.cfg.width, self.cfg.height, rs.format.bgr8, self.cfg.fps
            )
            pipe.start(cfg)
            self._devices[key] = pipe
            logger.info("camera %s opened (realsense%s)", key, f", serial={serial}" if serial else "")

    # ------------------------------------------------------------------ #
    def read(self) -> dict[str, np.ndarray]:
        """Return ``{key: np.uint8 (H, W, 3) RGB}`` for every key, in ``keys`` order."""
        return {key: self._read_one(key) for key in self.cfg.keys}

    def _read_one(self, key: str) -> np.ndarray:
        if self.cfg.backend == "mock":
            return np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)

        if self.cfg.backend == "opencv":
            ret, frame = self._devices[key].read()
            if not ret:
                raise RuntimeError(f"failed to read frame from camera {key!r}")
            return np.asarray(frame, dtype=np.uint8)[:, :, ::-1]  # BGR -> RGB

        if self.cfg.backend == "realsense":
            frames = self._devices[key].wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError(f"no color frame from camera {key!r}")
            return np.asanyarray(color.get_data())[:, :, ::-1]  # bgr8 -> RGB

        raise ValueError(f"unknown camera backend {self.cfg.backend!r}")

    def disconnect(self) -> None:
        for key, dev in self._devices.items():
            try:
                if self.cfg.backend == "opencv":
                    dev.release()
                elif self.cfg.backend == "realsense":
                    dev.stop()
            except Exception as exc:  # pragma: no cover
                logger.warning("error releasing camera %s: %s", key, exc)
        self._devices.clear()


def concat_cameras(frames: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Stack ``frames`` along the width into one image ``(H, 3W, 3)`` for FastWAM."""
    imgs = [np.asarray(frames[k], dtype=np.uint8) for k in keys]
    if len({im.shape[:2] for im in imgs}) > 1:
        raise ValueError(f"camera shapes differ: {[im.shape for im in imgs]}")
    return np.concatenate(imgs, axis=1)  # (H, 3W, 3)
