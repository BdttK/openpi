"""Dual-arm Piper driver for the bimanual station, via the AgileX ``piper_sdk``.

Wraps the official ``piper_sdk`` (V3 firmware) over CAN with a small, safe
joint+gripper control surface. Every SDK call is written against the **actual**
``piper_sdk`` API (``C_PiperInterface_V3``):

* ``C_PiperInterface_V3(can_name=..., dh_is_offset=..., start_sdk_joint_limit=...)``
* ``ConnectPort()`` / ``EnablePiper()`` / ``DisablePiper()`` / ``EmergencyStop()``
* ``MotionCtrl_2(ctrl_mode, move_mode, move_spd_rate_ctrl, is_mit_mode)`` then
  ``JointCtrl(j1..j6)`` (MOVE J) and ``GripperCtrl(width, effort, code, set_zero)``
* ``GetArmJointMsgs()`` -> ``.joint_state.joint_1..joint_6``
* ``GetArmGripperMsgs_V3()`` -> ``.gripper_state.grippers_val``

Units (from ``piper_sdk/assets/V3/INTERFACE_V3.MD``):

* ``JointCtrl``: 6 ints in **0.001 deg** (rad * 57295.7795)
* joint feedback: **0.001 deg**
* ``GripperCtrl``: width in **0.001 mm**, effort in 0.001 N*m (0..5000)
* gripper feedback (``grippers_val``): **0.001 mm**

The robot is driven with **joint + gripper** commands only; ee pose / rot6d are
not needed by the ``pi05_piper`` policy.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Units / limits (verified from the SDK source).
# --------------------------------------------------------------------------- #
_RAD_TO_0P001DEG = 57295.7795  # 1000 * 180 / pi
_0P001DEG_TO_RAD = math.pi / (180.0 * 1000.0)
_MM_TO_0P001MM = 1000.0
_Gripper0P001MM_TO_MM = 0.001

# Piper 6-DoF joint limits in radians (from ``JointCtrl`` docstring).
JOINT_LIMITS_RAD: list[tuple[float, float]] = [
    (-2.6179, 2.6179),      # joint 1  [-150, 150] deg
    (0.0, 3.14),            # joint 2  [0, 180] deg
    (-2.967, 0.0),          # joint 3  [-170, 0] deg
    (-1.745, 1.745),        # joint 4  [-100, 100] deg
    (-1.22, 1.22),          # joint 5  [-70, 70] deg
    (-2.09439, 2.09439),    # joint 6  [-120, 120] deg
]

# MotionCtrl_2 move modes.
_MOVE_J = 0x01
_CTRL_CAN = 0x01
_POS_VEL = 0x00


def _piper_interface_class():
    """Return ``C_PiperInterface_V3`` from the installed ``piper_sdk``."""
    try:
        from piper_sdk import C_PiperInterface_V3  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "piper_sdk not installed. Install it with: `pip install piper_sdk python-can`."
        ) from exc
    return C_PiperInterface_V3


@dataclass
class ArmConfig:
    """Per-arm hardware config. Only ``can_port`` / ``can_id`` are truly station-specific."""

    can_port: str = "can0"
    can_id: int = 1
    dh_is_offset: int = 1                 # 1 = new DH params (after S-V1.6-3)
    move_spd_rate_ctrl: int = 100         # MOVE speed %, 0..100
    gripper_effort: int = 1000            # 0.001 N*m (1000 = 1 N*m)
    gripper_open_mm: float = 100.0        # full-open gap in mm (data uses mm/100 -> [0,1])
    gripper_invert: bool = False          # flip gripper open/close direction if needed
    joint_limits_rad: list[tuple[float, float]] = field(
        default_factory=lambda: list(JOINT_LIMITS_RAD)
    )
    enable_sdk_joint_limit: bool = True
    enable_sdk_gripper_limit: bool = True


def _clamp_joints(joints_rad: np.ndarray, limits: list[tuple[float, float]]) -> np.ndarray:
    lo = np.array([l[0] for l in limits], dtype=np.float64)
    hi = np.array([l[1] for l in limits], dtype=np.float64)
    return np.clip(joints_rad, lo, hi)


def _euler_deg_to_rot6d(rpy_deg: np.ndarray) -> np.ndarray:
    """Convert Euler angles ``[rx, ry, rz]`` (deg, extrinsic XYZ) to rot6d.

    rot6d = first two columns of the 3x3 rotation matrix, flattened.
    TODO: confirm the Euler convention (extrinsic XYZ vs ZYX) matches the dataset.
    """
    rx, ry, rz = np.radians(rpy_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # R = Rz @ Ry @ Rx (extrinsic XYZ).
    R = np.array(
        [
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz],
            [-sy, cy * sx, cx * cy],
        ]
    )
    return R[:, :2].reshape(-1)


class PiperArm:
    """One Piper arm + its gripper, behind a minimal joint-control API."""

    def __init__(self, name: str, cfg: ArmConfig):
        self.name = name
        self.cfg = cfg
        self._iface: Any = None
        self._fk_calibrated = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        self._iface = _piper_interface_class()(
            can_name=self.cfg.can_port,
            dh_is_offset=self.cfg.dh_is_offset,
            start_sdk_joint_limit=self.cfg.enable_sdk_joint_limit,
            start_sdk_gripper_limit=self.cfg.enable_sdk_gripper_limit,
        )
        self._iface.ConnectPort()

        deadline = time.monotonic() + 5.0
        while not self._iface.EnablePiper():
            if time.monotonic() > deadline:
                self._iface.EmergencyStop()
                raise TimeoutError(f"[{self.name}] EnablePiper() timed out on {self.cfg.can_port}")
            time.sleep(0.01)
        logger.info("[%s] connected + enabled on %s", self.name, self.cfg.can_port)

    def disconnect(self) -> None:
        if self._iface is not None:
            self._iface.DisablePiper()
        self._iface = None

    def emergency_stop(self) -> None:
        if self._iface is not None:
            self._iface.EmergencyStop()

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def read_joints_rad(self) -> np.ndarray:
        """Return the 6 joint angles in radians (from 0.001-deg feedback)."""
        js = self._iface.GetArmJointMsgs().joint_state
        raw = np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=np.float64,
        )
        return raw * _0P001DEG_TO_RAD

    def read_gripper_norm(self) -> float:
        """Return gripper opening normalized to [0, 1] (dataset convention mm/100)."""
        msg = self._iface.GetArmGripperMsgs_V3()
        mm = msg.gripper_state.grippers_val * _Gripper0P001MM_TO_MM
        norm = float(np.clip(mm / self.cfg.gripper_open_mm, 0.0, 1.0))
        if self.cfg.gripper_invert:
            norm = 1.0 - norm
        return norm

    def read_ee(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ee_pos_m, ee_rot6d)`` from Piper FK, for the 32-dim FastWAM state.

        ``GetFK`` returns 6 joint poses; ``[-1]`` is the end effector
        ``[x, y, z (mm), rx, ry, rz (deg)]``. ee_pos is returned in meters and
        rot6d from the first two columns of the Euler rotation matrix.

        NOTE: the rot6d convention must match the dataset's
        ``convert_piper_hdf5_to_lerobot.py``; verify before relying on it.
        """
        if not self._fk_calibrated:
            self._iface.EnableFkCal()
            self._fk_calibrated = True
        ee = self._iface.GetFK("control")[-1]  # [x, y, z(mm), rx, ry, rz(deg)]
        xyz = np.asarray(ee[:3], dtype=np.float64) / 1000.0  # mm -> m
        rot6d = _euler_deg_to_rot6d(np.asarray(ee[3:6], dtype=np.float64))
        return xyz, rot6d

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def send_joint_command(self, joints_rad: np.ndarray, gripper_norm: float) -> None:
        """Send 6 joint targets (rad) + gripper (normalized [0,1]) to one arm."""
        joints_rad = np.asarray(joints_rad, dtype=np.float64).reshape(-1)
        if joints_rad.shape[0] != 6:
            raise ValueError(f"[{self.name}] expected 6 joints, got {joints_rad.shape[0]}")

        joints_rad = _clamp_joints(joints_rad, self.cfg.joint_limits_rad)
        raw = [int(round(r * _RAD_TO_0P001DEG)) for r in joints_rad]

        mm = float(np.clip(gripper_norm, 0.0, 1.0)) * self.cfg.gripper_open_mm
        if self.cfg.gripper_invert:
            mm = self.cfg.gripper_open_mm - mm
        raw_gripper = int(round(mm * _MM_TO_0P001MM))

        self._iface.MotionCtrl_2(_CTRL_CAN, _MOVE_J, self.cfg.move_spd_rate_ctrl, _POS_VEL)
        self._iface.JointCtrl(*raw)
        self._iface.GripperCtrl(raw_gripper, self.cfg.gripper_effort, 0x01, 0x00)


class CobotMagicPiper:
    """Dual-arm CobotMagic (Piper) robot."""

    def __init__(self, left: PiperArm, right: PiperArm):
        self.left = left
        self.right = right

    def connect(self) -> None:
        self.left.connect()
        self.right.connect()

    def disconnect(self) -> None:
        self.left.disconnect()
        self.right.disconnect()

    def safety_stop(self) -> None:
        self.left.emergency_stop()
        self.right.emergency_stop()

    def read_joint_gripper(self) -> np.ndarray:
        """Return the 14-dim joint+gripper vector ``[lj(6), lg, rj(6), rg]``."""
        return np.concatenate(
            [
                self.left.read_joints_rad(),
                [self.left.read_gripper_norm()],
                self.right.read_joints_rad(),
                [self.right.read_gripper_norm()],
            ]
        )

    def send_joint_command(
        self,
        left_joints: np.ndarray,
        left_gripper: float,
        right_joints: np.ndarray,
        right_gripper: float,
    ) -> None:
        self.left.send_joint_command(left_joints, left_gripper)
        self.right.send_joint_command(right_joints, right_gripper)


def build_robot(cfg: dict[str, Any]) -> CobotMagicPiper:
    """Build a :class:`CobotMagicPiper` from the ``robot`` block of the station config."""

    def arm(name: str) -> PiperArm:
        acfg = ArmConfig(
            can_port=cfg[name]["can_port"],
            can_id=cfg[name].get("can_id", 1),
            dh_is_offset=cfg.get("dh_is_offset", 1),
            move_spd_rate_ctrl=cfg.get("move_spd_rate_ctrl", 100),
            gripper_effort=cfg.get("gripper_effort", 1000),
            gripper_open_mm=cfg.get("gripper_open_mm", 100.0),
            gripper_invert=cfg.get("gripper_invert", False),
            enable_sdk_joint_limit=cfg.get("enable_sdk_joint_limit", True),
            enable_sdk_gripper_limit=cfg.get("enable_sdk_gripper_limit", True),
        )
        if name in cfg and "joint_limits_deg" in cfg[name]:
            acfg.joint_limits_rad = [
                (math.radians(lo), math.radians(hi)) for lo, hi in cfg[name]["joint_limits_deg"]
            ]
        return PiperArm(name, acfg)

    return CobotMagicPiper(left=arm("left"), right=arm("right"))
