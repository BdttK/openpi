"""Action / proprio layout for the bimanual Piper (CobotMagic) station.

The ``RoboSynChallenge/cobotmagic_Real_items_handover`` LeRobot dataset uses a
**32-dim** state and action for the dual-arm (ALOHA-style) robot. Per arm, 16
dims are laid out as::

    [0:6]   joint      (6)  -- radians
    [6]     gripper    (1)  -- normalized opening, mm/100 -> [0, 1]
    [7:10]  ee_pos     (3)  -- xyz
    [10:16] ee_rot6d   (6)

Left arm = indices ``[0:16]``, right arm = ``[16:32]``.

Two policies consume different spaces:

* ``pi05_piper`` (openpi): reads the 14 joint+gripper dims, emits a **14-dim**
  action ``[lj(6), lg, rj(6), rg]`` (ee dims dropped, can be zeroed on input).
* FastWAM (lerobot): consumes the full **32-dim** state (incl. ee from FK) and
  emits a **32-dim** action; the motors still only take the 14 joint+gripper dims.
"""

from __future__ import annotations

import numpy as np

ARM_DIM = 16
JOINT_DIM = 6
JOINT_SLICE = slice(0, 6)
GRIPPER_IDX = 6
EE_POS_SLICE = slice(7, 10)
EE_ROT6D_SLICE = slice(10, 16)
LEFT = slice(0, 16)
RIGHT = slice(16, 32)

# 14-dim joint+gripper layout (openpi pi05_piper action).
ACTION_DIM = 14
# 32-dim layout (dataset / FastWAM action).
ACTION_DIM_32 = 32


def split_action_14(action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Split a 14-dim action into ``(left_joints, left_gripper, right_joints, right_gripper)``."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} action dims, got {a.shape[0]}")
    return a[0:6], float(a[6]), a[7:13], float(a[13])


def split_action_32(action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Extract the 14 joint+gripper dims from a 32-dim action (FastWAM output)."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != ACTION_DIM_32:
        raise ValueError(f"expected {ACTION_DIM_32} action dims, got {a.shape[0]}")
    left, right = a[LEFT], a[RIGHT]
    return left[JOINT_SLICE], float(left[GRIPPER_IDX]), right[JOINT_SLICE], float(right[GRIPPER_IDX])


def build_proprio_32(
    left_joints: np.ndarray,
    left_gripper: float,
    right_joints: np.ndarray,
    right_gripper: float,
    *,
    left_ee_pos: np.ndarray | None = None,
    left_ee_rot6d: np.ndarray | None = None,
    right_ee_pos: np.ndarray | None = None,
    right_ee_rot6d: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 32-dim observation.

    ``ee_pos`` / ``ee_rot6d`` default to zeros (the openpi pi05_piper policy drops
    them). For FastWAM pass the real forward-kinematics values.
    """

    def arm(joints, gripper, ee_pos, ee_rot6d):
        v = np.zeros(ARM_DIM, dtype=np.float64)
        v[JOINT_SLICE] = joints
        v[GRIPPER_IDX] = gripper
        if ee_pos is not None:
            v[EE_POS_SLICE] = ee_pos
        if ee_rot6d is not None:
            v[EE_ROT6D_SLICE] = ee_rot6d
        return v

    return np.concatenate(
        [
            arm(left_joints, left_gripper, left_ee_pos, left_ee_rot6d),
            arm(right_joints, right_gripper, right_ee_pos, right_ee_rot6d),
        ]
    )
