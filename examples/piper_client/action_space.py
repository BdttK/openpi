"""Action / proprio layout for the bimanual Piper (CobotMagic) station.

The ``RoboSynChallenge/cobotmagic_Real_items_handover`` LeRobot dataset uses a
**32-dim** state and action for the dual-arm (ALOHA-style) robot. Per arm, 16
dims are laid out as::

    [0:6]   joint      (6)  -- radians
    [6]     gripper    (1)  -- normalized opening, mm/100 -> [0, 1]
    [7:10]  ee_pos     (3)  -- xyz        (unused by the pi05_piper policy)
    [10:16] ee_rot6d   (6)  --            (unused by the pi05_piper policy)

Left arm = indices ``[0:16]``, right arm = ``[16:32]``.

The ``pi05_piper`` policy only reads the 14 joint+gripper dims and emits a
**14-dim** action ``[left joints(6), left gripper, right joints(6), right gripper]``.
This module centralizes the indexing and unit layout so the driver and the
rollout loop agree.
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

# 14-dim joint+gripper action layout (the policy output).
ACTION_DIM = 14
# 32-dim observation layout (the policy input; ee dims zeroed).
PROPRIO_DIM = 2 * ARM_DIM


def split_action_14(action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Split a 14-dim action into ``(left_joints, left_gripper, right_joints, right_gripper)``."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} action dims, got {a.shape[0]}")
    return a[0:6], float(a[6]), a[7:13], float(a[13])


def build_proprio_32(
    left_joints: np.ndarray,
    left_gripper: float,
    right_joints: np.ndarray,
    right_gripper: float,
) -> np.ndarray:
    """Build the 32-dim observation. ee_pos / ee_rot6d are zeroed (unused by the policy)."""

    def arm(joints: np.ndarray, gripper: float) -> np.ndarray:
        v = np.zeros(ARM_DIM, dtype=np.float64)
        v[JOINT_SLICE] = joints
        v[GRIPPER_IDX] = gripper
        return v

    return np.concatenate([arm(left_joints, left_gripper), arm(right_joints, right_gripper)])
