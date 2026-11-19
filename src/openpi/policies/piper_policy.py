"""Policy for the bimanual Piper (ALOHA-style) robot.

Maps the 32-dim RoboSynChallenge `cobotmagic_Real_items_handover` state/action
to the 14-dim joint+gripper space used by the pi0.5 base model.

32-dim layout (per arm = 16 dims):

    [0:6]   left arm joints (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
    [6]     left gripper (0 = closed, 1 ~ 100 mm open)
    [7:10]  left ee_pos xyz
    [10:16] left ee_rot (rot6d)
    [16:22] right arm joints
    [22]    right gripper
    [23:26] right ee_pos xyz
    [26:32] right ee_rot (rot6d)

14-dim model layout (matches pi0.5 ALOHA pretraining):

    [0:6] left joints, [6] left gripper, [7:13] right joints, [13] right gripper.

The ee_pos / ee_rot dims are dropped: they are recoverable from the joint angles
via forward kinematics at runtime, so the policy only predicts joints + grippers.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

# Indices into the 32-dim vector that map to the 14-dim joint+gripper space.
_SELECT_14 = [0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 22]


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Inputs for the bimanual Piper policy.

    Expected inputs:
      - images: dict[name, img] with img in [channel, height, width]; names in EXPECTED_CAMERAS.
      - state: [32]
      - actions: [action_horizon, 32] (training only)
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        # Select the 14 joint+gripper dims out of the 32-dim state.
        state = np.asarray(data["state"], dtype=np.float32)[_SELECT_14]

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # [channel, height, width] -> [height, width, channel].
            return einops.rearrange(img, "c h w -> h w c")

        in_images = {name: convert_image(img) for name, img in data["images"].items()}
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        images = {
            "base_0_rgb": in_images["cam_high"],
            "left_wrist_0_rgb": in_images["cam_left_wrist"],
            "right_wrist_0_rgb": in_images["cam_right_wrist"],
        }
        image_masks = {name: np.True_ for name in images}

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)[:, _SELECT_14]  # [T, 32] -> [T, 14]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Outputs for the bimanual Piper policy (14-dim joint+gripper actions).

    The model outputs 32-dim actions (Pi0Config action_dim=32): the 14 real
    joint+gripper dims sit at [0:14], the rest are zero padding. Slice [0:14].
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)[:, :14]  # [T, 32] -> [T, 14]
        return {"actions": actions}
