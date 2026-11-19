"""openpi (pi05_piper) real-robot client for the bimanual Piper station.

Runs on the **Piper host** (not the GPU machine). Captures 3 cameras + dual-arm
joint/gripper state, streams them over WebSocket to the openpi policy server,
receives 14-dim action chunks, and executes them on the two Piper arms.

This reuses the hardware drivers from the FastWAM-RL checkout
(``deployment/robots/cobotmagic_piper.py``, ``deployment/cameras/camera_group.py``,
``deployment/action_space.py``), passed in via ``--deployment-dir``. The only
policy-side dependency is the lightweight ``openpi-client`` package.

Quick start (on the Piper host):

  1. Install the lightweight client (no JAX / torch needed):
     cd openpi/packages/openpi-client && pip install -e .

  2. Smoke test (no robot, no camera, zeros in / zeros out):
     python examples/piper_client/main.py --host <GPU_MACHINE_IP> --dry-run

  3. Real robot:
     python examples/piper_client/main.py --host <GPU_MACHINE_IP> \
         --deployment-dir /path/to/FastWAM-RL/deployment \
         --config /path/to/cobotmagic_real.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("piper_client")

# Task instruction. Override with --prompt.
PROMPT = "Pick up the pen, hand it over to the other arm and then place it in to the pen holder"

# Model output is 14-dim: [left joints(6), left gripper(1), right joints(6), right gripper(1)].
ACTION_DIM = 14
# Dataset proprio layout is 32-dim (16 per arm). The policy only reads the
# joint+gripper dims (indices 0..6 and 16..22); ee_pos/ee_rot6d are dropped,
# so we fill them with zeros and skip forward kinematics.
PROPRIO_DIM = 32


# --------------------------------------------------------------------------- #
# Observation building
# --------------------------------------------------------------------------- #
def _to_chw(img_hwc: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, 224, 224) uint8 [C, H, W] for the openpi policy."""
    resized = image_tools.resize_with_pad(np.asarray(img_hwc, dtype=np.uint8), 224, 224)
    return np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)


def build_observation(frames: dict, proprio_32: np.ndarray, prompt: str) -> dict:
    """Build the observation dict the pi05_piper policy expects."""
    return {
        "observation.images.cam_high": _to_chw(frames["cam_high"]),
        "observation.images.cam_left_wrist": _to_chw(frames["cam_left_wrist"]),
        "observation.images.cam_right_wrist": _to_chw(frames["cam_right_wrist"]),
        "observation.state": np.asarray(proprio_32, dtype=np.float32),
        "prompt": prompt,
    }


def read_proprio_32(robot) -> np.ndarray:
    """32-dim proprio with joints+gripper, ee_pos/ee_rot6d zeroed (unused by policy)."""
    from action_space import build_proprio

    zeros3 = np.zeros(3, dtype=np.float64)
    zeros6 = np.zeros(6, dtype=np.float64)
    return build_proprio(
        robot.left.read_joints_rad(), robot.left.read_gripper_norm(), zeros3, zeros6,
        robot.right.read_joints_rad(), robot.right.read_gripper_norm(), zeros3, zeros6,
    )


# --------------------------------------------------------------------------- #
# Rollout loop
# --------------------------------------------------------------------------- #
def rollout(
    client,
    robot,
    cameras,
    *,
    prompt: str,
    n_episodes: int,
    max_steps: int,
    control_hz: float,
    exec_horizon: int,
) -> None:
    """Run episodes with standard action-chunking: query -> execute N steps -> query."""
    from action_space import extract_joint_command

    step_s = 1.0 / max(1.0, control_hz)

    for ep in range(n_episodes):
        logger.info("=== episode %d/%d ===", ep + 1, n_episodes)
        pending: list = []  # queue of (lj, lg, rj, rg)

        for _ in range(max_steps):
            # Refill the queue when empty (re-query the policy server).
            if not pending:
                frames = cameras.read()
                proprio = read_proprio_32(robot) if robot is not None else np.zeros(PROPRIO_DIM)
                obs = build_observation(frames, proprio, prompt)
                action_chunk = client.infer(obs)["actions"]  # (action_horizon, 14)
                for a in action_chunk[:exec_horizon]:
                    pending.append(extract_joint_command(a, ACTION_DIM))

            lj, lg, rj, rg = pending.pop(0)

            if robot is not None:
                robot.send_joint_command(lj, lg, rj, rg)
            else:
                logger.info("dry-run action: lj=%s lg=%.3f rj=%s rg=%.3f",
                            np.round(lj, 3), lg, np.round(rj, 3), rg)

            time.sleep(step_s)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost", help="policy server host (GPU machine IP)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None, help="optional policy server API key")
    p.add_argument("--deployment-dir", default=None,
                   help="path to FastWAM-RL/deployment (for robots/ + cameras/ + action_space)")
    p.add_argument("--config", default=None,
                   help="YAML with `robot` and `cameras` blocks (same format as rollout_cobotmagic.py)")
    p.add_argument("--dry-run", action="store_true", help="no robot, no camera; zeros in / zeros out")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=300, help="max steps per episode")
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--exec-horizon", type=int, default=8, help="steps to execute per policy query")
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()

    # Resolve the FastWAM-RL deployment drivers.
    deployment_dir = Path(args.deployment_dir).resolve() if args.deployment_dir else None
    if deployment_dir is not None:
        sys.path.insert(0, str(deployment_dir))

    # Build hardware (or mocks for dry-run).
    robot = None
    cameras = None
    if args.dry_run:
        from cameras.camera_group import CameraGroup, CameraGroupConfig

        cameras = CameraGroup(CameraGroupConfig(keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                                                backend="mock"))
        logger.info("dry-run: mock cameras, no robot")
    else:
        if deployment_dir is None or args.config is None:
            p.error("--deployment-dir and --config are required unless --dry-run")
        import yaml

        from cameras.camera_group import build_camera_group
        from robots.cobotmagic_piper import build_robot

        cfg = yaml.safe_load(Path(args.config).read_text())
        cameras = build_camera_group(cfg["cameras"])
        robot = build_robot(cfg["robot"], PROPRIO_DIM)
        robot.connect()
        logger.info("robot + cameras connected")

    cameras.connect()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port,
                                                           api_key=args.api_key)
    logger.info("server metadata: %s", client.get_server_metadata())

    try:
        rollout(
            client,
            robot,
            cameras,
            prompt=args.prompt,
            n_episodes=args.episodes,
            max_steps=args.steps,
            control_hz=args.control_hz,
            exec_horizon=args.exec_horizon,
        )
    finally:
        cameras.disconnect()
        if robot is not None:
            robot.safety_stop()
            robot.disconnect()
        logger.info("done.")


if __name__ == "__main__":
    main()
