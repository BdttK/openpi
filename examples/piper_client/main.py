"""Real-robot client for the openpi ``pi05_piper`` policy on the bimanual Piper.

Runs on the **Piper host** (not the GPU machine). Captures 3 cameras + dual-arm
joint/gripper state, streams them over WebSocket to the openpi policy server,
receives 14-dim action chunks, and drives the two Piper arms.

Self-contained: imports ``robot`` / ``cameras`` / ``action_space`` from this
directory (no external deployment checkout). The only external deps are the
lightweight ``openpi-client``, ``piper_sdk``, and a camera backend.

Quick start (on the Piper host):

  1. Install deps:
     cd openpi/packages/openpi-client && pip install -e .
     pip install piper_sdk python-can pyrealsense2 PyYAML

  2. Smoke test (no robot, no camera):
     python examples/piper_client/main.py --host <GPU_MACHINE_IP> --dry-run

  3. Real robot (edit config.yaml first):
     python examples/piper_client/main.py --host <GPU_MACHINE_IP> --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from action_space import PROPRIO_DIM, build_proprio_32, split_action_14
from cameras import CameraGroup, CameraGroupConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("piper_client")

# Task instruction. Override with --prompt.
PROMPT = "Pick up the pen, hand it over to the other arm and then place it in to the pen holder"


# --------------------------------------------------------------------------- #
# Observation building
# --------------------------------------------------------------------------- #
def _to_chw(img_hwc: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, 224, 224) uint8 [C, H, W] for the openpi policy."""
    resized = image_tools.resize_with_pad(np.asarray(img_hwc, dtype=np.uint8), 224, 224)
    return np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)


def build_observation(frames: dict, proprio_32: np.ndarray, prompt: str) -> dict:
    """Build the observation dict the ``pi05_piper`` policy expects."""
    return {
        "observation.images.cam_high": _to_chw(frames["cam_high"]),
        "observation.images.cam_left_wrist": _to_chw(frames["cam_left_wrist"]),
        "observation.images.cam_right_wrist": _to_chw(frames["cam_right_wrist"]),
        "observation.state": np.asarray(proprio_32, dtype=np.float32),
        "prompt": prompt,
    }


def read_proprio_32(robot) -> np.ndarray:
    """Read 14-dim joint+gripper from the robot and build the 32-dim observation."""
    jg = robot.read_joint_gripper()  # [lj(6), lg, rj(6), rg]
    lj, lg, rj, rg = split_action_14(jg)
    return build_proprio_32(lj, lg, rj, rg)


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
    step_s = 1.0 / max(1.0, control_hz)

    for ep in range(n_episodes):
        logger.info("=== episode %d/%d ===", ep + 1, n_episodes)
        pending: list = []  # queue of (lj, lg, rj, rg)

        for _ in range(max_steps):
            if not pending:
                frames = cameras.read()
                proprio = read_proprio_32(robot) if robot is not None else np.zeros(PROPRIO_DIM)
                action_chunk = client.infer(build_observation(frames, proprio, prompt))["actions"]
                for a in action_chunk[:exec_horizon]:
                    pending.append(split_action_14(a))

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
    p.add_argument("--config", default="config.yaml", help="station config (robot + cameras)")
    p.add_argument("--dry-run", action="store_true", help="no robot, no camera; zeros in / zeros out")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=300, help="max steps per episode")
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--exec-horizon", type=int, default=8, help="steps executed per policy query")
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()

    # Build hardware (or mocks for dry-run).
    robot = None
    if args.dry_run:
        cameras = CameraGroup(CameraGroupConfig(keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                                                backend="mock"))
        logger.info("dry-run: mock cameras, no robot")
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        cameras = CameraGroup(CameraGroupConfig(**cfg["cameras"]))
        from robot import build_robot

        robot = build_robot(cfg["robot"])
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
