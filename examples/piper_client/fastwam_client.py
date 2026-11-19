"""FastWAM (lerobot) remote client for the bimanual Piper station.

Runs on the **Piper host**. Captures 3 cameras + dual-arm state (incl. FK ee
pose), sends them to a FastWAM WebSocket serve, receives 32-dim actions, and
drives the two arms with the joint+gripper dims.

Reuses ``robot.py`` / ``cameras.py`` / ``action_space.py`` from this directory.
The FastWAM policy is stateful (action chunking inside ``select_action``), so the
client calls ``reset`` once per episode and ``select_action`` every control step.

Usage (Piper host):
  python examples/piper_client/fastwam_client.py --host <GPU_MACHINE_IP> --dry-run
  python examples/piper_client/fastwam_client.py --host <GPU_MACHINE_IP> --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml

import msgpack
import msgpack_numpy
from websockets.sync.client import connect

from action_space import ACTION_DIM_32, build_proprio_32, split_action_32
from cameras import CameraGroup, CameraGroupConfig, concat_cameras

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fastwam_client")

PROMPT = "Pick up the pen, hand it over to the other arm and then place it in to the pen holder"


class FastWAMClient:
    """Thin WebSocket wrapper around the FastWAM serve."""

    def __init__(self, host: str, port: int):
        self._ws = connect(f"ws://{host}:{port}")

    def reset(self) -> None:
        self._ws.send(msgpack.packb({"type": "reset"}))
        self._ws.recv()

    def select_action(self, image_chw: np.ndarray, state: np.ndarray, task: str, seed: int = 0) -> np.ndarray:
        self._ws.send(
            msgpack.packb(
                {
                    "type": "select_action",
                    "input_image": np.asarray(image_chw, dtype=np.uint8),  # (C, H, 3W)
                    "state": np.asarray(state, dtype=np.float32),          # (32,)
                    "task": task,
                    "seed": seed,
                }
            )
        )
        resp = msgpack.unpackb(self._ws.recv(), raw=False)
        if "error" in resp:
            raise RuntimeError(f"serve error: {resp['error']}")
        return np.asarray(resp["action"], dtype=np.float64)  # (32,)

    def close(self) -> None:
        self._ws.close()


def read_state_32(robot) -> np.ndarray:
    """Read joints+gripper+FK ee and build the 32-dim state (FastWAM input)."""
    lj = robot.left.read_joints_rad()
    lg = robot.left.read_gripper_norm()
    lee_pos, lee_rot6d = robot.left.read_ee()
    rj = robot.right.read_joints_rad()
    rg = robot.right.read_gripper_norm()
    ree_pos, ree_rot6d = robot.right.read_ee()
    return build_proprio_32(
        lj, lg, rj, rg,
        left_ee_pos=lee_pos, left_ee_rot6d=lee_rot6d,
        right_ee_pos=ree_pos, right_ee_rot6d=ree_rot6d,
    )


def rollout(
    client: FastWAMClient,
    robot,
    cameras,
    *,
    prompt: str,
    n_episodes: int,
    max_steps: int,
    control_hz: float,
) -> None:
    """Run episodes. The policy does its own action-chunking via select_action."""
    step_s = 1.0 / max(1.0, control_hz)

    for ep in range(n_episodes):
        logger.info("=== episode %d/%d ===", ep + 1, n_episodes)
        client.reset()

        for _ in range(max_steps):
            frames = cameras.read()
            concat = concat_cameras(frames, cameras.cfg.keys)  # (H, 3W, 3)
            img_chw = np.asarray(concat, dtype=np.uint8).transpose(2, 0, 1)  # (C, H, 3W)

            state = read_state_32(robot) if robot is not None else np.zeros(ACTION_DIM_32)

            action = client.select_action(img_chw, state, prompt)  # (32,)
            lj, lg, rj, rg = split_action_32(action)

            if robot is not None:
                robot.send_joint_command(lj, lg, rj, rg)
            else:
                logger.info("dry-run action: lj=%s lg=%.3f rj=%s rg=%.3f",
                            np.round(lj, 3), lg, np.round(rj, 3), rg)

            time.sleep(step_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost", help="FastWAM serve host (GPU machine IP)")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--config", default="config.yaml", help="station config (robot + cameras)")
    p.add_argument("--dry-run", action="store_true", help="no robot, no camera; zeros in / zeros out")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=300, help="max steps per episode")
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()

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

    client = FastWAMClient(args.host, args.port)
    logger.info("connected to FastWAM serve at %s:%d", args.host, args.port)

    try:
        rollout(
            client,
            robot,
            cameras,
            prompt=args.prompt,
            n_episodes=args.episodes,
            max_steps=args.steps,
            control_hz=args.control_hz,
        )
    finally:
        cameras.disconnect()
        if robot is not None:
            robot.safety_stop()
            robot.disconnect()
        client.close()
        logger.info("done.")


if __name__ == "__main__":
    main()
