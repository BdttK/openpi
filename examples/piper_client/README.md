# Piper client for the `pi05_piper` policy

Runs on the **Piper host** (not the GPU machine). It captures 3 cameras + the
dual-arm joint/gripper state, streams them over WebSocket to an openpi policy
server, receives 14-dim action chunks, and drives the two Piper arms.

```
Piper host (this script)          GPU machine (training server / >=24GB GPU)
─────────────────────────────     ───────────────────────────────────────────
3x RealSense + 2x Piper           uv run scripts/serve_policy.py policy:checkpoint \
   │  obs (3 imgs + 32-dim)             --policy.config=pi05_piper \
   │ ───────────────────────────▶       --policy.dir=.../<step>
   │  actions (action_horizon, 14)
   ◀───────────────────────────
```

## Files (self-contained)

| File | Role |
|---|---|
| `main.py` | entry point: WebSocket client + rollout loop |
| `robot.py` | dual-arm Piper driver (`piper_sdk` V3) |
| `cameras.py` | camera group (mock / opencv / realsense) |
| `action_space.py` | 14-dim / 32-dim action & proprio layout |
| `config.yaml` | station config (CAN ports, camera serials, limits) |

## Dependencies (Piper host)

```bash
# lightweight openpi client (no JAX / torch)
cd openpi/packages/openpi-client && pip install -e .

# robot + camera + config
pip install piper_sdk python-can pyrealsense2 PyYAML
```

## Usage

**Smoke test first** (no robot, no camera — verifies the WebSocket + policy round-trip):

```bash
python examples/piper_client/main.py --host <GPU_MACHINE_IP> --dry-run
```

**Real robot** (edit `config.yaml` first):

```bash
python examples/piper_client/main.py --host <GPU_MACHINE_IP> --config config.yaml
```

Key flags: `--episodes`, `--steps`, `--control-hz` (default 10, matches dataset),
`--exec-horizon` (steps executed per query, default 8), `--prompt`.

## Conventions (must match the `pi05_piper` training data)

| Item | Convention |
|---|---|
| Observation `observation.state` | **32-dim**: per arm `[6 joint(rad), 1 gripper(0..1), 3 ee_pos, 6 rot6d]`. The policy only reads joint+gripper, so ee dims are sent as zeros (no FK). |
| Cameras | `cam_high` + `cam_left_wrist` + `cam_right_wrist`, each `[C,H,W]` uint8, resized to 224x224 with pad |
| Action `actions` | **14-dim**: `[left 6 joints(rad), left gripper(0..1), right 6 joints(rad), right gripper(0..1)]` |
| Joints | radians (the driver converts to Piper's 0.001-deg internally) |
| Gripper | normalized 0..1 (`mm / 100`); the driver converts to Piper's 0.001-mm |

## First-run verification checklist

1. Start the policy server on the GPU machine (see `serve_policy.py`).
2. Run `--dry-run` here and confirm the server logs inference + this script prints actions.
3. Before enabling the arms, send one fixed observation and print `action_chunk[0]`; compare
   against the current joint angles to confirm **absolute vs delta** convention (the server
   applies `AbsoluteActions`, so you should get absolute joint targets).
4. Then enable the arms, keep `--exec-horizon` small, and keep the e-stop reachable.
