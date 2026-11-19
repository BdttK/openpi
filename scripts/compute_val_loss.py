"""Compute validation loss for a fine-tuned openpi model.

Loads a training checkpoint, runs the model in eval mode over the val split
(`RoboSynChallenge/cobotmagic_Real_items_handover_val`), and prints the mean loss.

Usage:
  uv run scripts/compute_val_loss.py pi05_piper \
      --checkpoint checkpoints/pi05_piper/<exp>/<step> \
      --val-repo-id RoboSynChallenge/cobotmagic_Real_items_handover_val \
      --num-batches 20

Run this on a few saved checkpoints (e.g. step 200/400/600/800/1000) and pick
the one with the lowest val loss — this is your early-stopping signal.
"""

import argparse
import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", help="e.g. pi05_piper")
    parser.add_argument("--checkpoint", required=True, help="checkpoint step dir, e.g. checkpoints/pi05_piper/<exp>/<step>")
    parser.add_argument("--val-repo-id", default="RoboSynChallenge/cobotmagic_Real_items_handover_val")
    parser.add_argument("--num-batches", type=int, default=20)
    args = parser.parse_args()

    config = _config.get_config(args.config_name)

    # Point the data pipeline at the val split instead of the train split.
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, repo_id=args.val_repo_id),
    )

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False)

    model = config.model.load(_model.restore_params(pathlib.Path(args.checkpoint) / "params", dtype=jnp.bfloat16))

    rng = jax.random.key(0)
    losses = []
    for i, batch in enumerate(data_loader):
        if i >= args.num_batches:
            break
        observation, actions = batch
        loss = model.compute_loss(rng, observation, actions, train=False)
        losses.append(float(jax.device_get(jnp.mean(loss))))

    print(f"val loss (mean over {len(losses)} batches): {np.mean(losses):.4f}")


if __name__ == "__main__":
    main()
