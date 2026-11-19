"""Split cobotmagic_Real_items_handover (60 eps) into train (50) / val (10).

Writes two LeRobot datasets (readable via `repo_id` + HF_LEROBOT_HOME, which is
how openpi's data config resolves datasets):
  - RoboSynChallenge/cobotmagic_Real_items_handover_train  (50 episodes)
  - RoboSynChallenge/cobotmagic_Real_items_handover_val    (10 episodes)

The split is deterministic (seed=42); re-running reproduces the same split.

Usage:
  export HF_LEROBOT_HOME=~/robot-ws/data
  python scripts/split_cobotmagic_50_10.py

NOTE: the episode-filtering / save API depends on your installed `lerobot`
version. If `push_to_hub` or the `episodes=` kwarg is unavailable, split the
underlying `meta/info.json` + parquet by episode index instead (60 episodes,
so 10/50 is trivial to do manually).
"""

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC = "RoboSynChallenge/cobotmagic_Real_items_handover"
DST_TRAIN = "RoboSynChallenge/cobotmagic_Real_items_handover_train"
DST_VAL = "RoboSynChallenge/cobotmagic_Real_items_handover_val"
SEED = 42
N_VAL = 10


def main() -> None:
    ds = LeRobotDataset(SRC)
    episodes = sorted(ds.meta.episodes.keys())
    assert len(episodes) == 60, f"expected 60 episodes, got {len(episodes)}"

    rng = np.random.default_rng(SEED)
    order = episodes.copy()
    rng.shuffle(order)

    val_eps = sorted(order[:N_VAL])
    train_eps = sorted(order[N_VAL:])

    for eps, dst in ((train_eps, DST_TRAIN), (val_eps, DST_VAL)):
        subset = LeRobotDataset(SRC, episodes=eps)  # episode filter (lerobot >= 2.1)
        subset.push_to_hub(dst, private=True)
        print(f"{dst}: {len(eps)} episodes, first 5 = {eps[:5]}")

    print(f"\nSplit done (seed={SEED}): train={len(train_eps)} val={len(val_eps)}")


if __name__ == "__main__":
    main()
