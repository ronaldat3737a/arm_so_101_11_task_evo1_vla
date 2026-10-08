"""Episode-level train/val/test split per task, and the global episode indices after aggregation."""

from __future__ import annotations

import numpy as np


def split_task_episodes(
    num_episodes: int, excluded: list[int], n_val: int, n_test: int, n_train: int | None, seed: int
) -> dict[str, list[int]]:
    """Split one task's episodes. Excluded episodes are never used. Test and val are taken first so that
    they keep their full size when a task has fewer usable episodes; train gets the rest (capped at n_train)."""
    bad = set(excluded)
    unknown = bad - set(range(num_episodes))
    if unknown:
        raise ValueError(f"excluded episodes {sorted(unknown)} not in 0..{num_episodes - 1}")
    usable = [e for e in range(num_episodes) if e not in bad]
    if len(usable) < n_val + n_test + 1:
        raise ValueError(f"only {len(usable)} usable episodes, need > {n_val + n_test}")
    order = np.random.default_rng(seed).permutation(usable).tolist()
    test = sorted(order[:n_test])
    val = sorted(order[n_test : n_test + n_val])
    train = order[n_test + n_val :]
    unused = []
    if n_train is not None and len(train) > n_train:
        unused = sorted(train[n_train:])
        train = train[:n_train]
    return {"train": sorted(train), "val": val, "test": test, "excluded": sorted(bad), "unused": unused}


def episode_offsets(num_episodes_per_dataset: list[int]) -> list[int]:
    """Global index of each dataset's episode 0 when datasets are concatenated in order
    (as lerobot.datasets.aggregate.aggregate_datasets does)."""
    return np.concatenate([[0], np.cumsum(num_episodes_per_dataset)[:-1]]).astype(int).tolist()


def check_disjoint(splits: dict[str, list[int]]) -> None:
    names = ["train", "val", "test"]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = set(splits[a]) & set(splits[b])
            if inter:
                raise AssertionError(f"{a} and {b} share episodes {sorted(inter)}")
