#!/usr/bin/env python
"""Normalization statistics from TRAIN episodes only, for the merged dataset.

Evo-1 (lerobot.policies.evo1) normalizes STATE/ACTION with MIN_MAX using `dataset.meta.stats`
(= <root>/meta/stats.json), and the stats are then frozen into the checkpoint's preprocessor. aggregate_datasets
wrote stats over ALL episodes (including val/test), which leaks held-out episodes into normalization.

This script:
  1. reads the per-episode stats LeRobot already stored in meta/episodes/*.parquet (`stats/<feature>/<stat>`)
  2. aggregates the TRAIN episodes only with the official `lerobot.datasets.compute_stats.aggregate_stats`
  3. verifies min/max/mean/std of action and state against a direct recomputation from the train frames
  4. writes meta/stats.json (train-only) with the official writer; all-episode stats stay in meta/stats_all_episodes.json
  5. writes reports/norm_stats.json + reports/norm_stats.md

No custom normalization scheme: same keys and stat types LeRobot produces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AGGR_ROOT = ROOT / "datasets" / "so101_11tasks"
VECTOR_KEYS = ("action", "observation.state")


def per_episode_stats(root: Path) -> dict[int, dict]:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    out: dict[int, dict] = {}
    for f in files:
        tbl = pq.read_table(f)
        cols = [c for c in tbl.column_names if c.startswith("stats/")]
        rows = tbl.select(["episode_index", *cols]).to_pylist()
        for r in rows:
            st: dict = {}
            for c in cols:
                _, feat_stat = c.split("/", 1)
                feat, stat = feat_stat.rsplit("/", 1)
                if r[c] is None:
                    continue
                st.setdefault(feat, {})[stat] = np.asarray(r[c], dtype=np.float64 if stat != "count" else np.int64)
            out[int(r["episode_index"])] = st
    return out


def main() -> int:
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    split = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    train = sorted(split["global"]["train"])
    held_out = set(split["global"]["val"]) | set(split["global"]["test"]) | set(split["global"]["excluded"])
    assert not set(train) & held_out

    ep_stats = per_episode_stats(AGGR_ROOT)
    missing = [e for e in train if e not in ep_stats]
    if missing:
        print(f"STOP: per-episode stats missing for train episodes {missing[:10]}")
        return 1
    # Drop bookkeeping columns: LeRobot normalizes only features; keep the same feature set as stats.json.
    all_stats = json.loads((AGGR_ROOT / "meta" / "stats_all_episodes.json").read_text())
    keys = sorted(all_stats)
    train_stats = aggregate_stats([{k: v for k, v in ep_stats[e].items() if k in keys} for e in train])

    # verification against raw train frames (vector features only)
    import pyarrow as pa

    tbls = [pq.read_table(f, columns=["episode_index", *VECTOR_KEYS]) for f in sorted((AGGR_ROOT / "data").glob("chunk-*/file-*.parquet"))]
    data = pa.concat_tables(tbls)
    ep = np.asarray(data.column("episode_index").to_pylist())
    mask = np.isin(ep, train)
    verification = {}
    ok = True
    for k in VECTOR_KEYS:
        x = np.asarray(data.column(k).to_pylist(), dtype=np.float64)[mask]
        ref = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0)}
        res = {}
        for s, v in ref.items():
            d = float(np.abs(np.asarray(train_stats[k][s]).reshape(-1) - v).max())
            res[s] = d
            ok &= d < (1e-3 if s in ("min", "max") else 1e-2)
        res["count"] = int(np.asarray(train_stats[k]["count"]).reshape(-1)[0])
        ok &= res["count"] == int(mask.sum())
        verification[k] = res
    if not ok:
        print("STOP: train-only stats do not match recomputation:", json.dumps(verification, indent=2))
        return 1

    write_stats(train_stats, AGGR_ROOT)

    def tolist(d):
        return {k: {s: np.asarray(v).tolist() for s, v in st.items()} for k, st in d.items()}

    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    report = {
        "source": "per-episode stats in meta/episodes aggregated with lerobot.datasets.compute_stats.aggregate_stats",
        "n_train_episodes": len(train),
        "n_train_frames": int(mask.sum()),
        "held_out_episodes_excluded": len(held_out),
        "normalization_used_by_evo1": {"STATE": "MIN_MAX", "ACTION": "MIN_MAX", "VISUAL": "IDENTITY"},
        "note_quantiles": "q01..q99 are count-weighted averages of per-episode quantiles (LeRobot aggregate_stats); Evo-1 MIN_MAX uses min/max only.",
        "verification_max_abs_diff_vs_raw_frames": verification,
        "stats_file": str((AGGR_ROOT / "meta" / "stats.json").relative_to(ROOT)),
        "all_episode_stats_file": str((AGGR_ROOT / "meta" / "stats_all_episodes.json").relative_to(ROOT)),
        "train_stats": {k: v for k, v in tolist(train_stats).items() if k in VECTOR_KEYS},
        "all_episode_stats": {k: {s: all_stats[k][s] for s in ("min", "max", "mean", "std")} for k in VECTOR_KEYS},
    }
    (ROOT / "reports" / "norm_stats.json").write_text(json.dumps(report, indent=2))
    L = ["# Normalization stats (train episodes only)\n",
         f"Train episodes: {len(train)}, frames: {int(mask.sum())}. Held-out (val/test/excluded) episodes not used: {len(held_out)}.\n",
         "Evo-1 uses MIN_MAX for state/action (lerobot.policies.evo1 default), so min/max define [-1, 1].\n",
         "Verification vs. direct recomputation from train frames (max abs diff): " + json.dumps({k: {s: f'{v:.2e}' if isinstance(v, float) else v for s, v in r.items()} for k, r in verification.items()}) + "\n"]
    for k in VECTOR_KEYS:
        L += [f"\n## {k}\n", "| joint | train min | train max | train mean | train std | all-episode min | all-episode max |", "|---|---:|---:|---:|---:|---:|---:|"]
        ts = report["train_stats"][k]
        al = report["all_episode_stats"][k]
        for j, n in enumerate(joints):
            L.append(f"| {n} | {ts['min'][j]:.2f} | {ts['max'][j]:.2f} | {ts['mean'][j]:.2f} | {ts['std'][j]:.2f} | {al['min'][j]:.2f} | {al['max'][j]:.2f} |")
    (ROOT / "reports" / "norm_stats.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
