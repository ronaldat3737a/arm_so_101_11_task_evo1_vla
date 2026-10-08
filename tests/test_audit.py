"""Unit tests for src/data/audit.py on a synthetic LeRobot v3.0 snapshot (no network)."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.data.audit import ERROR, EXCLUDED, audit_snapshot, cross_dataset_checks

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
FPS = 30


def make_snapshot(tmp: Path, lengths=(40, 50, 45), task="Pick up the red cube.", static_ep=None, nan_ep=None):
    (tmp / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (tmp / "data" / "chunk-000").mkdir(parents=True)
    cam = {"dtype": "video", "shape": [360, 640, 3], "names": ["height", "width", "channels"],
           "info": {"video.height": 360, "video.width": 640, "video.codec": "av1", "video.pix_fmt": "yuv420p", "video.fps": FPS}}
    info = {
        "codebase_version": "v3.0", "fps": FPS, "robot_type": "so_follower",
        "total_episodes": len(lengths), "total_frames": int(sum(lengths)), "total_tasks": 1,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [6], "names": JOINTS},
            "observation.state": {"dtype": "float32", "shape": [6], "names": JOINTS},
            "observation.images.top": cam,
            "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (tmp / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({"task_index": [0], "task": [task]}), tmp / "meta" / "tasks.parquet")

    rng = np.random.default_rng(0)
    rows = {k: [] for k in ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]}
    eps = []
    start, t_video = 0, 0.0
    for e, n in enumerate(lengths):
        traj = np.cumsum(rng.normal(0, 2.0, size=(n, 6)), axis=0)
        if e == static_ep:
            traj = np.zeros((n, 6))
        state = np.vstack([traj[:1]] * 4 + [traj[:-4]])[:n]  # follower lags the target by 4 frames
        if e == nan_ep:
            state[3, 2] = np.nan
        rows["action"] += traj.astype(np.float32).tolist()
        rows["observation.state"] += state.astype(np.float32).tolist()
        rows["timestamp"] += (np.arange(n) / FPS).astype(np.float32).tolist()
        rows["frame_index"] += list(range(n))
        rows["episode_index"] += [e] * n
        rows["index"] += list(range(start, start + n))
        rows["task_index"] += [0] * n
        eps.append({
            "episode_index": e, "tasks": [task], "length": n, "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": start, "dataset_to_index": start + n,
            "videos/observation.images.top/chunk_index": 0, "videos/observation.images.top/file_index": 0,
            "videos/observation.images.top/from_timestamp": t_video,
            "videos/observation.images.top/to_timestamp": t_video + n / FPS,
        })
        start += n
        t_video += n / FPS
    pq.write_table(pa.Table.from_pylist(eps), tmp / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    tbl = pa.table({
        "action": pa.array(rows["action"], type=pa.list_(pa.float32(), 6)),
        "observation.state": pa.array(rows["observation.state"], type=pa.list_(pa.float32(), 6)),
        **{k: rows[k] for k in ["timestamp", "frame_index", "episode_index", "index", "task_index"]},
    })
    pq.write_table(tbl, tmp / "data" / "chunk-000" / "file-000.parquet")
    a = np.asarray(rows["action"], dtype=np.float64)
    s = np.asarray(rows["observation.state"], dtype=np.float64)
    stats = {k: {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(), "std": v.std(0).tolist()}
             for k, v in [("action", a), ("observation.state", s)]}
    stats["observation.images.top"] = {"mean": [[[0.5]]] * 3}
    (tmp / "meta" / "stats.json").write_text(json.dumps(stats))
    repo_files = {"videos/observation.images.top/chunk-000/file-000.mp4": 10_000_000}
    return tmp, repo_files


def codes(res, severity=None):
    return {i["code"] for i in res["issues"] if severity is None or i["severity"] == severity}


def test_clean_snapshot_has_no_errors(tmp_path):
    root, files = make_snapshot(tmp_path)
    res = audit_snapshot(root, files)
    assert codes(res, ERROR) == set()
    assert res["num_episodes"] == 3
    assert res["num_frames"] == 135
    assert res["fps"] == FPS
    assert res["features"]["vectors"]["action"]["shape"] == [6]
    assert res["features"]["cameras"]["observation.images.top"]["height"] == 360
    assert res["tasks"] == {"0": "Pick up the red cube."}
    assert res["action_state_lag"]["best_lag"] == 4  # synthetic follower lag recovered
    assert all(v == "ok" for v in res["stats_check"]["action"].values())


def test_static_episode_detected(tmp_path):
    root, files = make_snapshot(tmp_path, static_ep=1)
    res = audit_snapshot(root, files)
    assert res["static_episodes"] == [1]
    assert "static_episode" in codes(res, EXCLUDED)
    assert codes(res, ERROR) == set()  # no-op episodes are not integrity errors


def test_nan_detected(tmp_path):
    root, files = make_snapshot(tmp_path, nan_ep=0)
    res = audit_snapshot(root, files)
    assert res["missing_values"]["observation.state"]["nan"] == 1
    assert "nan_inf" in codes(res, ERROR)


def test_missing_video_file_detected(tmp_path):
    root, _ = make_snapshot(tmp_path)
    res = audit_snapshot(root, repo_files={})
    assert "video_file_missing" in codes(res, ERROR)


def test_total_frames_mismatch_detected(tmp_path):
    root, files = make_snapshot(tmp_path)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["total_frames"] += 1
    (root / "meta" / "info.json").write_text(json.dumps(info))
    assert "total_frames_mismatch" in codes(audit_snapshot(root, files), ERROR)


def test_cross_dataset_consistency(tmp_path):
    r1, f1 = make_snapshot(tmp_path / "a", task="task A")
    r2, f2 = make_snapshot(tmp_path / "b", task="task B")
    res = {"a": audit_snapshot(r1, f1), "b": audit_snapshot(r2, f2)}
    out = cross_dataset_checks(res, JOINTS, "so_follower")
    assert all(c["consistent"] for c in out["summary"].values() if isinstance(c, dict) and "consistent" in c)
    assert not [i for i in out["issues"] if i["severity"] == ERROR]


def test_cross_dataset_shared_instruction_is_error(tmp_path):
    r1, f1 = make_snapshot(tmp_path / "a", task="same")
    r2, f2 = make_snapshot(tmp_path / "b", task="same")
    out = cross_dataset_checks({"a": audit_snapshot(r1, f1), "b": audit_snapshot(r2, f2)}, JOINTS, "so_follower")
    assert any(i["code"] == "shared_instruction" for i in out["issues"])


def test_wrong_joint_names_is_error(tmp_path):
    r1, f1 = make_snapshot(tmp_path / "a")
    res = {"a": audit_snapshot(r1, f1)}
    out = cross_dataset_checks(res, list(reversed(JOINTS)), "so_follower")
    assert any(i["code"] == "joint_names" for i in out["issues"])


AUDIT = ROOT / "reports" / "dataset_audit.json"


@pytest.mark.skipif(not AUDIT.exists(), reason="run scripts/inspect_datasets.py first")
def test_tasks_yaml_matches_audit():
    """configs/tasks.yaml instructions/exclusions must equal what the audit found in the datasets."""
    audit = json.loads(AUDIT.read_text())
    tasks = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["tasks"]
    repos = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())["repos"]
    assert [t["dataset"] for t in tasks] == repos
    assert [t["task_id"] for t in tasks] == [f"task_{i:02d}" for i in range(1, 12)]
    for t in tasks:
        a = audit["datasets"][t["dataset"]]
        assert list(a["tasks"].values()) == [t["instruction"]]
        assert sorted(a["static_episodes"]) == sorted(t["excluded_episodes"])
    assert len({t["instruction"] for t in tasks}) == 11
