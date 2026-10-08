"""Launch the official `lerobot-train` CLI with project defaults and record resources + reproducibility metadata.

The training loop itself is LeRobot's (lerobot.scripts.lerobot_train). This module only builds the command line from a
YAML config, selects train episodes from reports/split_manifest.json, monitors peak RAM (process tree) and VRAM
(nvidia-smi), and writes reports/experiment_<timestamp>.json.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
AGGR_ROOT = ROOT / "datasets" / "so101_11tasks"


def train_episodes(task_ids: list[str] | str) -> list[int]:
    split = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    ids = sorted(split["tasks"]) if task_ids == "all" else task_ids
    eps: list[int] = []
    for t in ids:
        s = split["tasks"][t]
        eps += [s["global_episode_offset"] + e for e in s["train"]]
    return sorted(eps)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    return str(v)


def build_command(cfg: dict, output_dir: Path) -> list[str]:
    args = {
        "dataset.repo_id": "local/so101_11tasks",
        "dataset.root": str(AGGR_ROOT),
        "dataset.episodes": train_episodes(cfg["tasks"]),
        "dataset.video_backend": cfg.get("video_backend", "pyav"),
        "output_dir": str(output_dir),
        "job_name": output_dir.name,
        "wandb.enable": cfg.get("wandb", False),
        "policy.push_to_hub": False,
        "save_checkpoint": True,
    }
    args.update(cfg.get("lerobot_args", {}))
    exe = str(Path(sys.executable).parent / "lerobot-train")
    return [exe] + [f"--{k}={_fmt(v)}" for k, v in args.items()]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def environment_info() -> dict:
    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "lerobot": version("lerobot"),
        "transformers": version("transformers"),
        "git_commit": _git_commit() or "not a git repository",
    }
    audit = json.loads((ROOT / "reports" / "dataset_audit.json").read_text())
    info["dataset_revisions"] = {k: v["revision"] for k, v in audit["datasets"].items()}
    prov = AGGR_ROOT / "meta" / "so101_evo1_provenance.json"
    info["aggregated_dataset"] = json.loads(prov.read_text()) if prov.exists() else None
    return info


class Monitor:
    def __init__(self, pid: int, interval: float = 1.0):
        self.pid, self.interval = pid, interval
        self.peak_rss_gb = 0.0
        self.peak_vram_gb = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _vram(self) -> float:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            return float(out.split()[0]) / 1024
        except Exception:  # noqa: BLE001
            return 0.0

    def _run(self):
        while not self._stop.is_set():
            try:
                p = psutil.Process(self.pid)
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                self.peak_rss_gb = max(self.peak_rss_gb, rss / 2**30)
            except psutil.Error:
                pass
            self.peak_vram_gb = max(self.peak_vram_gb, self._vram())
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def run(config_path: str, overrides: dict | None = None, dry_run: bool = False) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    for k, v in (overrides or {}).items():
        cfg.setdefault("lerobot_args", {})[k] = v
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / cfg.get("output_root", "outputs") / f"{cfg['name']}_{ts}"
    cmd = build_command(cfg, output_dir)
    record = {
        "timestamp": ts,
        "config_file": str(config_path),
        "config": cfg,
        "command": " ".join(shlex.quote(c) for c in cmd),
        "n_train_episodes": len(train_episodes(cfg["tasks"])),
        "environment": environment_info(),
    }
    if dry_run:
        print(record["command"])
        return record
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log = ROOT / cfg.get("output_root", "outputs") / f"{cfg['name']}_{ts}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        with Monitor(proc.pid) as mon:
            rc = proc.wait()
    record.update({
        "return_code": rc,
        "wall_time_s": round(time.time() - t0, 1),
        "peak_rss_gb_process_tree": round(mon.peak_rss_gb, 3),
        "peak_vram_used_gb_nvidia_smi": round(mon.peak_vram_gb, 3),
        "log": str(log.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
    })
    (ROOT / "reports" / f"experiment_{ts}.json").write_text(json.dumps(record, indent=2, default=str))
    return record
