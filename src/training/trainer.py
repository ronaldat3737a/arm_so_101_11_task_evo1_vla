"""Thin training loop around official LeRobot components, adding what lerobot-train 0.6.1 lacks:
gradient accumulation, per-step RAM/VRAM logging, val-loss model selection and hooks for selective fine-tuning.

Everything model-related is official:
  - dataset:     lerobot.datasets.lerobot_dataset.LeRobotDataset (lazy video decoding, episode subset)
  - policy:      lerobot.policies.factory.make_policy (e.g. policy.type=evo1 -> Evo1Policy, unchanged)
  - processors:  lerobot.policies.factory.make_pre_post_processors(dataset_stats=train-only meta/stats.json)
  - optimizer / scheduler: the policy config's own presets (get_optimizer_preset / get_scheduler_preset)
  - checkpoints: policy.save_pretrained + processor.save_pretrained (LeRobot format, loadable with from_pretrained)
The batch preparation (uint8 -> float [0,1] -> preprocessor) mirrors lerobot_train.py lines 603-607.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import shutil
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

from src.training.launcher import AGGR_ROOT, ROOT, environment_info, train_episodes


def _rss_gb() -> float:
    p = psutil.Process()
    return (p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))) / 2**30


def build_policy_config(pcfg: dict):
    import lerobot.policies  # noqa: F401  registers all policy config classes
    from lerobot.configs.policies import PreTrainedConfig

    cls = PreTrainedConfig.get_choice_class(pcfg["type"])
    valid = {f.name for f in fields(cls)}
    pcfg = dict(pcfg)
    rev = pcfg.pop("vlm_revision", None)  # project key: pin the VLM snapshot for reproducibility
    if rev and pcfg.get("vlm_model_name"):
        from huggingface_hub import snapshot_download

        pcfg["vlm_model_name"] = snapshot_download(pcfg["vlm_model_name"], revision=rev)
    kwargs = {k: (tuple(v) if isinstance(v, list) and k == "image_resolution" else v) for k, v in pcfg.items() if k != "type"}
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(f"unknown {pcfg['type']} config fields: {sorted(unknown)}")
    return cls(**kwargs)


class ResumableSampler(torch.utils.data.Sampler):
    """Epoch permutations derived from (seed, epoch); can start mid-epoch so a resumed run consumes exactly the samples
    an uninterrupted run would have (no repeats, no skips), without decoding the skipped ones."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed = n, seed
        self.epoch, self.offset = 0, 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        yield from perm[self.offset :]

    def __len__(self):
        return self.n - self.offset


def _prepare(batch: dict, camera_keys, preprocessor) -> dict:
    for k in camera_keys:
        if k in batch and batch[k].dtype == torch.uint8:
            batch[k] = batch[k].to(dtype=torch.float32) / 255.0
    return preprocessor(batch)


class Trainer:
    def __init__(self, cfg: dict, run_dir: Path):
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self.cfg = cfg
        self.run_dir = run_dir
        t = cfg["training"]
        torch.manual_seed(t.get("seed", 1000))
        np.random.seed(t.get("seed", 1000))
        self.device = torch.device(cfg["policy"].get("device", "cuda"))

        meta = LeRobotDatasetMetadata("local/so101_11tasks", root=AGGR_ROOT)
        self.policy_cfg = build_policy_config(cfg["policy"])
        if cfg.get("init_from"):
            self.policy_cfg.pretrained_path = cfg["init_from"]
        dts = resolve_delta_timestamps(self.policy_cfg, meta)
        tasks = cfg["tasks"]
        self.train_eps = train_episodes(tasks)
        split = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
        ids = sorted(split["tasks"]) if tasks == "all" else tasks
        self.val_eps = sorted(split["tasks"][x]["global_episode_offset"] + e for x in ids for e in split["tasks"][x]["val"])
        common = dict(delta_timestamps=dts, video_backend=cfg.get("video_backend", "pyav"), download_videos=False, return_uint8=True)
        mixture = cfg.get("mixture", "REAL_ONLY")
        if mixture == "REAL_ONLY":
            self.ds = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, episodes=self.train_eps, **common)
        else:  # scripts/build_mixture.py: real train episodes + sim episodes, stats over exactly those
            mix = json.loads((ROOT / "reports" / f"mixture_{mixture}.json").read_text())
            self.train_eps = mix["train_episodes"]
            self.ds = LeRobotDataset(f"local/{mixture}", root=ROOT / mix["root"], episodes=self.train_eps, **common)
        # validation always on held-out REAL episodes
        self.val_ds = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, episodes=self.val_eps, **common)
        self.camera_keys = list(meta.camera_keys)

        self.policy = make_policy(self.policy_cfg, ds_meta=self.ds.meta)
        self.policy.to(self.device)
        self.pre, self.post = make_pre_post_processors(self.policy_cfg, dataset_stats=self.ds.meta.stats)

        params = self.policy.get_optim_params()
        self.optimizer = self.policy_cfg.get_optimizer_preset().build(params)
        self.total_updates = t["max_steps"]
        sched = self.policy_cfg.get_scheduler_preset()
        self.scheduler = sched.build(self.optimizer, self.total_updates) if sched is not None else None
        self.n_trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        self.n_total = sum(p.numel() for p in self.policy.parameters())

        self.sampler = ResumableSampler(len(self.ds), t.get("seed", 1000))
        self.loader = torch.utils.data.DataLoader(self.ds, batch_size=t["batch_size"], sampler=self.sampler, num_workers=t.get("num_workers", 0),
                                                  pin_memory=self.device.type == "cuda", drop_last=True)
        self.start_step, self.best, self.log = 0, math.inf, []
        rng = np.random.default_rng(0)
        self.val_idx = rng.choice(len(self.val_ds), size=min(t.get("val_samples", 64), len(self.val_ds)), replace=False).tolist()

    def _loss(self, batch) -> torch.Tensor:
        batch = _prepare(batch, self.camera_keys, self.pre)
        loss, _ = self.policy.forward(batch)
        return loss

    @torch.no_grad()
    def validate(self) -> float:
        self.policy.eval()
        losses = []
        bs = self.cfg["training"]["batch_size"]
        g = torch.Generator()
        g.manual_seed(1234)  # flow matching samples noise/time: fixed generator for comparable val loss
        state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if self.device.type == "cuda" else None
        torch.manual_seed(1234)
        for i in range(0, len(self.val_idx), bs):
            items = [self.val_ds[j] for j in self.val_idx[i : i + bs]]
            batch = torch.utils.data.default_collate(items)
            losses.append(float(self._loss(batch).item()))
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        self.policy.train()
        return float(np.mean(losses))

    def save(self, name: str, step: int, extra: dict | None = None, full_state: bool = False) -> Path:
        """Write to a temp dir and rename, so an interruption never leaves a half-written checkpoint."""
        final = self.run_dir / "checkpoints" / name
        tmp = self.run_dir / "checkpoints" / f".{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        d = tmp / "pretrained_model"
        d.mkdir(parents=True)
        self.policy.save_pretrained(d)
        self.pre.save_pretrained(d)
        self.post.save_pretrained(d)
        (tmp / "training_state.json").write_text(json.dumps({"step": step, **(extra or {})}, indent=2))
        if full_state:
            torch.save({
                "step": step, "best": self.best, "log": self.log,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "sampler": {"epoch": self.sampler.epoch, "offset": self.sampler.offset + self._consumed_in_epoch},
                "rng": {"torch": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state() if self.device.type == "cuda" else None,
                        "numpy": np.random.get_state()},
            }, tmp / "training_state.pt")
        if final.exists():
            old = self.run_dir / "checkpoints" / f".{name}.old"
            if old.exists():
                shutil.rmtree(old)
            final.rename(old)
            tmp.rename(final)
            shutil.rmtree(old)
        else:
            tmp.rename(final)
        return final / "pretrained_model"

    def _new_iter(self):
        # DataLoader draws a base seed from the default RNG when an iterator is created; keep the training RNG stream
        # untouched so an uninterrupted and a resumed run see identical randomness.
        state = torch.random.get_rng_state()
        it = iter(self.loader)
        torch.random.set_rng_state(state)
        return it

    def load_for_resume(self) -> None:
        from safetensors.torch import load_file

        ck = self.run_dir / "checkpoints" / "latest"
        sd = load_file(str(ck / "pretrained_model" / "model.safetensors"))
        missing, unexpected = self.policy.load_state_dict(sd, strict=False)
        if unexpected or [m for m in missing if not m.startswith("model.embedder")]:
            raise RuntimeError(f"resume weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        st = torch.load(ck / "training_state.pt", weights_only=False)
        self.optimizer.load_state_dict(st["optimizer"])
        if self.scheduler is not None and st["scheduler"] is not None:
            self.scheduler.load_state_dict(st["scheduler"])
        self.sampler.epoch, self.sampler.offset = st["sampler"]["epoch"], st["sampler"]["offset"]
        torch.random.set_rng_state(st["rng"]["torch"])
        if st["rng"]["cuda"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(st["rng"]["cuda"])
        np.random.set_state(st["rng"]["numpy"])
        self.start_step, self.best, self.log = st["step"], st["best"], st["log"]
        print(json.dumps({"resumed_from_step": self.start_step, "epoch": self.sampler.epoch, "sample_offset": self.sampler.offset}), flush=True)

    def train(self) -> dict:
        t = self.cfg["training"]
        accum = t.get("grad_accum_steps", 1)
        clip = self.policy_cfg.optimizer_grad_clip_norm if hasattr(self.policy_cfg, "optimizer_grad_clip_norm") else 1.0
        log_every, ckpt_every, val_every = t.get("log_interval", 10), t.get("ckpt_interval", 1000), t.get("val_interval", 500)
        max_wall = t.get("max_wall_minutes")
        log = self.log
        best = self.best
        peak_rss = peak_vram = 0.0
        self.policy.train()
        self._consumed_in_epoch = 0
        it = self._new_iter()
        t_start = time.time()
        step = self.start_step
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        while step < self.total_updates:
            t0 = time.time()
            data_s = 0.0
            loss_acc = 0.0
            for _ in range(accum):
                td = time.time()
                try:
                    batch = next(it)
                except StopIteration:
                    self.sampler.epoch += 1
                    self.sampler.offset = 0
                    self._consumed_in_epoch = 0
                    it = self._new_iter()
                    batch = next(it)
                self._consumed_in_epoch += t["batch_size"]
                data_s += time.time() - td
                loss = self._loss(batch) / accum
                loss.backward()
                loss_acc += float(loss.item())
            gn = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            step += 1
            if self.device.type == "cuda":
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 2**30)
            peak_rss = max(peak_rss, _rss_gb())
            if step % log_every == 0 or step == 1:
                rec = {"step": step, "loss": loss_acc, "grad_norm": float(gn), "lr": self.optimizer.param_groups[0]["lr"], "step_s": time.time() - t0,
                       "data_s": data_s, "peak_vram_alloc_gb": round(peak_vram, 3), "rss_gb": round(_rss_gb(), 3)}
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step % val_every == 0 or step == self.total_updates:
                vl = self.validate()
                log.append({"step": step, "val_loss": vl})
                print(json.dumps({"step": step, "val_loss": vl}), flush=True)
                if vl < best:
                    best = vl
                    self.best = best
                    self.save("best", step, {"val_loss": vl})
            if step % ckpt_every == 0 or step == self.total_updates:
                self.best = best
                self.save("latest", step, full_state=True)
            if max_wall and (time.time() - t_start) / 60 > max_wall:
                print(f"stopping: wall-time budget {max_wall} min reached", flush=True)
                self.save("latest", step, full_state=True)
                break
        wall = time.time() - t_start
        return {
            "steps": step, "grad_accum_steps": accum, "batch_size": t["batch_size"], "effective_batch": accum * t["batch_size"],
            "wall_time_s": round(wall, 1), "samples_per_s": round((step - self.start_step) * accum * t["batch_size"] / wall, 3), "resumed_from_step": self.start_step,
            "best_val_loss": best, "peak_vram_alloc_gb": round(peak_vram, 3), "peak_rss_gb": round(peak_rss, 3),
            "trainable_params": self.n_trainable, "total_params": self.n_total, "n_train_episodes": len(self.train_eps),
            "n_val_episodes": len(self.val_eps), "log": log,
        }


def run(config_path: str, overrides: dict | None = None, resume_dir: str | None = None) -> dict:
    if resume_dir:  # continue an interrupted run with its own saved config
        run_dir = Path(resume_dir) if Path(resume_dir).is_absolute() else ROOT / resume_dir
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        ts = run_dir.name.rsplit("_", 2)[-2] + "_" + run_dir.name.rsplit("_", 1)[-1]
        tr = Trainer(cfg, run_dir)
        tr.load_for_resume()
        return _finish(tr, cfg, run_dir, ts)
    cfg = yaml.safe_load(Path(config_path).read_text())
    for k, v in (overrides or {}).items():
        if "." in k:
            sect, key = k.split(".", 1)
            cfg.setdefault(sect, {})[key] = v
        else:
            cfg[k] = v
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / cfg.get("output_root", "outputs") / f"{cfg['name']}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    tr = Trainer(cfg, run_dir)
    return _finish(tr, cfg, run_dir, ts)


def _finish(tr: "Trainer", cfg: dict, run_dir: Path, ts: str) -> dict:
    print(json.dumps({"trainable_params": tr.n_trainable, "total_params": tr.n_total, "train_frames": len(tr.ds), "val_frames": len(tr.val_ds)}), flush=True)
    summary = tr.train()
    summary["run_dir"] = str(run_dir.relative_to(ROOT))
    summary["environment"] = environment_info()
    summary["config"] = cfg
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # canonical export: checkpoints/<export_name>/{best,latest,config.yaml,norm_stats.json,training_summary.json}
    if cfg.get("export_name"):
        exp = ROOT / "checkpoints" / cfg["export_name"]
        exp.mkdir(parents=True, exist_ok=True)
        for n in ("best", "latest"):
            src = run_dir / "checkpoints" / n / "pretrained_model"
            if src.exists():
                if (exp / n).exists():
                    shutil.rmtree(exp / n)
                shutil.copytree(src, exp / n)
        shutil.copy(run_dir / "config.yaml", exp / "config.yaml")
        shutil.copy(AGGR_ROOT / "meta" / "stats.json", exp / "norm_stats.json")
        shutil.copy(run_dir / "training_summary.json", exp / "training_summary.json")
    (ROOT / "reports" / f"experiment_{ts}.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "log"}, indent=2, default=str))
    return summary
