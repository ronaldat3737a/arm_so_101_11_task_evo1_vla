#!/usr/bin/env python
"""Run a trained LeRobot policy (Evo-1, ACT, ...) on the SO-101 follower, with a safety layer.

    camera + instruction + robot state -> policy (official predict_action) -> action chunk -> safety layer -> SO-101

Modes
  --dry-run --source replay   no hardware: observations from a held-out REAL test episode; prints/logs the actions that
                              WOULD be sent. Use it on any machine.
  --dry-run --source robot    real cameras + robot state (torque disabled, read-only). Never sends actions.
                              A pass writes reports/dryrun_pass.json, which --real requires.
  --real                      sends actions. Requires: a recent dry-run pass for the same checkpoint and calibration,
                              a calibrated robot, speed_scale <= 1 (default 0.1 from configs/deployment.yaml).

Safety: joint clipping to calibration ∩ data envelope, per-step rate limit (max_joint_delta * speed_scale), workspace
check via FK, LeRobot max_relative_target, inference/communication timeouts, e-stop (Ctrl+C, space/esc,
`touch /tmp/so101_STOP`), slow safe reset to the rest pose, torque off on disconnect.

Examples
  python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --dry-run --source replay
  python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --dry-run --source robot
  python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --real --speed-scale 0.1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deployment.safety import (  # noqa: E402
    JOINTS,
    EStop,
    SafetyConfig,
    SafetyLayer,
    Watchdog,
    WorkspaceChecker,
    calibration_limits,
    data_envelope,
)

FEATS = [f"{j}.pos" for j in JOINTS]


def checkpoint_hash(ckpt: Path) -> str:
    h = hashlib.sha256()
    for name in ("model.safetensors", "config.json"):
        p = ckpt / name
        if p.exists():
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()[:16]


class ReplaySource:
    """Held-out real episode as a stand-in for camera + robot (no hardware)."""

    def __init__(self, task_id: str, split: str = "test"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        sm = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
        s = sm["tasks"][task_id]
        self.ep = s["global_episode_offset"] + s[split][0]
        self.ds = LeRobotDataset("local/so101_11tasks", root=ROOT / "datasets" / "so101_11tasks", episodes=[self.ep], video_backend="pyav", download_videos=False)
        self.i = 0

    def read(self) -> dict:
        it = self.ds[min(self.i, len(self.ds) - 1)]
        self.i += 1
        return {
            "observation.state": it["observation.state"].numpy().astype(np.float32),
            "observation.images.top": (it["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            "observation.images.wrist": (it["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            "recorded_action": it["action"].numpy(),
        }

    def done(self) -> bool:
        return self.i >= len(self.ds)


def make_robot(dep: dict, cam_cfg: dict):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cams = {}
    for name, c in cam_cfg["cameras"].items():
        if c.get("exposure") is not None and shutil.which("v4l2-ctl"):
            subprocess.run(["v4l2-ctl", "-d", str(c["device"]), "-c", f"{c['exposure_ctrl']}={c['exposure']}"], check=False)
        cams[name] = OpenCVCameraConfig(index_or_path=c["device"], fps=c["fps"], width=c["width"], height=c["height"])
    r = dep["robot"]
    cfg = SO101FollowerConfig(port=r["port"], id=r["id"], cameras=cams, max_relative_target=r["max_relative_target"], disable_torque_on_disconnect=r["disable_torque_on_disconnect"])
    return SO101Follower(cfg)


def robot_obs(robot) -> dict:
    o = robot.get_observation()
    return {
        "observation.state": np.array([o[f] for f in FEATS], dtype=np.float32),
        "observation.images.top": np.asarray(o["top"]),
        "observation.images.wrist": np.asarray(o["wrist"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", required=True, help="task_01..task_11 (instruction from configs/tasks.yaml)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    ap.add_argument("--source", choices=["replay", "robot"], default="robot", help="dry-run observation source")
    ap.add_argument("--speed-scale", type=float)
    ap.add_argument("--max-joint-delta", type=float)
    ap.add_argument("--workspace-limit", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    ap.add_argument("--emergency-stop", default=None, help="stop file path (default from configs/deployment.yaml)")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--deployment-config", default=str(ROOT / "configs" / "deployment.yaml"))
    ap.add_argument("--camera-config", default=str(ROOT / "configs" / "camera.yaml"))
    ap.add_argument("--calibration-file", help="default: $HF_LEROBOT_CALIBRATION/robots/so_follower/<id>.json")
    a = ap.parse_args()

    import torch
    from lerobot.common.control_utils import predict_action
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    from src.evaluation.policy_eval import load_policy
    from src.simulation.tasks import INSTRUCTIONS

    dep = yaml.safe_load(Path(a.deployment_config).read_text())
    cam_cfg = yaml.safe_load(Path(a.camera_config).read_text())
    ctl = dep["control"]
    speed = a.speed_scale if a.speed_scale is not None else ctl["speed_scale"]
    instruction = INSTRUCTIONS[a.task]
    ckpt = Path(a.checkpoint)
    ck_hash = checkpoint_hash(ckpt)
    calib = Path(a.calibration_file) if a.calibration_file else HF_LEROBOT_CALIBRATION / "robots" / "so_follower" / f"{dep['robot']['id']}.json"

    # ---- joint box: calibration ∩ data envelope (replay dry-run without calibration: data envelope only)
    dlo, dhi = data_envelope(ROOT / "reports" / "norm_stats.json", ctl["data_envelope_margin_deg"])
    if calib.exists():
        clo, chi = calibration_limits(calib)
        lo, hi = np.maximum(dlo, clo), np.minimum(dhi, chi)
        calib_note = str(calib)
    elif a.real or a.source == "robot":
        print(f"STOP: calibration file {calib} not found. Run: lerobot-calibrate --robot.type=so101_follower --robot.port=... --robot.id={dep['robot']['id']}")
        return 2
    else:
        lo, hi = dlo, dhi
        calib_note = "none (replay dry-run: data envelope only)"
    ws_lo = np.array(a.workspace_limit[:3]) if a.workspace_limit else np.array(ctl["workspace_low"])
    ws_hi = np.array(a.workspace_limit[3:]) if a.workspace_limit else np.array(ctl["workspace_high"])
    safety = SafetyLayer(SafetyConfig(lo, hi, a.max_joint_delta or ctl["max_joint_delta_deg"], ctl["max_gripper_delta"], speed, ws_lo, ws_hi), WorkspaceChecker())

    if a.real:
        pf = ROOT / dep["safety"]["dry_run_pass_file"]
        ok = False
        if pf.exists():
            p = json.loads(pf.read_text())
            age_h = (dt.datetime.now() - dt.datetime.fromisoformat(p["time"])).total_seconds() / 3600
            ok = p.get("passed") and p.get("source") == "robot" and p.get("checkpoint_hash") == ck_hash and p.get("calibration") == str(calib) and age_h <= dep["safety"]["dry_run_max_age_hours"]
        if not ok:
            print("STOP: --real requires a recent passing `--dry-run --source robot` for this checkpoint and calibration.")
            return 2

    estop = EStop(a.emergency_stop or dep["safety"]["stop_file"])
    policy, pre, post, cfg = load_policy(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(cfg.device)
    use_amp = bool(getattr(cfg, "use_amp", False))

    robot = None
    source = None
    if a.dry_run and a.source == "replay":
        source = ReplaySource(a.task)
    else:
        robot = make_robot(dep, cam_cfg)
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            robot.disconnect()
            print("STOP: motors are not calibrated (calibration file and motor registers disagree)")
            return 2
        if a.dry_run:
            robot.bus.disable_torque()  # read-only: the arm can be moved by hand, nothing is commanded

    log = []
    lat, acts = [], []
    max_steps = a.max_steps or int(ctl["max_episode_s"] * ctl["fps"])
    period = 1.0 / ctl["fps"]
    reason = "max_steps"
    try:
        if a.real:  # safe reset: slowly to the rest pose
            reset_layer = SafetyLayer(SafetyConfig(lo, hi, ctl["max_joint_delta_deg"], ctl["max_gripper_delta"], ctl["reset_speed_scale"], ws_lo, ws_hi), safety.ws)
            rest = np.array(ctl["rest_pose"], dtype=np.float64)
            for _ in range(int(10 * ctl["fps"])):
                cur = robot_obs(robot)["observation.state"].astype(np.float64)
                cmd, _ = reset_layer.filter(rest, cur)
                robot.send_action(dict(zip(FEATS, cmd.tolist(), strict=True)))
                if np.abs(cur - rest).max() < 2.0 or estop.active:
                    break
                time.sleep(period)
        policy.reset()
        for step in range(max_steps):
            t0 = time.perf_counter()
            if estop.active:
                reason = f"emergency stop: {estop.reason}"
                break
            with Watchdog("communication", ctl["communication_timeout_s"] if robot else 10.0):
                obs = source.read() if source else robot_obs(robot)
            cur = obs["observation.state"].astype(np.float64)
            o = {k: v for k, v in obs.items() if k.startswith("observation.")}
            with Watchdog("inference", ctl["inference_timeout_s"] if step > 0 else 30.0) as w:
                act = predict_action(o, policy, device, pre, post, use_amp, task=instruction, robot_type="so_follower")
            act = act.squeeze(0).float().cpu().numpy()[:6].astype(np.float64)
            lat.append(w.last_s * 1000)
            cmd, rep = safety.filter(act, cur)
            if rep["estop"]:
                estop.trigger(rep["estop"])
            acts.append(act)
            entry = {"step": step, "state": cur.round(2).tolist(), "policy_action": act.round(2).tolist(), "command": cmd.round(2).tolist(),
                     "clipped": rep["clipped_joints"], "rate_limited": rep["rate_limited"], "workspace_reject": rep["workspace_reject"], "latency_ms": round(lat[-1], 1)}
            if source is not None:
                entry["recorded_action"] = obs["recorded_action"].round(2).tolist()
            log.append(entry)
            if step % 15 == 0:
                print(f"[{'REAL' if a.real else 'DRY'}] step {step:4d} joints {np.round(act[:5], 1)} gripper {act[5]:5.1f} -> cmd {np.round(cmd, 1)} lat {lat[-1]:.0f} ms {'CLIP ' + ','.join(rep['clipped_joints']) if rep['clipped_joints'] else ''}", flush=True)
            if a.real:
                robot.send_action(dict(zip(FEATS, cmd.tolist(), strict=True)))
            if source is not None and source.done():
                reason = "replay episode finished"
                break
            dt_left = period - (time.perf_counter() - t0)
            if dt_left > 0 and not source:
                time.sleep(dt_left)
    except TimeoutError as exc:
        reason = f"timeout: {exc}"
        estop.trigger(reason)
    finally:
        if robot is not None:
            if a.real and robot.is_connected:
                cur = robot_obs(robot)["observation.state"]
                robot.send_action(dict(zip(FEATS, cur.tolist(), strict=True)))  # hold where it is
            robot.disconnect()

    A = np.asarray(acts) if acts else np.zeros((0, 6))
    summary = {
        "time": dt.datetime.now().isoformat(), "mode": "real" if a.real else "dry-run", "source": a.source if a.dry_run else "robot",
        "checkpoint": str(ckpt), "checkpoint_hash": ck_hash, "calibration": calib_note if not calib.exists() else str(calib), "task": a.task, "instruction": instruction,
        "steps": len(log), "stop_reason": reason, "speed_scale": speed,
        "latency_ms": {"p50": float(np.percentile(lat, 50)) if lat else None, "p95": float(np.percentile(lat, 95)) if lat else None, "max": float(max(lat)) if lat else None},
        "action_range": {"min": A.min(0).round(2).tolist() if len(A) else None, "max": A.max(0).round(2).tolist() if len(A) else None},
        "joint_box": {"low": lo.round(2).tolist(), "high": hi.round(2).tolist()},
        "interventions": {"clipped_steps": sum(bool(x["clipped"]) for x in log), "rate_limited_steps": sum(bool(x["rate_limited"]) for x in log), "workspace_rejects": sum(x["workspace_reject"] for x in log)},
    }
    if source is not None and log:
        rec = np.array([x["recorded_action"] for x in log])
        summary["replay_mae_vs_recorded"] = float(np.abs(A - rec).mean())
    passed = len(log) > 0 and not reason.startswith(("timeout", "emergency")) and np.all(np.isfinite(A))
    summary["passed"] = bool(passed)
    out = ROOT / "reports" / f"deploy_{summary['mode']}_{a.task}_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "log": log}, indent=2))
    if a.dry_run and a.source == "robot" and passed:  # only a hardware dry-run can unlock --real
        (ROOT / dep["safety"]["dry_run_pass_file"]).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
