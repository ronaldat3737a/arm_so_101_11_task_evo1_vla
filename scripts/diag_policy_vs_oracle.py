#!/usr/bin/env python
"""Where does the closed-loop policy leave the oracle demonstration of the SAME layout?

For every policy trace on the v2 training layouts (scripts/evaluate_policy.py --seed-map ... --trace) the oracle
demonstration of the identical seed is taken from datasets/sim_so101_v2_all (episode = (task-1)*40 + rank of the seed
among the successful generation seeds; lengths verified against reports/sim_v2_report_task_XX.json).

Grasp point = gripperframe site + GripperIK offset (as the oracle), by MuJoCo forward kinematics of the recorded
joint states. Oracle phases (first grasp only):
  approach  start .. last step > 4 cm above the grasp height
  descend   .. gripper closing onset (action drops 10 below its open level)
  close     20 steps from the closing onset
  carry     rest (transport / place / release)
Policy steps are aligned to the oracle path by nearest grasp point with forward-progress constraint (window
[-5, +60] oracle steps). Divergence onset = first policy step whose distance to the oracle path stays > --thr cm for
15 steps; its phase = phase of the matched oracle step.
At pot tips (episode end with pot_tipped_over): grasp point relative to the pot (initial pose), gripper state, joint
deviation from the matched oracle state.

    SO101_SIM_CONFIG=configs/simulation_v2.yaml python scripts/diag_policy_vs_oracle.py \
        --runs evo1_rpsv2c_trainlayouts_v2scene_bf16_s8,evo1_rpsv2c_trainlayouts_v2scene_bf16_s8_n10
Writes reports/diag_policy_vs_oracle.{json,md}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.action_adapter import canonical_to_sim  # noqa: E402
from src.simulation.env import SO101Env  # noqa: E402
from src.simulation.ik import GripperIK  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TARGET = {"task_01": "red_cube", "task_02": "blue_cube", "task_03": "blue_cube", "task_04": "banana", "task_05": "blue_cube",
          "task_06": "red_cube", "task_07": "yellow_cube", "task_08": None, "task_09": "red_cube", "task_10": "orange_block_a", "task_11": None}


def grasp_points(env, ik, states: np.ndarray) -> np.ndarray:
    m, d = env.model, env.data
    q0 = d.qpos.copy()
    out = np.zeros((len(states), 3))
    sid = m.site("gripperframe").id
    for k, s in enumerate(states):
        d.qpos[:6] = canonical_to_sim(np.asarray(s, dtype=np.float64))
        mujoco.mj_kinematics(m, d)
        out[k] = d.site_xpos[sid] + d.site_xmat[sid].reshape(3, 3) @ ik.offset
    d.qpos[:] = q0
    mujoco.mj_kinematics(m, d)
    return out


def oracle_phases(gp: np.ndarray, grip_a: np.ndarray) -> tuple[np.ndarray, dict]:
    T = len(gp)
    t_o = int(np.argmax(grip_a > 30)) if (grip_a > 30).any() else 0
    t_c = T - 1
    run_max = -1e9
    for t in range(t_o, T):
        run_max = max(run_max, grip_a[t])
        if grip_a[t] < run_max - 10:
            t_c = t
            break
    above = np.where(gp[:t_c, 2] > gp[t_c, 2] + 0.04)[0]
    t_d = int(above[-1]) + 1 if len(above) else 0
    ph = np.array(["carry"] * T, dtype=object)
    ph[:t_d] = "approach"
    ph[t_d:t_c] = "descend"
    ph[t_c:min(T, t_c + 20)] = "close"
    return ph, {"t_descend": t_d, "t_close": t_c, "grasp_z": float(gp[t_c, 2])}


def closing_onset(grip_a: np.ndarray) -> int | None:
    run_max = -1e9
    seen_open = False
    for t, g in enumerate(grip_a):
        if g > 30:
            seen_open = True
        if seen_open:
            run_max = max(run_max, g)
            if g < run_max - 10:
                return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="evo1_rpsv2c_trainlayouts_v2scene_bf16_s8,evo1_rpsv2c_trainlayouts_v2scene_bf16_s8_n10")
    ap.add_argument("--seed-map", default=str(ROOT / "reports" / "seed_map_sim_v2_train.json"))
    ap.add_argument("--data", default=str(ROOT / "datasets" / "sim_so101_v2_all"))
    ap.add_argument("--thr", type=float, default=2.0, help="divergence threshold, cm")
    a = ap.parse_args()

    seed_map = json.loads(Path(a.seed_map).read_text())
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                    for f in glob.glob(f"{a.data}/data/**/*.parquet", recursive=True)])
    rows = []
    for run in a.runs.split(","):
        ev = json.loads((ROOT / "reports" / f"evaluation_{run}_position_visual.json").read_text())
        for t, seeds in sorted(seed_map.items()):
            i = int(t.split("_")[1])
            rep = json.loads((ROOT / "reports" / f"sim_v2_report_task_{i:02d}.json").read_text())
            succ = [x["seed"] for x in rep["tasks"][t]["attempts"] if x["success"]]
            env = SO101Env(t, split="train", profile="position_visual", render=False)
            ik = GripperIK(env.model)
            res = {r["seed"]: r for r in ev["results"][t]["train"]}
            for s in seeds:
                ep = (i - 1) * 40 + succ.index(s)
                g = df[df.episode_index == ep].sort_values("frame_index")
                o_state, o_act = np.stack(g["observation.state"].values), np.stack(g["action"].values)
                tr = np.load(ROOT / "outputs" / "traces" / f"{run}_{t}_position_visual_train_seed{s}.npz")
                p_state, p_act = tr["state"], tr["action"]
                env.reset(seed=s)
                pot = env.body_pos("pot").copy() if "pot" in env.obj_info else None
                tgt = env.body_pos(TARGET[t]).copy() if TARGET[t] else None
                gpo, gpp = grasp_points(env, ik, o_state), grasp_points(env, ik, p_state)
                ph, ev_o = oracle_phases(gpo, o_act[:, 5])
                # progress-constrained alignment
                idx, dist, prev = [], [], 0
                for p in gpp:
                    lo, hi = max(0, prev - 5), min(len(gpo), prev + 61)
                    dd = np.linalg.norm(gpo[lo:hi] - p, axis=1)
                    k = lo + int(np.argmin(dd))
                    idx.append(k)
                    dist.append(float(dd[k - lo]) * 100)
                    prev = max(prev, k)
                dist = np.asarray(dist)
                above = dist > a.thr
                onset = next((j for j in range(len(dist) - 15) if above[j:j + 15].all()), None)
                row = {"run": run, "task": t, "seed": s, "oracle_episode": ep, "success": res[s]["success"],
                       "failure": (res[s]["failure_reason"] or "success").split("(")[0].split(":")[0].strip(),
                       "policy_steps": len(p_state), "oracle_steps": len(o_state), "oracle_events": ev_o,
                       "max_oracle_progress_phase": ph[max(idx)], "max_oracle_progress_frac": round(max(idx) / (len(gpo) - 1), 3),
                       "mean_dev_cm": round(float(dist.mean()), 2)}
                if onset is not None:
                    k = idx[onset]
                    row["divergence"] = {"policy_step": onset, "oracle_step": k, "phase": ph[k], "dev_cm": round(float(dist[onset]), 2),
                                         "joint_dev_deg": dict(zip(JOINTS, (p_state[onset] - o_state[k]).round(1).tolist(), strict=True))}
                else:
                    row["divergence"] = None
                pc = closing_onset(p_act[:, 5])
                if tgt is not None:
                    row["oracle_close_offset_cm"] = {"xy": round(float(np.linalg.norm(gpo[ev_o["t_close"], :2] - tgt[:2])) * 100, 2),
                                                     "z": round(float(gpo[ev_o["t_close"], 2] - tgt[2]) * 100, 2)}
                    if pc is not None:
                        row["policy_close"] = {"step": pc, "xy_to_target_cm": round(float(np.linalg.norm(gpp[pc, :2] - tgt[:2])) * 100, 2),
                                               "z_to_target_cm": round(float(gpp[pc, 2] - tgt[2]) * 100, 2), "aligned_phase": ph[idx[pc]]}
                    else:
                        row["policy_close"] = None
                if row["failure"] == "pot_tipped_over" and pot is not None:
                    e = len(p_state) - 1
                    w = slice(max(0, e - 15), e + 1)
                    r_h = float(np.linalg.norm(gpp[e, :2] - pot[:2]))
                    r_in, h = env.obj_info["pot"]["inner_radius"], env.obj_info["pot"]["height"]
                    r_out = env.obj_info["pot"]["footprint_radius"] - 0.02
                    where = "inside opening" if r_h < r_in - 0.01 else ("over / at the wall" if r_h <= r_out + 0.015 else "outside the pot")
                    k = idx[e]
                    row["tip"] = {"step": e, "grasp_point_r_from_pot_axis_cm": round(r_h * 100, 2), "pot_inner_radius_cm": round(r_in * 100, 2),
                                  "pot_outer_radius_cm": round(r_out * 100, 2), "grasp_point_z_cm": round(float(gpp[e, 2]) * 100, 2),
                                  "pot_height_cm": round(h * 100, 2), "where": where, "aligned_oracle_phase": ph[k], "dev_from_oracle_path_cm": round(float(dist[e]), 2),
                                  "gripper_cmd": round(float(p_act[w, 5].mean()), 1), "gripper_state": round(float(p_state[w, 5].mean()), 1),
                                  "joint_dev_deg": dict(zip(JOINTS, (p_state[e] - o_state[k]).round(1).tolist(), strict=True)),
                                  "oracle_min_r_from_pot_axis_cm_while_below_rim": round(float(np.min(np.linalg.norm(gpo[gpo[:, 2] < h + 0.02, :2] - pot[:2], axis=1))) * 100, 2) if (gpo[:, 2] < h + 0.02).any() else None}
                rows.append(row)
            env.close()

    def agg(rs):
        div = [r["divergence"] for r in rs if r["divergence"]]
        tips = [r["tip"] for r in rs if "tip" in r]
        pcs = [r["policy_close"] for r in rs if r.get("policy_close")]
        out = {"episodes": len(rs), "success": sum(r["success"] for r in rs), "diverged": len(div),
               "divergence_phase": dict(Counter(d["phase"] for d in div)),
               "divergence_dev_cm_median": round(float(np.median([d["dev_cm"] for d in div])), 2) if div else None,
               "max_progress_phase": dict(Counter(r["max_oracle_progress_phase"] for r in rs)),
               "mean_dev_cm_median": round(float(np.median([r["mean_dev_cm"] for r in rs])), 2)}
        if div:
            out["divergence_joint_dev_abs_median_deg"] = {j: round(float(np.median([abs(d["joint_dev_deg"][j]) for d in div])), 1) for j in JOINTS}
        if pcs:
            out["policy_close"] = {"n": len(pcs), "xy_to_target_cm_median": round(float(np.median([p["xy_to_target_cm"] for p in pcs])), 2),
                                   "z_to_target_cm_median": round(float(np.median([p["z_to_target_cm"] for p in pcs])), 2),
                                   "aligned_phase": dict(Counter(p["aligned_phase"] for p in pcs))}
        if tips:
            out["tips"] = {"n": len(tips), "where": dict(Counter(x["where"] for x in tips)), "aligned_phase": dict(Counter(x["aligned_oracle_phase"] for x in tips)),
                           "r_from_axis_cm_median": round(float(np.median([x["grasp_point_r_from_pot_axis_cm"] for x in tips])), 2),
                           "z_cm_median": round(float(np.median([x["grasp_point_z_cm"] for x in tips])), 2),
                           "dev_from_oracle_cm_median": round(float(np.median([x["dev_from_oracle_path_cm"] for x in tips])), 2),
                           "gripper_closed_on_something": sum(x["gripper_cmd"] < x["gripper_state"] - 4 for x in tips),
                           "joint_dev_abs_median_deg": {j: round(float(np.median([abs(x["joint_dev_deg"][j]) for x in tips])), 1) for j in JOINTS},
                           "joint_dev_signed_median_deg": {j: round(float(np.median([x["joint_dev_deg"][j] for x in tips])), 1) for j in JOINTS}}
        return out

    summary = {"all": agg(rows), "per_task": {t: agg([r for r in rows if r["task"] == t]) for t in sorted(seed_map)},
               "oracle_close_offset_cm_median": {"xy": round(float(np.median([r["oracle_close_offset_cm"]["xy"] for r in rows if "oracle_close_offset_cm" in r])), 2),
                                                 "z": round(float(np.median([r["oracle_close_offset_cm"]["z"] for r in rows if "oracle_close_offset_cm" in r])), 2)}}
    (ROOT / "reports" / "diag_policy_vs_oracle.json").write_text(json.dumps({"threshold_cm": a.thr, "runs": a.runs.split(","), "summary": summary, "episodes": rows}, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
