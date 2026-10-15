#!/usr/bin/env python
"""Resumable CLP loop (CLP-inspired adaptation to Evo-1): prune one more language-model layer, benchmark, continue only
if acceptable.

Every step prunes the REFERENCE checkpoint with the cumulative drop list (no renumbering ambiguity), runs the offline
test evaluation and records the result in reports/clp_progress.json immediately. Finished steps are skipped on rerun,
so the loop can be interrupted at any time (only the running evaluation, about 12 min, is lost).

    python scripts/clp_run.py --order 4,3,5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="checkpoints/evo1_so101_s2lite_selective_llm/best")
    ap.add_argument("--ref-report", default="reports/offline_evo1_s2lite_selective_llm_test.json")
    ap.add_argument("--order", default="4,3,5", help="source LM layers in pruning order (from the sensitivity sweep)")
    ap.add_argument("--tol", type=float, default=0.3, help="accept if MAE <= reference MAE + tol (deg)")
    a = ap.parse_args()

    prog_path = ROOT / "reports" / "clp_progress.json"
    ref = json.loads((ROOT / a.ref_report).read_text())
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {
        "method": "CLP-inspired adaptation to Evo-1 (not official Evo-1 CLP)", "reference": a.ref,
        "reference_mae": ref["mean_action_mae"], "reference_hold": ref["mean_hold_state_baseline_mae"],
        "reference_latency_p50_ms": ref["latency_ms"]["p50"], "tol_deg": a.tol, "steps": []}
    done = {tuple(s["dropped"]): s for s in prog["steps"]}
    dropped: list[int] = []
    for layer in [int(x) for x in a.order.split(",")]:
        cand = dropped + [layer]
        key = tuple(cand)
        tag = "lm_drop" + "_".join(map(str, cand))
        if key not in done:
            out = f"checkpoints/evo1_clp/{tag}"
            run([sys.executable, "scripts/clp_prune.py", "--src", a.ref, "--drop", ",".join(map(str, cand)), "--out", out])
            run([sys.executable, "scripts/evaluate_offline.py", "--checkpoint", out, "--task", "all", "--split", "test",
                 "--frames-per-episode", "4", "--name", f"evo1_clp_{tag}"])
            r = json.loads((ROOT / "reports" / f"offline_evo1_clp_{tag}_test.json").read_text())
            pt = r["per_task"].values()
            step = {"dropped": cand, "checkpoint": out, "vlm_num_layers": 14 - len(cand), "mae": r["mean_action_mae"],
                    "d_mae": r["mean_action_mae"] - prog["reference_mae"],
                    "tasks_better_than_hold": sum(v["action_mae"] < v["hold_state_baseline_mae"] for v in pt),
                    "delta_cosine": sum(v["delta_cosine"] for v in pt) / len(r["per_task"]),
                    "latency_p50_ms": r["latency_ms"]["p50"], "resources": r["resources"],
                    "report": f"reports/offline_evo1_clp_{tag}_test.json"}
            step["accepted"] = step["d_mae"] <= a.tol
            prog["steps"].append(step)
            prog_path.write_text(json.dumps(prog, indent=2))
            done[key] = step
            print(json.dumps(step), flush=True)
        if not done[key]["accepted"]:
            print(f"STOP: dropping {cand} not acceptable (dMAE {done[key]['d_mae']:+.2f} > {a.tol})", flush=True)
            break
        dropped = cand
    prog["final_accepted_drop"] = dropped
    prog_path.write_text(json.dumps(prog, indent=2))
    print(json.dumps({"final_accepted_drop": dropped}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
