#!/usr/bin/env python
"""CLP-inspired adaptation to Evo-1: remove language-model decoder layers of the InternVL3 embedder.

This is NOT an official Evo-1 CLP implementation. Layers are chosen from our own CKA + skip-layer sensitivity
measurements (reports/cka_sensitivity_*.md), one at a time, each followed by a benchmark.

The pruned checkpoint stays loadable by the official `Evo1Policy.from_pretrained` without code changes:
the remaining layers are renumbered contiguously and `vlm_num_layers` is reduced, so the official embedder truncates
the base VLM to the new depth and then strictly loads the renumbered weights. (Vision-encoder layers cannot be pruned
this way: their count comes from the base VLM config, not from the Evo-1 config.)

    python scripts/clp_prune.py --src checkpoints/evo1_so101_s2lite_selective_llm/best --drop 4 --out checkpoints/evo1_clp_lm4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

PREFIX = "model.embedder.model.language_model.layers."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="pretrained_model directory")
    ap.add_argument("--drop", required=True, help="comma-separated layer indices of the SOURCE checkpoint to remove")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    cfg = json.loads((src / "config.json").read_text())
    n = int(cfg["vlm_num_layers"])
    drop = sorted({int(x) for x in a.drop.split(",")})
    assert all(0 <= d < n for d in drop), (drop, n)
    keep = [i for i in range(n) if i not in drop]
    new_index = {old: new for new, old in enumerate(keep)}

    sd = load_file(str(src / "model.safetensors"))
    new_sd, removed = {}, 0
    pat = re.compile(re.escape(PREFIX) + r"(\d+)\.(.*)")
    for k, v in sd.items():
        m = pat.match(k)
        if not m:
            new_sd[k] = v
            continue
        old = int(m.group(1))
        if old in new_index:
            new_sd[f"{PREFIX}{new_index[old]}.{m.group(2)}"] = v.contiguous()
        else:
            removed += v.numel()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("model.safetensors"))
    save_file(new_sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg["vlm_num_layers"] = len(keep)
    (out / "config.json").write_text(json.dumps(cfg, indent=4))
    info = {"method": "CLP-inspired adaptation to Evo-1 (not official Evo-1 CLP)", "source": str(src),
            "dropped_source_layers": drop, "kept_source_layers": keep, "vlm_num_layers": len(keep), "removed_params": removed}
    prev = src / "clp.json"
    if prev.exists():  # keep the history when pruning an already pruned checkpoint
        info["previous"] = json.loads(prev.read_text())
    (out / "clp.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
