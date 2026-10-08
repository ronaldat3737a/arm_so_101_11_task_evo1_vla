#!/usr/bin/env python
"""Download the official SO-101 MuJoCo model (MJCF + STL meshes) from the hardware repository
TheRobotStudio/SO-ARM100 (Simulation/SO101), pinned to a commit, into simulation/assets/so101/.

"New calibration" files are used: every joint's virtual zero is the middle of its range, which is the
LeRobot `use_degrees=True` convention our datasets are recorded in (docs/so101_interface.md).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "TheRobotStudio/SO-ARM100"
COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"  # main @ 2026-10-06
SUBDIR = "Simulation/SO101"
DEST = ROOT / "simulation" / "assets" / "so101"
FILES = [
    "README.md",
    "joints_properties.xml",
    "scene.xml",
    "so101_new_calib.xml",
    "so101_new_calib_camera.xml",
]


def fetch(url: str, retries: int = 5) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"retry {url}: {exc}")
            time.sleep(2 + 2 * i)
    raise RuntimeError("unreachable")


def main() -> int:
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    stls = [t["path"][len(SUBDIR) + 1 :] for t in tree["tree"] if t["path"].startswith(f"{SUBDIR}/assets/") and t["path"].endswith(".stl")]
    manifest = {"repo": REPO, "commit": COMMIT, "subdir": SUBDIR, "files": {}}
    for rel in FILES + stls:
        out = DEST / rel
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(fetch(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{SUBDIR}/{rel}"))
        manifest["files"][rel] = hashlib.sha256(out.read_bytes()).hexdigest()
        print("ok", rel)
    (DEST / "SOURCE.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
