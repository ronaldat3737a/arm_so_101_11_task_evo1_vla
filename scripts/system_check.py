#!/usr/bin/env python
"""Detect hardware/software and write reports/system_check.md (+ .json).

Uses only the standard library (+ psutil/torch/lerobot when importable) so it can run in any environment,
including before dependencies are installed.

Usage:
    python scripts/system_check.py
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Thresholds used for the recommendations below. Sources:
#   Isaac Sim 5.1 minimum spec: 32 GB RAM, 16 GB VRAM (RTX 4080), Ubuntu 22.04/24.04
#     https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
ISAAC_MIN_RAM_GB = 32
ISAAC_MIN_VRAM_GB = 16
ISAAC_OS = ("22.04", "24.04")


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def os_release() -> dict:
    out = {}
    p = Path("/etc/os-release")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v.strip('"')
    return out


def mem_gb() -> tuple[float | None, float | None]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / 2**30, vm.available / 2**30
    except ImportError:
        pass
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"] / 2**30, info["MemAvailable"] / 2**30
    except Exception:  # noqa: BLE001
        return None, None


def swap_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("SwapTotal"):
                return int(line.split()[1]) / 2**20
    except Exception:  # noqa: BLE001
        return None
    return None


def nvidia() -> dict:
    q = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not q:
        return {"available": False}
    gpus = []
    for line in q.splitlines():
        name, total, used, driver, cc = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "vram_total_gb": float(total) / 1024, "vram_used_gb": float(used) / 1024, "driver": driver, "compute_capability": cc})
    header = run(["nvidia-smi"]) or ""
    cuda_drv = None
    for tok in header.split("|"):
        if "CUDA Version" in tok:
            cuda_drv = tok.split("CUDA Version:")[-1].strip()
    return {"available": True, "gpus": gpus, "driver_cuda_version": cuda_drv}


def torch_info() -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"installed": False, "error": str(exc)}
    out = {"installed": True, "version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["bf16_supported"] = torch.cuda.is_bf16_supported()
        free, total = torch.cuda.mem_get_info()
        out["vram_free_gb"] = free / 2**30
        out["vram_total_gb"] = total / 2**30
        try:
            x = torch.randn(256, 256, device="cuda")
            out["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
        except Exception as exc:  # noqa: BLE001
            out["cuda_matmul_ok"] = f"failed: {exc}"
    return out


def pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def module_ok(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    osr = os_release()
    total_ram, avail_ram = mem_gb()
    disk = shutil.disk_usage(ROOT)
    nv = nvidia()
    t = torch_info()
    vram = max((g["vram_total_gb"] for g in nv.get("gpus", [])), default=0.0)

    r = {
        "os": {"pretty": osr.get("PRETTY_NAME", platform.platform()), "version_id": osr.get("VERSION_ID"), "kernel": platform.release()},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "cpu": {"model": next((line.split(":")[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else platform.processor(), "logical_cores": os.cpu_count()},
        "ram_gb": {"total": total_ram, "available": avail_ram, "swap": swap_gb()},
        "disk_gb": {"path": str(ROOT), "total": disk.total / 2**30, "free": disk.free / 2**30},
        "nvidia": nv,
        "nvcc": run(["nvcc", "--version"]),
        "torch": t,
        "packages": {p: pkg_version(p) for p in ["lerobot", "transformers", "accelerate", "huggingface-hub", "datasets", "torchcodec", "av", "flash-attn", "feetech-servo-sdk", "isaacsim", "isaaclab", "leisaac"]},
        "isaac": {
            "isaacsim_importable": module_ok("isaacsim"),
            "isaaclab_importable": module_ok("isaaclab"),
            "leisaac_importable": module_ok("leisaac"),
            "ISAACSIM_PATH": os.environ.get("ISAACSIM_PATH"),
        },
        "tools": {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"), "git": shutil.which("git"), "git-lfs": shutil.which("git-lfs"), "uv": shutil.which("uv"), "conda": shutil.which("conda")},
    }

    # Recommendations / blockers
    blockers, warnings, recs = [], [], []
    if not nv.get("available"):
        blockers.append("No NVIDIA GPU detected: Evo-1 training and Isaac Sim need CUDA.")
    if t.get("installed") and not t.get("cuda_available"):
        blockers.append("PyTorch is installed but torch.cuda.is_available() is False.")
    if not t.get("installed"):
        warnings.append("PyTorch not importable in this interpreter (run inside the project env: `uv run python scripts/system_check.py`).")
    if vram and vram < 8:
        recs.append(
            f"VRAM {vram:.1f} GB: Evo-1 stage1 only (VLM frozen), batch_size 1-2 with gradient accumulation, bf16 autocast "
            "(policy.use_amp=true), gradient checkpointing on, flash-attn optional. Full stage2 VLM finetune "
            "(~0.8B params with AdamW) does not fit; run it on a >=24 GB GPU."
        )
    if total_ram and total_ram < 16:
        recs.append(f"RAM {total_ram:.1f} GB: num_workers<=2, prefetch_factor<=2, never cache decoded frames in RAM.")
    if (total_ram or 0) < ISAAC_MIN_RAM_GB or vram < ISAAC_MIN_VRAM_GB or osr.get("VERSION_ID") not in ISAAC_OS:
        blockers.append(
            f"Isaac Sim 5.1 minimum spec not met (needs >= {ISAAC_MIN_RAM_GB} GB RAM, >= {ISAAC_MIN_VRAM_GB} GB VRAM, "
            f"Ubuntu {'/'.join(ISAAC_OS)}; have {total_ram:.1f} GB RAM, {vram:.1f} GB VRAM, {r['os']['pretty']}). "
            "IsaacLab/LeIsaac simulation must run on another machine, or a lighter simulator must be chosen."
        )
    if not r["tools"]["ffmpeg"]:
        warnings.append("ffmpeg not on PATH. Not required for decoding (LeRobot uses torchcodec/PyAV wheels), but needed for MP4 encoding of evaluation videos via ffmpeg CLI.")
    if not r["tools"]["git-lfs"]:
        warnings.append("git-lfs not installed. Not required: datasets/models are fetched with huggingface_hub.")
    if disk.free / 2**30 < 50:
        warnings.append(f"Only {disk.free / 2**30:.0f} GB free disk; datasets (~7 GB) + merged copy + checkpoints need ~50 GB.")
    r["blockers"], r["warnings"], r["recommendations"] = blockers, warnings, recs

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "system_check.json").write_text(json.dumps(r, indent=2, default=str))

    def f(x, nd=1):
        return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    L = ["# System check\n", "| item | value |", "|------|-------|"]
    L += [
        f"| OS | {r['os']['pretty']} (kernel {r['os']['kernel']}) |",
        f"| Python (this interpreter) | {r['python']['version']} — `{r['python']['executable']}` |",
        f"| CPU | {r['cpu']['model']} ({r['cpu']['logical_cores']} threads) |",
        f"| RAM total / available / swap | {f(total_ram)} / {f(avail_ram)} / {f(r['ram_gb']['swap'])} GB |",
        f"| Disk free (project) | {f(disk.free / 2**30, 0)} of {f(disk.total / 2**30, 0)} GB |",
    ]
    for g in nv.get("gpus", []):
        L.append(f"| GPU | {g['name']}, {g['vram_total_gb']:.1f} GB VRAM, compute {g['compute_capability']} |")
        L.append(f"| NVIDIA driver / driver CUDA | {g['driver']} / {nv.get('driver_cuda_version')} |")
    if not nv.get("available"):
        L.append("| GPU | none detected |")
    L.append(f"| nvcc (CUDA toolkit) | {'not installed' if not r['nvcc'] else r['nvcc'].splitlines()[-1]} |")
    if t.get("installed"):
        L.append(f"| PyTorch | {t['version']} (CUDA build {t['cuda_build']}), cuda_available={t['cuda_available']}, bf16={t.get('bf16_supported')}, matmul={t.get('cuda_matmul_ok')} |")
    else:
        L.append("| PyTorch | not importable in this interpreter |")
    for k, v in r["packages"].items():
        L.append(f"| pkg `{k}` | {v or 'not installed'} |")
    for k, v in r["isaac"].items():
        L.append(f"| {k} | {v} |")
    for k, v in r["tools"].items():
        L.append(f"| tool `{k}` | {v or 'not found'} |")
    L.append("\n## Blockers\n")
    L += [f"- BLOCKED: {b}" for b in blockers] or ["- none"]
    L.append("\n## Warnings\n")
    L += [f"- {w}" for w in warnings] or ["- none"]
    L.append("\n## Recommendations\n")
    L += [f"- {x}" for x in recs] or ["- none"]
    (out / "system_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
