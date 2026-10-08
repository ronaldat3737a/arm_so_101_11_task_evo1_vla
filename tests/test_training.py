import json
from pathlib import Path

import pytest
import yaml

from src.training.launcher import build_command, train_episodes
from src.training.trainer import build_policy_config

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "reports" / "split_manifest.json"


@pytest.mark.skipif(not SPLIT.exists(), reason="needs reports/split_manifest.json")
def test_train_episodes_exclude_held_out():
    s = json.loads(SPLIT.read_text())
    eps = set(train_episodes("all"))
    held = set(s["global"]["val"]) | set(s["global"]["test"]) | set(s["global"]["excluded"])
    assert len(eps) == 432 and not eps & held
    t1 = train_episodes(["task_01"])
    assert len(t1) == 40 and max(t1) < 50


def test_evo1_yaml_is_a_valid_evo1_config():
    cfg = yaml.safe_load((ROOT / "configs" / "evo1.yaml").read_text())
    pc = build_policy_config(cfg["policy"])
    assert pc.type == "evo1" and pc.training_stage == "stage1"
    assert pc.finetune_vlm is False and pc.finetune_action_head is True  # stage1 = frozen VLM
    assert pc.chunk_size == 50 and tuple(pc.image_resolution) == (448, 448)  # official evo1 requires the VLM native size
    assert pc.binarize_gripper is False


def test_unknown_policy_field_rejected():
    with pytest.raises(ValueError):
        build_policy_config({"type": "evo1", "not_a_field": 1})


@pytest.mark.skipif(not SPLIT.exists(), reason="needs reports/split_manifest.json")
def test_act_command_uses_official_cli_and_train_episodes():
    cfg = yaml.safe_load((ROOT / "configs" / "act_baseline.yaml").read_text())
    cmd = build_command(cfg, ROOT / "outputs" / "x")
    assert cmd[0].endswith("lerobot-train")
    ep_arg = next(c for c in cmd if c.startswith("--dataset.episodes="))
    assert json.loads(ep_arg.split("=", 1)[1]) == train_episodes(["task_01"])
    assert "--policy.type=act" in cmd and "--num_workers=0" in cmd
