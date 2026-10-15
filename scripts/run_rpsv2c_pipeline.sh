#!/usr/bin/env bash
# Continued training of the REAL_PLUS_SIM(v2) selective_llm policy with twice the cached sim frames (resumable; every
# step is skipped when its output exists; training resumes from checkpoints/latest).
#   1 hidden cache of v2 sim frames at offset 5      -> datasets/evo1_hidden_cache_L10/simv2_s10_o5
#   2 train from checkpoints/evo1_so101_s2lite_selective_llm_rpsv2/best, 20k steps, new cosine schedule,
#     parts train_s10_o0 (real) + simv2_s10_o0 + simv2_s10_o5
#   3 evals (BF16 eager, no compile, 8 flow steps): offline real test, offline on trained sim episodes,
#     closed loop on the exact training layouts and on the unseen test positions (v2 scene, same seeds as before)
set -u
cd "$(dirname "$0")/.."
V2=configs/simulation_v2.yaml
CK=checkpoints/evo1_so101_s2lite_selective_llm_rpsv2c/best

uv run python scripts/cache_evo1_hidden.py --start 10 --parts sim --sim-label simv2 --sim-root datasets/sim_so101_v2_all --offset 5 \
  || { echo "CACHE FAILED"; exit 1; }
N=$(ls datasets/evo1_hidden_cache_L10/simv2_s10_o5/shard_*.safetensors | wc -l)
S=$(python3 -c "import json;print(json.load(open('datasets/evo1_hidden_cache_L10/cache_meta.json'))['parts']['simv2_s10_o5']['shards'])")
[ "$N" = "$S" ] || { echo "CACHE INCOMPLETE $N/$S"; exit 1; }
echo "CACHE OK $N/$S"

if [ ! -f "$CK/model.safetensors" ]; then
  RUN=$(ls -d outputs/evo1_s2lite_selective_llm_rpsv2c_* 2>/dev/null | tail -1)
  if [ -n "$RUN" ] && [ -f "$RUN/checkpoints/latest/training_state.pt" ]; then
    uv run python scripts/train_evo1_selective.py --resume "$RUN" || { echo "TRAIN FAILED"; exit 1; }
  else
    uv run python scripts/train_evo1_selective.py --init checkpoints/evo1_so101_s2lite_selective_llm_rpsv2/best --train-from 12 \
      --steps 20000 --parts train_s10_o0,simv2_s10_o0,simv2_s10_o5 --seed 3000 \
      --name evo1_s2lite_selective_llm_rpsv2c --export evo1_so101_s2lite_selective_llm_rpsv2c || { echo "TRAIN FAILED"; exit 1; }
  fi
fi
echo "TRAIN OK"

[ -f reports/offline_evo1_s2lite_selective_llm_rpsv2c_test.json ] || uv run python scripts/evaluate_offline.py --checkpoint $CK \
  --task all --split test --frames-per-episode 4 --name evo1_s2lite_selective_llm_rpsv2c
[ -f reports/offline_evo1_rpsv2c_simv2train_test.json ] || uv run python scripts/evaluate_offline.py --checkpoint $CK \
  --dataset-root datasets/sim_so101_v2_all --episodes-per-task 5 --task all --frames-per-episode 4 --num-inference-timesteps 8 \
  --name evo1_rpsv2c_simv2train
[ -f reports/evaluation_evo1_rpsv2c_trainlayouts_v2scene_bf16_s8_position_visual.json ] || SO101_SIM_CONFIG=$V2 uv run python \
  scripts/evaluate_policy.py --checkpoint $CK --task all --splits train --profile position_visual --episodes 5 \
  --seed-map reports/seed_map_sim_v2_train.json --name evo1_rpsv2c_trainlayouts_v2scene_bf16_s8 --num-inference-timesteps 8 --video --trace \
  || { echo "SIM EVAL TRAIN FAILED"; exit 1; }
[ -f reports/evaluation_evo1_rpsv2c_v2scene_bf16_s8_position.json ] || SO101_SIM_CONFIG=$V2 uv run python scripts/evaluate_policy.py \
  --checkpoint $CK --task all --splits test --profile position --episodes 5 --name evo1_rpsv2c_v2scene_bf16_s8 \
  --num-inference-timesteps 8 --video --trace || { echo "SIM EVAL TEST FAILED"; exit 1; }
echo PIPELINE_DONE
