#!/usr/bin/env bash
# Resumable v2-scene pipeline (after scripts/generate_sim_v2.sh). Every step is skipped when its output exists, so the
# script can be interrupted and rerun at any time.
#   1 aggregate per-task v2 datasets  -> datasets/sim_so101_v2_all (+ reports/sim_v2_dataset_report.json)
#   2 hidden-state cache of v2 frames -> datasets/evo1_hidden_cache_L10/simv2_s10_o0 (finished shards skipped)
#   3 selective_llm REAL_PLUS_SIM(v2) from stage-1 run 2, 10k steps (same recipe as the v1 REAL_PLUS_SIM run)
#   4 offline test eval (real episodes)
#   5 MuJoCo eval on the v2 scene, BEST_LOCAL inference (W8A8 vision, compile, 8 flow steps), same seeds:
#     REAL_ONLY selective_llm (scene change alone) and REAL_PLUS_SIM(v2)
set -u
cd "$(dirname "$0")/.."
V2=configs/simulation_v2.yaml
CK_RO=checkpoints/evo1_so101_s2lite_selective_llm/best
CK_V2=checkpoints/evo1_so101_s2lite_selective_llm_rpsv2/best

python3 -c "import json,sys;sys.exit(0 if json.load(open('reports/sim_v2_dataset_report.json')).get('passed') else 1)" 2>/dev/null \
  || uv run python scripts/aggregate_sim_v2.py || { echo "AGGREGATE FAILED"; exit 1; }

uv run python scripts/cache_evo1_hidden.py --start 10 --parts sim --sim-label simv2 --sim-root datasets/sim_so101_v2_all \
  || { echo "CACHE FAILED"; exit 1; }
N=$(ls datasets/evo1_hidden_cache_L10/simv2_s10_o0/shard_*.safetensors | wc -l)
S=$(python3 -c "import json;print(json.load(open('datasets/evo1_hidden_cache_L10/cache_meta.json'))['parts']['simv2_s10_o0']['shards'])")
[ "$N" = "$S" ] || { echo "CACHE INCOMPLETE $N/$S"; exit 1; }

if [ ! -f "$CK_V2/model.safetensors" ]; then
  RUN=$(ls -d outputs/evo1_s2lite_selective_llm_rpsv2_* 2>/dev/null | tail -1)
  if [ -n "$RUN" ] && [ -f "$RUN/checkpoints/latest/training_state.pt" ]; then
    uv run python scripts/train_evo1_selective.py --resume "$RUN" || { echo "TRAIN FAILED"; exit 1; }
  else
    uv run python scripts/train_evo1_selective.py --train-from 12 --steps 10000 --parts train_s10_o0,simv2_s10_o0 \
      --name evo1_s2lite_selective_llm_rpsv2 --export evo1_so101_s2lite_selective_llm_rpsv2 || { echo "TRAIN FAILED"; exit 1; }
  fi
fi

[ -f reports/offline_evo1_s2lite_selective_llm_rpsv2_test.json ] || uv run python scripts/evaluate_offline.py --checkpoint $CK_V2 \
  --task all --split test --frames-per-episode 4 --name evo1_s2lite_selective_llm_rpsv2

for pair in "realonly:$CK_RO" "rpsv2:$CK_V2"; do
  tag=${pair%%:*}; ck=${pair#*:}
  [ -f reports/evaluation_evo1_v2scene_${tag}_w8vis_s8_position.json ] && continue
  SO101_SIM_CONFIG=$V2 uv run python scripts/evaluate_policy.py --checkpoint $ck --task all --splits test --profile position \
    --episodes 5 --name evo1_v2scene_${tag}_w8vis_s8 --num-inference-timesteps 8 --quant w8a8_vision --compile --video --trace \
    || { echo "SIM EVAL $tag FAILED"; exit 1; }
done
echo PIPELINE_DONE
