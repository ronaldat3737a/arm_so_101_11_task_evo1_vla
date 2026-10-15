#!/usr/bin/env bash
# v3 round (after generate_sim_v3.sh + topup_sim_v3.sh). Resumable: every step is skipped when its output exists;
# training resumes from checkpoints/latest.
#   1 aggregate datasets/sim_so101_v3/task_XX(+_topupN) -> datasets/sim_so101_v3_all (+ decode check)
#   2 seed map of the v3 training layouts (verified reproduction)
#   3 hidden cache of v3 sim frames (stride 10) -> datasets/evo1_hidden_cache_L10/simv3_s10_o0
#   4 train rpsv3: init rpsv2c, 20k steps, real train_s10_o0 + simv3_s10_o0
#   5 evals, BF16 eager (no compile), 8 flow steps, scene v3:
#       offline real test; offline on trained v3 sim episodes;
#       closed loop on the exact v3 training layouts: rpsv3 and the rpsv2c baseline (same layouts);
#       closed loop on unseen test positions (profile position, seeds 1000-1004): rpsv3 and the rpsv2c baseline
# Stop criterion agreed with the user: on the v3 training layouts success >= 10/55 AND clearly fewer pot tips than rpsv2c.
set -u
cd "$(dirname "$0")/.."
V3=configs/simulation_v3.yaml
CK=checkpoints/evo1_so101_s2lite_selective_llm_rpsv3/best
BASE=checkpoints/evo1_so101_s2lite_selective_llm_rpsv2c/best

python3 -c "import json,sys;sys.exit(0 if json.load(open('reports/sim_v3_dataset_report.json')).get('passed') else 1)" 2>/dev/null \
  || uv run python scripts/aggregate_sim_v2.py --src datasets/sim_so101_v3 --out datasets/sim_so101_v3_all \
       --report sim_v3_dataset_report --sim-config $V3 || { echo "AGGREGATE FAILED"; exit 1; }
echo "AGGREGATE OK"

[ -f reports/seed_map_sim_v3_train.json ] || SO101_SIM_CONFIG=$V3 uv run python scripts/make_seed_map.py \
  --report-prefix sim_v3_report_task_ --profile position_wide --out reports/seed_map_sim_v3_train.json || { echo "SEED MAP FAILED"; exit 1; }

uv run python scripts/cache_evo1_hidden.py --start 10 --parts sim --sim-label simv3 --sim-root datasets/sim_so101_v3_all \
  || { echo "CACHE FAILED"; exit 1; }
N=$(ls datasets/evo1_hidden_cache_L10/simv3_s10_o0/shard_*.safetensors | wc -l)
S=$(python3 -c "import json;print(json.load(open('datasets/evo1_hidden_cache_L10/cache_meta.json'))['parts']['simv3_s10_o0']['shards'])")
[ "$N" = "$S" ] || { echo "CACHE INCOMPLETE $N/$S"; exit 1; }
echo "CACHE OK $N/$S"

if [ ! -f "$CK/model.safetensors" ]; then
  RUN=$(ls -d outputs/evo1_s2lite_selective_llm_rpsv3_* 2>/dev/null | tail -1)
  if [ -n "$RUN" ] && [ -f "$RUN/checkpoints/latest/training_state.pt" ]; then
    uv run python scripts/train_evo1_selective.py --resume "$RUN" || { echo "TRAIN FAILED"; exit 1; }
  else
    uv run python scripts/train_evo1_selective.py --init $BASE --train-from 12 --steps 20000 --parts train_s10_o0,simv3_s10_o0 \
      --seed 4000 --name evo1_s2lite_selective_llm_rpsv3 --export evo1_so101_s2lite_selective_llm_rpsv3 || { echo "TRAIN FAILED"; exit 1; }
  fi
fi
echo "TRAIN OK"

[ -f reports/offline_evo1_s2lite_selective_llm_rpsv3_test.json ] || uv run python scripts/evaluate_offline.py --checkpoint $CK \
  --task all --split test --frames-per-episode 4 --name evo1_s2lite_selective_llm_rpsv3
[ -f reports/offline_evo1_rpsv3_simv3train_test.json ] || uv run python scripts/evaluate_offline.py --checkpoint $CK \
  --dataset-root datasets/sim_so101_v3_all --episodes-per-task 5 --task all --frames-per-episode 4 --num-inference-timesteps 8 \
  --name evo1_rpsv3_simv3train
for pair in "rpsv3:$CK" "rpsv2c:$BASE"; do
  tag=${pair%%:*}; ck=${pair#*:}
  name=evo1_${tag}_v3trainlayouts_v3scene_bf16_s8
  [ -f reports/evaluation_${name}_position_wide.json ] || SO101_SIM_CONFIG=$V3 uv run python scripts/evaluate_policy.py --checkpoint $ck \
    --task all --splits train --profile position_wide --episodes 5 --seed-map reports/seed_map_sim_v3_train.json --name $name \
    --num-inference-timesteps 8 --video --trace || { echo "SIM EVAL $name FAILED"; exit 1; }
  name=evo1_${tag}_v3scene_test_bf16_s8
  [ -f reports/evaluation_${name}_position.json ] || SO101_SIM_CONFIG=$V3 uv run python scripts/evaluate_policy.py --checkpoint $ck \
    --task all --splits test --profile position --episodes 5 --name $name --num-inference-timesteps 8 --video --trace \
    || { echo "SIM EVAL $name FAILED"; exit 1; }
done
echo PIPELINE_DONE
