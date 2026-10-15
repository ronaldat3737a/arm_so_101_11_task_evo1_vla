#!/usr/bin/env bash
# Physics-fix evaluation (resumable; finished runs are skipped). REAL_PLUS_SIM(v2) selective_llm, BF16 eager (no
# torch.compile: avoids the 10-11 GB RSS of compiled runs; W8A8 without compile is slower than BF16), 8 flow steps,
# same seeds / unseen test positions. A = v2 scene (old physics, same inference) as control, B = v3 scene (new physics).
set -u
cd "$(dirname "$0")/.."
CK=checkpoints/evo1_so101_s2lite_selective_llm_rpsv2/best
for pair in "v2:configs/simulation_v2.yaml" "v3:configs/simulation_v3.yaml"; do
  tag=${pair%%:*}; cfg=${pair#*:}
  name=evo1_rpsv2_${tag}scene_bf16_s8
  [ -f reports/evaluation_${name}_position.json ] && { echo "$name done, skip"; continue; }
  SO101_SIM_CONFIG=$cfg uv run python scripts/evaluate_policy.py --checkpoint $CK --task all --splits test --profile position \
    --episodes 5 --name $name --num-inference-timesteps 8 --video --trace || { echo "$name FAILED"; exit 1; }
done
echo EVAL_DONE
