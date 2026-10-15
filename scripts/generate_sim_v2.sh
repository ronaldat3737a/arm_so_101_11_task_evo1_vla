#!/usr/bin/env bash
# Resumable generation of the v2 sim dataset (scene configs/simulation_v2.yaml), one LeRobotDataset per task.
# A task whose dataset is finalized with the target episode count is skipped; an unfinished one is regenerated.
# The v1 dataset datasets/sim_so101_all is never touched.
set -u
cd "$(dirname "$0")/.."
N=${N:-40}
export SO101_SIM_CONFIG=configs/simulation_v2.yaml
for i in 01 02 03 04 05 06 07 08 09 10 11; do
  out=datasets/sim_so101_v2/task_$i
  if [ -f "$out/meta/info.json" ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$out/meta/info.json'))['total_episodes']>=$N else 1)"; then
    echo "task_$i complete, skip"; continue
  fi
  uv run python scripts/generate_sim_data.py --tasks task_$i --episodes-per-task $N --seed0 $((200000 + 10#$i * 1000)) \
      --out $out --force --report-name sim_v2_report_task_$i || { echo "task_$i FAILED"; exit 1; }
done
echo ALL_DONE
