#!/usr/bin/env bash
# Resumable generation of the v3 sim dataset: scene configs/simulation_v3.yaml (v2 visuals + fixed pot physics),
# profile position_wide (wider positions / yaw +-45, train cells only), DART recovery demos (p 0.5, 2-4 cm, +-15 deg,
# 15-25 steps), hover 15 steps above containers (tasks 05/06), release 2-3 cm above the rim (tasks 01/03).
# 1000 episodes: tasks 05/06 122 each (container entry over-sampled), the other 9 tasks 84 each.
# One LeRobotDataset per task in datasets/sim_so101_v3/task_XX; a finalized task with its target count is skipped.
set -u
cd "$(dirname "$0")/.."
export SO101_SIM_CONFIG=configs/simulation_v3.yaml
for i in 01 02 03 04 05 06 07 08 09 10 11; do
  if [ "$i" = 05 ] || [ "$i" = 06 ]; then N=122; else N=84; fi
  out=datasets/sim_so101_v3/task_$i
  if [ -f "$out/meta/info.json" ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$out/meta/info.json'))['total_episodes']>=$N else 1)"; then
    echo "task_$i complete, skip"; continue
  fi
  if [ -f "reports/sim_v3_report_task_$i.json" ] && [ -f "$out/meta/info.json" ]; then
    echo "task_$i finished below target (see reports/sim_v3_report_task_$i.json), skip"; continue
  fi
  uv run python scripts/generate_sim_data.py --tasks task_$i --episodes-per-task $N --max-attempts-factor 3 --profile position_wide \
      --seed0 300000 --dart-prob 0.5 --dart-dxy 0.02 0.04 --dart-dyaw 15 --dart-steps 15 25 --hover-steps 15 --drop-height 0.02 0.03 \
      --out $out --force --report-name sim_v3_report_task_$i || { echo "task_$i FAILED"; exit 1; }
done
echo ALL_DONE
