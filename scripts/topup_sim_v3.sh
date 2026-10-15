#!/usr/bin/env bash
# Top-up for v3 tasks that ended below their target (only successful oracle episodes are kept). Each round writes a
# separate dataset datasets/sim_so101_v3/task_XX_topupN with fresh seeds (seed0 400000 + 100000*N) and
# --max-attempts-factor 6, until the task's total reaches the target. Rerunnable: finished top-ups are counted.
set -u
cd "$(dirname "$0")/.."
export SO101_SIM_CONFIG=configs/simulation_v3.yaml
count() { python3 - "$1" <<'PY'
import json, glob, sys
t = sys.argv[1]
n = 0
for d in glob.glob(f"datasets/sim_so101_v3/{t}") + glob.glob(f"datasets/sim_so101_v3/{t}_topup*"):
    try:
        n += json.load(open(f"{d}/meta/info.json"))["total_episodes"]
    except Exception:
        pass
print(n)
PY
}
for i in 01 02 03 04 05 06 07 08 09 10 11; do
  if [ "$i" = 05 ] || [ "$i" = 06 ]; then N=122; else N=84; fi
  for round in 1 2 3 4; do
    have=$(count task_$i); miss=$((N - have))
    [ "$miss" -le 0 ] && break
    out=datasets/sim_so101_v3/task_${i}_topup$round
    [ -f "$out/meta/info.json" ] && [ -f "reports/sim_v3_report_task_${i}_topup$round.json" ] && continue
    echo "task_$i: have $have / $N -> top-up round $round for $miss"
    uv run python scripts/generate_sim_data.py --tasks task_$i --episodes-per-task $miss --max-attempts-factor 6 --profile position_wide \
        --seed0 $((400000 + 100000 * round)) --dart-prob 0.5 --dart-dxy 0.02 0.04 --dart-dyaw 15 --dart-steps 15 25 --hover-steps 15 \
        --drop-height 0.02 0.03 --out $out --force --report-name sim_v3_report_task_${i}_topup$round || { echo "topup task_$i FAILED"; exit 1; }
  done
  echo "task_$i total $(count task_$i) / $N"
done
echo TOPUP_DONE
