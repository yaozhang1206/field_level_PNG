#!/bin/bash
# Work through a prioritised Stage 3 task list in consecutive 2-hour interactive sessions (one at a time).
#   nohup scripts/stage3_interactive_driver.sh [running_job_id] > logs/stage3_driver.log 2>&1 &
# Each task is "<sample> <ids> [extra args]|<results json>"; a task is done when the JSON holds all its ids.
cd /u/yzhang116/field_level_PNG
B="python scripts/run_stage3_halos.py --res 16 --lap --rsd --n_warmup 300 --n_samples 1000"
R=results/stage3
TASKS=(
 "fiducial_HR_z0.5_LRG 0-29 --mock|$R/fiducial_HR_z0.5_LRG_res16_p1m4d_lpt2_rsd_lap_p1_mock0.json"
 "fiducial_HR_z1_QSO_b2.9 0-29 --mock|$R/fiducial_HR_z1_QSO_b2.9_res16_p1m4d_lpt2_rsd_lap_p1.6_mock0.json"
 "fiducial_HR_z0.5_LRG 26-55|$R/fiducial_HR_z0.5_LRG_res16_p1m4d_lpt2_rsd_lap_p1.json"
 "fiducial_z0.5_M2e13 21-50 --mmin 6e13|$R/fiducial_z0.5_M2e13_res16_p1m4d_lpt2_rsd_lap_p1_M6e+13.json"
 "fiducial_HR_z1_QSO_b2.9 20-49|$R/fiducial_HR_z1_QSO_b2.9_res16_p1m4d_lpt2_rsd_lap_p1.6.json"
 "fiducial_HR_z1_QSO_b2.44 20-49|$R/fiducial_HR_z1_QSO_b2.44_res16_p1m4d_lpt2_rsd_lap_p1.6.json"
)
missing() {
  python3 -c "
import json, sys
lo, hi = map(int, sys.argv[2].split('-'))
try: have = set(json.load(open(sys.argv[1]))['runs'])
except Exception: have = set()
print(sum(str(i) not in have for i in range(lo, hi + 1)))" "$1" "$2"
}
JOB=$1
while true; do
  while [ -n "$JOB" ] && [ -n "$(squeue -j "$JOB" -h 2>/dev/null)" ]; do sleep 60; done
  NEXT=""
  for t in "${TASKS[@]}"; do
    args=${t%%|*}; file=${t##*|}
    ids=$(echo "$args" | awk '{print $2}')
    m=$(missing "$file" "$ids")
    if [ "$m" -gt 0 ]; then NEXT="$args"; echo "$(date) next: $args ($m left)"; break; fi
  done
  if [ -z "$NEXT" ]; then echo "$(date) all tasks done"; break; fi
  S=$(echo "$NEXT" | awk '{print $1}'); I=$(echo "$NEXT" | awk '{print $2}'); X=$(echo "$NEXT" | cut -d' ' -f3-)
  [ "$X" = "$NEXT" ] && X=""
  JOB=$(sbatch --parsable --job-name=s3int --account=bdne-dtai-gh --partition=ghx4-interactive --gpus-per-node=1 \
        --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120g --time=2:00:00 --no-requeue -o logs/%x.o%j -e logs/%x.o%j \
        --wrap="cd /u/yzhang116/field_level_PNG && source env/setup_env.sh && unset JAX_PLATFORMS && timeout 6900 $B --sample $S --ids $I $X")
  echo "$(date) submitted ${JOB:-FAILED}"
  [ -z "$JOB" ] && sleep 300
done
