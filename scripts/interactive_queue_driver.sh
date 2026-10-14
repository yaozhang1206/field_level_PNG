#!/bin/bash
# Run a queue of GPU tasks in consecutive 2-hour interactive sessions (one at a time; the partition allows one job).
#   nohup scripts/interactive_queue_driver.sh [running_job_id] > logs/interactive_queue.log 2>&1 &
# The queue file (re-read before every submission, so tasks can be added or reordered while it runs) has lines
#   name|done-check shell command (exit 0 = finished)|run command (executed in the repo with the env loaded)
# Lines starting with # are ignored. Every task must be resumable: a session ends after 115 minutes.
cd /u/yzhang116/field_level_PNG
Q=${QUEUE:-logs/interactive_queue.txt}
JOB=$1
declare -A FAIL
LAST=""
while true; do
  while [ -n "$JOB" ] && [ -n "$(squeue -j "$JOB" -h 2>/dev/null)" ]; do sleep 60; done
  # a session that ended in under 10 minutes counts as a failure; a task with 3 failures in a row is skipped
  if [ -n "$JOB" ] && [ -n "$LAST" ]; then
    el=$(sacct -j "$JOB" -X -n -o ElapsedRaw 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$el" ] && [ "$el" -lt 600 ]; then FAIL[$LAST]=$(( ${FAIL[$LAST]:-0} + 1 )); else FAIL[$LAST]=0; fi
    [ "${FAIL[$LAST]:-0}" -ge 3 ] && echo "$(date) $LAST failed 3 times in a row; skipping it (see logs/q_$LAST.o$JOB)"
  fi
  NEXT=""
  while IFS='|' read -r name check cmd; do
    [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
    [ "${FAIL[$name]:-0}" -ge 3 ] && continue
    if ! bash -c "$check" >/dev/null 2>&1; then NEXT="$name"; RUN="$cmd"; break; fi
  done < "$Q"
  LAST=$NEXT
  if [ -z "$NEXT" ]; then echo "$(date) queue empty; waiting"; sleep 600; JOB=""; continue; fi
  JOB=$(sbatch --parsable --job-name="q_$NEXT" --account=bdne-dtai-gh --partition=ghx4-interactive --gpus-per-node=1 \
        --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120g --time=2:00:00 --no-requeue -o logs/%x.o%j -e logs/%x.o%j \
        --wrap="cd /u/yzhang116/field_level_PNG && source env/setup_env.sh && unset JAX_PLATFORMS && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 && timeout 6900 $RUN")
  echo "$(date) $NEXT -> ${JOB:-SUBMIT FAILED}"
  [ -z "$JOB" ] && sleep 300
done
