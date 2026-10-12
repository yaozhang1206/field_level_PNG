#!/bin/bash
# Fetch one batch of Quijote Rockstar catalogues (Globus: Quijote_simulations2 -> NCSA Delta), cut, delete raw.
#   scripts/stage3_fetch_quijote.sh <set> <snap> <z> <id_from> <id_to> <cut> [<cut> ...]
#   snap: 0=z3 1=z2 2=z1 3=z0.5 4=z0.   Sets: fiducial, fiducial_HR, LC_p, LC_m, ...
#   cut:  <tracer>[:b1=<target>|:mmin=<Msun/h>]   e.g.  QSO:b1=2.9   LRG:mmin=2e13   LRG  (DESI b1(z))
# Raw files land in $RAW/<set>/<i>/out_<snap>_pid.list and are removed after the last cut is saved.
set -e
SET=$1; SNAP=$2; Z=$3; FROM=$4; TO=$5; shift 5; CUTS="$@"
TRACER=$(echo "$1" | cut -d: -f1)
Q2=e0eae0aa-5bca-11ea-9683-0e56c063f437
DELTA=2d66a243-4a3f-4578-9d7f-1935fb5fba8f
RAW=/work/hdd/bdne/yzhang116/quijote_raw
OUT=/work/hdd/bdne/yzhang116/quijote_cuts
cd "$(dirname "$0")/.."
source env/setup_env.sh >/dev/null 2>&1
mkdir -p "$RAW/$SET" logs
BATCH=$(mktemp)
IDS=""
for i in $(seq $FROM $TO); do
  f="$RAW/$SET/$i/out_${SNAP}_pid.list"
  if [ ! -s "$f" ]; then
    echo "/Halos/Rockstar/$SET/$i/out_${SNAP}_pid.list $f" >> "$BATCH"
  fi
  IDS="$IDS,$i"
done
IDS=${IDS#,}
if [ -s "$BATCH" ]; then
  echo "transferring $(wc -l < "$BATCH") files for $SET snap $SNAP ids $FROM-$TO"
  TASK=$(globus transfer "$Q2" "$DELTA" --batch "$BATCH" --label "quijote_${SET}_${SNAP}_${FROM}_${TO}" --jmespath task_id --format unix)
  echo "task $TASK"
  globus task wait "$TASK" --timeout 7200 --polling-interval 30
  globus task show "$TASK" | grep -E "Status|Files|Bytes"
fi
rm -f "$BATCH"
# skip empty (broken) realisations
GOOD=""
for i in $(echo $IDS | tr ',' ' '); do
  f="$RAW/$SET/$i/out_${SNAP}_pid.list"
  if [ -s "$f" ]; then GOOD="$GOOD,$i"; else echo "skip $i (empty/missing)"; fi
done
GOOD=${GOOD#,}
N=$(echo $CUTS | wc -w); k=0
for CUT in $CUTS; do
  k=$((k+1)); DEL=""; [ $k -eq $N ] && DEL="--delete_raw"
  TR=$(echo "$CUT" | cut -d: -f1); OPT=$(echo "$CUT" | cut -s -d: -f2); EXTRA=""; TAG="$TR"
  case "$OPT" in
    b1=*)   EXTRA="--b1 ${OPT#b1=}";     TAG="${TR}_b${OPT#b1=}" ;;
    mmin=*) EXTRA="--mmin ${OPT#mmin=}"; TAG="${TR}_M${OPT#mmin=}" ;;
  esac
  echo "== cut $TAG"
  python scripts/stage3_cut_quijote.py --in_dir "$RAW/$SET" --pattern "{i}/out_${SNAP}_pid.list" --ids "$GOOD" \
    --z "$Z" --tracer "$TR" --tag "$TAG" --set "$SET" --out_root "$OUT" $EXTRA $DEL 2>&1 | grep -v -i "cuda\|jax\|Traceback\|File \"\|plugin\|RuntimeError"
done
find "$RAW/$SET" -type d -empty -delete 2>/dev/null || true
