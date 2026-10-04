#!/usr/bin/env bash
#SBATCH --job-name=tcb-cablese
#SBATCH --output=cablese_%A_%a.log
#SBATCH --error=cablese_%A_%a.log
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=04:00:00
#SBATCH --export=ALL

# Cablese output-recovery experiment — 50 passages tiled EXACTLY over 24 slots
# (slot i owns passages [i*b + min(i,r), ...limit b + (i<r))).
BUNDLE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && sleep $(( (SLURM_ARRAY_TASK_ID * 7 + RANDOM) % 120 ))
: "${TCB_CABLESE_RUN_ID:?export TCB_CABLESE_RUN_ID=<stamp> before submitting; do NOT pre-create results/cablese_<runid>}"
cd "$BUNDLE_DIR"
export PYTHONUNBUFFERED=1
export TCB_TREAT_401_TRANSIENT=1
export TCB_RETRY_ATTEMPTS=15
RESULTS="results/cablese_${TCB_CABLESE_RUN_ID}"

if [ "${TCB_CABLESE_MODE:-}" = "merge" ]; then
  # merge is one scheduler pass on slot 0 only
  [ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ] && exit 0
  exec python3 -m ab.cablese cablese-merge --results-dir "$RESULTS"
fi

TOTAL="${TCB_CABLESE_PASSAGES:-50}"
SLOTS=24
BASE=$((TOTAL / SLOTS))
REM=$((TOTAL % SLOTS))
ID="${SLURM_ARRAY_TASK_ID:?array submission required}"
LIMIT=$((BASE + (ID < REM ? 1 : 0)))
OFFSET=$((ID * BASE + (ID < REM ? ID : REM)))
if [ "$LIMIT" -eq 0 ]; then
  echo "slot $ID: no passages assigned (TOTAL=$TOTAL); exiting"
  exit 0
fi
exec python3 -m ab.cablese cablese ${TCB_MAX_SECONDS:+--max-seconds $TCB_MAX_SECONDS} --max-tokens "${TCB_MAX_TOKENS:-4000}" --passages "$TOTAL" \
  --task-offset "$OFFSET" --task-limit "$LIMIT" \
  --slice-id "$(printf '%04d' "$ID")" --results-dir "$RESULTS"
