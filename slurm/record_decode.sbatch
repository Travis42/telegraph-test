#!/usr/bin/env bash
#SBATCH --job-name=tcb-recdec
#SBATCH --output=record_decode_%A_%a.log
#SBATCH --error=record_decode_%A_%a.log
#SBATCH --array=0-0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=04:00:00
#SBATCH --export=ALL

# Record-decode experiment (SPEC-RECORD-DECODE) — single slot is fine:
# ~750 relay calls over the 50 cbl2 passages (1 expansion + ~14.5
# anchored answers each). Slicing contract identical to cablese for
# future reuse: set TCB_RECORD_DECODE_SLOTS>1 to tile the passages.
BUNDLE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && sleep $(( (SLURM_ARRAY_TASK_ID * 7 + RANDOM) % 120 ))
: "${TCB_RECORD_DECODE_RUN_ID:?export TCB_RECORD_DECODE_RUN_ID=<stamp> before submitting; do NOT pre-create results/record_decode_<runid>}"
cd "$BUNDLE_DIR"
export PYTHONUNBUFFERED=1
export TCB_TREAT_401_TRANSIENT=1
export TCB_RETRY_ATTEMPTS=15
RESULTS="results/record_decode_${TCB_RECORD_DECODE_RUN_ID}"
SOURCE="${TCB_RECORD_DECODE_SOURCE:-results/cablese_cbl2}"

if [ "${TCB_RECORD_DECODE_MODE:-}" = "merge" ]; then
  # merge is one scheduler pass on slot 0 only
  [ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ] && exit 0
  exec python3 -m ab.record_decode record-decode-merge --results-dir "$RESULTS"
fi

TOTAL="${TCB_RECORD_DECODE_PASSAGES:-50}"
SLOTS="${TCB_RECORD_DECODE_SLOTS:-1}"
BASE=$((TOTAL / SLOTS))
REM=$((TOTAL % SLOTS))
ID="${SLURM_ARRAY_TASK_ID:?array submission required}"
LIMIT=$((BASE + (ID < REM ? 1 : 0)))
OFFSET=$((ID * BASE + (ID < REM ? ID : REM)))
if [ "$LIMIT" -eq 0 ]; then
  echo "slot $ID: no passages assigned (TOTAL=$TOTAL); exiting"
  exit 0
fi
if [ "$SLOTS" -gt 1 ]; then
  exec python3 -m ab.record_decode record-decode ${TCB_MAX_SECONDS:+--max-seconds $TCB_MAX_SECONDS} --max-tokens "${TCB_MAX_TOKENS:-4000}" --source-run "$SOURCE" --passages "$TOTAL" \
    --task-offset "$OFFSET" --task-limit "$LIMIT" \
    --slice-id "$(printf '%04d' "$ID")" --results-dir "$RESULTS"
fi
exec python3 -m ab.record_decode record-decode ${TCB_MAX_SECONDS:+--max-seconds $TCB_MAX_SECONDS} --max-tokens "${TCB_MAX_TOKENS:-4000}" --source-run "$SOURCE" --passages "$TOTAL" \
  --task-offset "$OFFSET" --task-limit "$LIMIT" --results-dir "$RESULTS"
