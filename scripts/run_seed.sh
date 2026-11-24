#!/usr/bin/env bash
# PLAN-week14 M1+M2 (or M3+M4) for one seed run, unattended:
#   train (guarded: same recipe as v2) -> 10 dev-200 evals -> selection (release rule)
#   -> Spider test for the selected checkpoint (and step 900) -> BIRD dev for the selected checkpoint.
# Official Spider scoring runs afterwards in the CPU venv (scripts/official_eval.py --split test).
#
#   bash scripts/run_seed.sh v2-s1 1 [HH:MM]   # optional: do not start before this time today
# Waits until the GPU is free (another app using > FREE_MIB MiB holds the start).
set -euo pipefail
RUN=$1; SEED=$2; NOT_BEFORE=${3:-}
cd "$(dirname "$0")/.."
PY=~/venvs/tinysql/bin/python
LOG=results/${RUN}_runs.log
FREE_MIB=${FREE_MIB:-1800}
say() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

if [[ -n "$NOT_BEFORE" ]]; then
  while [[ $(date +%H:%M) < "$NOT_BEFORE" ]]; do sleep 300; done
fi
until used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) && (( used < FREE_MIB )); do
  say "waiting: GPU memory in use ${used} MiB"; sleep 300
done

say "start train $RUN seed $SEED"
$PY scripts/train_grpo.py --run-name "$RUN" --seed "$SEED" --pass-rate data/train_pass_rate.jsonl \
  --same-recipe-as checkpoints/v2/run_args.json > "results/train_${RUN}.log" 2>&1
say "done train"

for n in $(seq 100 100 1000); do
  out="results/dev200_Qwen__Qwen3-1.7B_${RUN}-checkpoint-${n}.json"
  [[ -f $out ]] && continue
  $PY scripts/eval_baseline.py --model Qwen/Qwen3-1.7B --subset data/dev200_ids.json \
    --adapter "checkpoints/${RUN}/checkpoint-${n}" >> "results/eval_${RUN}.log" 2>&1
done
say "done dev-200"
$PY scripts/compare_checkpoints.py "$RUN" --write-selection | tee "results/checkpoint_eval_${RUN}.txt"
SEL=$($PY -c "import json; print(json.load(open('results/selection_${RUN}.json'))['selected'])")
say "selected checkpoint-$SEL"

for n in $(printf '%s\n' "$SEL" 900 | sort -u); do
  say "start test checkpoint-$n"
  $PY scripts/eval_baseline.py --model Qwen/Qwen3-1.7B --split test \
    --adapter "checkpoints/${RUN}/checkpoint-${n}" >> "results/eval_${RUN}.log" 2>&1
done
say "start BIRD checkpoint-$SEL"
$PY scripts/eval_bird.py --adapter "checkpoints/${RUN}/checkpoint-${SEL}" --tag "B-${RUN}" >> "results/eval_${RUN}.log" 2>&1
say "done all"
