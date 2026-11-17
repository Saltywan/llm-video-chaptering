#!/usr/bin/env bash
# Batched zero-shot validation for the seq2seq backbones on the existing pack v1.
# Validation only. Log failures; do not repair, retry or train.
set -uo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
run=artifacts/zeroshot-validation-v2
pack=artifacts/windows/zeroshot-validation-v1
mkdir "$run" || exit 1
exec > "$run/queue.log" 2>&1
run_step() {
  local name="$1"; shift
  printf '%s\t%s\tSTART\n' "$(date -u +%FT%TZ)" "$name" >> "$run/status.tsv"
  "$@" > "$run/$name.log" 2>&1
  local code=$?
  printf '%s\t%s\tEXIT_%s\n' "$(date -u +%FT%TZ)" "$name" "$code" >> "$run/status.tsv"
  if (( code != 0 )); then
    printf '%s failed (exit %s); see %s/%s.log. No retry attempted.\n' "$name" "$code" "$run" "$name" >> "$run/issues.txt"
  fi
  return "$code"
}
for spec in "E2 $((16 * 8192 * 8192))" "E1 $((8 * 8192 * 8192))"; do
  set -- $spec
  alias="$1"; budget="$2"
  run_step "$alias-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py \
    --alias "$alias" --windows "$pack/validation_windows.jsonl" \
    --output "$run/$alias" --max-new-tokens 4096 --square-budget "$budget" || continue
  run_step "$alias-score" env CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/evaluate_predictions.py \
    --references "$pack/validation_records.jsonl" --predictions "$run/$alias/predictions.jsonl" \
    --output "$run/$alias-metrics.json" --semantic-model FacebookAI/roberta-large \
    --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
    --semantic-layers 17 --semantic-device cpu || true
done
printf '%s\tQUEUE_FINISHED\n' "$(date -u +%FT%TZ)" >> "$run/status.tsv"
