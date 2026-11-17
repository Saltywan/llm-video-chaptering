#!/usr/bin/env bash
# Unattended validation-only queue. Log failures; do not repair, retry or train.
set -uo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
run=artifacts/zeroshot-validation-v1
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
run_step prepare env CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/prepare_windows.py \
  --data artifacts/data_audit/2026-11-16-resolved-v1 --output "$pack" \
  --train-videos 32 --validation-videos 1446 || exit 1
for alias in M1 M2 E2 E1; do
  output="$run/$alias"
  if [[ "$alias" == M1 || "$alias" == M2 ]]; then
    run_step "$alias-infer" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "$VLLM_PYTHON" -u scripts/infer_vllm_pair.py \
      --alias "$alias" --arms ZS --windows "$pack/validation_windows.jsonl" \
      --output "$output" --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 || continue
    predictions="$output/ZS/predictions.jsonl"
  else
    run_step "$alias-infer" .venv/bin/python -u scripts/infer_local.py \
      --alias "$alias" --windows "$pack/validation_windows.jsonl" \
      --output "$output" --max-new-tokens 4096 || continue
    predictions="$output/predictions.jsonl"
  fi
  run_step "$alias-score" env CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/evaluate_predictions.py \
    --references "$pack/validation_records.jsonl" --predictions "$predictions" \
    --output "$run/$alias-metrics.json" --semantic-model FacebookAI/roberta-large \
    --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
    --semantic-layers 17 --semantic-device cpu || true
done
printf '%s\tQUEUE_FINISHED\n' "$(date -u +%FT%TZ)" >> "$run/status.tsv"
