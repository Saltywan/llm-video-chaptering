#!/usr/bin/env bash
# Stage 3, unattended and strictly sequential on GPU0. Development/validation only.
#   A. self_thought_v3 pilots for M1, M2 on the dev pack (thought generation, training, paired eval)
#   B. native seq2seq pilots for E2, E1 on the dev pack (training, zero-shot and adapted eval)
#   C. full-validation zero-shot on window pack common-v2 for all four backbones
# Failures are logged and the queue moves on. No retry, no repair, no test data.
set -uo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
run=artifacts/stage3-v1
dev=artifacts/windows/dev-v2
full=artifacts/windows/common-v2
mkdir "$run" || exit 1
exec > "$run/queue.log" 2>&1

# Wait for the batched zero-shot queue to release GPU0 and for the full pack to exist.
printf '%s\tWAITING\n' "$(date -u +%FT%TZ)" >> "$run/status.tsv"
until grep -q QUEUE_FINISHED artifacts/zeroshot-validation-v2/status.tsv 2>/dev/null; do sleep 60; done
until [[ -f "$full/manifest.json" ]]; do sleep 60; done

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
cpu() { env CUDA_VISIBLE_DEVICES= "$@"; }
vllm() {
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn "$VLLM_PYTHON" -u scripts/infer_vllm_pair.py "$@"
}
score() {  # name references predictions [compare-predictions]
  local name="$1" refs="$2" preds="$3" compare="${4:-}"
  local extra=()
  [[ -n "$compare" ]] && extra=(--compare-predictions "$compare")
  run_step "$name" cpu .venv/bin/python -u scripts/evaluate_predictions.py \
    --references "$refs" --predictions "$preds" "${extra[@]}" --output "$run/$name.json" \
    --semantic-model FacebookAI/roberta-large \
    --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
    --semantic-layers 17 --semantic-device cpu
}

# Gold-free copy of the dev training windows, for generating each model's own thoughts.
run_step dev-train-goldfree cpu .venv/bin/python -c '
import json, sys
with open(sys.argv[1]) as source, open(sys.argv[2], "x") as out:
    for line in source:
        out.write(json.dumps({"window": json.loads(line)["window"]}, ensure_ascii=False) + "\n")
' "$dev/train_windows.jsonl" "$run/dev_train_goldfree_windows.jsonl" || exit 1

# A. Causal self_thought_v3 pilots.
for alias in M1 M2; do
  run_step "$alias-thoughts" vllm --alias "$alias" --arms ZS \
    --windows "$run/dev_train_goldfree_windows.jsonl" --output "$run/$alias-thoughts" \
    --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 || continue
  run_step "$alias-train" .venv/bin/python -u scripts/train_lora.py --alias "$alias" \
    --windows "$dev/train_windows.jsonl" --thoughts "$run/$alias-thoughts/ZS/windows.jsonl" \
    --output "$run/$alias-self-thought-v3" --max-steps 0 --grad-accum 4 \
    --max-sequence-tokens 24000 --checkpoint-every 200 || continue
  run_step "$alias-dev-infer" vllm --alias "$alias" --arms ZS FT \
    --adapter "$run/$alias-self-thought-v3/adapter" \
    --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev" \
    --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 || continue
  score "$alias-dev-paired" "$dev/validation_records.jsonl" \
    "$run/$alias-dev/FT/predictions.jsonl" "$run/$alias-dev/ZS/predictions.jsonl" || true
done

# B. Seq2seq pilots (native recipe, batched greedy inference for both arms).
for spec in "E2 $((16 * 8192 * 8192))" "E1 $((8 * 8192 * 8192))"; do
  set -- $spec
  alias="$1"; budget="$2"
  run_step "$alias-dev-zs-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias "$alias" \
    --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev-ZS" --square-budget "$budget" || true
  run_step "$alias-train" .venv/bin/python -u scripts/train_lora.py --alias "$alias" \
    --windows "$dev/train_windows.jsonl" --output "$run/$alias-native-v1" \
    --max-steps 0 --grad-accum 4 --checkpoint-every 200 || continue
  run_step "$alias-dev-ft-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias "$alias" \
    --adapter "$run/$alias-native-v1/adapter" \
    --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev-FT" --square-budget "$budget" || continue
  score "$alias-dev-paired" "$dev/validation_records.jsonl" \
    "$run/$alias-dev-FT/predictions.jsonl" "$run/$alias-dev-ZS/predictions.jsonl" || true
done

# C. Full-validation zero-shot on pack common-v2.
for alias in M1 M2; do
  run_step "$alias-full-zs-infer" vllm --alias "$alias" --arms ZS \
    --windows "$full/validation_windows.jsonl" --output "$run/$alias-full" \
    --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 || continue
  score "$alias-full-zs" "$full/validation_records.jsonl" "$run/$alias-full/ZS/predictions.jsonl" || true
done
for spec in "E2 $((16 * 8192 * 8192))" "E1 $((8 * 8192 * 8192))"; do
  set -- $spec
  alias="$1"; budget="$2"
  run_step "$alias-full-zs-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias "$alias" \
    --windows "$full/validation_windows.jsonl" --output "$run/$alias-full" --square-budget "$budget" || continue
  score "$alias-full-zs" "$full/validation_records.jsonl" "$run/$alias-full/predictions.jsonl" || true
done
printf '%s\tQUEUE_FINISHED\n' "$(date -u +%FT%TZ)" >> "$run/status.tsv"
