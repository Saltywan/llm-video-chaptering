#!/usr/bin/env bash
# Stage 5, unattended and strictly sequential on GPU0. Development/validation only.
# Replaces the rest of stage 3 and all of stage 4 after the self_thought_v3 diagnosis.
#   A. direct final-answer LoRA pilots for M1, M2 (native template, fast training settings)
#   B. seq2seq pilots for E2, E1
#   C. full-validation zero-shot on window pack common-v2 for all four backbones
#   D. lenient envelope-v1 rescoring of every finished run (CPU)
# Failures are logged and the queue moves on. No retry, no repair, no test data.
set -uo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
s3=artifacts/stage3-v1
run=artifacts/stage5-v1
dev=artifacts/windows/dev-v2
full=artifacts/windows/common-v2
v1=artifacts/windows/zeroshot-validation-v1
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
# One-pass LoRA training. First try the fast settings (padded batches, no gradient
# checkpointing); if that step fails, run the conservative settings in a new directory.
train() {  # alias output-stem
  local alias="$1" stem="$2"
  local common=(--alias "$alias" --windows "$dev/train_windows.jsonl" --max-steps 0
                --target-modules q_proj,k_proj,v_proj,o_proj --learning-rate 2e-4
                --lr-schedule cosine --checkpoint-every 100)
  if run_step "$alias-train-fast" .venv/bin/python -u scripts/train_lora.py "${common[@]}" \
       --output "$run/$stem-fast" --batch-tokens 48000 --no-gradient-checkpointing --warmup-steps 10; then
    adapter="$run/$stem-fast/adapter"
  elif run_step "$alias-train-safe" .venv/bin/python -u scripts/train_lora.py "${common[@]}" \
       --output "$run/$stem-safe" --grad-accum 8 --warmup-steps 10; then
    adapter="$run/$stem-safe/adapter"
  else
    return 1
  fi
}

# A. Causal direct final-answer pilots (thinking stays enabled and unforced at inference).
if train M1 M1-direct; then
  run_step M1-dev-infer vllm --alias M1 --arms ZS FT --adapter "$adapter" \
    --windows "$dev/validation_windows.jsonl" --output "$run/M1-dev" \
    --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 \
    && score M1-dev-paired "$dev/validation_records.jsonl" \
         "$run/M1-dev/FT/predictions.jsonl" "$run/M1-dev/ZS/predictions.jsonl" || true
fi
if train M2 M2-direct; then
  run_step M2-dev-infer vllm --alias M2 --arms ZS FT --adapter "$adapter" \
    --windows "$dev/validation_windows.jsonl" --output "$run/M2-dev" \
    --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 \
    && score M2-dev-paired "$dev/validation_records.jsonl" \
         "$run/M2-dev/FT/predictions.jsonl" "$run/M2-dev/ZS/predictions.jsonl" || true
fi

# B. Seq2seq pilots (batched greedy inference for both arms).
for spec in "E2 $((16 * 8192 * 8192))" "E1 $((8 * 8192 * 8192))"; do
  set -- $spec
  alias="$1"; budget="$2"
  run_step "$alias-dev-zs-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias "$alias" \
    --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev-ZS" --square-budget "$budget" || true
  train "$alias" "$alias-native" || continue
  run_step "$alias-dev-ft-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias "$alias" \
    --adapter "$adapter" --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev-FT" \
    --square-budget "$budget" || continue
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

# D. Lenient rescoring: label, window pack, per-window results.
while read -r label pack results; do
  [[ -f "$results" ]] || { printf 'envelope %s skipped: %s missing\n' "$label" "$results" >> "$run/issues.txt"; continue; }
  run_step "envelope-$label" cpu .venv/bin/python -u scripts/rescore_envelope.py \
    --windows "$pack/validation_windows.jsonl" --results "$results" \
    --output "$run/envelope-$label.predictions.jsonl" || continue
  score "envelope-$label-metrics" "$pack/validation_records.jsonl" "$run/envelope-$label.predictions.jsonl" || true
done <<EOF
packv1-E2-zs $v1 artifacts/zeroshot-validation-v2/E2/windows.jsonl
packv1-E1-zs $v1 artifacts/zeroshot-validation-v2/E1/windows.jsonl
dev-M1-zs $dev $run/M1-dev/ZS/windows.jsonl
dev-M1-ft $dev $run/M1-dev/FT/windows.jsonl
dev-M2-zs $dev $run/M2-dev/ZS/windows.jsonl
dev-M2-ft $dev $run/M2-dev/FT/windows.jsonl
dev-E2-zs $dev $run/E2-dev-ZS/windows.jsonl
dev-E2-ft $dev $run/E2-dev-FT/windows.jsonl
dev-E1-zs $dev $run/E1-dev-ZS/windows.jsonl
dev-E1-ft $dev $run/E1-dev-FT/windows.jsonl
full-M1-zs $full $run/M1-full/ZS/windows.jsonl
full-M2-zs $full $run/M2-full/ZS/windows.jsonl
full-E2-zs $full $run/E2-full/windows.jsonl
full-E1-zs $full $run/E1-full/windows.jsonl
EOF
printf '%s\tQUEUE_FINISHED\n' "$(date -u +%FT%TZ)" >> "$run/status.tsv"
