#!/usr/bin/env bash
# Stage 6, unattended and strictly sequential on GPU0. Development/validation only, no test data.
#   0. wait for stage 5 sections A/B; stop stage 5 before its section C (full-validation
#      zero-shot on GPU0, superseded by the GPU1 endpoint runs and too slow for E1)
#   1. speed benchmarks: checkpointing + padded token batches (+ selective logits for M1/M2)
#   2. one shared training-video subset for all four backbones, sized to a 9.5 h summed budget
#   3. full LoRA training on that subset, then dev-200 FT inference paired with the stage-5 dev ZS
#   4. lenient envelope-v1 rescoring of the dev and pack-v1 runs (CPU)
# Failures are logged and the queue moves on. No retry, no repair.
set -uo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
s5=artifacts/stage5-v1
run=artifacts/stage6-v1
dev=artifacts/windows/dev-v2
full=artifacts/windows/common-v2
v1=artifacts/windows/zeroshot-validation-v1
subset="$full/train_subset_stage6.jsonl"
mkdir "$run" || exit 1
exec > "$run/queue.log" 2>&1

note() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >> "$run/status.tsv"; }
run_step() {
  local name="$1"; shift
  note "$name	START"
  "$@" > "$run/$name.log" 2>&1
  local code=$?
  note "$name	EXIT_$code"
  (( code == 0 )) || printf '%s failed (exit %s); see %s/%s.log. No retry attempted.\n' "$name" "$code" "$run" "$name" >> "$run/issues.txt"
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

# 0. Hand-over from stage 5 (same owner, same queue family; verified by exact command line).
note "waiting for stage 5 sections A/B"
until grep -q -E 'M1-full-zs-infer	START|QUEUE_FINISHED' "$s5/status.tsv"; do sleep 30; done
queue_pid=$(pgrep -u "$USER" -f '^bash scripts/run_stage5.sh$' || true)
if [[ -n "$queue_pid" ]]; then
  kill "$queue_pid"
  sleep 5
  for pid in $(pgrep -u "$USER" -f "infer_vllm_pair.py .*--output $s5/M1-full" || true); do kill "$pid"; done
  note "stage 5 stopped before section C (queue pid $queue_pid)"
  printf '%s\tSTOPPED_BY_STAGE6 before section C (full-validation ZS on GPU0)\n' "$(date -u +%FT%TZ)" >> "$s5/status.tsv"
fi
for _ in $(seq 120); do
  used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u="$CUDA_VISIBLE_DEVICES" '$1==u{print $2}')
  (( used <= 16 )) && break
  sleep 10
done
note "GPU0 memory after hand-over: ${used} MiB"

common=(--target-modules q_proj,k_proj,v_proj,o_proj --learning-rate 2e-4 --lr-schedule cosine --warmup-steps 10)
declare -A tokens=([M1]=49152 [M2]=32768 [E2]=65536 [E1]=32768)
declare -A mode
selective() { [[ "$1" == M1 || "$1" == M2 ]] && echo --selective-logits; }
rows() { [[ "$1" == M1 || "$1" == M2 ]] && echo --max-batch-windows 8; }

# 1. Benchmarks (64 windows) in the exact full-run mode; fall back to one window per step.
for alias in M1 M2 E2 E1; do
  if run_step "$alias-bench-batched" .venv/bin/python -u scripts/train_lora.py --alias "$alias" \
       --windows "$full/train_windows.jsonl" --max-steps 64 "${common[@]}" $(selective "$alias") \
       --batch-tokens "${tokens[$alias]}" $(rows "$alias") --grad-accum 2 --output "$run/$alias-bench-batched"; then
    mode[$alias]=batched
  elif run_step "$alias-bench-safe" .venv/bin/python -u scripts/train_lora.py --alias "$alias" \
       --windows "$full/train_windows.jsonl" --max-steps 32 "${common[@]}" \
       --grad-accum 8 --output "$run/$alias-bench-safe"; then
    mode[$alias]=safe
  fi
done
note "modes: M1=${mode[M1]:-none} M2=${mode[M2]:-none} E2=${mode[E2]:-none} E1=${mode[E1]:-none}"

# 2. Shared subset (only backbones with a working mode count toward the budget).
bench=()
for alias in M1 M2 E2 E1; do
  [[ -n "${mode[$alias]:-}" ]] && bench+=(--benchmark "$alias" "$run/$alias-bench-${mode[$alias]}")
done
run_step plan-subset cpu .venv/bin/python -u scripts/plan_train_subset.py --windows "$full/train_windows.jsonl" \
  "${bench[@]}" --budget-hours 9.5 --output "$subset" || exit 1

# 3. Full training on the subset, then paired dev-200 evaluation against the stage-5 dev ZS arm.
train_full() {  # alias
  local alias="$1"
  local extra=(--grad-accum 8)   # safe mode: the stage-5 pilot settings, full logits
  [[ "${mode[$alias]}" == batched ]] && extra=(--batch-tokens "${tokens[$alias]}" $(rows "$alias") $(selective "$alias") --grad-accum 2)
  run_step "$alias-train" .venv/bin/python -u scripts/train_lora.py --alias "$alias" --windows "$subset" \
    --max-steps 0 "${common[@]}" "${extra[@]}" --checkpoint-every 50 \
    --output "$run/$alias-full-lora"
}
for alias in M1 E2 M2 E1; do
  [[ -n "${mode[$alias]:-}" ]] || continue
  train_full "$alias" || continue
  adapter="$run/$alias-full-lora/adapter"
  if [[ "$alias" == M* ]]; then
    run_step "$alias-dev-ft-infer" vllm --alias "$alias" --arms FT --adapter "$adapter" \
      --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev" \
      --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 || continue
    score "$alias-dev-paired" "$dev/validation_records.jsonl" \
      "$run/$alias-dev/FT/predictions.jsonl" "$s5/$alias-dev/ZS/predictions.jsonl" || true
  else
    budget=$((16 * 8192 * 8192)); [[ "$alias" == E1 ]] && budget=$((8 * 8192 * 8192))
    run_step "$alias-dev-ft-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias "$alias" \
      --adapter "$adapter" --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev-FT" \
      --square-budget "$budget" || continue
    score "$alias-dev-paired" "$dev/validation_records.jsonl" \
      "$run/$alias-dev-FT/predictions.jsonl" "$s5/$alias-dev-ZS/predictions.jsonl" || true
  fi
done

# 4. Lenient rescoring: label, window pack, per-window results.
while read -r label pack results; do
  [[ -f "$results" ]] || { printf 'envelope %s skipped: %s missing\n' "$label" "$results" >> "$run/issues.txt"; continue; }
  run_step "envelope-$label" cpu .venv/bin/python -u scripts/rescore_envelope.py \
    --windows "$pack/validation_windows.jsonl" --results "$results" \
    --output "$run/envelope-$label.predictions.jsonl" || continue
  score "envelope-$label-metrics" "$pack/validation_records.jsonl" "$run/envelope-$label.predictions.jsonl" || true
done <<EOF
packv1-E2-zs $v1 artifacts/zeroshot-validation-v2/E2/windows.jsonl
packv1-E1-zs $v1 artifacts/zeroshot-validation-v2/E1/windows.jsonl
dev-E2-zs $dev $s5/E2-dev-ZS/windows.jsonl
dev-E1-zs $dev $s5/E1-dev-ZS/windows.jsonl
dev-E1-pilot-ft $dev $s5/E1-dev-FT/windows.jsonl
dev-M1-zs $dev $s5/M1-dev/ZS/windows.jsonl
dev-M2-zs $dev $s5/M2-dev/ZS/windows.jsonl
dev-M1-full-ft $dev $run/M1-dev/FT/windows.jsonl
dev-M2-full-ft $dev $run/M2-dev/FT/windows.jsonl
dev-E2-full-ft $dev $run/E2-dev-FT/windows.jsonl
dev-E1-full-ft $dev $run/E1-dev-FT/windows.jsonl
EOF
note QUEUE_FINISHED
