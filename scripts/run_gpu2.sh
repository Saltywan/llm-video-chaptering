#!/usr/bin/env bash
# Physical GPU2 queue (authorized 2026-11-17), bound by UUID with its own lock.
#   1. RFT (rft_self_samples_v1) for M1 and M2: select best-of-4 own samples on training windows,
#      one-pass LoRA, dev-200 FT inference paired with the stage-5 dev ZS arm. Validation only.
#   2. E1 zero-shot TEST run with the frozen driver/settings (moved off GPU0), then E1 pair scoring.
set -uo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-d7e5651f-45ed-fd48-2bd4-7f38cc45e13b
run=artifacts/rft-v1
s5=artifacts/stage5-v1
dev=artifacts/windows/dev-v2
test=artifacts/heldout-test-v1
pack=artifacts/windows/test-v1
exec >> "$run/gpu2-queue.log" 2>&1

note() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >> "$run/status.tsv"; }
run_step() {
  local name="$1"; shift
  note "$name	START"
  "$@" > "$run/$name.log" 2>&1
  local code=$?
  note "$name	EXIT_$code"
  (( code == 0 )) || printf '%s failed (exit %s); see %s/%s.log. No retry.\n' "$name" "$code" "$run" "$name" >> "$run/issues.txt"
  return "$code"
}
cpu() { env CUDA_VISIBLE_DEVICES= "$@"; }
vllm() {
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn "$VLLM_PYTHON" -u scripts/infer_vllm_pair.py "$@"
}
score() {  # name references predictions output [compare]
  local name="$1" refs="$2" preds="$3" output="$4" compare="${5:-}"
  local extra=()
  [[ -n "$compare" ]] && extra=(--compare-predictions "$compare")
  run_step "$name" cpu .venv/bin/python -u scripts/evaluate_predictions.py \
    --references "$refs" --predictions "$preds" "${extra[@]}" --output "$output" \
    --semantic-model FacebookAI/roberta-large \
    --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
    --semantic-layers 17 --semantic-device cpu
}

note "waiting for RFT samples"
until [[ -f "$run/samples/M1.DONE" && -f "$run/samples/M2.DONE" ]]; do sleep 30; done
for alias in M1 M2; do
  data="$dev/rft_${alias}_v1.jsonl"
  run_step "$alias-rft-select" cpu .venv/bin/python -u scripts/rft_build.py select \
    --train-windows "$dev/train_windows.jsonl" --manifest "$run/train_manifest.jsonl" \
    --samples "$run"/samples/"$alias"-seed*/windows.jsonl --min-f1 0.5 --output "$data" || continue
  run_step "$alias-rft-train" .venv/bin/python -u scripts/train_lora.py --alias "$alias" --windows "$data" \
    --max-steps 0 --target-modules q_proj,k_proj,v_proj,o_proj --learning-rate 2e-4 --lr-schedule cosine \
    --warmup-steps 3 --grad-accum 8 --checkpoint-every 100 --output "$run/$alias-rft-lora" || continue
  run_step "$alias-rft-dev-infer" vllm --alias "$alias" --arms FT --adapter "$run/$alias-rft-lora/adapter" \
    --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev" \
    --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 \
    --purpose "RFT dev-200 validation; not held-out test" || continue
  score "$alias-rft-dev-paired" "$dev/validation_records.jsonl" "$run/$alias-dev/FT/predictions.jsonl" \
    "$run/$alias-rft-dev-paired.json" "$s5/$alias-dev/ZS/predictions.jsonl" || true
done

# E1 zero-shot test, once, frozen settings (GPU0's queue skips it via a placeholder directory).
if cpu .venv/bin/python scripts/freeze.py --verify; then
  run_step E1-test-zs-infer .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias E1 \
    --windows "$pack/test_windows.jsonl" --output "$test/E1-ZS-gpu2" --square-budget $((8 * 8192 * 8192)) \
    --purpose "held-out test; frozen configuration configs/frozen_test.json; run once (physical GPU2)"
  until grep -q -E 'E1-test-ft-infer	EXIT' "$test/status.tsv"; do sleep 60; done
  if [[ -f "$test/E1-FT/predictions.jsonl" && -f "$test/E1-ZS-gpu2/predictions.jsonl" ]]; then
    score E1-test-paired "$pack/test_records.jsonl" "$test/E1-FT/predictions.jsonl" "$test/E1-test-paired.json" \
      "$test/E1-ZS-gpu2/predictions.jsonl" || true
    run_step envelope-E1-ZS cpu .venv/bin/python -u scripts/rescore_envelope.py --windows "$pack/test_windows.jsonl" \
      --results "$test/E1-ZS-gpu2/windows.jsonl" --output "$test/envelope-E1-ZS.predictions.jsonl" \
      && score envelope-E1-ZS-metrics "$pack/test_records.jsonl" "$test/envelope-E1-ZS.predictions.jsonl" \
           "$test/envelope-E1-ZS-metrics.json" || true
  fi
else
  note "freeze verification failed; E1 test ZS not run"
fi
note QUEUE_FINISHED
