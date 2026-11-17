#!/usr/bin/env bash
# held-out test pass: eight frozen configurations, each run once on GPU0.
# Waits for stage 5 section B (E1 pilot dev evaluation), stops stage 5 before its section C,
# then runs ZS+FT per backbone (fast backbones first) and scores each pair on CPU.
# The test pack must already exist (scripts/prepare_test.py, built after configs/frozen_test.json).
# Failures are logged, never rerun with changes.
set -uo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
s5=artifacts/stage5-v1
run=artifacts/heldout-test-v1
pack=artifacts/windows/test-v1
purpose="held-out test; frozen configuration configs/frozen_test.json; run once"
mkdir "$run" || exit 1
exec > "$run/queue.log" 2>&1

note() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >> "$run/status.tsv"; }
run_step() {
  local name="$1"; shift
  note "$name	START"
  "$@" > "$run/$name.log" 2>&1
  local code=$?
  note "$name	EXIT_$code"
  (( code == 0 )) || printf '%s failed (exit %s); see %s/%s.log. Not rerun.\n' "$name" "$code" "$run" "$name" >> "$run/issues.txt"
  return "$code"
}
cpu() { env CUDA_VISIBLE_DEVICES= "$@"; }
vllm() {
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn "$VLLM_PYTHON" -u scripts/infer_vllm_pair.py "$@"
}
score() {  # name predictions [compare-predictions]
  local name="$1" preds="$2" compare="${3:-}"
  local extra=()
  [[ -n "$compare" ]] && extra=(--compare-predictions "$compare")
  run_step "$name" cpu .venv/bin/python -u scripts/evaluate_predictions.py \
    --references "$pack/test_records.jsonl" --predictions "$preds" "${extra[@]}" --output "$run/$name.json" \
    --semantic-model FacebookAI/roberta-large \
    --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
    --semantic-layers 17 --semantic-device cpu
}
frozen() { cpu .venv/bin/python scripts/freeze.py --verify; }
adapter() { cpu .venv/bin/python -c "import json,sys;print(json.load(open('configs/frozen_test.json'))['configurations'][sys.argv[1]+'-FT']['adapter'])" "$1"; }

note "waiting for stage 5 section B and the test pack"
until grep -q -E 'M1-full-zs-infer	START|QUEUE_FINISHED' "$s5/status.tsv" && [[ -f "$pack/manifest.json" ]]; do sleep 30; done
grep -q '"preparation_gate_passed": true' "$pack/manifest.json" || { note "test pack gate failed"; exit 1; }
queue_pid=$(pgrep -u "$USER" -f '^bash scripts/run_stage5.sh$' || true)
if [[ -n "$queue_pid" ]]; then
  kill "$queue_pid"
  sleep 5
  for pid in $(pgrep -u "$USER" -f "infer_vllm_pair.py .*--output $s5/M1-full" || true); do kill "$pid"; done
  note "stage 5 stopped before section C (queue pid $queue_pid)"
  printf '%s\tSTOPPED_BY_TEST_QUEUE before section C (full-validation ZS on GPU0)\n' "$(date -u +%FT%TZ)" >> "$s5/status.tsv"
fi
for _ in $(seq 120); do
  used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u="$CUDA_VISIBLE_DEVICES" '$1==u{print $2}')
  (( used <= 16 )) && break
  sleep 10
done
note "GPU0 memory after hand-over: ${used} MiB"

for alias in M2 M1; do
  frozen || { note "freeze verification failed"; exit 1; }
  run_step "$alias-test-infer" vllm --alias "$alias" --arms ZS FT --adapter "$(adapter "$alias")" \
    --windows "$pack/test_windows.jsonl" --output "$run/$alias" \
    --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 --purpose "$purpose" || continue
  score "$alias-test-paired" "$run/$alias/FT/predictions.jsonl" "$run/$alias/ZS/predictions.jsonl" || true
done
for spec in "E2 $((16 * 8192 * 8192))" "E1 $((8 * 8192 * 8192))"; do
  set -- $spec
  alias="$1"; budget="$2"
  frozen || { note "freeze verification failed"; exit 1; }
  run_step "$alias-test-ft-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias "$alias" \
    --adapter "$(adapter "$alias")" --windows "$pack/test_windows.jsonl" --output "$run/$alias-FT" \
    --square-budget "$budget" --purpose "$purpose"
  run_step "$alias-test-zs-infer" .venv/bin/python -u scripts/infer_seq2seq_batched.py --alias "$alias" \
    --windows "$pack/test_windows.jsonl" --output "$run/$alias-ZS" --square-budget "$budget" --purpose "$purpose"
  if [[ -f "$run/$alias-FT/predictions.jsonl" && -f "$run/$alias-ZS/predictions.jsonl" ]]; then
    score "$alias-test-paired" "$run/$alias-FT/predictions.jsonl" "$run/$alias-ZS/predictions.jsonl" || true
  fi
done

# Secondary: lenient envelope-v1 reading of every test run.
for item in M2/ZS M2/FT M1/ZS M1/FT E2-ZS E2-FT E1-ZS E1-FT; do
  results="$run/$item/windows.jsonl"; label="${item//\//-}"
  [[ -f "$results" ]] || continue
  run_step "envelope-$label" cpu .venv/bin/python -u scripts/rescore_envelope.py --windows "$pack/test_windows.jsonl" \
    --results "$results" --output "$run/envelope-$label.predictions.jsonl" || continue
  score "envelope-$label-metrics" "$run/envelope-$label.predictions.jsonl" || true
done
note QUEUE_FINISHED
