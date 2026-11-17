#!/usr/bin/env bash
# Stage 4, unattended, after stage 3. Development/validation only.
#   A. seed-noise check: M1/M2 dev runs with two more seeds (GPU0, sequential)
#   B. lenient envelope-v1 rescoring of every finished run (CPU)
# Failures are logged and the queue moves on. No retry, no repair, no test data.
set -uo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
s3=artifacts/stage3-v1
run=artifacts/stage4-v1
dev=artifacts/windows/dev-v2
full=artifacts/windows/common-v2
v1=artifacts/windows/zeroshot-validation-v1
mkdir "$run" || exit 1
exec > "$run/queue.log" 2>&1
printf '%s\tWAITING\n' "$(date -u +%FT%TZ)" >> "$run/status.tsv"
until grep -q QUEUE_FINISHED "$s3/status.tsv" 2>/dev/null; do sleep 60; done

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

# A. Two more seeds for the causal dev runs (seed 393 is the stage-3 run).
for alias in M1 M2; do
  adapter="$s3/$alias-self-thought-v3/adapter"
  for seed in 1393 2393; do
    out="$run/$alias-dev-seed$seed"
    if [[ -f "$adapter/adapter_model.safetensors" ]]; then
      arms=(ZS FT); extra=(--adapter "$adapter")
    else
      arms=(ZS); extra=()
    fi
    run_step "$alias-dev-seed$seed-infer" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn "$VLLM_PYTHON" -u \
      scripts/infer_vllm_pair.py --alias "$alias" --arms "${arms[@]}" "${extra[@]}" --seed "$seed" \
      --windows "$dev/validation_windows.jsonl" --output "$out" \
      --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 || continue
    if [[ -f "$out/FT/predictions.jsonl" ]]; then
      score "$alias-dev-seed$seed-paired" "$dev/validation_records.jsonl" \
        "$out/FT/predictions.jsonl" "$out/ZS/predictions.jsonl" || true
    else
      score "$alias-dev-seed$seed-zs" "$dev/validation_records.jsonl" "$out/ZS/predictions.jsonl" || true
    fi
  done
done

# B. Lenient rescoring: label, window pack, per-window results.
while read -r label pack results; do
  [[ -f "$results" ]] || { printf 'envelope %s skipped: %s missing\n' "$label" "$results" >> "$run/issues.txt"; continue; }
  run_step "envelope-$label" cpu .venv/bin/python -u scripts/rescore_envelope.py \
    --windows "$pack/validation_windows.jsonl" --results "$results" \
    --output "$run/envelope-$label.predictions.jsonl" || continue
  score "envelope-$label-metrics" "$pack/validation_records.jsonl" "$run/envelope-$label.predictions.jsonl" || true
done <<EOF
packv1-M1-zs $v1 artifacts/zeroshot-validation-v1/M1/ZS/windows.jsonl
packv1-M2-zs $v1 artifacts/zeroshot-validation-v1/M2/ZS/windows.jsonl
packv1-E2-zs $v1 artifacts/zeroshot-validation-v2/E2/windows.jsonl
packv1-E1-zs $v1 artifacts/zeroshot-validation-v2/E1/windows.jsonl
dev-M1-zs $dev $s3/M1-dev/ZS/windows.jsonl
dev-M1-ft $dev $s3/M1-dev/FT/windows.jsonl
dev-M2-zs $dev $s3/M2-dev/ZS/windows.jsonl
dev-M2-ft $dev $s3/M2-dev/FT/windows.jsonl
dev-E2-zs $dev $s3/E2-dev-ZS/windows.jsonl
dev-E2-ft $dev $s3/E2-dev-FT/windows.jsonl
dev-E1-zs $dev $s3/E1-dev-ZS/windows.jsonl
dev-E1-ft $dev $s3/E1-dev-FT/windows.jsonl
full-M1-zs $full $s3/M1-full/ZS/windows.jsonl
full-M2-zs $full $s3/M2-full/ZS/windows.jsonl
full-E2-zs $full $s3/E2-full/windows.jsonl
full-E1-zs $full $s3/E1-full/windows.jsonl
EOF
printf '%s\tQUEUE_FINISHED\n' "$(date -u +%FT%TZ)" >> "$run/status.tsv"
