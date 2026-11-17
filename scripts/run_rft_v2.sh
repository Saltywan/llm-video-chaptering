#!/usr/bin/env bash
# RFT v2 (rft_self_samples_v1 recipe, scaled): one backbone per owned GPU, bound by UUID.
#   usage: scripts/run_rft_v2.sh M1 GPU-05f1cfd1-...   (GPU0)   |   M2 GPU-d7e5651f-...   (GPU2)
# 1. Sample 4 seeds on the 1,006 pilot training windows that v1 did not sample, using the local
#    vLLM driver with the same sampling settings and per-window seed scheme as the v1 endpoint runs.
# 2. Select best-of-N own samples over all 1,306 pilot windows (v1 endpoint + v2 local samples),
#    owned-start F1 >= 0.5, unchanged recipe.
# 3. One-pass LoRA with the v1 hyperparameters. 4. Dev-200 inference, paired with stage-5 ZS and with RFT v1.
# Validation only; training split only for sampling. Not a held-out test run.
set -uo pipefail
alias="$1" gpu="$2"
case "$alias:$gpu" in
  M1:GPU-05f1cfd1-0219-e850-913d-159b6ed72d60|M2:GPU-d7e5651f-45ed-fd48-2bd4-7f38cc45e13b) ;;
  *) echo "unexpected alias/GPU pairing" >&2; exit 2 ;;
esac
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src CUDA_VISIBLE_DEVICES="$gpu"
run=artifacts/rft-v2
v1=artifacts/rft-v1
s5=artifacts/stage5-v1
dev=artifacts/windows/dev-v2
exec >> "$run/$alias-queue.log" 2>&1

note() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >> "$run/status-$alias.tsv"; }
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
score() {  # name references predictions output compare
  run_step "$1" cpu .venv/bin/python -u scripts/evaluate_predictions.py \
    --references "$2" --predictions "$3" --compare-predictions "$5" --output "$4" \
    --semantic-model FacebookAI/roberta-large \
    --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
    --semantic-layers 17 --semantic-device cpu
}

for seed in 1001 2002 3003 4004; do
  run_step "$alias-sample-seed$seed" vllm --alias "$alias" --arms ZS --windows "$run/train_manifest.jsonl" \
    --output "$run/samples/$alias-seed$seed" --seed "$seed" --max-num-seqs 256 --batch-size 1006 \
    --max-new-tokens 32768 --purpose "RFT v2 sampling on training windows only; not evaluation"
done
samples=("$v1"/samples/"$alias"-seed*/windows.jsonl)
for path in "$run"/samples/"$alias"-seed*/ZS/windows.jsonl; do [[ -f "$path" ]] && samples+=("$path"); done
data="$dev/rft_${alias}_v2.jsonl"
run_step "$alias-rft-select" cpu .venv/bin/python -u scripts/rft_build.py select \
  --train-windows "$dev/train_windows.jsonl" --manifest "$run/union_manifest.jsonl" \
  --samples "${samples[@]}" --min-f1 0.5 --output "$data" || { note QUEUE_FINISHED; exit 1; }
run_step "$alias-rft-train" .venv/bin/python -u scripts/train_lora.py --alias "$alias" --windows "$data" \
  --max-steps 0 --target-modules q_proj,k_proj,v_proj,o_proj --learning-rate 2e-4 --lr-schedule cosine \
  --warmup-steps 3 --grad-accum 8 --checkpoint-every 100 --output "$run/$alias-rft-lora" || { note QUEUE_FINISHED; exit 1; }
run_step "$alias-rft-dev-infer" vllm --alias "$alias" --arms FT --adapter "$run/$alias-rft-lora/adapter" \
  --windows "$dev/validation_windows.jsonl" --output "$run/$alias-dev" \
  --max-num-seqs 100 --batch-size 100 --max-new-tokens 32768 \
  --purpose "RFT v2 dev-200 validation; not held-out test" || { note QUEUE_FINISHED; exit 1; }
score "$alias-rft-dev-paired" "$dev/validation_records.jsonl" "$run/$alias-dev/FT/predictions.jsonl" \
  "$run/$alias-rft-dev-paired.json" "$s5/$alias-dev/ZS/predictions.jsonl" || true
score "$alias-v2-vs-v1-paired" "$dev/validation_records.jsonl" "$run/$alias-dev/FT/predictions.jsonl" \
  "$run/$alias-v2-vs-v1-paired.json" "$v1/$alias-dev/FT/predictions.jsonl" || true
note QUEUE_FINISHED
