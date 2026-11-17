#!/usr/bin/env bash
# CPU only: lenient envelope-v1 rescoring of finished dev/pack-v1 runs (secondary metric).
set -uo pipefail
cd <server-project-dir>
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=src
out=artifacts/envelope-dev-v1; mkdir "$out" || exit 1
s5=artifacts/stage5-v1; dev=artifacts/windows/dev-v2; v1=artifacts/windows/zeroshot-validation-v1
while read -r label pack results; do
  [[ -f "$results" ]] || { echo "$label missing" >> "$out/issues.txt"; continue; }
  .venv/bin/python -u scripts/rescore_envelope.py --windows "$pack/validation_windows.jsonl" --results "$results" \
    --output "$out/$label.predictions.jsonl" > "$out/$label.log" 2>&1 || { echo "$label rescore failed" >> "$out/issues.txt"; continue; }
  .venv/bin/python -u scripts/evaluate_predictions.py --references "$pack/validation_records.jsonl" \
    --predictions "$out/$label.predictions.jsonl" --output "$out/$label-metrics.json" \
    --semantic-model FacebookAI/roberta-large --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
    --semantic-layers 17 --semantic-device cpu >> "$out/$label.log" 2>&1 || echo "$label score failed" >> "$out/issues.txt"
done <<LIST
dev-E2-zs $dev $s5/E2-dev-ZS/windows.jsonl
dev-E2-ft $dev $s5/E2-dev-FT/windows.jsonl
dev-E1-zs $dev $s5/E1-dev-ZS/windows.jsonl
dev-M1-zs $dev $s5/M1-dev/ZS/windows.jsonl
dev-M1-ft $dev $s5/M1-dev/FT/windows.jsonl
dev-M2-zs $dev $s5/M2-dev/ZS/windows.jsonl
dev-M2-ft $dev $s5/M2-dev/FT/windows.jsonl
packv1-E1-zs $v1 artifacts/zeroshot-validation-v2/E1/windows.jsonl
packv1-E2-zs $v1 artifacts/zeroshot-validation-v2/E2/windows.jsonl
LIST
echo done > "$out/DONE"
