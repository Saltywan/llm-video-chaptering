#!/usr/bin/env bash
# One encoder-decoder at a time: bounded learning, validation, then CPU metrics.
set -euo pipefail
case "${1:-}" in
  E2|E1) alias="$1" ;;
  *) echo 'Usage: run_learning_validation.sh E2|E1' >&2; exit 2 ;;
esac
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
pack=artifacts/windows/pilot-v1
learning="artifacts/learning/${alias}-v1"
zs="artifacts/inference/${alias}-ZS-pilot-v1"
ft="artifacts/inference/${alias}-FT-pilot-v1"
for output in "$learning" "$zs" "$ft" "artifacts/inference/${alias}-paired-pilot-v1.json"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    exit 1
  fi
done
.venv/bin/python -u scripts/train_lora.py --alias "$alias" \
  --windows "$pack/train_windows.jsonl" --output "$learning" --max-steps 64
.venv/bin/python -u scripts/infer_local.py --alias "$alias" \
  --windows "$pack/validation_windows.jsonl" --output "$zs" \
  --max-new-tokens 4096 --reasoning-budget 1024
CUDA_VISIBLE_DEVICES= .venv/bin/python - "$zs/summary.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['status_counts'].get('error', 0) < report['requested_videos'], \
    'All requests failed operationally; inspect before continuing'
PY
.venv/bin/python -u scripts/infer_local.py --alias "$alias" \
  --windows "$pack/validation_windows.jsonl" --adapter "$learning/adapter" \
  --output "$ft" --max-new-tokens 4096 --reasoning-budget 1024
CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/evaluate_predictions.py \
  --references "$pack/validation_records.jsonl" --predictions "$ft/predictions.jsonl" \
  --compare-predictions "$zs/predictions.jsonl" \
  --output "artifacts/inference/${alias}-paired-pilot-v1.json" \
  --semantic-model FacebookAI/roberta-large \
  --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
  --semantic-layers 17 --semantic-device cpu
