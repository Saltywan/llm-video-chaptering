#!/usr/bin/env bash
set -euo pipefail
cd <server-project-dir>

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
.venv/bin/python -u scripts/infer_local.py --alias M1 --windows artifacts/windows/pilot-v1/validation_windows.jsonl --output artifacts/inference/M1-ZS-pilot-v1 --max-new-tokens 4096 --reasoning-budget 1024

CUDA_VISIBLE_DEVICES= .venv/bin/python -c 'import json; d=json.load(open("artifacts/inference/M1-ZS-pilot-v1/summary.json")); assert d["status_counts"].get("error",0)<d["requested_videos"], "All requests failed operationally; inspect before continuing"'

.venv/bin/python -u scripts/infer_local.py --alias M1 --windows artifacts/windows/pilot-v1/validation_windows.jsonl --adapter artifacts/learning/M1-v1/adapter --output artifacts/inference/M1-FT-pilot-v1 --max-new-tokens 4096 --reasoning-budget 1024

CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/evaluate_predictions.py --references artifacts/windows/pilot-v1/validation_records.jsonl --predictions artifacts/inference/M1-FT-pilot-v1/predictions.jsonl --compare-predictions artifacts/inference/M1-ZS-pilot-v1/predictions.jsonl --output artifacts/inference/M1-paired-pilot-v1.json --semantic-model FacebookAI/roberta-large --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 --semantic-layers 17 --semantic-device cpu
