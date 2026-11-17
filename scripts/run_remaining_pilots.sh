#!/usr/bin/env bash
# Continue the three remaining bounded pilots, with one GPU workload at a time.
set -euo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
.venv/bin/python -u scripts/train_lora.py --alias M2 \
  --windows artifacts/windows/pilot-v1/train_windows.jsonl \
  --output artifacts/learning/M2-v1 --max-steps 64
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn "$VLLM_PYTHON" -u scripts/infer_vllm_pair.py \
  --alias M2 --windows artifacts/windows/pilot-v1/validation_windows.jsonl \
  --adapter artifacts/learning/M2-v1/adapter \
  --output artifacts/inference/M2-vllm-native-pilot-v1 \
  --max-num-seqs 100 --max-new-tokens 32768
CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/evaluate_predictions.py \
  --references artifacts/windows/pilot-v1/validation_records.jsonl \
  --predictions artifacts/inference/M2-vllm-native-pilot-v1/FT/predictions.jsonl \
  --compare-predictions artifacts/inference/M2-vllm-native-pilot-v1/ZS/predictions.jsonl \
  --output artifacts/inference/M2-vllm-native-pilot-v1/paired.json \
  --semantic-model FacebookAI/roberta-large \
  --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
  --semantic-layers 17 --semantic-device cpu
bash scripts/run_learning_validation.sh E2
bash scripts/run_learning_validation.sh E1
