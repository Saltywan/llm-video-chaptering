#!/usr/bin/env bash
# One reviewed training-format change; all other pilot settings are preserved.
set -euo pipefail
cd <server-project-dir>
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=GPU-05f1cfd1-0219-e850-913d-159b6ed72d60
CUDA_VISIBLE_DEVICES= .venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
audit = json.loads(Path('artifacts/runtime/gemma-final-phase-v2-audit.json').read_text())
digest = hashlib.sha256(Path('artifacts/windows/pilot-v1/train_windows.jsonl').read_bytes()).hexdigest()
assert audit['passed'] and audit['gemma_final_context'] == 'closed_thought_v2'
assert audit['windows_sha256'] == digest and audit['models']['M2']['windows'] == 74
PY
snapshot=artifacts/source_snapshots/m2-closed-v2
mkdir -p "$snapshot"
cp scripts/train_lora.py src/chaptering/modeling.py src/chaptering/supervision.py "$snapshot/"
.venv/bin/python -u scripts/train_lora.py --alias M2 \
  --windows artifacts/windows/pilot-v1/train_windows.jsonl \
  --output artifacts/learning/M2-v2 --max-steps 64 --gemma-final-context closed_thought_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn "$VLLM_PYTHON" -u scripts/infer_vllm_pair.py \
  --alias M2 --windows artifacts/windows/pilot-v1/validation_windows.jsonl \
  --adapter artifacts/learning/M2-v2/adapter \
  --output artifacts/inference/M2-vllm-native-pilot-v2 \
  --max-num-seqs 100 --max-new-tokens 32768
CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/evaluate_predictions.py \
  --references artifacts/windows/pilot-v1/validation_records.jsonl \
  --predictions artifacts/inference/M2-vllm-native-pilot-v2/FT/predictions.jsonl \
  --compare-predictions artifacts/inference/M2-vllm-native-pilot-v2/ZS/predictions.jsonl \
  --output artifacts/inference/M2-vllm-native-pilot-v2/paired.json \
  --semantic-model FacebookAI/roberta-large \
  --semantic-revision 722cf37b1afa9454edce342e7895e588b6ff1d59 \
  --semantic-layers 17 --semantic-device cpu
