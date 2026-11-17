#!/usr/bin/env bash
# RFT sampling through the existing GPU1 vLLM services (never managed here). Training windows only.
# Keys come from the environment (RFT_KEY_M1, RFT_KEY_M2), are piped to each process's stdin,
# and are never written to files or logs. Four seeds per model, run concurrently.
set -uo pipefail
cd <server-project-dir>
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=src
out=artifacts/rft-v1
manifest="$out/train_manifest.jsonl"
mkdir -p "$out/samples"
declare -A url=([M1]=http://127.0.0.1:48761/v1 [M2]=http://127.0.0.1:48762/v1)
for alias in M1 M2; do
  key_var="RFT_KEY_$alias"
  for seed in 1001 2002 3003 4004; do
    printf '%s\n' "${!key_var}" | .venv/bin/python -u scripts/infer_endpoint_zs.py --alias "$alias" \
      --windows "$manifest" --output "$out/samples/$alias-seed$seed" --base-url "${url[$alias]}" \
      --concurrency 16 --seed "$seed" --purpose "RFT sampling on training windows only; not evaluation" \
      > "$out/samples/$alias-seed$seed.log" 2>&1 &
  done
done
unset RFT_KEY_M1 RFT_KEY_M2
wait
for alias in M1 M2; do touch "$out/samples/$alias.DONE"; done
