#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="../model/experiment/20250819_qwen_2_5_7b_par_beaver_general/checkpoint-2583/"

python -m reward.serve_online.reward_server \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8300 \
  --device cuda:1 \
  --dtype bfloat16 \
  --attn-impl flash_attention_2 \
  --max-length 4096 \
  --padding max_length \
  --truncation \
  --micro-batch-size 16 \
  --log-file reward_qwen_server.log \
  --log-rotation "500 MB" \
  --workers 1
