#!/usr/bin/env bash
set -euo pipefail

cd /ossfs/workspace/yijian3/495804/Pragma-VL

pip install -U bitsandbytes
pip install -U accelerate

# ------------------------------------------------------------------------------
# Distributed config
# ------------------------------------------------------------------------------

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# ------------------------------------------------------------------------------
# Paths and run config
# ------------------------------------------------------------------------------

DEEPSPEED_CONFIG="zero2.json"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
SOURCE_DIR="../asset/0819_train_beaverori_replaced"

RUN_NAME="20250825_qwen_2_5_7b_par_beaver_general_verl"
OUTPUT_DIR="../model/experiment/${RUN_NAME}"

ENTRY_FILE="reward/trainer.py"

echo "Run name: ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"

# ------------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------

LEARNING_RATE="1e-6"
BATCH_SIZE="32"
GRAD_ACCUM_STEPS="1"

# Dataset definitions: "name:type:sampling_rate" separated by spaces
DATASET_DEFINITIONS=(
  "bt_data:bt:1.0"
  "mse_data:mse:1.0"
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

build_dataset_sources_json() {
  local -a defs=("$@")
  local json="["
  local first=1

  for definition in "${defs[@]}"; do
    IFS=":" read -r name type rate <<< "${definition}"

    if [[ ${first} -eq 1 ]]; then
      first=0
    else
      json+=","
    fi

    json+=$(printf '{"name":"%s","type":"%s","sampling_rate":%s}' "${name}" "${type}" "${rate}")
  done

  json+="]"
  printf "%s" "${json}"
}

DATASET_SOURCES_JSON="$(build_dataset_sources_json "${DATASET_DEFINITIONS[@]}")"
echo "Dataset sources JSON: ${DATASET_SOURCES_JSON}"

# ------------------------------------------------------------------------------
# CLI args
# ------------------------------------------------------------------------------

args=(
  --deepspeed "${DEEPSPEED_CONFIG}"
  --model_name_or_path "${MODEL_PATH}"
  --source_dir "${SOURCE_DIR}"
  --dataset_sources "'${DATASET_SOURCES_JSON}'"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"

  --tune_mm_vision True
  --tune_mm_mlp True
  --tune_mm_llm True

  --use_lora True
  --lora_r 128
  --lora_alpha 256
  --lora_target_modules "q_proj,k_proj,v_proj,qkv"

  --bf16
  --num_train_epochs 7
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay 0
  --warmup_ratio 0.01
  --lr_scheduler_type cosine
  --max_grad_norm 1.0
  --gradient_checkpointing True
  --remove_unused_columns False
  --ddp_find_unused_parameters False
  --label_names labels

  --logging_steps 10
  --save_strategy steps
  --save_steps 500
  --save_total_limit 50

  --eval_strategy no
  --per_device_eval_batch_size "$((BATCH_SIZE * 2))"
  --dataloader_num_workers 0

  --model_max_length 8192
  --max_pixels 200704
  --min_pixels 200704

  --report_to tensorboard
)

# ------------------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------------------

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${ENTRY_FILE}" "${args[@]}"

echo "Training finished."
