#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_focusft_qwen25_7b.sh /path/to/train.parquet ./checkpoints/focusft-qwen25-7b
#
# The defaults below match the main FocuSFT configuration reported in the paper.

TRAIN_DATA="${1:?Usage: $0 <train.parquet[,more.parquet]> [output_dir] [model_name_or_path]}"
OUTPUT_DIR="${2:-./checkpoints/focusft-qwen25-7b}"
MODEL_NAME_OR_PATH="${3:-Qwen/Qwen2.5-7B}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  scripts/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_data "${TRAIN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length 4096 \
  --num_train_epochs 5 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --bf16 true \
  --logging_steps 10 \
  --save_strategy no \
  --seed 1234 \
  --attn_implementation sdpa \
  --num_inner_steps 2 \
  --inner_lr 1.0 \
  --inner_grad_clip 1.0 \
  --inner_rank 32 \
  --inner_alpha 64.0 \
  --inner_dropout 0.0 \
  --inner_layer_fraction 0.35 \
  --inner_target_modules gate_proj,up_proj,down_proj \
  --inner_gradient_checkpointing true \
  --sync_inner false
