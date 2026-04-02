#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

REPO_ROOT="/workspace/cs336_assignment5"
RUN_SCRIPT="${REPO_ROOT}/cs336_alignment/run_sft.py"

COMMON_ARGS=(
  --wandb_mode online
  --wandb_project cs336-alignment
  --max_lr 5e-5
  --min_lr 1e-5
  --warmup_iters 100
  --gpu_memory_utilization 0.7
  --micro_batch_size 2
  --grad_acc_steps 16
  --gradient_checkpointing
  --train_log_token_entropy
  --eval_limit 400
  --eval_interval 1000
  --save_final_model
)

for train_limit in 128 256 512 1024; do
  uv run python "${RUN_SCRIPT}" \
    --experiment_name "sft_math12k_n${train_limit}" \
    --train_limit "${train_limit}" \
    --max_iters 2000 \
    "${COMMON_ARGS[@]}"
done

uv run python "${RUN_SCRIPT}" \
  --experiment_name "sft_math12k_all" \
  --max_iters 4000 \
  "${COMMON_ARGS[@]}"
