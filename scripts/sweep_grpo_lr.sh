#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

REPO_ROOT="/workspace/cs336_assignment5"
RUN_SCRIPT="${REPO_ROOT}/cs336_alignment/run_grpo.py"

COMMON_ARGS=(
  --wandb_mode online \
  --wandb_project cs336_alignment_grpo \
  --n_grpo_steps 200 \
  --rollout_batch_size 256 \
  --group_size 8 \
  --epochs_per_rollout_batch 1 \
  --train_batch_size 256 \
  --grad_acc_steps 128 \
  --gpu_memory_utilization 0.7 \
  --sampling_max_tokens 1024 \
  --eval_limit 1024 \
  --eval_interval 1 \
  --gradient_checkpointing \
  --loss_type reinforce_with_baseline \
  --use_std_normalization \
  # --save_final_model
)

for lr in 5e-6 1e-5 2e-5 5e-5; do
  lr_tag="${lr//./p}"
  uv run python "${RUN_SCRIPT}" \
    --experiment_name "grpo_reinforce_with_baseline_lr${lr_tag}" \
    --lr "${lr}" \
    "${COMMON_ARGS[@]}"
done
