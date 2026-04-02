#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

REPO_ROOT="/workspace/cs336_assignment5"
RUN_SCRIPT="${REPO_ROOT}/cs336_alignment/run_sft.py"

COMMON_ARGS=(
  --model_id Qwen/Qwen2.5-Math-1.5B
  --wandb_mode online
  --wandb_project cs336-alignment
  --max_lr 2e-5
  --min_lr 2e-5
  --warmup_iters 0
  --gpu_memory_utilization 0.7
  --micro_batch_size 1
  --grad_acc_steps 32
  --gradient_checkpointing
  --train_log_token_entropy
  --eval_limit 1000
  --eval_interval 1000
  --n_ei_steps 5
  --ei_dataset_name math
  --save_final_model
)

# Baseline runs to isolate effect of Db (EI batch size), with fixed G=1 and EI-SFT iters=200.
for db in 512 1024 2048; do
  uv run python "${RUN_SCRIPT}" \
    --experiment_name "ei_math_db${db}_g1_iters400" \
    --ei_batch_size "${db}" \
    --ei_num_rollouts 1 \
    --ei_sft_max_iters 800 \
    "${COMMON_ARGS[@]}"
done

# Targeted ablations at medium Db to assess effect of rollout count G and EI SFT max iters.
uv run python "${RUN_SCRIPT}" \
  --experiment_name "ei_math_db1024_g1_iters800" \
  --ei_batch_size 1024 \
  --ei_num_rollouts 1 \
  --ei_sft_max_iters 1600 \
  "${COMMON_ARGS[@]}"

uv run python "${RUN_SCRIPT}" \
  --experiment_name "ei_math_db1024_g4_iters800" \
  --ei_batch_size 1024 \
  --ei_num_rollouts 4 \
  --ei_sft_max_iters 1600 \
  "${COMMON_ARGS[@]}"
