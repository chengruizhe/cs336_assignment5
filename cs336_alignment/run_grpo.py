import argparse
import json
import logging
import math
import pathlib
import random
from datetime import datetime
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from cs336_alignment.dataset_prep import MathSFTDataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.run_math_baseline import evaluate_vllm
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    tokenize_prompt_and_output,
)
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from cs336_alignment.run_sft import (
    init_vllm,
    load_policy_into_vllm_instance,
    maybe_init_wandb,
    maybe_save_final_model,
    run_final_full_val_eval,
    log_metrics,
    configure_wandb_metrics,
    math_sft_val_data,
    build_policy_and_tokenizer,
    cycle_dataloader,
    filter_correct_records,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    parser.add_argument("--prompt_type", default="r1_zero")
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="runs/grpo")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--train_limit", type=int, default=None)

    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--grad_acc_steps", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--loss_type",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        default="reinforce_with_baseline",
    )
    parser.add_argument(
        "--use_std_normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--vllm_max_model_len", type=int, default=2048)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train_log_token_entropy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--wandb_mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
    parser.add_argument("--wandb_project", default="cs336-alignment")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument(
        "--save_final_model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def compute_grpo_metrics(
    *,
    log_probs: torch.Tensor,
    token_entropy: torch.Tensor | None,
    response_mask: torch.Tensor,
    loss: torch.Tensor,
    grad_norm: float | None = None,
    clip_fraction: float | None = None,
) -> dict[str, float]:
    mask = response_mask.float()
    response_tokens = max(int(mask.sum().item()), 1)
    mean_response_log_prob = float((log_probs * mask).sum().item() / response_tokens)
    metrics = {
        "train/loss": float(loss.item()),
        "train/perplexity": float(math.exp(-mean_response_log_prob)),
        "train/response_tokens": response_tokens,
    }
    if token_entropy is not None:
        response_entropy = token_entropy * mask
        metrics["train/mean_response_entropy"] = float(
            response_entropy.sum().item() / response_tokens
        )
    if grad_norm is not None:
        metrics["train/grad_norm"] = float(grad_norm)
    if clip_fraction is not None:
        metrics["train/clip_fraction"] = float(clip_fraction)
    return metrics


def summarize_eval_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"eval/num_examples": 0.0}

    reward_keys = sorted(records[0]["rewards"].keys())
    summary = {"eval/num_examples": float(len(records))}
    for key in reward_keys:
        summary[f"eval/{key}_mean"] = float(
            sum(record["rewards"][key] for record in records) / len(records)
        )
    return summary


def maybe_run_eval(
    *,
    step: int,
    policy: torch.nn.Module,
    llm: LLM,
    val_prompts: list[str],
    val_answers: list[str],
    eval_sampling_params: SamplingParams,
    eval_dir: pathlib.Path,
    logger: logging.Logger,
    metrics_path: pathlib.Path,
    wandb_run,
) -> None:
    policy.eval()
    load_policy_into_vllm_instance(policy, llm)
    output_path = eval_dir / f"step_{step:07d}.json"
    records = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=val_prompts,
        answers=val_answers,
        eval_sampling_params=eval_sampling_params,
        output_path=output_path,
    )
    summary = summarize_eval_records(records)
    log_metrics(
        summary,
        step=step,
        metrics_path=metrics_path,
        logger=logger,
        wandb_run=wandb_run,
    )
    policy.train()


def sample_grpo_rollouts(
    *,
    llm: LLM,
    reward_fn,
    prompts: list[str],
    answers: list[str],
    output_path: pathlib.Path,
    sampling_temperature: float,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    group_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample `group_size` rollouts per prompt in one vLLM request.

    Using `n=group_size` avoids issuing separate identical prompt requests under a
    fixed seed, which can otherwise collapse each GRPO group to identical samples
    and yield zero advantages.
    """
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
        n=group_size,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    records: list[dict[str, Any]] = []
    for prompt, output, answer in zip(prompts, outputs, answers, strict=True):
        if len(output.outputs) != group_size:
            raise ValueError(
                f"Expected {group_size} rollouts for one prompt, got {len(output.outputs)}"
            )
        for candidate in output.outputs:
            response = candidate.text
            records.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "rewards": reward_fn(response, answer),
                }
            )

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=4)

    return records


def build_optimizer(
    args: argparse.Namespace,
    policy: torch.nn.Module,
) -> torch.optim.Optimizer:
    opt = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    opt.zero_grad(set_to_none=True)
    return opt


def cache_rollout_old_log_probs(
    *,
    policy: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    responses: list[str],
    micro_train_batch_size: int = 4,
    device: str,
) -> dict[str, torch.Tensor]:
    tokenized = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)

    policy.eval()
    batch_size = input_ids.shape[0]
    assert micro_train_batch_size > 0, "micro_train_batch_size must be positive"
    old_log_probs_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, batch_size, micro_train_batch_size):
            batch_input_ids = input_ids[i : i + micro_train_batch_size]
            batch_labels = labels[i : i + micro_train_batch_size]
            old_policy_outputs = get_response_log_probs(
                model=policy,
                input_ids=batch_input_ids,
                labels=batch_labels,
                return_token_entropy=False,
            )
            old_log_probs_chunks.append(old_policy_outputs["log_probs"])

    old_log_probs = torch.cat(old_log_probs_chunks, dim=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
    }


def run_grpo(
    *,
    args: argparse.Namespace,
    policy: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    opt: torch.optim.Optimizer,
    llm: LLM,
    eval_sampling_params: SamplingParams,
    eval_dir: pathlib.Path,
    val_prompts: list[str],
    val_answers: list[str],
    metrics_path: pathlib.Path,
    logger: logging.Logger,
    wandb_run,
) -> int:
    assert (
        args.train_batch_size % args.grad_acc_steps == 0
    ), "train_batch_size must be divisible by grad_acc_steps"
    micro_train_batch_size = args.train_batch_size // args.grad_acc_steps
    assert (
        args.rollout_batch_size % args.group_size == 0
    ), "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    assert (
        args.train_batch_size >= args.group_size
    ), "train_batch_size must be greater than or equal to group_size"
    n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size

    logger.info(
        "Starting GRPO: n_grpo_steps=%s, rollout_batch_size=%s, "
        "group_size=%s, epochs_per_rollout_batch=%s, loss_type=%s",
        args.n_grpo_steps,
        args.rollout_batch_size,
        args.group_size,
        args.epochs_per_rollout_batch,
        args.loss_type,
    )
    rng = random.Random(args.seed)

    train_dataset = MathSFTDataset(
        split="train",
        dataset_type="math_12k",
        prompt_type=args.prompt_type,
        limit=args.train_limit,
        seed=args.seed,
    )

    global_step = 0
    optimizer_step = 0
    for grpo_step in range(1, args.n_grpo_steps + 1):
        sampled_idx = rng.sample(
            range(len(train_dataset)), k=n_prompts_per_rollout_batch
        )
        dataset = Subset(train_dataset, indices=sampled_idx)

        step_prompts = [row["problem"] for row in dataset]
        step_answers = [row["answer"] for row in dataset]
        policy.eval()
        load_policy_into_vllm_instance(policy, llm)
        grpo_output_path = eval_dir / f"grpo_step_{grpo_step:03d}_samples.json"
        sampled_records = sample_grpo_rollouts(
            llm=llm,
            reward_fn=r1_zero_reward_fn,
            output_path=grpo_output_path,
            prompts=step_prompts,
            answers=step_answers,
            sampling_temperature=args.sampling_temperature,
            sampling_min_tokens=args.sampling_min_tokens,
            sampling_max_tokens=args.sampling_max_tokens,
            group_size=args.group_size,
            seed=args.seed + grpo_step,
        )

        correct_records = filter_correct_records(sampled_records)
        correct_ratio = len(correct_records) / len(sampled_records)
        logger.info(
            "GRPO step %s | sampled=%s, correct=%s (%.2f%%)",
            grpo_step,
            len(sampled_records),
            len(correct_records),
            correct_ratio * 100.0,
        )

        rollout_prompts = [record["prompt"] for record in sampled_records]
        rollout_responses = [record["response"] for record in sampled_records]
        rollout_cache = cache_rollout_old_log_probs(
            policy=policy,
            tokenizer=tokenizer,
            prompts=rollout_prompts,
            responses=rollout_responses,
            device=args.train_device,
        )
        input_ids = rollout_cache["input_ids"]
        labels = rollout_cache["labels"]
        old_log_probs = rollout_cache["old_log_probs"]
        response_mask = rollout_cache["response_mask"]
        logger.info(
            "Cached old-policy log probs for GRPO step %s (batch=%s, tokens=%s)",
            grpo_step,
            old_log_probs.shape[0],
            int(response_mask.sum().item()),
        )

        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=[record["response"] for record in sampled_records],
            repeated_ground_truths=[
                answer
                for answer in step_answers
                for _ in range(args.group_size)
            ],
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )
        raw_rewards = raw_rewards.to(args.train_device)
        advantages = advantages.to(args.train_device)
        reward_metadata = {
            f"grpo/{key}": float(value) for key, value in reward_metadata.items()
        }
        mean_total_reward = float(
            sum(
                float(record["rewards"].get("reward", 0.0))
                for record in sampled_records
            )
            / max(len(sampled_records), 1)
        )
        mean_format_reward = float(
            sum(
                float(record["rewards"].get("format_reward", 0.0))
                for record in sampled_records
            )
            / max(len(sampled_records), 1)
        )
        mean_answer_reward = float(
            sum(
                float(record["rewards"].get("answer_reward", 0.0))
                for record in sampled_records
            )
            / max(len(sampled_records), 1)
        )
        metrics = {
            "grpo/step": grpo_step,
            "grpo/num_questions": len(step_prompts),
            "grpo/num_rollouts": args.group_size,
            "grpo/num_sampled": len(sampled_records),
            "grpo/num_correct": len(correct_records),
            "grpo/correct_ratio": correct_ratio,
            "grpo/mean_total_reward": mean_total_reward,
            "grpo/mean_format_reward": mean_format_reward,
            "grpo/mean_answer_reward": mean_answer_reward,
        } | reward_metadata
        log_metrics(
            metrics,
            step=grpo_step,
            metrics_path=metrics_path,
            logger=logger,
            wandb_run=wandb_run,
        )

        total_iters = args.epochs_per_rollout_batch * n_microbatches_per_rollout_batch
        progress_bar = tqdm(total=total_iters, desc="GRPO", dynamic_ncols=True)
        last_grad_norm: float | None = None
        microbatch_step = 0
        policy.train()
        for _ in range(args.epochs_per_rollout_batch):
            for step in range(n_microbatches_per_rollout_batch):
                microbatch_step += 1
                batch_indices = list(
                    range(
                        step * micro_train_batch_size,
                        (step + 1) * micro_train_batch_size,
                    )
                )
                batch_input_ids = input_ids[batch_indices]
                batch_labels = labels[batch_indices]
                batch_response_mask = response_mask[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_raw_rewards = raw_rewards[batch_indices]

                policy_log_probs = get_response_log_probs(
                    model=policy,
                    input_ids=batch_input_ids,
                    labels=batch_labels,
                    return_token_entropy=False,
                )["log_probs"]

                loss, grpo_metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=batch_response_mask,
                    gradient_accumulation_steps=args.grad_acc_steps,
                    loss_type=args.loss_type,
                    raw_rewards=batch_raw_rewards,
                    advantages=batch_advantages,
                    old_log_probs=batch_old_log_probs,
                    cliprange=args.cliprange,
                )

                if microbatch_step % args.grad_acc_steps == 0:
                    if args.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            policy.parameters(),
                            max_norm=args.max_grad_norm,
                        )
                        last_grad_norm = float(grad_norm.item())
                    else:
                        last_grad_norm = None
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    if optimizer_step % args.eval_interval == 0:
                        maybe_run_eval(
                            step=optimizer_step,
                            policy=policy,
                            llm=llm,
                            val_prompts=val_prompts,
                            val_answers=val_answers,
                            eval_sampling_params=eval_sampling_params,
                            eval_dir=eval_dir,
                            logger=logger,
                            metrics_path=metrics_path,
                            wandb_run=wandb_run,
                        )

                if microbatch_step % args.log_interval == 0:
                    clip_fraction = None
                    clipped = grpo_metadata.get("clipped")
                    if clipped is not None:
                        clipped_mask = clipped.float()
                        mask = batch_response_mask.float()
                        denom = max(float(mask.sum().item()), 1.0)
                        clip_fraction = float(
                            (clipped_mask * mask).sum().item() / denom
                        )
                    metrics = compute_grpo_metrics(
                        log_probs=policy_log_probs,
                        token_entropy=None,
                        response_mask=batch_response_mask,
                        loss=loss,
                        grad_norm=last_grad_norm,
                        clip_fraction=clip_fraction,
                    )
                    metrics["train/optimizer_step"] = optimizer_step
                    metrics["train/lr"] = opt.param_groups[0]["lr"]
                    log_metrics(
                        metrics,
                        step=global_step,
                        metrics_path=metrics_path,
                        logger=logger,
                        wandb_run=wandb_run,
                    )
                    progress_bar.set_postfix(
                        loss=f"{metrics['train/loss']:.4f}",
                        ppl=f"{metrics['train/perplexity']:.2f}",
                        lr=f"{metrics['train/lr']:.2e}",
                    )

                progress_bar.update(1)
                global_step += 1

        progress_bar.close()
    return global_step


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    args.experiment_name = f"{args.experiment_name}_{timestamp}"

    run_dir = pathlib.Path(args.output_dir) / args.experiment_name
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger = setup_logging(run_dir, name="cs336_alignment.run_grpo")
    wandb_run = maybe_init_wandb(args, run_dir)
    configure_wandb_metrics(wandb_run, ("train", "eval", "grpo", "final"))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer, policy = build_policy_and_tokenizer(args, logger)

    logger.info("Initializing vLLM on %s", args.vllm_device)
    llm = init_vllm(
        args.model_id,
        args.vllm_device,
        args.seed,
        args.gpu_memory_utilization,
        args.vllm_max_model_len,
    )
    eval_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        top_p=1.0,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    logger.info("Loading validation dataset")
    val_prompts, val_answers = math_sft_val_data(
        seed=args.seed,
        prompt_type=args.prompt_type,
        limit=args.eval_limit,
    )
    logger.info("Validation size=%s", len(val_prompts))
    opt = build_optimizer(args, policy)

    logger.info("Loading training dataset")
    train_dataset = MathSFTDataset(
        split="train",
        dataset_type="math_12k",
        prompt_type=args.prompt_type,
        limit=args.train_limit,
        seed=args.seed,
    )
    logger.info("Training size=%s (split=%s)", len(train_dataset), "train")
    logger.info(
        "Starting GRPO training: max_iters=%s, batch_size=%s, "
        "grad_acc_steps=%s, val_examples=%s, train_split=%s",
        args.n_grpo_steps,
        args.train_batch_size,
        args.grad_acc_steps,
        len(val_prompts),
        "train",
    )
    final_step = run_grpo(
        args=args,
        policy=policy,
        tokenizer=tokenizer,
        opt=opt,
        llm=llm,
        val_prompts=val_prompts,
        val_answers=val_answers,
        eval_sampling_params=eval_sampling_params,
        eval_dir=eval_dir,
        metrics_path=metrics_path,
        logger=logger,
        wandb_run=wandb_run,
    )

    run_final_full_val_eval(
        seed=args.seed,
        prompt_type=args.prompt_type,
        policy=policy,
        llm=llm,
        eval_sampling_params=eval_sampling_params,
        eval_dir=eval_dir,
        logger=logger,
        metrics_path=metrics_path,
        wandb_run=wandb_run,
        step=final_step,
    )
    maybe_save_final_model(
        policy=policy,
        tokenizer=tokenizer,
        run_dir=run_dir,
        logger=logger,
        enabled=args.save_final_model,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

COMMAND = """
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1 uv run python /workspace/cs336_assignment5/cs336_alignment/run_grpo.py \
  --experiment_name grpo_math12k \
  --wandb_mode online \
  --wandb_project cs336-alignment \
  --n_grpo_steps 200 \
  --lr 1e-5 \
  --rollout_batch_size 256 \
  --group_size 8 \
  --epochs_per_rollout_batch 1 \
  --train_batch_size 256 \
  --grad_acc_steps 128 \
  --gpu_memory_utilization 0.8 \
  --sampling_max_tokens 1024 \
  --eval_limit 1024 \
  --eval_interval 10 \
  --gradient_checkpointing \
  --loss_type reinforce_with_baseline \
  --use_std_normalization \
  --save_final_model
"""
