import argparse
import json
import logging
import math
import pathlib
from datetime import datetime
from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.dataset_prep import MathSFTDataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.run_math_baseline import evaluate_vllm
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    parser.add_argument("--prompt_type", default="r1_zero")
    parser.add_argument("--train_filtered", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument(
        "--val_limit", "--eval_limit", dest="val_limit", type=int, default=None
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="runs/sft")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_limit", type=int, default=None)

    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_acc_steps", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
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


def setup_logging(run_dir: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger("cs336_alignment.run_sft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def maybe_init_wandb(args: argparse.Namespace, run_dir: pathlib.Path):
    if args.wandb_mode == "disabled":
        return None

    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.experiment_name,
        dir=str(run_dir),
        config=vars(args),
        tags=args.wandb_tags,
        mode=args.wandb_mode,
    )


def log_metrics(
    metrics: dict[str, float | int],
    *,
    step: int,
    metrics_path: pathlib.Path,
    logger: logging.Logger,
    wandb_run,
) -> None:
    serializable_metrics = dict(metrics)
    serializable_metrics["step"] = step
    append_jsonl(metrics_path, serializable_metrics)

    metric_text = ", ".join(
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in serializable_metrics.items()
    )
    logger.info(metric_text)

    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    input_texts = [item["problem"] for item in batch]
    target_texts = [item["reasoning_trace"] for item in batch]
    return tokenize_prompt_and_output(
        prompt_strs=input_texts,
        output_strs=target_texts,
        tokenizer=tokenizer,
    )


def math_sft_val_data(
    seed: int,
    prompt_type: str,
    limit: int | None = None,
) -> tuple[list[str], list[str]]:
    math_val = MathSFTDataset(
        split="val",
        prompt_type=prompt_type,
        limit=limit,
        seed=seed,
    )
    prompts = [row["problem"] for row in math_val]
    answers = [row["expected_answer"] for row in math_val]
    return prompts, answers


def cycle_dataloader(dataloader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        yield from dataloader


def compute_sft_metrics(
    *,
    log_probs: torch.Tensor,
    token_entropy: torch.Tensor | None,
    response_mask: torch.Tensor,
    loss: torch.Tensor,
) -> dict[str, float]:
    mask = response_mask.float()
    response_tokens = max(int(mask.sum().item()), 1)
    response_log_probs = log_probs * mask
    mean_response_log_prob = float(response_log_probs.sum().item() / response_tokens)
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


def get_lr_for_step(
    *,
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    max_iters: int,
) -> float:
    if max_iters <= 0:
        return min_lr

    if warmup_iters > 0 and step <= warmup_iters:
        return max_lr * step / warmup_iters

    if step >= max_iters:
        return min_lr

    decay_span = max(max_iters - warmup_iters, 1)
    decay_progress = min(max(step - warmup_iters, 0) / decay_span, 1.0)
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (max_lr - min_lr) * cosine_coeff


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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


def run_final_full_val_eval(
    *,
    seed: int,
    prompt_type: str,
    policy: torch.nn.Module,
    llm: LLM,
    eval_sampling_params: SamplingParams,
    eval_dir: pathlib.Path,
    logger: logging.Logger,
    metrics_path: pathlib.Path,
    wandb_run,
    step: int,
) -> None:
    logger.info("Running final evaluation on full validation set")
    full_val_prompts, full_val_answers = math_sft_val_data(
        seed=seed,
        prompt_type=prompt_type,
        limit=None,
    )
    policy.eval()
    load_policy_into_vllm_instance(policy, llm)
    final_eval_output_path = eval_dir / "final_full_val.json"
    final_records = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=full_val_prompts,
        answers=full_val_answers,
        eval_sampling_params=eval_sampling_params,
        output_path=final_eval_output_path,
    )
    final_summary = summarize_eval_records(final_records)
    final_metrics = {f"final/{key}": value for key, value in final_summary.items()}
    log_metrics(
        final_metrics,
        step=step,
        metrics_path=metrics_path,
        logger=logger,
        wandb_run=wandb_run,
    )


def maybe_save_final_model(
    *,
    policy: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    run_dir: pathlib.Path,
    logger: logging.Logger,
    enabled: bool,
) -> None:
    if not enabled:
        logger.info("Skipping final model save (--no-save_final_model)")
        return

    output_dir = run_dir / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving final model checkpoint to %s", output_dir)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def build_policy_and_tokenizer(
    args: argparse.Namespace, logger: logging.Logger
) -> tuple[PreTrainedTokenizerBase, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading policy model from %s", args.model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.train_device)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.train()
    return tokenizer, policy


def build_train_and_val_data(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger,
) -> tuple[DataLoader, list[str], list[str], str]:
    train_split = "train_filtered" if args.train_filtered else "train"
    train_dataset = MathSFTDataset(
        split=train_split,
        prompt_type=args.prompt_type,
        limit=args.train_limit,
        seed=args.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        drop_last=True,
    )
    val_prompts, val_answers = math_sft_val_data(
        seed=args.seed,
        prompt_type=args.prompt_type,
        limit=args.val_limit,
    )
    logger.info(
        "Dataset sizes | train=%s (split=%s), val=%s",
        len(train_dataset),
        train_split,
        len(val_prompts),
    )
    return train_dataloader, val_prompts, val_answers, train_split


def build_optimizer(
    args: argparse.Namespace, policy: torch.nn.Module
) -> torch.optim.Optimizer:
    initial_lr = get_lr_for_step(
        step=1,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        max_iters=args.max_iters,
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=initial_lr)
    opt.zero_grad(set_to_none=True)
    return opt


def run_sft_loop(
    *,
    args: argparse.Namespace,
    policy: torch.nn.Module,
    opt: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    llm: LLM,
    val_prompts: list[str],
    val_answers: list[str],
    eval_sampling_params: SamplingParams,
    eval_dir: pathlib.Path,
    metrics_path: pathlib.Path,
    logger: logging.Logger,
    wandb_run,
) -> None:
    progress_bar = tqdm(total=args.max_iters, desc="SFT", dynamic_ncols=True)

    for step, batch in enumerate(cycle_dataloader(train_dataloader), start=1):
        if step > args.max_iters:
            break

        current_lr = get_lr_for_step(
            step=step,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_iters=args.warmup_iters,
            max_iters=args.max_iters,
        )
        set_optimizer_lr(opt, current_lr)

        input_ids = batch["input_ids"].to(args.train_device)
        response_mask = batch["response_mask"].to(args.train_device)
        labels = batch["labels"].to(args.train_device)

        policy_outputs = get_response_log_probs(
            model=policy,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=args.train_log_token_entropy,
        )
        log_probs = policy_outputs["log_probs"]
        token_entropy = policy_outputs["token_entropy"]

        loss, _ = sft_microbatch_train_step(
            policy_log_probs=log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=args.grad_acc_steps,
            normalize_constant=1.0,
        )

        if step % args.grad_acc_steps == 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    max_norm=args.max_grad_norm,
                )
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step == 1 or step % args.log_interval == 0:
            train_metrics = compute_sft_metrics(
                log_probs=log_probs.detach(),
                token_entropy=None if token_entropy is None else token_entropy.detach(),
                response_mask=response_mask.detach(),
                loss=loss.detach(),
            )
            train_metrics["train/optimizer_step"] = step // args.grad_acc_steps
            train_metrics["train/lr"] = current_lr
            log_metrics(
                train_metrics,
                step=step,
                metrics_path=metrics_path,
                logger=logger,
                wandb_run=wandb_run,
            )
            progress_bar.set_postfix(
                loss=f"{train_metrics['train/loss']:.4f}",
                ppl=f"{train_metrics['train/perplexity']:.2f}",
                lr=f"{current_lr:.2e}",
            )

        if step % args.eval_interval == 0:
            maybe_run_eval(
                step=step,
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
        progress_bar.update(1)

    progress_bar.close()
    if args.max_iters % args.grad_acc_steps != 0:
        logger.info("Applying final optimizer step for leftover accumulated gradients")
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=args.max_grad_norm,
            )
        opt.step()
        opt.zero_grad(set_to_none=True)


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

    logger = setup_logging(run_dir)
    wandb_run = maybe_init_wandb(args, run_dir)

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
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    logger.info("Loading training and validation datasets")
    train_dataloader, val_prompts, val_answers, _ = build_train_and_val_data(
        args=args,
        tokenizer=tokenizer,
        logger=logger,
    )
    opt = build_optimizer(args, policy)

    logger.info(
        "Starting SFT training: max_iters=%s, micro_batch_size=%s, grad_acc_steps=%s, val_examples=%s",
        args.max_iters,
        args.micro_batch_size,
        args.grad_acc_steps,
        len(val_prompts),
    )

    run_sft_loop(
        args=args,
        policy=policy,
        opt=opt,
        train_dataloader=train_dataloader,
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
        step=args.max_iters,
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
CUDA_VISIBLE_DEVICES=0,1 uv run python /workspace/cs336_assignment5/cs336_alignment/run_sft.py \
  --experiment_name sft_math12k \
  --wandb_mode online \
  --wandb_project cs336-alignment \
  --max_lr 5e-5 \
  --min_lr 1e-5 \
  --warmup_iters 200 \
  --max_iters 4000 \
  --gpu_memory_utilization 0.7 \
  --micro_batch_size 2 \
  --grad_acc_steps 16 \
  --gradient_checkpointing \
  --train_log_token_entropy \
  --eval_limit 1000 \
  --eval_interval 1000
"""
