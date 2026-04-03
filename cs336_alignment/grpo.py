from typing import Callable, Literal
import torch
import einops


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)

    reward_dicts = [
        reward_fn(r, a)
        for r, a in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.Tensor([r["reward"] for r in reward_dicts])
    grouped_rewards = einops.rearrange(raw_rewards, "(r g) -> r g", g=group_size)
    advantages = grouped_rewards - grouped_rewards.mean(dim=-1, keepdim=True)

    if normalize_by_std:
        advantages /= grouped_rewards.std(dim=-1, keepdim=True) + advantage_eps
    metadata = {
        "mean_reward": advantages.mean(),
        "min_reward": advantages.min(),
        "max_reward": advantages.max(),
        "mean_raw_reward": raw_rewards.mean(),
    }
    return advantages.flatten(), raw_rewards.flatten(), metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    seq_len = policy_log_probs.shape[-1]
    rewards = einops.repeat(raw_rewards_or_advantages, "b 1 -> b s", s=seq_len)
    return -policy_log_probs * rewards


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert old_log_probs.shape == policy_log_probs.shape

    seq_len = policy_log_probs.shape[-1]
    prob_ratio = torch.exp(policy_log_probs - old_log_probs)
    advantages = einops.repeat(advantages, "b 1 -> b s", s=seq_len)
    clipped_prob_ratio = torch.clip(prob_ratio, min=1 - cliprange, max=1 + cliprange)

    regular_val = prob_ratio * advantages
    clipped_val = clipped_prob_ratio * advantages
    min_val = torch.minimum(regular_val, clipped_val)
    clipped = clipped_val < regular_val
    return -min_val, {"clipped": clipped}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    match loss_type:
        case "no_baseline":
            assert raw_rewards is not None
            return (
                compute_naive_policy_gradient_loss(
                    raw_rewards_or_advantages=raw_rewards,
                    policy_log_probs=policy_log_probs,
                ),
                {},
            )
        case "reinforce_with_baseline":
            assert advantages is not None
            return (
                compute_naive_policy_gradient_loss(
                    raw_rewards_or_advantages=advantages,
                    policy_log_probs=policy_log_probs,
                ),
                {},
            )
        case "grpo_clip":
            assert advantages is not None
            assert old_log_probs is not None
            assert cliprange is not None
            return compute_grpo_clip_loss(
                advantages=advantages,
                policy_log_probs=policy_log_probs,
                old_log_probs=old_log_probs,
                cliprange=cliprange,
            )
        case _:
            raise ValueError(f"Unknown loss_type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    count = mask.sum(dim=dim)
    return (tensor * mask).sum(dim=dim) / count


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = masked_mean(loss, mask=response_mask, dim=-1)
    final_loss = loss.mean() / gradient_accumulation_steps
    final_loss.backward()
    return final_loss, metadata
