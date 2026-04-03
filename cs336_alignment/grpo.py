from typing import Callable
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
    return - policy_log_probs * rewards
    