import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return (-torch.exp(log_prob) * log_prob).sum(dim=-1)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    result = {}
    logits = model(input_ids).logits
    vocab_size = logits.shape[-1]
    token_nll = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    )
    result["log_probs"] = -token_nll.reshape_as(labels)
    result["token_entropy"] = compute_entropy(logits) if return_token_entropy else None
    return result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch."""
    loss = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        dim=-1,
        normalize_constant=normalize_constant,
    ).mean()
    loss /= gradient_accumulation_steps

    loss.backward()
    metadata = {}
    return loss, metadata


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    result = {}
    prompt_token_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_token_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    combined_token_ids = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(
            prompt_token_ids, output_token_ids, strict=True
        )
    ]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id

    tokens = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in combined_token_ids],
        batch_first=True,
        padding_value=pad_token_id,
    )

    result["input_ids"] = tokens[:, :-1]
    result["labels"] = tokens[:, 1:]

    mask = torch.zeros_like(result["labels"], dtype=torch.bool)
    for i, (prompt_ids, output_ids) in enumerate(
        zip(prompt_token_ids, output_token_ids, strict=True)
    ):
        prompt_len = len(prompt_ids)
        output_len = len(output_ids)
        mask[i, prompt_len - 1 : prompt_len + output_len - 1] = True

    result["response_mask"] = mask
    return result
