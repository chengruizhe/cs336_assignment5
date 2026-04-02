from typing import Callable, Any
import pathlib
import json
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.dataset_prep import load_math
from cs336_alignment.prompting import format_prompt


def download_qwen2_5_math_1_5b():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="Qwen/Qwen2.5-Math-1.5B",
        local_dir="/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        local_dir_use_symlinks=False,
    )


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams,
    output_path: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    assert len(prompts) == len(
        answers
    ), "Prompts and answers must have the same length."
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)

    records = []
    for prompt, output, answer in zip(prompts, outputs, answers, strict=True):
        response = output.outputs[0].text
        rewards = reward_fn(response, answer)
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "rewards": rewards,
            }
        )

    if output_path is not None:
        with open(output_path, "w") as fp:
            json.dump(records, fp, indent=4)
    else:
        print("First 10 records:")
        for record in records[:10]:
            print(record)

    return records


def main():
    math_test = load_math(split="test")
    prompts = [format_prompt(q, "r1_zero") for q in math_test["problem"]]
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    reward_fn = r1_zero_reward_fn
    vllm_model = LLM("Qwen/Qwen2.5-Math-1.5B")
    output_path = pathlib.Path(
        "cs336_alignment/eval_outputs/qwen2.5_1.5B_math_test/results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=reward_fn,
        prompts=prompts,
        answers=math_test["answer"],
        eval_sampling_params=sampling_params,
        output_path=output_path,
    )

    count_correct = 0
    count_format_correct = 0
    count_wrong = 0
    results = json.loads(output_path.read_text())
    for record in results:
        rewards = record["rewards"]
        if rewards["answer_reward"] == 1.0:
            count_correct += 1
        elif rewards["format_reward"] == 1.0:
            count_format_correct += 1
        else:
            count_wrong += 1
    print(
        f"Total: {len(results)}, Correct: {count_correct}, Format Correct: {count_format_correct}, Wrong: {count_wrong}"
    )


if __name__ == "__main__":
    main()
