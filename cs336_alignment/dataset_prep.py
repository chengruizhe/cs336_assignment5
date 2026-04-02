from typing import Any
import json

from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from cs336_alignment.prompting import format_prompt


def load_math(split: str | None = None) -> HFDataset:
    return load_dataset("hiyouga/math12k", split=split)


def _load_hf_json_array_dataset(repo_id: str, file_path: str) -> HFDataset:
    def _normalize_scalar_types(value):
        if isinstance(value, list):
            return [_normalize_scalar_types(item) for item in value]
        if isinstance(value, dict):
            return {k: _normalize_scalar_types(v) for k, v in value.items()}
        if value is None or isinstance(value, str):
            return value
        return str(value)

    local_path = hf_hub_download(
        repo_id=repo_id, filename=file_path, repo_type="dataset"
    )
    with open(local_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        return HFDataset.from_list(records)

    type_map: dict[str, set[type]] = {}
    for record in records:
        for key, value in record.items():
            type_map.setdefault(key, set()).add(type(value))

    for key, types in type_map.items():
        if len(types) <= 1:
            continue
        if list in types:
            for record in records:
                value = record.get(key)
                if not isinstance(value, list):
                    record[key] = [value]
        else:
            for record in records:
                value = record.get(key)
                record[key] = json.dumps(value, ensure_ascii=False)

    for record in records:
        for key, value in record.items():
            record[key] = _normalize_scalar_types(value)

    return HFDataset.from_list(records)


def load_math_sft(split: str | None = None) -> HFDataset | DatasetDict:
    repo_id = "garg-aayush/sft-cs336-assign5-datasets"
    split_to_file = {
        "train": "sft-reason/sft_gpt-oss-120b.jsonl",
        "train_filtered": "sft-reason/sft_gpt-oss-120b_filtered.jsonl",
        "val": "sft-reason/val.jsonl",
    }

    if split is None:
        return DatasetDict(
            {
                split_name: _load_hf_json_array_dataset(repo_id, file_path)
                for split_name, file_path in split_to_file.items()
            }
        )

    if split not in split_to_file:
        valid_splits = ", ".join(split_to_file.keys())
        raise ValueError(f"Unsupported split: {split}. Expected one of: {valid_splits}")

    return _load_hf_json_array_dataset(repo_id, split_to_file[split])


class MathSFTDataset(Dataset):
    def __init__(
        self,
        *,
        split: str,
        prompt_type: str = "r1_zero",
        limit: int | None = None,
        seed: int = 42,
    ):
        self.prompt_type = prompt_type
        self._data = load_math_sft(split=split).shuffle(seed=seed)
        if limit is not None:
            self._data = self._data.select(range(min(limit, len(self._data))))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._data[idx]
        prompt = format_prompt(
            input_text=item["problem"],
            name=self.prompt_type,
        )
        result = item.copy()
        result["problem"] = prompt
        return result


class QADataset(Dataset):
    def __init__(
        self,
        *,
        prompts: list[str],
        answers: list[str],
        prompt_type: str = "r1_zero",
    ):
        self.prompt_type = prompt_type
        self.prompts = prompts
        self.answers = answers
        assert len(self.prompts) == len(
            self.answers
        ), "Prompts and answers must have the same length."

    def __len__(self) -> int:
        return len(self.answers)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = format_prompt(
            input_text=self.prompts[idx],
            name=self.prompt_type,
        )
        return {
            "problem": prompt,
            "expected_answer": self.answers[idx],
        }
