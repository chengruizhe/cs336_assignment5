from pathlib import Path


def format_prompt(input_text: str, name: str) -> str:
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    prompt_path = prompts_dir / f"{name}.prompt"
    with open(prompt_path, "r", encoding="utf-8") as fp:
        prompt = fp.read()
    return prompt.format(question=input_text)
