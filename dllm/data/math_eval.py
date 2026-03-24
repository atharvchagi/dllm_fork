"""Load math eval datasets from Hugging Face into SFT chat format.

Run:
    PYTHONPATH=. python -m dllm.data.math_eval
"""

from datasets import DatasetDict, load_dataset


def _pick_field(example: dict, candidates: list[str], required: bool = True) -> str:
    for name in candidates:
        value = example.get(name, None)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    if required:
        raise KeyError(f"Missing required field in candidates={candidates}")
    return ""


def _map_example_to_messages(example: dict) -> dict:
    question = _pick_field(example, ["question", "problem", "prompt"])
    answer = _pick_field(example, ["answer", "solution", "output"])
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def _map_dataset_to_messages(dataset: DatasetDict) -> DatasetDict:
    out = {}
    for split, ds in dataset.items():
        out[split] = ds.map(
            _map_example_to_messages,
            remove_columns=ds.column_names,
            num_proc=4,
        )
    return DatasetDict(out)


def load_dataset_gsm8k(dataset_name_or_path: str, name: str = "main") -> DatasetDict:
    dataset = load_dataset(dataset_name_or_path, name=name)
    return _map_dataset_to_messages(dataset)


def load_dataset_math500(dataset_name_or_path: str) -> DatasetDict:
    dataset = load_dataset(dataset_name_or_path)
    return _map_dataset_to_messages(dataset)


if __name__ == "__main__":
    ds_gsm8k = load_dataset_gsm8k("openai/gsm8k", name="main")
    ds_math500 = load_dataset_math500("HuggingFaceH4/MATH-500")
    print({"gsm8k_splits": list(ds_gsm8k.keys()), "math500_splits": list(ds_math500.keys())})