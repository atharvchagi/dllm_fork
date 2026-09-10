"""Load FineMath from Hugging Face into SFT chat format.

Run:
    PYTHONPATH=. python -m dllm.data.finemath
"""

from datasets import DatasetDict, load_dataset


DEFAULT_FINEMATH_PROMPT = "Provide a clear mathematical explanation."


def _has_nonempty_text(example: dict) -> bool:
    text = example.get("text", None)
    if text is None:
        return False
    return bool(str(text).strip())


def _map_text_to_messages(example: dict, prompt: str) -> dict:
    text = str(example["text"]).strip()
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ]
    }


def load_dataset_finemath(
    dataset_name_or_path: str,
    name: str = "finemath-3plus",
    prompt: str = DEFAULT_FINEMATH_PROMPT,
) -> DatasetDict:
    """Load FineMath and normalize rows to the repo's SFT `messages` schema."""
    dataset = load_dataset(dataset_name_or_path, name=name)

    out = {}
    for split, ds in dataset.items():
        filtered = ds.filter(
            _has_nonempty_text,
            num_proc=4,
            desc=f"Dropping empty text rows in {split}",
        )
        out[split] = filtered.map(
            _map_text_to_messages,
            fn_kwargs={"prompt": prompt},
            remove_columns=filtered.column_names,
            num_proc=4,
            desc=f"Mapping {split} split to messages format",
        )

    return DatasetDict(out)


if __name__ == "__main__":
    ds = load_dataset_finemath("HuggingFaceTB/finemath", name="finemath-3plus")
    print({"splits": list(ds.keys()), "train_rows": len(ds["train"])})