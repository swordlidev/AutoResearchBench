from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Align with the existing baselines: fixed cache/data roots and time budget.
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "ssl_transformer_data")
DATASET_DIR = os.path.join(DATA_DIR, "sst2")
TIME_BUDGET = 180  # seconds

DATASET_NAME = "glue"
DATASET_SUBSET = "sst2"
SPLIT_MAPPING = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str, lowercase: bool) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_PATTERN.findall(text)


def build_vocab(
    sentences: list[str],
    lowercase: bool,
    min_freq: int,
    max_vocab_size: int,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for sentence in sentences:
        counter.update(tokenize(sentence, lowercase=lowercase))

    special_tokens = ["[PAD]", "[UNK]", "[CLS]"]
    vocab = {token: index for index, token in enumerate(special_tokens)}
    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    for token, freq in sorted_tokens:
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def export_split(
    dataset_name: str,
    subset_name: str,
    source_split: str,
    target_path: Path,
    max_examples: int | None,
) -> tuple[int, list[str]]:
    dataset = load_dataset(dataset_name, subset_name, split=source_split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    sentences: list[str] = []
    count = 0
    with target_path.open("w", encoding="utf-8") as f:
        for example in dataset:
            sentence = str(example["sentence"]).strip()
            label = example.get("label")
            record = {
                "sentence": sentence,
                "label": None if label is None or int(label) < 0 else int(label),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            sentences.append(sentence)
            count += 1
    return count, sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare SST-2")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME)
    parser.add_argument("--dataset-subset", type=str, default=DATASET_SUBSET)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.set_defaults(lowercase=True)
    args = parser.parse_args()

    root_dir = args.data_dir.resolve() / "sst2"
    root_dir.mkdir(parents=True, exist_ok=True)

    split_limits = {
        "train": args.max_train,
        "validation": args.max_val,
        "test": args.max_test,
    }
    split_counts: dict[str, int] = {}
    train_sentences: list[str] = []

    for source_split, target_split in SPLIT_MAPPING.items():
        target_path = root_dir / f"{target_split}.jsonl"
        print(f"[prepare] exporting {source_split} -> {target_path}")
        count, sentences = export_split(
            dataset_name=args.dataset_name,
            subset_name=args.dataset_subset,
            source_split=source_split,
            target_path=target_path,
            max_examples=split_limits[source_split],
        )
        split_counts[target_split] = count
        if source_split == "train":
            train_sentences = sentences
        print(f"[prepare] {target_split}: {count}")

    vocab = build_vocab(
        train_sentences,
        lowercase=args.lowercase,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    vocab_path = root_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_subset": args.dataset_subset,
        "lowercase": args.lowercase,
        "min_freq": args.min_freq,
        "max_vocab_size": args.max_vocab_size,
        "vocab_size": len(vocab),
        "splits": split_counts,
        "train_file": str((root_dir / "train.jsonl").resolve()),
        "val_file": str((root_dir / "val.jsonl").resolve()),
        "test_file": str((root_dir / "test.jsonl").resolve()),
        "vocab_file": str(vocab_path.resolve()),
    }
    metadata_path = root_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[prepare] done")
    print(f"[prepare] dataset dir: {root_dir}")
    print(f"[prepare] train file:  {root_dir / 'train.jsonl'}")
    print(f"[prepare] val file:    {root_dir / 'val.jsonl'}")
    print(f"[prepare] test file:   {root_dir / 'test.jsonl'}")
    print(f"[prepare] vocab file:  {vocab_path}")


if __name__ == "__main__":
    main()
