from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path

import numpy as np

from rinlu.data.conll import read_jsonl


def is_emoji_character(character: str) -> bool:
    value = ord(character)
    return (
        0x1F000 <= value <= 0x1FAFF
        or 0x2600 <= value <= 0x27BF
        or value in {0x200D, 0xFE0F}
    )


def contains_emoji(text: str) -> bool:
    return any(is_emoji_character(character) for character in text)


def contains_punctuation(text: str) -> bool:
    return any(unicodedata.category(character).startswith("P") for character in text)


def is_code_mixed(tags: tuple[str, ...]) -> bool:
    observed = set(tags)
    return "Hin" in observed and "Eng" in observed


def _percentile(values: list[int], percentile: int) -> float:
    return float(np.percentile(np.asarray(values), percentile)) if values else 0.0


def _split_summary(records) -> dict:
    token_lengths = [len(record.tokens) for record in records]
    character_lengths = [len(record.text) for record in records]
    return {
        "examples": len(records),
        "mean_source_tokens": float(np.mean(token_lengths)),
        "p95_source_tokens": _percentile(token_lengths, 95),
        "mean_characters": float(np.mean(character_lengths)),
        "p95_characters": _percentile(character_lengths, 95),
        "emoji_examples": sum(contains_emoji(record.text) for record in records),
        "punctuation_examples": sum(contains_punctuation(record.text) for record in records),
        "code_mixed_hin_eng_examples": sum(is_code_mixed(record.language_tags) for record in records),
    }


def _coverage(reference, target) -> dict:
    train_tokens = {token.casefold() for record in reference for token in record.tokens}
    target_tokens = [token.casefold() for record in target for token in record.tokens]
    train_characters = {character for record in reference for character in record.text.casefold()}
    target_characters = [character for record in target for character in record.text.casefold()]
    oov_tokens = [token for token in target_tokens if token not in train_tokens]
    oov_characters = [character for character in target_characters if character not in train_characters]
    return {
        "word_token_oov_rate": len(oov_tokens) / max(1, len(target_tokens)),
        "character_oov_rate": len(oov_characters) / max(1, len(target_characters)),
        "unique_train_word_tokens": len(train_tokens),
        "unique_train_characters": len(train_characters),
        "unique_oov_word_tokens": len(set(oov_tokens)),
    }


def build_audit(data_dir: Path) -> dict:
    train = read_jsonl(data_dir / "train.jsonl")
    dev = read_jsonl(data_dir / "dev.jsonl")
    test = read_jsonl(data_dir / "test_labeled.jsonl")
    return {
        "splits": {
            "train": _split_summary(train),
            "dev": _split_summary(dev),
            "test": _split_summary(test),
        },
        "coverage_against_train": {
            "dev": _coverage(train, dev),
            "test": _coverage(train, test),
        },
        "interpretation": (
            "Word OOV counts exact case-folded source tokens. Character OOV counts exact "
            "Unicode characters. These are corpus coverage diagnostics, not BPE fragmentation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit token and character coverage")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result["coverage_against_train"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
