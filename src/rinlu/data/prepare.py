from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rinlu.data.conll import (
    KNOWN_LANGUAGE_TAGS,
    attach_labels,
    count_empty_token_lines,
    count_repairable_token_lines,
    duplicate_key,
    parse_conll,
    read_test_labels,
    sha256_file,
    write_jsonl,
)


EXPECTED_SPLIT_SIZES = {"train": 14_000, "dev": 3_000, "test": 3_000}


def _text_index(records):
    index = defaultdict(list)
    for record in records:
        index[duplicate_key(record.text)].append(record)
    return index


def _label_counts(records):
    return dict(sorted(Counter(record.label for record in records).items()))


def _tag_counts(records):
    return dict(sorted(Counter(tag for record in records for tag in record.language_tags).items()))


def prepare_dataset(
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    test_labels_path: Path,
    output_dir: Path,
    enforce_expected_sizes: bool = True,
) -> dict:
    raw = {
        "train": parse_conll(train_path, "train", require_labels=True),
        "dev": parse_conll(dev_path, "dev", require_labels=True),
        "test": parse_conll(test_path, "test", require_labels=False),
    }
    if enforce_expected_sizes:
        for split, expected in EXPECTED_SPLIT_SIZES.items():
            if len(raw[split]) != expected:
                raise ValueError(
                    f"{split}: expected {expected} examples, found {len(raw[split])}"
                )

    labels = read_test_labels(test_labels_path)
    labelled_test = attach_labels(raw["test"], labels)

    unknown_tags = sorted(
        {
            tag
            for records in raw.values()
            for record in records
            for tag in record.language_tags
            if tag not in KNOWN_LANGUAGE_TAGS
        }
    )
    if unknown_tags:
        raise ValueError(f"unknown language tags: {unknown_tags}")

    indexes = {split: _text_index(records) for split, records in raw.items()}
    overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = sorted(set(indexes[left]) & set(indexes[right]))
        overlap[f"{left}_{right}"] = [
            {
                "text_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                left: [record.uid for record in indexes[left][key]],
                right: [record.uid for record in indexes[right][key]],
            }
            for key in shared
        ]

    test_keys = set(indexes["test"])
    seen_train_keys: set[str] = set()
    cleaned_train = []
    removals = []
    for record in raw["train"]:
        key = duplicate_key(record.text)
        reason = None
        if key in test_keys:
            reason = "exact_train_test_text_leakage"
        elif key in seen_train_keys:
            reason = "duplicate_inside_train"
        if reason:
            removals.append(
                {
                    "uid": record.uid,
                    "reason": reason,
                    "text_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                }
            )
            continue
        seen_train_keys.add(key)
        cleaned_train.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", cleaned_train)
    write_jsonl(output_dir / "dev.jsonl", raw["dev"])
    write_jsonl(output_dir / "test.jsonl", raw["test"], include_label=False)
    write_jsonl(output_dir / "test_labeled.jsonl", labelled_test)

    manifest = {
        "schema_version": 1,
        "input_sha256": {
            "train": sha256_file(train_path),
            "dev": sha256_file(dev_path),
            "test": sha256_file(test_path),
            "test_labels": sha256_file(test_labels_path),
        },
        "source_anomalies": {
            "empty_token_lines_skipped": {
                "train": count_empty_token_lines(train_path),
                "dev": count_empty_token_lines(dev_path),
                "test": count_empty_token_lines(test_path),
            },
            "mojibake_token_lines_repaired": {
                "train": count_repairable_token_lines(train_path),
                "dev": count_repairable_token_lines(dev_path),
                "test": count_repairable_token_lines(test_path),
            },
        },
        "counts_before_cleaning": {split: len(records) for split, records in raw.items()},
        "counts_after_cleaning": {
            "train": len(cleaned_train),
            "dev": len(raw["dev"]),
            "test": len(raw["test"]),
        },
        "label_counts": {
            "train": _label_counts(cleaned_train),
            "dev": _label_counts(raw["dev"]),
            "test": _label_counts(labelled_test),
        },
        "language_tag_counts": {
            split: _tag_counts(records) for split, records in raw.items()
        },
        "exact_cross_split_overlap": overlap,
        "removed_from_train": removals,
        "test_labels_separated": True,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the supplied Hinglish sentiment corpus")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.train, args.dev, args.test, args.test_labels, args.output_dir
    )
    print(json.dumps(manifest["counts_after_cleaning"], indent=2))
    print(f"removed_from_train={len(manifest['removed_from_train'])}")


if __name__ == "__main__":
    main()
