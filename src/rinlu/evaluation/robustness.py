from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np

from rinlu.data.conll import read_jsonl
from rinlu.evaluation.audit import contains_emoji, contains_punctuation, is_code_mixed
from rinlu.evaluation.metrics import classification_metrics


_ASCII_WORD = re.compile(r"[A-Za-z]+")
_VOWELS = set("aeiouAEIOU")


def _stable_number(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


def drop_internal_vowel(text: str, uid: str) -> str:
    """Apply deterministic shorthand-like vowel deletion to eligible Latin words."""
    candidates = []
    for match in _ASCII_WORD.finditer(text):
        word = match.group(0)
        positions = [i for i, char in enumerate(word[1:-1], start=1) if char in _VOWELS]
        if len(word) >= 5 and positions:
            candidates.append((match, positions))
    if not candidates:
        return text
    match, positions = candidates[_stable_number(uid + ":word") % len(candidates)]
    remove_at = positions[_stable_number(uid + ":vowel") % len(positions)]
    word = match.group(0)
    changed = word[:remove_at] + word[remove_at + 1 :]
    return text[: match.start()] + changed + text[match.end() :]


def repeat_letter(text: str, uid: str) -> str:
    """Apply deterministic character elongation to one eligible Latin word."""
    candidates = [match for match in _ASCII_WORD.finditer(text) if len(match.group(0)) >= 3]
    if not candidates:
        return text
    match = candidates[_stable_number(uid + ":repeat-word") % len(candidates)]
    word = match.group(0)
    position = 1 + _stable_number(uid + ":repeat-position") % (len(word) - 1)
    changed = word[:position] + word[position] * 3 + word[position + 1 :]
    return text[: match.start()] + changed + text[match.end() :]


def remove_emoji(text: str) -> str:
    kept = []
    for character in text:
        value = ord(character)
        if (
            0x1F000 <= value <= 0x1FAFF
            or 0x2600 <= value <= 0x27BF
            or value in {0x200D, 0xFE0F}
        ):
            continue
        kept.append(character)
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def remove_punctuation(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in text
        ),
    ).strip()


def _evaluate(model, records, texts) -> tuple[dict, list[str], np.ndarray]:
    labels = [record.label for record in records]
    predictions = model.predict(texts).tolist()
    probabilities = model.predict_proba(texts)
    return (
        classification_metrics(labels, predictions, probabilities, model.classes_),
        predictions,
        probabilities,
    )


def run_robustness(model_path: Path, data_path: Path, output_dir: Path) -> dict:
    model = joblib.load(model_path)
    records = read_jsonl(data_path)
    clean_texts = [record.text for record in records]
    clean_metrics, clean_predictions, clean_probabilities = _evaluate(model, records, clean_texts)

    variants = {
        "vowel_drop": [drop_internal_vowel(record.text, record.uid) for record in records],
        "repeated_letter": [repeat_letter(record.text, record.uid) for record in records],
        "punctuation_removed_ablation": [remove_punctuation(record.text) for record in records],
    }
    results = {
        "clean": clean_metrics,
        "synthetic_meaning_preserving": {},
        "input_ablation": {},
        "natural_slices": {},
        "limitations": [
            "Vowel deletion and letter repetition are deterministic synthetic perturbations, not human messages.",
            "Emoji removal is an ablation only; it is not assumed to preserve the original meaning.",
            "Emoji pragmatic flips require a separately human-annotated contrast set and are not claimed here.",
        ],
    }
    for name, texts in variants.items():
        metrics, predictions, _ = _evaluate(model, records, texts)
        entry = {
            "metrics": metrics,
            "macro_f1_change_from_clean": metrics["macro_f1"] - clean_metrics["macro_f1"],
            "prediction_stability": float(np.mean(np.asarray(predictions) == np.asarray(clean_predictions))),
            "changed_inputs": sum(a != b for a, b in zip(clean_texts, texts)),
        }
        if name.endswith("ablation"):
            results["input_ablation"][name] = entry
        else:
            results["synthetic_meaning_preserving"][name] = entry

    slice_masks = {
        "code_mixed_hin_eng": [is_code_mixed(record.language_tags) for record in records],
        "emoji_present": [contains_emoji(record.text) for record in records],
        "punctuation_present": [contains_punctuation(record.text) for record in records],
    }
    for name, mask in slice_masks.items():
        indexes = [index for index, include in enumerate(mask) if include]
        if not indexes:
            continue
        subset = [records[index] for index in indexes]
        texts = [clean_texts[index] for index in indexes]
        metrics, _, _ = _evaluate(model, subset, texts)
        results["natural_slices"][name] = {"examples": len(indexes), "metrics": metrics}

    emoji_indexes = [index for index, record in enumerate(records) if contains_emoji(record.text)]
    if emoji_indexes:
        subset = [records[index] for index in emoji_indexes]
        stripped = [remove_emoji(record.text) for record in subset]
        original_predictions = [clean_predictions[index] for index in emoji_indexes]
        metrics, predictions, _ = _evaluate(model, subset, stripped)
        results["input_ablation"]["emoji_removed_on_emoji_subset"] = {
            "examples": len(subset),
            "metrics": metrics,
            "macro_f1_change_from_same_subset_clean": (
                metrics["macro_f1"]
                - results["natural_slices"]["emoji_present"]["metrics"]["macro_f1"]
            ),
            "prediction_stability": float(
                np.mean(np.asarray(predictions) == np.asarray(original_predictions))
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "robustness_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with (output_dir / "high_confidence_errors.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["uid", "gold", "prediction", "confidence", "code_mixed", "emoji", "text"]
        )
        errors = []
        for record, prediction, probability in zip(
            records, clean_predictions, clean_probabilities
        ):
            if prediction != record.label:
                errors.append((float(np.max(probability)), record, prediction))
        for confidence, record, prediction in sorted(errors, reverse=True, key=lambda row: row[0]):
            writer.writerow(
                [
                    record.uid,
                    record.label,
                    prediction,
                    confidence,
                    is_code_mixed(record.language_tags),
                    contains_emoji(record.text),
                    record.text,
                ]
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hinglish sentiment robustness")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = run_robustness(args.model, args.data, args.output_dir)
    compact = {
        "clean_macro_f1": results["clean"]["macro_f1"],
        "synthetic": {
            name: value["metrics"]["macro_f1"]
            for name, value in results["synthetic_meaning_preserving"].items()
        },
        "natural_slices": {
            name: {"examples": value["examples"], "macro_f1": value["metrics"]["macro_f1"]}
            for name, value in results["natural_slices"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

