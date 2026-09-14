"""Evaluate a frozen Hugging Face sentiment checkpoint; never train its weights."""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import joblib
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from rinlu.data.tasks import load_sentiment


LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def sample_rows(rows, limit, seed):
    if not limit or limit >= len(rows):
        return list(rows)
    grouped = {label: [] for label in LABELS.values()}
    for row in rows:
        grouped[row["label"]].append(row)
    rng, chosen = random.Random(seed), []
    for label in grouped:
        rng.shuffle(grouped[label])
        chosen.extend(grouped[label][:limit // len(grouped)])
    remaining = [row for label in grouped for row in grouped[label] if row not in chosen]
    rng.shuffle(remaining); chosen.extend(remaining[:limit - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def predict_batches(model, tokenizer, rows, device, batch_size, max_length):
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            text = [row["text"] for row in rows[start:start + batch_size]]
            encoded = tokenizer(text, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities.append(model(**encoded).logits.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities)


def label_predictions(probabilities):
    return [LABELS[index] for index in probabilities.argmax(-1)]


def char_probabilities(model, rows):
    raw = model.predict_proba([row["text"] for row in rows])
    columns = [list(model.classes_).index(LABELS[index]) for index in range(3)]
    return raw[:, columns]


def select_ensemble(external_dev, char_dev, dev_rows, external_test, char_test, test_rows):
    expected = [row["label"] for row in dev_rows]
    candidates = []
    for weight in np.linspace(0, 1, 21):
        mixed = weight * external_dev + (1 - weight) * char_dev
        score = f1_score(expected, label_predictions(mixed), average="macro")
        candidates.append({"external_weight": float(weight), "dev_macro_f1": score})
    selected = max(candidates, key=lambda row: (row["dev_macro_f1"], -row["external_weight"]))
    weight = selected["external_weight"]
    predicted = label_predictions(weight * external_test + (1 - weight) * char_test)
    expected = [row["label"] for row in test_rows]
    return {"formula": "weight * frozen_mbert + (1 - weight) * char_tfidf",
            "selection_split": "dev", "grid": candidates, "selected": selected,
            "test_accuracy": accuracy_score(expected, predicted),
            "test_macro_f1": f1_score(expected, predicted, average="macro")}


def latency_report(model, tokenizer, rows, device, max_length, runs):
    selected = [rows[index % len(rows)] for index in range(runs + 1)]
    samples = []
    with torch.inference_mode():
        for index, row in enumerate(selected):
            started = time.perf_counter_ns()
            encoded = tokenizer(row["text"], truncation=True, max_length=max_length,
                                return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded).logits.argmax(-1).cpu()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if index:
                samples.append(elapsed)
    return {"p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "runs": runs,
            "scope": "tokenization + warm batch-one model + argmax; no loading or file I/O"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/external/codemixed_bert.json"))
    parser.add_argument("--limit", type=int, default=0, help="Stratified smoke subset; 0 evaluates all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--char-model", type=Path, default=Path("models/current/sentiment.joblib"))
    parser.add_argument("--ensemble", action="store_true",
                        help="Select frozen-mBERT/character probability weight on dev")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.latency_runs < 1:
        parser.error("invalid evaluation limits")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=args.local_files_only).to(device).eval()
    splits, audit = load_sentiment(args.data)
    rows = sample_rows(splits["test"], args.limit, args.seed)
    probabilities = predict_batches(model, tokenizer, rows, device, args.batch_size,
                                    args.max_length)
    predicted = label_predictions(probabilities)
    expected = [row["label"] for row in rows]
    report = {
        "role": "frozen external comparison; no training performed",
        "model": args.model,
        "checkpoint_revision": getattr(model.config, "_commit_hash", None),
        "model_card_sail_2017": {"accuracy": 0.588889, "f1_unspecified_average": 0.582678},
        "label_mapping": LABELS,
        "test_source_audit": audit,
        "examples": len(rows),
        "subset": "full" if len(rows) == len(splits["test"]) else "seeded stratified smoke subset",
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(expected, predicted, average="macro"),
        "per_class": classification_report(expected, predicted, output_dict=True,
                                             zero_division=0),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "latency": latency_report(model, tokenizer, rows, device, args.max_length,
                                  args.latency_runs),
        "platform": platform.platform(),
        "privacy": "No WhatsApp text is loaded by this evaluator."
    }
    if args.ensemble:
        bundle = joblib.load(args.char_model)
        char_model = bundle["model"]
        dev_rows = sample_rows(splits["dev"], args.limit, args.seed)
        external_dev = predict_batches(model, tokenizer, dev_rows, device,
                                       args.batch_size, args.max_length)
        report["ensemble"] = select_ensemble(
            external_dev, char_probabilities(char_model, dev_rows), dev_rows,
            probabilities, char_probabilities(char_model, rows), rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {key: report[key] for key in
                      ("examples", "subset", "accuracy", "macro_f1", "parameters",
                       "device", "latency")}
    if "ensemble" in report:
        summary["ensemble"] = {key: report["ensemble"][key] for key in
                               ("selected", "test_accuracy", "test_macro_f1")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
