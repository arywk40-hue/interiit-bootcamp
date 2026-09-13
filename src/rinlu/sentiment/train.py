from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np

from rinlu.data.conll import read_jsonl
from rinlu.evaluation.metrics import artifact_size_bytes, benchmark_batch_one, classification_metrics
from rinlu.sentiment.features import MODEL_NAMES, build_model


def _write_predictions(path, records, predictions, probabilities, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "gold", "prediction", "confidence", *classes])
        for record, prediction, row in zip(records, predictions, probabilities):
            writer.writerow(
                [record.uid, record.label, prediction, float(np.max(row)), *map(float, row)]
            )


def evaluate_model(model, records):
    texts = [record.text for record in records]
    labels = [record.label for record in records]
    predictions = model.predict(texts).tolist()
    probabilities = model.predict_proba(texts)
    metrics = classification_metrics(labels, predictions, probabilities, model.classes_)
    return metrics, predictions, probabilities


def select_candidate(results: dict, macro_f1_tolerance: float = 0.002) -> tuple[str, list[str]]:
    best_f1 = max(result["dev"]["macro_f1"] for result in results.values())
    eligible = [
        name
        for name, result in results.items()
        if best_f1 - result["dev"]["macro_f1"] <= macro_f1_tolerance
    ]
    selected = min(eligible, key=lambda name: results[name]["latency"]["p95_ms"])
    return selected, eligible


def train_candidates(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    latency_runs: int,
    macro_f1_tolerance: float,
) -> dict:
    train = read_jsonl(data_dir / "train.jsonl")
    dev = read_jsonl(data_dir / "dev.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    trained = {}
    for name in MODEL_NAMES:
        model = build_model(name, random_state=seed)
        started = time.perf_counter()
        model.fit([record.text for record in train], [record.label for record in train])
        fit_seconds = time.perf_counter() - started
        metrics, predictions, probabilities = evaluate_model(model, dev)
        artifact = output_dir / f"{name}.joblib"
        joblib.dump(model, artifact, compress=3)
        _write_predictions(
            output_dir / f"{name}_dev_predictions.csv",
            dev,
            predictions,
            probabilities,
            model.classes_,
        )
        results[name] = {
            "fit_seconds": fit_seconds,
            "dev": metrics,
            "latency": benchmark_batch_one(
                model, [record.text for record in dev], runs=latency_runs
            ),
            "artifact_bytes": artifact_size_bytes(artifact),
        }
        trained[name] = model
        print(f"{name}: dev_macro_f1={metrics['macro_f1']:.6f}")

    selected, eligible = select_candidate(results, macro_f1_tolerance)
    summary = {
        "selection_split": "dev",
        "selection_rule": (
            "lowest p95 batch-one latency among models within the configured "
            "macro-F1 tolerance of the best dev score"
        ),
        "macro_f1_tolerance": macro_f1_tolerance,
        "eligible_models": eligible,
        "selected_model": selected,
        "seed": seed,
        "models": results,
    }
    with (output_dir / "dev_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def finalize_selected(
    data_dir: Path, output_dir: Path, selected_name: str, seed: int, latency_runs: int
) -> dict:
    train = read_jsonl(data_dir / "train.jsonl")
    dev = read_jsonl(data_dir / "dev.jsonl")
    test = read_jsonl(data_dir / "test_labeled.jsonl")
    combined = train + dev
    model = build_model(selected_name, random_state=seed)
    started = time.perf_counter()
    model.fit([record.text for record in combined], [record.label for record in combined])
    fit_seconds = time.perf_counter() - started
    metrics, predictions, probabilities = evaluate_model(model, test)
    artifact = output_dir / "final_model.joblib"
    joblib.dump(model, artifact, compress=3)
    _write_predictions(
        output_dir / "test_predictions.csv", test, predictions, probabilities, model.classes_
    )
    result = {
        "model": selected_name,
        "training_split": "cleaned_train_plus_dev",
        "evaluation_split": "test_once_after_dev_selection",
        "seed": seed,
        "fit_seconds": fit_seconds,
        "test": metrics,
        "latency": benchmark_batch_one(
            model, [record.text for record in test], runs=latency_runs
        ),
        "artifact_bytes": artifact_size_bytes(artifact),
        "classes": model.classes_.tolist(),
    }
    with (output_dir / "final_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "selected_model.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"artifact": "final_model.joblib", "model": selected_name, "classes": result["classes"]},
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"final {selected_name}: test_macro_f1={metrics['macro_f1']:.6f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reproducible Hinglish sentiment baselines")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-runs", type=int, default=300)
    parser.add_argument("--selection-tolerance", type=float, default=0.002)
    parser.add_argument(
        "--final-evaluate",
        action="store_true",
        help="After dev selection, retrain the selected model on train+dev and evaluate test once",
    )
    args = parser.parse_args()
    summary = train_candidates(
        args.data_dir,
        args.output_dir,
        args.seed,
        args.latency_runs,
        args.selection_tolerance,
    )
    if args.final_evaluate:
        finalize_selected(
            args.data_dir,
            args.output_dir,
            summary["selected_model"],
            args.seed,
            args.latency_runs,
        )


if __name__ == "__main__":
    main()
