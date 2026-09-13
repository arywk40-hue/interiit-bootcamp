from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from rinlu.data.conll import read_jsonl
from rinlu.evaluation.metrics import artifact_size_bytes, benchmark_batch_one
from rinlu.sentiment.features import MODEL_NAMES, build_model
from rinlu.sentiment.train import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a preregistered train+dev candidate")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=MODEL_NAMES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-runs", type=int, default=500)
    args = parser.parse_args()

    train = read_jsonl(args.data_dir / "train.jsonl")
    dev = read_jsonl(args.data_dir / "dev.jsonl")
    test = read_jsonl(args.data_dir / "test_labeled.jsonl")
    model = build_model(args.model_name, random_state=args.seed)
    started = time.perf_counter()
    model.fit(
        [record.text for record in train + dev],
        [record.label for record in train + dev],
    )
    fit_seconds = time.perf_counter() - started
    metrics, _, _ = evaluate_model(model, test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / f"{args.model_name}_train_dev.joblib"
    joblib.dump(model, artifact, compress=3)
    result = {
        "model": args.model_name,
        "training_split": "cleaned_train_plus_dev",
        "purpose": "preregistered secondary contrast-set candidate; not selected as final clean-set model",
        "seed": args.seed,
        "fit_seconds": fit_seconds,
        "test": metrics,
        "latency": benchmark_batch_one(
            model, [record.text for record in test], runs=args.latency_runs
        ),
        "artifact": artifact.name,
        "artifact_bytes": artifact_size_bytes(artifact),
    }
    result_path = args.output_dir / f"{args.model_name}_train_dev_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

