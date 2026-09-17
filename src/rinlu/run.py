"""One CLI for four small tasks. Start with: python -m rinlu.run --help."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from rinlu.data.tasks import LOADERS
from rinlu.evaluation.tasks import benchmark, evaluate, parameter_count, predict_one, robustness
from rinlu.qa import SpanReader
from rinlu.sentiment.features import build_model
from rinlu.summarization import ExtractiveSummarizer


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def split_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train(task, args):
    splits, audit = LOADERS[task](args.data)
    if any(not rows for rows in splits.values()):
        raise ValueError(f"{task} has an empty split")
    config = json.loads(args.config.read_text())
    candidates, scores = {}, {}
    if task in ("sentiment", "intent"):
        for name in config["candidate_models"]:
            model = build_model(name, random_state=config["seed"], config=config)
            model.fit([r["text"] for r in splits["train"]], [r["label"] for r in splits["train"]])
            metrics, _ = evaluate(task, model, splits["dev"])
            latency = benchmark(task, model, splits["dev"], args.runs)
            scores[name] = {"dev": metrics, "latency": latency}
            candidates[name] = model
            print(f"{task}/{name}: dev macro-F1={metrics['macro_f1']:.4f}", flush=True)
        best = max(s["dev"]["macro_f1"] for s in scores.values())
        eligible = [name for name in scores if best - scores[name]["dev"]["macro_f1"] <= config["selection_tolerance"]]
        selected = min(eligible, key=lambda name: scores[name]["latency"]["p95_ms"])
        model = candidates[selected]
    else:
        model = SpanReader(**config["qa"]) if task == "qa" else ExtractiveSummarizer(**config["summarization"])
        model.fit(splits["train"])
        selected = "linear_span_ranker" if task == "qa" else "linear_extractive_summarizer"
        metrics, _ = evaluate(task, model, splits["dev"])
        scores[selected] = {"dev": metrics, "latency": benchmark(task, model, splits["dev"], args.runs)}
    # Retain train-only fitting so dev remains independent and repeatable.
    bundle = {"task": task, "model": model, "config": config, "selected": selected,
              "split_sha256": {s: split_hash(rows) for s, rows in splits.items()}}
    artifact = args.models / f"{task}.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact, compress=3)
    report = {"task": task, "training_split": "train_only", "selected": selected, "candidates": scores,
              "counts": {s: len(rows) for s, rows in splits.items()}, "audit": audit,
              "split_sha256": bundle["split_sha256"], "artifact_bytes": artifact.stat().st_size,
              "parameters": parameter_count(model), "config": config, "code_sha256": code_hash(),
              "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl")}}
    write_json(args.reports / f"{task}_training.json", report)
    print(f"Saved {artifact}", flush=True)


def test(task, args):
    bundle = joblib.load(args.models / f"{task}.joblib")
    splits, _ = LOADERS[task](args.data)
    if bundle["split_sha256"] != {s: split_hash(rows) for s, rows in splits.items()}:
        raise ValueError("Dataset changed since training; use the matching data or retrain")
    model, rows = bundle["model"], splits["test"]
    metrics, predictions = evaluate(task, model, rows)
    result = {"task": task, "test": metrics, "latency": benchmark(task, model, rows, args.runs),
              "parameters": parameter_count(model), "robustness_diagnostics": robustness(task, model, rows),
              "inference_code_sha256": code_hash(),
              "artifact_sha256": hashlib.sha256((args.models / f"{task}.joblib").read_bytes()).hexdigest()}
    result["meets_measured_p95_under_10_ms"] = result["latency"]["p95_ms"] < 10
    result["meets_500m_learned_scalars"] = result["parameters"]["total_learned_scalars"] <= 500_000_000
    write_json(args.reports / f"{task}_test.json", result)
    # Local-only: QA answers and summaries contain extracts of source text.
    write_json(args.reports / f"{task}_predictions.json",
               [{"uid": row["uid"], "prediction": prediction} for row, prediction in zip(rows, predictions)])
    print(json.dumps({"task": task, "test": {k: v for k, v in metrics.items() if k != "per_class"},
                      "p95_ms": result["latency"]["p95_ms"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test", "predict"))
    parser.add_argument("--task", choices=(*LOADERS, "all"), required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--models", type=Path, default=Path("models/current"))
    parser.add_argument("--reports", type=Path, default=Path("reports/current"))
    parser.add_argument("--config", type=Path, default=Path("configs/sentiment.json"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--text", "--question", dest="text", default="",
                        help="Input text; --question is an alias for QA")
    parser.add_argument("--text-file", type=Path, help="Read a UTF-8 dialogue/question from a file")
    parser.add_argument("--context", default="")
    parser.add_argument("--context-file", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.command == "predict":
        if args.task == "all":
            parser.error("predict requires one explicit task")
        artifact = args.models / f"{args.task}.joblib"
        if args.task == "summarization" and not artifact.exists():
            config = json.loads(args.config.read_text())
            model = ExtractiveSummarizer(**config["summarization"])
        else:
            bundle = joblib.load(artifact)
            if bundle["task"] != args.task:
                raise ValueError("Model task mismatch")
            model = bundle["model"]
        text = args.text_file.read_text() if args.text_file else args.text
        context = args.context_file.read_text() if args.context_file else args.context
        result = predict_one(args.task, model, {"text": text, "context": context})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    tasks = list(LOADERS) if args.task == "all" else [args.task]
    missing = {}
    with threadpool_limits(limits=1):
        for task in tasks:
            try:
                (train if args.command == "train" else test)(task, args)
            except FileNotFoundError as error:
                if args.task != "all":
                    raise
                missing[task] = str(error)
                print(f"PENDING {task}: {error}", flush=True)
    if missing:
        write_json(args.reports / "pending.json", missing)
        raise SystemExit(2)
    if args.task == "all":
        write_json(args.reports / "pending.json", {})


if __name__ == "__main__":
    main()
