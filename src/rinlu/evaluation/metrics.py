from __future__ import annotations

import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def expected_calibration_error(
    labels: list[str], probabilities: np.ndarray, classes: np.ndarray, bins: int = 10
) -> float:
    label_to_index = {label: index for index, label in enumerate(classes)}
    true_indexes = np.array([label_to_index[label] for label in labels])
    predicted_indexes = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted_indexes == true_indexes
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def classification_metrics(
    labels: list[str], predictions: list[str], probabilities: np.ndarray, classes: np.ndarray
) -> dict:
    ordered = ["negative", "neutral", "positive"]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "expected_calibration_error_10_bins": expected_calibration_error(
            labels, probabilities, classes
        ),
        "per_class": classification_report(
            labels, predictions, labels=ordered, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": ordered,
            "values": confusion_matrix(labels, predictions, labels=ordered).tolist(),
        },
    }


def benchmark_batch_one(model, texts: list[str], warmup: int = 30, runs: int = 300) -> dict:
    if not texts:
        raise ValueError("latency benchmark requires at least one text")
    for index in range(warmup):
        model.predict_proba([texts[index % len(texts)]])
    samples_ms = []
    for index in range(runs):
        text = texts[index % len(texts)]
        started = time.perf_counter_ns()
        model.predict_proba([text])
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    values = np.asarray(samples_ms)
    total_seconds = values.sum() / 1000.0
    return {
        "warmup_runs": warmup,
        "measured_runs": runs,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "sequential_throughput_per_second": float(runs / total_seconds),
        "scope": "end_to_end_vectorization_plus_prediction_batch_size_1",
        "hardware": {
            "processor": platform.processor() or "unknown",
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def artifact_size_bytes(path: Path) -> int:
    return path.stat().st_size


def answer_tokens(text: str) -> list[str]:
    """Case-insensitive lexical tokens for QA/summary overlap (not model preprocessing)."""
    return re.findall(r"\w+", text.casefold())


def token_f1(prediction: str, reference: str) -> float:
    left, right = Counter(answer_tokens(prediction)), Counter(answer_tokens(reference))
    if not left or not right:
        return float(left == right)
    common = sum((left & right).values())
    return 2 * common / (sum(left.values()) + sum(right.values()))


def rouge_scores(prediction: str, reference: str) -> dict:
    """ROUGE-1/2/L F1 without stemming, using the same explicit tokenization."""
    a, b = answer_tokens(prediction), answer_tokens(reference)
    bigrams_a, bigrams_b = Counter(zip(a, a[1:])), Counter(zip(b, b[1:]))
    denominator = sum(bigrams_a.values()) + sum(bigrams_b.values())
    rouge2 = 2 * sum((bigrams_a & bigrams_b).values()) / denominator if denominator else 0.0
    previous = [0] * (len(b) + 1)
    for word in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if word == other else max(previous[j], current[-1]))
        previous = current
    rouge_l = 2 * previous[-1] / (len(a) + len(b)) if a or b else 1.0
    return {"rouge1_f1": token_f1(prediction, reference), "rouge2_f1": rouge2, "rougeL_f1": rouge_l}
