from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib


def predict(model, text: str) -> dict:
    started = time.perf_counter_ns()
    probabilities = model.predict_proba([text])[0]
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    best = int(probabilities.argmax())
    return {
        "label": str(model.classes_[best]),
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
        "latency_ms": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hinglish sentiment inference")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    print(json.dumps(predict(model, args.text), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

