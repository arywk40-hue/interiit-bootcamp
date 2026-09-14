"""Measure sustained concurrent throughput with already-loaded local models."""
import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from threadpoolctl import threadpool_limits

from rinlu.data.tasks import load_intent, load_qa, load_sentiment
from rinlu.evaluation.tasks import predict_one
from export_byte_model import build_output, cpu_session, feed_dict, fixed_batch


LOADERS = {"sentiment": load_sentiment, "intent": load_intent, "qa": load_qa}
LENGTHS = {"sentiment": 160, "intent": 160, "qa": 256}


def measure(call, total, workers):
    """Keep each worker busy without creating an artificial unbounded queue."""
    counts = [total // workers + (index < total % workers) for index in range(workers)]

    def loop(worker, count):
        samples = []
        for offset in range(count):
            started = time.perf_counter_ns()
            call(worker + offset * workers)
            samples.append((time.perf_counter_ns() - started) / 1e6)
        return samples

    for index in range(min(20, total)):
        call(index)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(loop, worker, count)
                   for worker, count in enumerate(counts) if count]
        samples = [value for future in futures for value in future.result()]
    seconds = time.perf_counter() - started
    return {"workers": workers, "requests": len(samples),
            "wall_seconds": seconds, "requests_per_second": len(samples) / seconds,
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("models/current"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("models/byte_multitask/onnx"))
    parser.add_argument("--output", type=Path, default=Path("reports/current/throughput.json"))
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--tasks", default="sentiment,intent,qa")
    args = parser.parse_args()
    workers = [int(value) for value in args.workers.split(",")]
    tasks = args.tasks.split(",")
    if args.requests < 1 or not workers or min(workers) < 1:
        parser.error("requests and worker counts must be positive")
    if not tasks or any(task not in LOADERS for task in tasks):
        parser.error("tasks must be a comma-separated subset of sentiment,intent,qa")
    report = {"scope": ("sustained in-process calls to one shared loaded model; preprocessing, "
                         "inference and output construction included; startup and file I/O excluded"),
              "platform": platform.platform(), "python": platform.python_version(),
              "requests_per_setting": args.requests, "results": {}}
    with threadpool_limits(limits=1):
        for task in tasks:
            loader = LOADERS[task]
            rows, _ = loader(args.data); rows = rows["test"]
            bundle = joblib.load(args.baseline_dir / f"{task}.joblib")
            baseline = bundle["model"]
            baseline_call = lambda index, t=task, m=baseline, values=rows: predict_one(
                t, m, values[index % len(values)])
            report["results"][f"baseline_{task}"] = [
                measure(baseline_call, args.requests, count) for count in workers]
            session = cpu_session(ort, args.onnx_dir / f"{task}.int8.onnx")

            def neural_call(index, t=task, values=rows, graph=session):
                row = values[index % len(values)]
                batch = fixed_batch(row["text"], t, LENGTHS[t], row.get("context", ""))
                return build_output(t, graph.run(None, feed_dict(batch)))

            report["results"][f"neural_{task}"] = [
                measure(neural_call, args.requests, count) for count in workers]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
