"""Run bounded local inference with the trained int8 byte model."""
import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from rinlu.neural import collate_bytes, encode_bytes
from rinlu.summarization import sentence_spans


LENGTHS = {"sentiment": 160, "intent": 160, "qa": 256, "summarization": 160}


def read_value(value, path):
    if value is not None:
        return value
    return path.read_text(encoding="utf-8") if path else ""


def fixed_feed(text, task, length, context=""):
    encoded = encode_bytes(text, task, length, context)
    batch = collate_bytes([encoded])
    for key, value in batch.items():
        if value.shape[1] < length:
            batch[key] = torch.nn.functional.pad(value, (0, length - value.shape[1]))
    return ({key: value.numpy().astype(np.int64) for key, value in batch.items()}, encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "intent", "qa", "summarization"], required=True)
    parser.add_argument("--text"); parser.add_argument("--text-file", type=Path)
    parser.add_argument("--context"); parser.add_argument("--context-file", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/byte_multitask/model.pt"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("models/byte_multitask/onnx"))
    args = parser.parse_args()
    text = read_value(args.text, args.text_file)
    context = read_value(args.context, args.context_file)
    if not text.strip() or (args.task == "qa" and not context.strip()):
        parser.error("nonblank text and QA context are required")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.task == "summarization" and args.task not in saved.get("supervised_tasks", []):
        raise SystemExit("Neural summarization is disabled until human GupShup h2h labels train its head; use the extractive baseline meanwhile.")
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1; options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.onnx_dir / f"{args.task}.int8.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    warm_feed, _ = fixed_feed(text, args.task, LENGTHS[args.task], context)
    session.run(None, warm_feed)
    started = time.perf_counter_ns()
    feed, encoded = fixed_feed(text, args.task, LENGTHS[args.task], context)
    output = session.run(None, feed)
    if args.task in ("sentiment", "intent"):
        labels = saved["sentiment_labels"] if args.task == "sentiment" else saved["intent_labels"]
        reverse = {index: label for label, index in labels.items()}
        logits = output[0][0]; shifted = np.exp(logits - logits.max())
        probabilities = shifted / shifted.sum(); prediction = int(probabilities.argmax())
        result = {"label": reverse[prediction],
                  "probabilities": {reverse[i]: float(value) for i, value in enumerate(probabilities)}}
    elif args.task == "qa":
        start = int(output[0].argmax(-1)[0]); end_scores = output[1][0].copy()
        end_scores[:start] = -1e4; end = int(end_scores.argmax())
        raw = unicodedata.normalize("NFC", context).encode("utf-8")
        left, right = start - encoded["context_start"], end - encoded["context_start"] + 1
        answer = raw[max(0, left):max(0, right)].decode("utf-8", errors="ignore")
        result = {"answer": answer, "byte_start": max(0, left), "byte_end": max(0, right),
                  "extractive": True}
    else:
        spans = sentence_spans(text)
        candidates = []
        for index, (left, right) in enumerate(spans):
            candidate_feed, _ = fixed_feed(text[left:right], args.task, LENGTHS[args.task])
            score = float(session.run(None, candidate_feed)[0][0])
            candidates.append((score, index, left, right))
        selected, words = [], 0
        for _, index, left, right in sorted(candidates, reverse=True):
            count = len(text[left:right].split())
            if words + count <= 80:
                selected.append((index, left, right)); words += count
            if len(selected) == 3:
                break
        selected.sort()
        result = {"summary": " ".join(text[left:right] for _, left, right in selected),
                  "spans": [[left, right] for _, left, right in selected], "extractive": True}
    result.update(task=args.task,
                  latency_ms=(time.perf_counter_ns() - started) / 1e6,
                  max_input_bytes=LENGTHS[args.task], quantized=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
