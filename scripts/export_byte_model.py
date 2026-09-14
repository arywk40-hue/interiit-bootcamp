"""Export the trained byte model to task-specific ONNX graphs and dynamic int8."""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from rinlu.data.tasks import load_intent, load_qa, load_sentiment, load_summary
from rinlu.evaluation.metrics import token_f1
from rinlu.evaluation.robustness import (drop_internal_vowel, remove_emoji,
                                         remove_punctuation, repeat_letter)
from rinlu.neural import ByteMultiTaskConfig, ByteMultiTaskModel, collate_bytes, encode_bytes


class TaskGraph(nn.Module):
    def __init__(self, model, task):
        super().__init__(); self.model, self.task = model, task

    def forward(self, input_ids, symbol_ids, segment_ids, attention_mask):
        output = self.model(self.task, input_ids, symbol_ids, segment_ids, attention_mask)
        if self.task == "qa":
            return output["start_logits"], output["end_logits"], output["answerable_logits"]
        return output["logits"]


class UnifiedGraph(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, input_ids, symbol_ids, segment_ids, attention_mask):
        output = self.model.forward_all(input_ids, symbol_ids, segment_ids, attention_mask)
        return tuple(output[name] for name in (
            "sentiment_logits", "intent_logits", "summary_logits", "start_logits",
            "end_logits", "answerable_logits"))


def encoded_example(task, length):
    if task == "qa":
        row = encode_bytes("delivery kab hogi?", task, length,
                           "Order processing complete hai. Delivery kal hogi. " * 20)
    elif task == "summarization":
        row = encode_bytes("A: order late hai 😒\nB: delivery kal hogi.", task, length)
    else:
        row = encode_bytes("service bahut acchi hai 😄! " * 20, task, length)
    batch = collate_bytes([row])
    # Fixed-width graphs make latency and input contracts unambiguous.
    for key in batch:
        if batch[key].shape[1] < length:
            batch[key] = torch.nn.functional.pad(batch[key], (0, length - batch[key].shape[1]))
    return batch


def fixed_batch(text, task, length, context=""):
    batch = collate_bytes([encode_bytes(text, task, length, context)])
    for key in batch:
        if batch[key].shape[1] < length:
            batch[key] = torch.nn.functional.pad(batch[key], (0, length - batch[key].shape[1]))
    return batch


def feed_dict(batch):
    return {key: value.numpy().astype(np.int64) for key, value in batch.items()}


def cpu_session(ort, path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def build_output(task, arrays):
    if task in ("sentiment", "intent"):
        return int(arrays[0].argmax(-1)[0])
    if task == "qa":
        start = int(arrays[0].argmax(-1)[0])
        end_scores = arrays[1][0].copy(); end_scores[:start] = -1e4
        return start, int(end_scores.argmax())
    return float(arrays[0][0])


def quality(session, task, length, data_root, saved, limit, split):
    if task == "summarization":
        try:
            rows, _ = load_summary(data_root)
        except FileNotFoundError:
            return {"status": "not measured: official GupShup h2h files unavailable"}
        return {"status": "not measured: summary candidate labels are not trained",
                "available_dialogues": len(rows[split]), "split": split}
    loader = {"sentiment": load_sentiment, "intent": load_intent, "qa": load_qa}[task]
    rows, _ = loader(data_root); rows = rows[split][:limit or None]
    if task in ("sentiment", "intent"):
        label_map = (saved["sentiment_labels"] if task == "sentiment"
                     else saved["intent_labels"])
        reverse = {index: label for label, index in label_map.items()}
        def predict(values):
            return [reverse[build_output(task, session.run(
                None, feed_dict(fixed_batch(text, task, length))))] for text in values]
        expected = [row["label"] for row in rows]
        clean_text = [row["text"] for row in rows]
        predicted = predict(clean_text)
        result = {"split": split, "examples": len(expected),
                  "accuracy": float(accuracy_score(expected, predicted)),
                  "macro_f1": float(f1_score(expected, predicted, average="macro")),
                  "robustness": {}}
        transforms = {
            "vowel_drop": lambda row: drop_internal_vowel(row["text"], row["uid"]),
            "repeat_letter": lambda row: repeat_letter(row["text"], row["uid"]),
            "emoji_ablation": lambda row: remove_emoji(row["text"]),
            "punctuation_ablation": lambda row: remove_punctuation(row["text"]),
        }
        for name, transform in transforms.items():
            changed = [(row, transform(row)) for row in rows]
            changed = [(row, text) for row, text in changed
                       if text.strip() and text != row["text"]]
            if not changed:
                result["robustness"][name] = {"changed_inputs": 0}
                continue
            gold = [row["label"] for row, _ in changed]
            clean = predict([row["text"] for row, _ in changed])
            perturbed = predict([text for _, text in changed])
            clean_f1 = float(f1_score(gold, clean, average="macro"))
            perturbed_f1 = float(f1_score(gold, perturbed, average="macro"))
            result["robustness"][name] = {
                "changed_inputs": len(changed), "clean_macro_f1": clean_f1,
                "perturbed_macro_f1": perturbed_f1,
                "delta": perturbed_f1 - clean_f1,
                "prediction_stability": float(np.mean(np.asarray(clean) == np.asarray(perturbed))),
            }
        return result
    scores = []
    for row in rows:
        context = unicodedata.normalize("NFC", row["context"])
        answer = unicodedata.normalize("NFC", row["answer"])
        match = re.search(re.escape(answer), context, re.I)
        encoded = encode_bytes(row["text"], "qa", length, context)
        gold_end = (encoded["context_start"]
                    + len(context[:match.end()].encode("utf-8")) - 1) if match else length
        if not match or gold_end >= len(encoded["input_ids"]) - 1:
            continue
        output = session.run(None, feed_dict(fixed_batch(row["text"], task, length, context)))
        start, end = build_output(task, output)
        raw = context.encode("utf-8")
        left, right = start - encoded["context_start"], end - encoded["context_start"] + 1
        predicted = raw[max(0, left):max(0, right)].decode("utf-8", errors="ignore")
        scores.append(token_f1(predicted, answer))
    return {"split": split, "examples": len(scores),
            "token_f1": float(sum(scores) / len(scores)) if scores else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/byte_multitask/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/byte_multitask/onnx"))
    parser.add_argument("--report", type=Path, default=Path("reports/neural/export.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--quality-limit", type=int, default=0)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    if args.runs < 1: parser.error("runs must be positive")
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:
        raise SystemExit("Install the project export dependencies: pip install -e '.[export]'") from error
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    torch.set_num_threads(1)
    config = ByteMultiTaskConfig(**{key: value for key, value in saved["config"].items()
                                    if key in ByteMultiTaskConfig.__dataclass_fields__})
    model = ByteMultiTaskModel(config).eval(); model.load_state_dict(saved["state_dict"])
    args.output.mkdir(parents=True, exist_ok=True); report = {
        "checkpoint": str(args.checkpoint), "parameters": model.parameter_report(), "tasks": {}}
    lengths = {"sentiment": 160, "intent": 160, "qa": 256, "summarization": 160}
    for task, length in lengths.items():
        graph, batch = TaskGraph(model, task).eval(), encoded_example(task, length)
        inputs = tuple(batch[key] for key in
                       ("input_ids", "symbol_ids", "segment_ids", "attention_mask"))
        fp32, int8 = args.output / f"{task}.onnx", args.output / f"{task}.int8.onnx"
        output_names = (["start_logits", "end_logits", "answerable_logits"]
                        if task == "qa" else ["logits"])
        torch.onnx.export(graph, inputs, fp32, input_names=[
            "input_ids", "symbol_ids", "segment_ids", "attention_mask"],
            output_names=output_names, opset_version=17, do_constant_folding=True)
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        fp32_session = cpu_session(ort, fp32)
        session = cpu_session(ort, int8)
        feed = feed_dict(batch)
        with torch.inference_mode():
            torch_output = graph(*inputs)
        torch_arrays = ([value.detach().numpy() for value in torch_output]
                        if isinstance(torch_output, tuple) else [torch_output.detach().numpy()])
        fp32_arrays = fp32_session.run(None, feed)
        int8_arrays = session.run(None, feed)
        fp32_error = max(float(np.max(np.abs(left - right)))
                           for left, right in zip(torch_arrays, fp32_arrays))
        int8_error = max(float(np.max(np.abs(left - right)))
                          for left, right in zip(torch_arrays, int8_arrays))
        samples = []
        for _ in range(args.runs):
            started = time.perf_counter_ns(); arrays = session.run(None, feed); build_output(task, arrays)
            samples.append((time.perf_counter_ns() - started) / 1e6)
        complete = []
        for _ in range(args.runs):
            started = time.perf_counter_ns()
            fresh = encoded_example(task, length)
            arrays = session.run(None, feed_dict(fresh)); build_output(task, arrays)
            complete.append((time.perf_counter_ns() - started) / 1e6)
        report["tasks"][task] = {
            "input_bytes": length, "fp32_bytes": fp32.stat().st_size,
            "int8_bytes": int8.stat().st_size, "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)),
            "end_to_end_p50_ms": float(np.percentile(complete, 50)),
            "end_to_end_p95_ms": float(np.percentile(complete, 95)),
            "end_to_end_p99_ms": float(np.percentile(complete, 99)), "runs": args.runs,
            "fp32_max_abs_error_vs_pytorch": fp32_error,
            "int8_max_abs_error_vs_pytorch": int8_error,
            "scope": "warm batch-one CPU; end-to-end includes byte encoding and output construction; file I/O excluded",
            "int8_quality": quality(session, task, length, args.data, saved,
                                    args.quality_limit, args.split)}
        print(json.dumps({"task": task, **report["tasks"][task]}), flush=True)
    # A single deployment graph shares one encoder copy across all four heads.
    unified_batch = encoded_example("qa", 256)
    unified_inputs = tuple(unified_batch[key] for key in
                           ("input_ids", "symbol_ids", "segment_ids", "attention_mask"))
    unified_fp32 = args.output / "unified.onnx"
    unified_int8 = args.output / "unified.int8.onnx"
    torch.onnx.export(
        UnifiedGraph(model).eval(), unified_inputs, unified_fp32,
        input_names=["input_ids", "symbol_ids", "segment_ids", "attention_mask"],
        output_names=["sentiment_logits", "intent_logits", "summary_logits",
                      "start_logits", "end_logits", "answerable_logits"],
        opset_version=17, do_constant_folding=True,
    )
    quantize_dynamic(str(unified_fp32), str(unified_int8), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    report["unified_artifact"] = {
        "input_bytes": 256, "fp32_bytes": unified_fp32.stat().st_size,
        "int8_bytes": unified_int8.stat().st_size,
        "description": "one shared encoder and all four heads; select the requested output",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
