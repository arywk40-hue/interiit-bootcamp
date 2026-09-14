"""Task metrics and actual batch-one CPU timing, shared by the four-task CLI."""
import platform
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from threadpoolctl import threadpool_info

from rinlu.evaluation.metrics import answer_tokens, rouge_scores, token_f1
from rinlu.evaluation.robustness import drop_internal_vowel, repeat_letter, remove_emoji, remove_punctuation


def predict_one(task, model, row):
    if task in ("sentiment", "intent"):
        if not row["text"].strip() or len(row["text"]) > 2000:
            raise ValueError("Classification requires 1–2,000 nonblank characters")
        probabilities = model.predict_proba([row["text"]])[0]
        return {"label": str(model.classes_[probabilities.argmax()]),
                "probabilities": dict(zip(model.classes_.tolist(), probabilities.tolist()))}
    if task == "qa":
        return model.predict(row["text"], row.get("context", ""))
    return model.predict(row["text"])


def evaluate(task, model, rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty split")
    predictions = [predict_one(task, model, row) for row in rows]
    if task in ("sentiment", "intent"):
        gold, predicted = [r["label"] for r in rows], [p["label"] for p in predictions]
        labels = sorted(set(gold) | set(model.classes_))
        result = {"accuracy": accuracy_score(gold, predicted),
                  "macro_f1": f1_score(gold, predicted, labels=labels, average="macro", zero_division=0),
                  "per_class": classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0)}
    elif task == "qa":
        result = {"exact_match": float(np.mean([answer_tokens(p["answer"]) == answer_tokens(r["answer"])
                                               for p, r in zip(predictions, rows)])),
                  "token_f1": float(np.mean([token_f1(p["answer"], r["answer"]) for p, r in zip(predictions, rows)])),
                  "answer_rate": float(np.mean([not p["no_answer"] for p in predictions])),
                  "no_answer_quality": "not measured: source has no annotated unanswerable questions"}
    else:
        scores = [rouge_scores(p["summary"], r["summary"]) for p, r in zip(predictions, rows)]
        result = {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}
    result["examples"] = len(rows)
    return result, predictions


def benchmark(task, model, rows, runs=100):
    if runs < 1 or not rows:
        raise ValueError("Benchmark needs positive runs and nonempty inputs")
    # Spread measurements across the split, rather than only its first examples.
    indexes = np.linspace(0, len(rows) - 1, runs, dtype=int)
    for index in indexes[:min(20, runs)]:
        predict_one(task, model, rows[index])
    timings = []
    for index in indexes:
        start = time.perf_counter_ns()
        predict_one(task, model, rows[index])
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return {"p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)),
            "p99_ms": float(np.percentile(timings, 99)), "mean_ms": float(np.mean(timings)),
            "sequential_requests_per_second": 1000 / float(np.mean(timings)),
            "measured_runs": runs, "scope": "preprocessing + prediction + output construction; warm model; no I/O",
            "platform": platform.platform(), "python": platform.python_version(),
            "threadpools": threadpool_info()}


def robustness(task, model, rows):
    """Diagnostic perturbations only. No claim that every edit preserves meaning."""
    transformations = {"vowel_drop": drop_internal_vowel, "repeat_letter": repeat_letter,
                       "emoji_ablation": lambda text, uid: remove_emoji(text),
                       "punctuation_ablation": lambda text, uid: remove_punctuation(text)}
    results = {}
    for name, transform in transformations.items():
        variants = [dict(row, text=transform(row["text"], row["uid"])) for row in rows]
        # Degenerate inputs after stripping symbols are not valid inference requests.
        paired = [(a, b) for a, b in zip(rows, variants) if b["text"].strip()]
        changed = [(a, b) for a, b in paired if a["text"] != b["text"]]
        if not changed:
            results[name] = {"changed_inputs": 0}
            continue
        original, perturbed = map(list, zip(*changed))
        clean_metrics, _ = evaluate(task, model, original)
        variant_metrics, _ = evaluate(task, model, perturbed)
        key = "macro_f1" if task in ("sentiment", "intent") else "token_f1" if task == "qa" else "rougeL_f1"
        results[name] = {"changed_inputs": len(changed), "metric": key,
                         "clean_on_same_inputs": clean_metrics[key], "perturbed": variant_metrics[key],
                         "delta": variant_metrics[key] - clean_metrics[key]}
    return results


def parameter_count(model):
    if hasattr(model, "named_steps"):
        classifier = model.named_steps["classifier"]
        coefficients = sum(e.coef_.size + e.intercept_.size for e in classifier.estimators_)
        # TF-IDF IDF values are learned too; vocabulary strings consume artifact memory.
        idf_values = sum(v.idf_.size for v in model.get_params().values() if hasattr(v, "idf_"))
        return {"classifier_coefficients_and_intercepts": coefficients, "idf_values": idf_values,
                "total_learned_scalars": coefficients + idf_values}
    ranker = model.ranker
    count = 0 if ranker is None else int(ranker.coef_.size + np.asarray(ranker.intercept_).size)
    return {"total_learned_scalars": count}
