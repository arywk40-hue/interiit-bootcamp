"""Adapt character features to private WhatsApp text without inventing labels.

The private messages fit only the TF-IDF vocabulary/IDF. Task labels still come
from the public labelled training split. Raw messages are never copied to output.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from rinlu.data.tasks import load_intent, load_sentiment
from rinlu.evaluation.tasks import benchmark, evaluate, parameter_count, robustness
from rinlu.sentiment.features import build_model


URL = re.compile(r"https?://\S+|www\.\S+", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s]?)?\d{10}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def read_private(paths):
    seen, messages, audit = set(), [], []
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="replace")
        phone_hits, email_hits, url_hits = len(PHONE.findall(raw)), len(EMAIL.findall(raw)), len(URL.findall(raw))
        for line in raw.splitlines():
            text = URL.sub("[url]", EMAIL.sub("[email]", PHONE.sub("[phone]", line))).strip()
            key = " ".join(text.casefold().split())
            if key and key not in seen:
                seen.add(key); messages.append(text)
        audit.append({"file": path.name, "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                      "lines": len(raw.splitlines()), "phone_matches": phone_hits,
                      "email_matches": email_hits, "url_matches_redacted": url_hits})
    return messages, audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("sentiment", "intent"), required=True)
    p.add_argument("--chat", type=Path, action="append", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("models/domain_adapted"))
    p.add_argument("--report-dir", type=Path, default=Path("reports/domain_adapted"))
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--skip-robustness", action="store_true",
                   help="Skip slow perturbation diagnostics during local adaptation")
    a = p.parse_args()
    chats, private_audit = read_private(a.chat)
    loader = load_sentiment if a.task == "sentiment" else load_intent
    splits, source_audit = loader(a.data_dir)
    model = build_model("char_tfidf", random_state=42)
    features, classifier = model.named_steps["features"], model.named_steps["classifier"]
    train_text = [row["text"] for row in splits["train"]]
    started = time.perf_counter(); features.fit(chats + train_text)
    classifier.fit(features.transform(train_text), [row["label"] for row in splits["train"]])
    fit_seconds = time.perf_counter() - started
    a.output_dir.mkdir(parents=True, exist_ok=True); artifact = a.output_dir / f"{a.task}.joblib"
    joblib.dump({"task": a.task, "model": model, "private_text_in_vocabulary": True}, artifact, compress=3)
    with threadpool_limits(limits=1):
        dev, _ = evaluate(a.task, model, splits["dev"]); test, _ = evaluate(a.task, model, splits["test"])
        latency = benchmark(a.task, model, splits["test"], a.runs)
        stress = {} if a.skip_robustness else robustness(a.task, model, splits["test"])
    report = {"task": a.task, "method": "unlabelled WhatsApp TF-IDF adaptation; public labels only",
              "private_unique_messages": len(chats), "private_sources": private_audit,
              "source_audit": source_audit, "fit_seconds": fit_seconds, "dev": dev, "test": test,
              "latency": latency, "parameters": parameter_count(model), "artifact_bytes": artifact.stat().st_size,
              "robustness_diagnostics": stress,
              "robustness_skipped": a.skip_robustness,
              "privacy_warning": "Artifact vocabulary contains private-source n-grams; do not publish without consent."}
    a.report_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / f"{a.task}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"task": a.task, "private_unique_messages": len(chats),
                      "dev_macro_f1": dev["macro_f1"], "test_macro_f1": test["macro_f1"],
                      "p95_ms": latency["p95_ms"], "artifact": str(artifact)}, indent=2))


if __name__ == "__main__":
    main()
