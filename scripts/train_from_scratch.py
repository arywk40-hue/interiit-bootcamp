"""Pretrain and jointly train the byte model without loading existing weights."""
import argparse
import copy
import csv
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from rinlu.data.conll import duplicate_key
from rinlu.data.tasks import load_intent, load_qa, load_sentiment, load_summary
from rinlu.evaluation.metrics import rouge_scores, token_f1
from rinlu.neural import (BYTE_OFFSET, MASK, VOCAB_SIZE, ByteMultiTaskConfig,
                          ByteMultiTaskModel, collate_bytes, encode_bytes,
                          mask_unicode_characters)
from rinlu.summarization import sentence_spans


URL = re.compile(r"https?://\S+|www\.\S+", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s]?)?\d{10}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
SENTIMENT_LABELS = {"negative": 0, "neutral": 1, "positive": 2}


class RowDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limited(rows, limit, seed):
    values = list(rows)
    random.Random(seed).shuffle(values)
    return values[:limit] if limit else values


def balanced_text_limit(rows, limit, seed):
    """Deterministically interleave sources so one large corpus cannot erase others."""
    if not limit or len(rows) <= limit:
        values = list(rows)
        random.Random(seed).shuffle(values)
        return values
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    randomizer = random.Random(seed)
    for values in by_source.values():
        randomizer.shuffle(values)
    names = sorted(by_source)
    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for name in names:
            if offset < len(by_source[name]):
                selected.append(by_source[name][offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    randomizer.shuffle(selected)
    return selected


def read_pretrain_sources(data_dir, private_paths, heldout, seed, limit):
    sources = {}
    phinc = data_dir / "raw/pretrain/PHINC.csv"
    if phinc.exists():
        with phinc.open(encoding="utf-8", errors="replace") as handle:
            sources["phinc_hinglish"] = [row["Sentence"] for row in csv.DictReader(handle)
                                            if row.get("Sentence", "").strip()]
    gupshup = data_dir / "raw/pretrain/gupshup_train.jsonl"
    if gupshup.exists():
        values = []
        for line in gupshup.open(encoding="utf-8", errors="replace"):
            text = json.loads(line).get("text", "").strip()
            if text: values.append(text)
        sources["gupshup_hf_train"] = values
    private_audit = []
    for path in private_paths:
        raw = path.read_text(encoding="utf-8", errors="replace")
        values = [URL.sub("[url]", EMAIL.sub("[email]", PHONE.sub("[phone]", line))).strip()
                  for line in raw.splitlines()]
        sources[f"private:{path.name}"] = [text for text in values if text]
        private_audit.append({"file": path.name, "sha256": sha256(path),
                              "lines": len(raw.splitlines())})
    seen, rows, counts = set(), [], Counter()
    for source, values in sources.items():
        for text in values:
            key = duplicate_key(text)
            if not key or key in heldout or key in seen:
                continue
            seen.add(key); rows.append({"text": text, "source": source}); counts[source] += 1
    random.Random(seed).shuffle(rows)
    if limit: rows = rows[:limit]
    selected_counts = Counter(row["source"] for row in rows)
    return rows, {"eligible_by_source": dict(counts), "selected_by_source": dict(selected_counts),
                  "unique_selected": len(rows), "private_sources": private_audit}


def masked_batch(rows, max_length, generator):
    encoded = [encode_bytes(row["text"], "masked_byte", max_length=max_length) for row in rows]
    labels = []
    for row in encoded:
        row["input_ids"], target = mask_unicode_characters(row, generator)
        labels.append(target)
    batch = collate_bytes(encoded)
    width = batch["input_ids"].shape[1]
    batch["labels"] = torch.tensor(
        [target + [-100] * (width - len(target)) for target in labels], dtype=torch.long
    )
    return batch


def classification_batch(rows, task, label_map, max_length):
    batch = collate_bytes([encode_bytes(row["text"], task, max_length=max_length)
                           for row in rows])
    batch["labels"] = torch.tensor([label_map[row["label"]] for row in rows])
    return batch


def qa_target(row, max_length):
    context = unicodedata.normalize("NFC", row["context"])
    answer = unicodedata.normalize("NFC", row["answer"])
    match = re.search(re.escape(answer), context, re.I)
    if not match: return None
    encoded = encode_bytes(row["text"], "qa", max_length=max_length, context=context)
    start = encoded["context_start"] + len(context[:match.start()].encode("utf-8"))
    end = encoded["context_start"] + len(context[:match.end()].encode("utf-8")) - 1
    if end >= len(encoded["input_ids"]) - 1: return None
    return encoded, start, end


def qa_batch(rows, max_length):
    prepared = [qa_target(row, max_length) for row in rows]
    if any(item is None for item in prepared): raise ValueError("QA batch contains truncated target")
    batch = collate_bytes([item[0] for item in prepared])
    batch["start"] = torch.tensor([item[1] for item in prepared])
    batch["end"] = torch.tensor([item[2] for item in prepared])
    return batch


def summary_candidates(rows):
    candidates = []
    for row in rows:
        spans = sentence_spans(row["text"])
        for index, (start, end) in enumerate(spans):
            text = row["text"][start:end]
            candidates.append({"uid": f"{row['uid']}:{index}", "text": text,
                               "target": token_f1(text, row["summary"])})
    return candidates


def summary_batch(rows, max_length):
    batch = collate_bytes([encode_bytes(row["text"], "summarization", max_length=max_length)
                           for row in rows])
    batch["targets"] = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
    return batch


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def class_weights(rows, label_map):
    counts = Counter(row["label"] for row in rows)
    weights = torch.zeros(len(label_map), dtype=torch.float32)
    for label, index in label_map.items():
        if counts[label]:
            weights[index] = (len(rows) / (len(counts) * counts[label])) ** 0.5
    return weights


def canonical_corpus_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["source"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(unicodedata.normalize("NFC", row["text"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_pretraining_corpus(seed_text, extra, heldout, limit, seed):
    seen, eligible = set(), []
    for row in seed_text + extra:
        key = duplicate_key(row["text"])
        if key and key not in heldout and key not in seen:
            seen.add(key); eligible.append(row)
    selected = balanced_text_limit(eligible, limit, seed)
    if any(duplicate_key(row["text"]) in heldout for row in selected):
        raise AssertionError("held-out text entered the pretraining corpus")
    return selected


def classification_metrics(model, rows, task, label_map, device, max_length, batch_size):
    reverse, expected, predicted = {value: key for key, value in label_map.items()}, [], []
    loader = DataLoader(RowDataset(rows), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda values: classification_batch(values, task, label_map, max_length))
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels"); logits = model(task, **move(batch, device))["logits"]
            expected.extend(reverse[value] for value in labels.tolist())
            predicted.extend(reverse[value] for value in logits.argmax(-1).cpu().tolist())
    return {"examples": len(expected), "accuracy": accuracy_score(expected, predicted),
            "macro_f1": f1_score(expected, predicted, average="macro")}


def qa_metrics(model, rows, device, max_length):
    scores, usable = [], 0; model.eval()
    with torch.inference_mode():
        for row in rows:
            target = qa_target(row, max_length)
            if target is None: continue
            encoded, _, _ = target; batch = move(collate_bytes([encoded]), device)
            output = model("qa", **batch)
            start = int(output["start_logits"].argmax(-1)[0]); end_logits = output["end_logits"][0].clone()
            end_logits[:start] = -1e4; end = int(end_logits.argmax())
            raw = unicodedata.normalize("NFC", row["context"]).encode("utf-8")
            left, right = start - encoded["context_start"], end - encoded["context_start"] + 1
            answer = raw[max(0, left):max(0, right)].decode("utf-8", errors="ignore")
            scores.append(token_f1(answer, row["answer"])); usable += 1
    return {"examples": usable, "token_f1": sum(scores) / usable if usable else 0}


def summary_metrics(model, rows, device, max_length):
    metrics = []; model.eval()
    with torch.inference_mode():
        for row in rows:
            spans = sentence_spans(row["text"])
            if not spans:
                continue
            chunks = [row["text"][start:end] for start, end in spans]
            batch = move(collate_bytes([encode_bytes(text, "summarization", max_length)
                                        for text in chunks]), device)
            scores = model("summarization", **batch)["logits"].cpu().tolist()
            selected, words = [], 0
            for index in sorted(range(len(spans)), key=lambda i: scores[i], reverse=True):
                count = len(chunks[index].split())
                if words + count <= 80:
                    selected.append(index); words += count
                if len(selected) == 3:
                    break
            prediction = " ".join(chunks[index] for index in sorted(selected))
            metrics.append(rouge_scores(prediction, row["summary"]))
    return {"examples": len(metrics), **({key: sum(item[key] for item in metrics) / len(metrics)
             for key in metrics[0]} if metrics else {"rouge1_f1": 0, "rouge2_f1": 0,
                                                     "rougeL_f1": 0})}


def train(args):
    random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    config_data = json.loads(args.config.read_text())
    config = ByteMultiTaskConfig(**{key: value for key, value in config_data.items()
                                    if key in ByteMultiTaskConfig.__dataclass_fields__})
    model = ByteMultiTaskModel(config)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                                    weight_decay=args.weight_decay)
    sentiment, sentiment_audit = load_sentiment(args.data)
    intent, intent_audit = load_intent(args.data)
    qa, qa_audit = load_qa(args.data)
    try:
        summary, summary_audit = load_summary(args.data)
        summary_status = f"available ({len(summary['train'])} train dialogues)"
    except FileNotFoundError:
        summary, summary_audit = None, None
        summary_status = "missing official GupShup h2h files; summary head not supervised"
    intent_labels = {label: index for index, label in enumerate(sorted({
        row["label"] for split in intent.values() for row in split}))}
    if len(intent_labels) != config.intent_labels:
        raise ValueError(f"config expects {config.intent_labels} intent labels, found {len(intent_labels)}")
    heldout = {duplicate_key(row.get(field, "")) for splits in (sentiment, intent, qa)
               for split in ("dev", "test") for row in splits[split]
               for field in ("text", "context") if row.get(field)}
    # Train splits are valid unlabelled pretraining text in addition to supervised use.
    seed_text = ([{"text": row["text"], "source": "sentiment_train"} for row in sentiment["train"]]
                 + [{"text": row["text"], "source": "intent_train"} for row in intent["train"]]
                 + [{"text": row["context"], "source": "qa_train"} for row in qa["train"]])
    extra, pretrain_audit = read_pretrain_sources(args.data, args.private, heldout,
                                                  args.seed, 0)
    pretrain = build_pretraining_corpus(seed_text, extra, heldout,
                                        args.pretrain_examples, args.seed)
    pretrain_audit["final_selected_by_source"] = dict(Counter(
        row["source"] for row in pretrain))
    pretrain_audit["corpus_sha256"] = canonical_corpus_hash(pretrain)
    pretrain_audit["heldout_normalized_keys"] = len(heldout)
    pretrain_audit["heldout_overlap"] = sum(
        duplicate_key(row["text"]) in heldout for row in pretrain
    )
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=True)
        if resumed["config"] != config_data:
            raise ValueError("resume checkpoint architecture does not match the current config")
        if resumed["corpus_sha256"] != pretrain_audit["corpus_sha256"]:
            raise ValueError("resume checkpoint pretraining corpus does not match this run")
        model.load_state_dict(resumed["state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state"])
    generator = torch.Generator().manual_seed(args.seed)
    pretrain_loader = DataLoader(RowDataset(pretrain), batch_size=args.batch_size, shuffle=True,
                                 generator=generator, collate_fn=lambda rows: masked_batch(
                                     rows, args.pretrain_length, generator))
    history, started = [], time.perf_counter()
    for epoch in range(args.pretrain_epochs):
        model.train(); total = 0
        for batch in pretrain_loader:
            batch = move(batch, device); labels = batch.pop("labels")
            loss = F.cross_entropy(model("masked_byte", **batch)["logits"].transpose(1, 2),
                                   labels, ignore_index=-100)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step(); total += float(loss.detach())
        history.append({"phase": "masked_byte", "epoch": epoch + 1,
                        "loss": total / max(1, len(pretrain_loader))})
        print(json.dumps(history[-1]), flush=True)
    supervised = {
        "sentiment": (limited(sentiment["train"], args.supervised_examples, args.seed),
                      SENTIMENT_LABELS),
        "intent": (limited(intent["train"], args.supervised_examples, args.seed), intent_labels),
    }
    qa_train = [row for row in qa["train"] if qa_target(row, args.qa_length) is not None]
    qa_train = limited(qa_train, args.supervised_examples, args.seed)
    summary_train = (limited(summary_candidates(summary["train"]), args.supervised_examples,
                             args.seed) if summary else [])
    intent_weight = class_weights(supervised["intent"][0], intent_labels).to(device)
    best = ({"epoch": resumed["best_epoch"], "dev_score": resumed["best_dev_score"],
             "state_dict": copy.deepcopy(model.state_dict()),
             "optimizer_state": copy.deepcopy(optimizer.state_dict())} if resumed else None)
    epoch_offset = resumed["best_epoch"] if resumed else 0
    stale_epochs = 0
    for epoch in range(args.supervised_epochs):
        loaders = {}
        for task, (rows, labels) in supervised.items():
            loaders[task] = DataLoader(RowDataset(rows), batch_size=args.batch_size, shuffle=True,
                collate_fn=lambda values, t=task, m=labels: classification_batch(
                    values, t, m, args.classification_length))
        loaders["qa"] = DataLoader(RowDataset(qa_train), batch_size=max(1, args.batch_size // 4),
                                    shuffle=True, collate_fn=lambda rows: qa_batch(rows, args.qa_length))
        if summary_train:
            loaders["summarization"] = DataLoader(
                RowDataset(summary_train), batch_size=args.batch_size, shuffle=True,
                collate_fn=lambda rows: summary_batch(rows, args.classification_length))
        schedule = [task for task, loader in loaders.items() for _ in range(len(loader))]
        random.Random(args.seed + epoch).shuffle(schedule); iterators = {k: iter(v) for k, v in loaders.items()}
        losses = Counter(); counts = Counter(); model.train()
        for task in schedule:
            batch = move(next(iterators[task]), device)
            if task in supervised:
                labels = batch.pop("labels")
                weight = intent_weight if task == "intent" else None
                loss = F.cross_entropy(model(task, **batch)["logits"], labels, weight=weight)
            elif task == "qa":
                starts, ends = batch.pop("start"), batch.pop("end"); output = model("qa", **batch)
                loss = (F.cross_entropy(output["start_logits"], starts)
                        + F.cross_entropy(output["end_logits"], ends)) / 2
            else:
                targets = batch.pop("targets")
                loss = F.binary_cross_entropy_with_logits(
                    model("summarization", **batch)["logits"], targets)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step(); losses[task] += float(loss.detach()); counts[task] += 1
        metrics = {"sentiment": classification_metrics(model, sentiment["dev"], "sentiment",
                                                        SENTIMENT_LABELS, device,
                                                        args.classification_length, args.batch_size),
                   "intent": classification_metrics(model, intent["dev"], "intent", intent_labels,
                                                     device, args.classification_length, args.batch_size),
                   "qa": qa_metrics(model, qa["dev"], device, args.qa_length)}
        if summary:
            metrics["summarization"] = summary_metrics(
                model, summary["dev"], device, args.classification_length)
        selected_epoch = epoch_offset + epoch + 1
        history.append({"phase": "supervised", "epoch": selected_epoch,
                        "loss": {task: losses[task] / counts[task] for task in counts}, "dev": metrics})
        print(json.dumps(history[-1]), flush=True)
        score_parts = [metrics["sentiment"]["macro_f1"], metrics["intent"]["macro_f1"],
                       metrics["qa"]["token_f1"]]
        if summary:
            score_parts.append(metrics["summarization"]["rougeL_f1"])
        dev_score = sum(score_parts) / len(score_parts)
        if best is None or dev_score > best["dev_score"]:
            best = {"epoch": selected_epoch, "dev_score": dev_score,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state": copy.deepcopy(optimizer.state_dict())}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best is None:
        raise RuntimeError("supervised training produced no checkpoint")
    model.load_state_dict(best["state_dict"])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "optimizer_state": best["optimizer_state"],
                "config": config_data, "seed": args.seed,
                "corpus_sha256": pretrain_audit["corpus_sha256"],
                "best_epoch": best["epoch"], "best_dev_score": best["dev_score"],
                "supervised_tasks": ["sentiment", "intent", "qa"]
                                    + (["summarization"] if summary_train else []),
                "intent_labels": intent_labels, "sentiment_labels": SENTIMENT_LABELS,
                "random_initialization": True}
    torch.save(checkpoint, args.output / "model.pt")
    report = {"random_initialization": True, "pretrained_checkpoint": None,
              "parameters": model.parameter_report(), "device": str(device),
              "resumed_from": str(args.resume) if args.resume else None,
              "seconds": time.perf_counter() - started, "history": history,
              "selection": {"development_only": True, "best_epoch": best["epoch"],
                            "mean_dev_score": best["dev_score"],
                            "heldout_test_evaluated": False},
              "data": {"pretrain_examples": len(pretrain), "pretrain": pretrain_audit,
                       "sentiment": sentiment_audit, "intent": intent_audit, "qa": qa_audit,
                       "qa_train_targets_within_256_bytes": len(qa_train),
                       "summarization": summary_status,
                       "summarization_audit": summary_audit,
                       "summary_training_candidates": len(summary_train)},
              "limits": vars(args) | {"data": str(args.data), "config": str(args.config),
                                       "output": str(args.output),
                                       "resume": str(args.resume) if args.resume else None,
                                       "private": [path.name for path in args.private]}}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"artifact": str(args.output / 'model.pt'), "seconds": report["seconds"],
                      "parameters": report["parameters"]["unique_parameters"],
                      "last": history[-1]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/byte_multitask.json"))
    parser.add_argument("--output", type=Path, default=Path("models/byte_multitask"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--private", type=Path, action="append", default=[])
    parser.add_argument("--pretrain-examples", type=int, default=10000)
    parser.add_argument("--supervised-examples", type=int, default=3000)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--supervised-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-length", type=int, default=128)
    parser.add_argument("--classification-length", type=int, default=160)
    parser.add_argument("--qa-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.pretrain_epochs < 0:
        parser.error("pretraining epochs cannot be negative")
    if min(args.supervised_epochs, args.batch_size, args.threads, args.patience) < 1:
        parser.error("supervised epochs, batch size, thread count and patience must be positive")
    train(args)


if __name__ == "__main__":
    main()
