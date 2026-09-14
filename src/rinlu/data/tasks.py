"""Small source-specific loaders. Every model sees train text only during fitting."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from rinlu.data.conll import duplicate_key, read_jsonl
from rinlu.data.prepare import prepare_dataset


def clean_splits(splits, field="text"):
    """Prefer held-out examples; remove normalized repeats from earlier splits."""
    seen, removed = set(), {}
    for split in ("test", "dev", "train"):
        kept = []
        for row in splits[split]:
            key = duplicate_key(row[field])
            if key not in seen:
                kept.append(row)
                seen.add(key)
        removed[split] = len(splits[split]) - len(kept)
        splits[split] = kept
    return splits, removed


def load_sentiment(root):
    raw, out = root / "raw/sentiment", root / "processed/sentiment_current"
    audit = prepare_dataset(raw / "train_14k_split_conll.txt", raw / "dev_3k_split_conll.txt",
                            raw / "Hindi_test_unalbelled_conll_updated.txt",
                            raw / "test_labels_hinglish.txt", out)
    splits = {}
    for split in ("train", "dev", "test"):
        filename = "test_labeled.jsonl" if split == "test" else split + ".jsonl"
        splits[split] = [dict(uid=r.uid, text=r.text, label=r.label) for r in read_jsonl(out / filename)]
    return splits, audit


def load_intent(root):
    splits = {}
    for split, name in (("train", "train"), ("dev", "validation"), ("test", "test")):
        with (root / "raw/intent" / f"{name}.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        splits[split] = []
        for index, row in enumerate(rows):
            # The outer IN tag is the utterance intent; slots/nested intents are not new labels.
            match = re.match(r"\[IN:([A-Z_]+)\s", row["cs_parse"])
            if not match or not row["cs_query"].strip():
                raise ValueError(f"Malformed Hinglish-TOP row {name}:{index}")
            splits[split].append(dict(uid=f"{name}:{index}", text=row["cs_query"], label=match[1]))
    splits, removed = clean_splits(splits)
    return splits, {"source": "Hinglish-TOP human-only", "duplicates_removed": removed,
                    "labels": {s: dict(Counter(r['label'] for r in rows)) for s, rows in splits.items()}}


def load_qa(root):
    source = root / "raw/qa/code_mixed_qa_train.json"
    rows = json.loads(source.read_text())["questions"]
    usable, excluded, seen = [], Counter(), set()
    for row in rows:
        if row["language"] != "Hindi":
            excluded["other_language"] += 1
            continue
        context, answer = row["context"].strip(), row["answer"].strip()
        if not context or context.lower() == "general":
            excluded["no_text_context"] += 1
            continue
        if not answer or not re.search(re.escape(answer), context, re.IGNORECASE):
            excluded["answer_not_a_contiguous_span"] += 1
            continue
        key = (duplicate_key(context), duplicate_key(row["query"]))
        if key in seen:
            excluded["duplicate_question_context"] += 1
            continue
        seen.add(key)
        group = hashlib.sha256(key[0].encode()).hexdigest()
        usable.append(dict(uid=str(row["id"]), text=row["query"], context=context,
                           answer=answer, group=group))
    # Partition complete articles, not individual questions about the same article.
    groups = sorted({r["group"] for r in usable})
    random.Random(42).shuffle(groups)
    train_end, dev_end = int(len(groups) * .7), int(len(groups) * .85)
    assignment = {g: "train" if i < train_end else "dev" if i < dev_end else "test"
                  for i, g in enumerate(groups)}
    splits = {s: [r for r in usable if assignment[r["group"]] == s] for s in ("train", "dev", "test")}
    expansion_path = root / "raw/qa/human_expansion.jsonl"
    expansion_count = 0
    if expansion_path.exists():
        expansion_groups = {}
        for line_number, line in enumerate(expansion_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"uid", "text", "context", "answer", "group", "split",
                        "reviewer_ids", "consent_to_use"}
            if required - row.keys():
                raise ValueError(f"human QA row {line_number} is missing {sorted(required - row.keys())}")
            if row["split"] not in splits:
                raise ValueError(f"human QA row {line_number} has invalid split")
            if not row["consent_to_use"] or len(set(row["reviewer_ids"])) < 2:
                raise ValueError(f"human QA row {line_number} needs consent and two reviewers")
            if not re.search(re.escape(row["answer"]), row["context"], re.IGNORECASE):
                raise ValueError(f"human QA row {line_number} answer is not an exact context span")
            key = (duplicate_key(row["context"]), duplicate_key(row["text"]))
            if key in seen:
                raise ValueError(f"human QA row {line_number} duplicates an existing question/context")
            seen.add(key)
            group = "human:" + hashlib.sha256(str(row["group"]).encode()).hexdigest()
            previous_split = expansion_groups.setdefault(group, row["split"])
            if previous_split != row["split"]:
                raise ValueError(f"human QA group crosses splits at row {line_number}")
            splits[row["split"]].append(dict(uid=str(row["uid"]), text=row["text"],
                context=row["context"], answer=row["answer"], group=group))
            assignment[group] = row["split"]
            expansion_count += 1
    return splits, {"source": "CMQA original train, custom context-disjoint 70/15/15 split",
                    "excluded": dict(excluded), "context_groups": len(assignment),
                    "human_expansion_rows": expansion_count, "group_split": assignment}


def load_summary(root):
    folder = root / "raw/summarization"
    splits = {}
    for split in ("train", "dev", "test"):
        source, target = folder / f"{split}.source", folder / f"{split}.target"
        if not source.exists() or not target.exists():
            raise FileNotFoundError(f"Need GupShup h2h files {source} and {target}; see README.md")
        texts, summaries = source.read_text().splitlines(), target.read_text().splitlines()
        if len(texts) != len(summaries) or not texts:
            raise ValueError(f"Unaligned/empty GupShup split: {split}")
        splits[split] = []
        for i, (text, summary) in enumerate(zip(texts, summaries)):
            if not text.strip() or not summary.strip():
                raise ValueError(f"Empty GupShup row: {split}:{i}")
            splits[split].append(dict(uid=f"{split}:{i}", text=text, summary=summary))
    splits, removed = clean_splits(splits)
    return splits, {"source": "user-provided GupShup h2h", "duplicates_removed": removed}


LOADERS = {"sentiment": load_sentiment, "intent": load_intent, "qa": load_qa,
           "summarization": load_summary}
