from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from rinlu.data.conll import SENTIMENT_LABELS, sha256_file
from rinlu.evaluation.audit import contains_emoji
from rinlu.evaluation.robustness import remove_emoji, remove_punctuation
from rinlu.sentiment.features import symbol_view


PAIR_FIELDS = ("pair_id", "variant_type", "text_a", "text_b", "writer_id", "notes")
ANNOTATION_FIELDS = ("item_id", "text", "label")
DRAFT_REVIEW_FIELDS = (
    "pair_id", "category", "draft_text_a", "draft_text_b", "surface_variation",
    "decision", "reviewed_text_a", "reviewed_text_b", "human_reviewer_id", "review_notes",
)
ALLOWED_VARIANTS = {
    "spelling_same",
    "shorthand_same",
    "emoji_flip",
    "emoji_same",
    "punctuation_flip",
    "punctuation_same",
    "context_flip",
}


def _normalized_space(text: str) -> str:
    return " ".join(text.split())


def _canonical_pair_payload(rows: list[dict]) -> bytes:
    selected = [
        {field: row[field].strip() for field in PAIR_FIELDS}
        for row in sorted(rows, key=lambda row: row["pair_id"])
    ]
    return json.dumps(
        selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def content_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(_canonical_pair_payload(rows)).hexdigest()


def read_pairs(path: Path, min_pairs: int = 100) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PAIR_FIELDS:
            raise ValueError(f"{path}: expected columns {','.join(PAIR_FIELDS)}")
        rows = [{field: row[field].strip() for field in PAIR_FIELDS} for row in reader]
    if len(rows) < min_pairs:
        raise ValueError(f"{path}: requires at least {min_pairs} pairs, found {len(rows)}")
    ids = [row["pair_id"] for row in rows]
    if any(not pair_id for pair_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: pair IDs must be non-empty and unique")
    for row in rows:
        pair_id = row["pair_id"]
        variant = row["variant_type"]
        text_a, text_b = row["text_a"], row["text_b"]
        if variant not in ALLOWED_VARIANTS:
            raise ValueError(f"{path}: {pair_id}: invalid variant type {variant!r}")
        if not text_a or not text_b or text_a == text_b:
            raise ValueError(f"{path}: {pair_id}: pair texts must be non-empty and different")
        if not row["writer_id"]:
            raise ValueError(f"{path}: {pair_id}: writer_id is required")
        similarity = SequenceMatcher(None, text_a, text_b).ratio()
        if similarity < 0.5:
            raise ValueError(
                f"{path}: {pair_id}: similarity {similarity:.3f} is too low for a minimal pair"
            )
        if variant.startswith("emoji_"):
            if not (contains_emoji(text_a) or contains_emoji(text_b)):
                raise ValueError(f"{path}: {pair_id}: emoji pair contains no detected emoji")
            if _normalized_space(remove_emoji(text_a)) != _normalized_space(remove_emoji(text_b)):
                raise ValueError(
                    f"{path}: {pair_id}: emoji pair changes non-emoji content"
                )
        elif variant.startswith("punctuation_"):
            if _normalized_space(remove_punctuation(text_a)) != _normalized_space(
                remove_punctuation(text_b)
            ):
                raise ValueError(
                    f"{path}: {pair_id}: punctuation pair changes non-punctuation content"
                )
        elif variant == "context_flip":
            pass
        elif symbol_view(text_a) != symbol_view(text_b):
            raise ValueError(
                f"{path}: {pair_id}: spelling/shorthand pair also changes symbols"
            )
    return rows


def prepare_draft_review(drafts_path: Path, output: Path) -> int:
    """Create a label-free human editing sheet from explicitly model-drafted pairs."""
    rows, ids, pairs = [], set(), set()
    allowed_categories = {"spelling_preserving", "shorthand_preserving",
                          "emoji_flip", "punctuation_context_flip"}
    for line_number, line in enumerate(drafts_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"pair_id", "category", "text_a", "text_b", "surface_variation",
                    "provenance", "annotator_1_label", "annotator_2_label",
                    "adjudicated_label"}
        if required - row.keys():
            raise ValueError(f"draft row {line_number} is missing {sorted(required - row.keys())}")
        if row["provenance"] != "model_drafted_pending_human_review":
            raise ValueError(f"draft row {line_number} has unsupported provenance")
        if any(row[field] is not None for field in
               ("annotator_1_label", "annotator_2_label", "adjudicated_label")):
            raise ValueError(f"draft row {line_number} already contains purported human labels")
        pair_id = str(row["pair_id"]).strip()
        pair = (row["text_a"].strip(), row["text_b"].strip())
        if not pair_id or pair_id in ids or pair in pairs or pair[0] == pair[1]:
            raise ValueError(f"draft row {line_number} has duplicate/invalid pair content")
        if row["category"] not in allowed_categories:
            raise ValueError(f"draft row {line_number} has unsupported category")
        ids.add(pair_id); pairs.add(pair)
        rows.append({"pair_id": pair_id, "category": row["category"],
                     "draft_text_a": pair[0], "draft_text_b": pair[1],
                     "surface_variation": row["surface_variation"], "decision": "",
                     "reviewed_text_a": "", "reviewed_text_b": "",
                     "human_reviewer_id": "", "review_notes": ""})
    _write_csv(output, DRAFT_REVIEW_FIELDS, rows)
    return len(rows)


def finalize_draft_review(review_path: Path, output: Path, min_pairs: int = 100) -> int:
    """Convert completed human editing decisions to the existing freeze input schema."""
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DRAFT_REVIEW_FIELDS:
            raise ValueError(f"{review_path}: unexpected review columns")
        reviewed = [{field: row[field].strip() for field in DRAFT_REVIEW_FIELDS}
                    for row in reader]
    output_rows = []
    base_variants = {"spelling_preserving": "spelling_same",
                     "shorthand_preserving": "shorthand_same",
                     "emoji_flip": "emoji_flip"}
    for row in reviewed:
        if row["decision"] not in {"accept", "rewrite", "reject"}:
            raise ValueError(f"{row['pair_id']}: decision must be accept, rewrite or reject")
        if not row["human_reviewer_id"]:
            raise ValueError(f"{row['pair_id']}: human_reviewer_id is required")
        if row["decision"] == "reject":
            continue
        if row["decision"] == "rewrite":
            text_a, text_b = row["reviewed_text_a"], row["reviewed_text_b"]
            if not text_a or not text_b:
                raise ValueError(f"{row['pair_id']}: rewritten texts are required")
        else:
            text_a, text_b = row["draft_text_a"], row["draft_text_b"]
        if row["category"] == "punctuation_context_flip":
            variant = ("punctuation_flip" if _normalized_space(remove_punctuation(text_a))
                       == _normalized_space(remove_punctuation(text_b)) else "context_flip")
        else:
            variant = base_variants.get(row["category"])
        if not variant:
            raise ValueError(f"{row['pair_id']}: unsupported category")
        output_rows.append({"pair_id": row["pair_id"], "variant_type": variant,
                            "text_a": text_a, "text_b": text_b,
                            "writer_id": row["human_reviewer_id"],
                            "notes": ("human-reviewed model draft; " + row["decision"]
                                      + ("; " + row["review_notes"] if row["review_notes"] else ""))})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        _write_csv(temporary_path, PAIR_FIELDS, output_rows)
        read_pairs(temporary_path, min_pairs=min_pairs)
        temporary_path.replace(output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(output_rows)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _item_id(dataset_hash: str, pair_id: str, side: str) -> str:
    payload = f"{dataset_hash}:{pair_id}:{side}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def freeze_pairs(
    pairs_path: Path,
    output_dir: Path,
    model_artifacts: dict[str, Path],
    config_paths: list[Path],
    min_pairs: int = 100,
) -> dict:
    rows = read_pairs(pairs_path, min_pairs=min_pairs)
    dataset_hash = content_sha256(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_pairs = output_dir / "pairs_unlabeled_frozen.csv"
    shutil.copyfile(pairs_path, frozen_pairs)

    mapping = []
    items = []
    for row in rows:
        for side in ("a", "b"):
            item = {
                "pair_id": row["pair_id"],
                "side": side,
                "item_id": _item_id(dataset_hash, row["pair_id"], side),
                "variant_type": row["variant_type"],
                "text": row[f"text_{side}"],
            }
            mapping.append(item)
            items.append({"item_id": item["item_id"], "text": item["text"], "label": ""})
    _write_csv(
        output_dir / "private_pair_mapping.csv",
        ("pair_id", "side", "item_id", "variant_type", "text"),
        mapping,
    )
    for name, seed in (("annotator_a", 104729), ("annotator_b", 130363)):
        shuffled = [dict(item) for item in items]
        random.Random(seed).shuffle(shuffled)
        _write_csv(output_dir / f"{name}.csv", ANNOTATION_FIELDS, shuffled)

    manifest = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_count": len(rows),
        "item_count": 2 * len(rows),
        "pairs_file_sha256": sha256_file(frozen_pairs),
        "canonical_pair_content_sha256": dataset_hash,
        "pair_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["pair_id"] for row in rows)).encode("utf-8")
        ).hexdigest(),
        "models": {
            name: {
                "file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(model_artifacts.items())
        },
        "configs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in config_paths
        },
        "annotation_protocol": {
            "blindness": "annotators receive separately shuffled individual messages without pair IDs or variant types",
            "labels": list(SENTIMENT_LABELS),
            "agreement": ["three_class_cohen_kappa", "same_vs_flip_cohen_kappa"],
            "tie_break": (
                "third annotator labels disagreements independently; majority wins; "
                "three-way label splits are excluded as ambiguous"
            ),
        },
        "preregistered_metrics": [
            "individual_message_macro_f1",
            "pair_exact_accuracy",
            "flip_sensitivity_on_gold_flip_pairs",
            "directional_flip_accuracy",
            "stability_on_gold_same_pairs",
            "bootstrap_95_percent_confidence_intervals_by_pair",
        ],
    }
    with (output_dir / "freeze_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def verify_freeze(frozen_dir: Path) -> dict:
    manifest_path = frozen_dir / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs_path = frozen_dir / "pairs_unlabeled_frozen.csv"
    rows = read_pairs(pairs_path, min_pairs=manifest["pair_count"])
    checks = {
        "pairs_file_sha256": sha256_file(pairs_path) == manifest["pairs_file_sha256"],
        "canonical_pair_content_sha256": (
            content_sha256(rows) == manifest["canonical_pair_content_sha256"]
        ),
        "pair_count": len(rows) == manifest["pair_count"],
    }
    if not all(checks.values()):
        raise ValueError(f"frozen contrast set verification failed: {checks}")
    return checks


def _read_annotations(path: Path) -> dict[str, str]:
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ANNOTATION_FIELDS:
            raise ValueError(f"{path}: expected columns {','.join(ANNOTATION_FIELDS)}")
        for row in reader:
            item_id, label = row["item_id"].strip(), row["label"].strip()
            if not item_id or item_id in labels:
                raise ValueError(f"{path}: item IDs must be non-empty and unique")
            if label not in SENTIMENT_LABELS:
                raise ValueError(f"{path}: {item_id}: invalid or missing label")
            labels[item_id] = label
    return labels


def prepare_adjudication(
    frozen_dir: Path, annotator_a: Path, annotator_b: Path, output: Path
) -> int:
    verify_freeze(frozen_dir)
    labels_a = _read_annotations(annotator_a)
    labels_b = _read_annotations(annotator_b)
    if set(labels_a) != set(labels_b):
        raise ValueError("annotator forms contain different item IDs")
    with (frozen_dir / "private_pair_mapping.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        mapping = {row["item_id"]: row for row in csv.DictReader(handle)}
    disagreements = [
        {"item_id": item_id, "text": mapping[item_id]["text"], "label": ""}
        for item_id in labels_a
        if labels_a[item_id] != labels_b[item_id]
    ]
    random.Random(161803).shuffle(disagreements)
    _write_csv(output, ANNOTATION_FIELDS, disagreements)
    return len(disagreements)


def _safe_kappa(left: list[str], right: list[str], labels: list[str]):
    value = float(cohen_kappa_score(left, right, labels=labels))
    return None if np.isnan(value) else value


def _rate(values: list[bool], seed: int = 42, bootstrap_runs: int = 2000) -> dict:
    if not values:
        return {"count": 0, "rate": None, "bootstrap_95_ci": None}
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(bootstrap_runs, len(data)), replace=True).mean(axis=1)
    return {
        "count": len(values),
        "rate": float(data.mean()),
        "bootstrap_95_ci": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }


def score_contrast_set(
    frozen_dir: Path,
    annotator_a: Path,
    annotator_b: Path,
    adjudicator: Path,
    model_artifacts: dict[str, Path],
    output_dir: Path,
) -> dict:
    verify_freeze(frozen_dir)
    manifest = json.loads((frozen_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    for name, path in model_artifacts.items():
        registered = manifest["models"].get(name)
        if not registered or sha256_file(path) != registered["sha256"]:
            raise ValueError(f"model {name!r} does not match the frozen preregistration")
    labels_a = _read_annotations(annotator_a)
    labels_b = _read_annotations(annotator_b)
    adjudicated = _read_annotations(adjudicator) if adjudicator.stat().st_size else {}

    with (frozen_dir / "private_pair_mapping.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        mapping_rows = list(csv.DictReader(handle))
    expected_ids = {row["item_id"] for row in mapping_rows}
    if set(labels_a) != expected_ids or set(labels_b) != expected_ids:
        raise ValueError("completed annotation forms do not match the frozen items")

    kappa_sentiment = _safe_kappa(
        [labels_a[item] for item in sorted(expected_ids)],
        [labels_b[item] for item in sorted(expected_ids)],
        list(SENTIMENT_LABELS),
    )
    by_pair = {}
    for row in mapping_rows:
        by_pair.setdefault(row["pair_id"], {})[row["side"]] = row
    relation_a, relation_b = [], []
    for pair in by_pair.values():
        relation_a.append(
            "same" if labels_a[pair["a"]["item_id"]] == labels_a[pair["b"]["item_id"]] else "flip"
        )
        relation_b.append(
            "same" if labels_b[pair["a"]["item_id"]] == labels_b[pair["b"]["item_id"]] else "flip"
        )
    kappa_relation = _safe_kappa(relation_a, relation_b, ["same", "flip"])

    final_labels = {}
    unresolved = []
    for item_id in expected_ids:
        left, right = labels_a[item_id], labels_b[item_id]
        if left == right:
            final_labels[item_id] = left
            continue
        third = adjudicated.get(item_id)
        if third in {left, right}:
            final_labels[item_id] = third
        else:
            unresolved.append(item_id)

    resolved_pairs = [
        (pair_id, pair)
        for pair_id, pair in by_pair.items()
        if pair["a"]["item_id"] in final_labels and pair["b"]["item_id"] in final_labels
    ]
    model_results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model_path in model_artifacts.items():
        model = joblib.load(model_path)
        if isinstance(model, dict):
            if model.get("task") != "sentiment":
                raise ValueError("Sentiment contrasts require a sentiment model bundle")
            model = model["model"]
        flat_rows = []
        for pair_id, pair in resolved_pairs:
            for side in ("a", "b"):
                row = pair[side]
                flat_rows.append(
                    {
                        "pair_id": pair_id,
                        "side": side,
                        "item_id": row["item_id"],
                        "variant_type": row["variant_type"],
                        "text": row["text"],
                        "gold": final_labels[row["item_id"]],
                    }
                )
        predictions = model.predict([row["text"] for row in flat_rows]).tolist()
        for row, prediction in zip(flat_rows, predictions):
            row["prediction"] = prediction
        predicted_by_item = {row["item_id"]: row["prediction"] for row in flat_rows}
        gold = [row["gold"] for row in flat_rows]
        pair_exact, flip_sensitivity, directional_flip, same_stability, same_exact = [], [], [], [], []
        intended_relation_match = []
        for _, pair in resolved_pairs:
            item_a, item_b = pair["a"]["item_id"], pair["b"]["item_id"]
            gold_a, gold_b = final_labels[item_a], final_labels[item_b]
            pred_a, pred_b = predicted_by_item[item_a], predicted_by_item[item_b]
            gold_flip = gold_a != gold_b
            predicted_flip = pred_a != pred_b
            exact = pred_a == gold_a and pred_b == gold_b
            pair_exact.append(exact)
            intended = "flip" if pair["a"]["variant_type"].endswith("_flip") else "same"
            intended_relation_match.append(("flip" if gold_flip else "same") == intended)
            if gold_flip:
                flip_sensitivity.append(predicted_flip)
                directional_flip.append(exact)
            else:
                same_stability.append(not predicted_flip)
                same_exact.append(exact)
        model_results[model_name] = {
            "individual_message_macro_f1": float(
                f1_score(gold, predictions, labels=list(SENTIMENT_LABELS), average="macro")
            ),
            "pair_exact_accuracy": _rate(pair_exact),
            "flip_sensitivity_on_gold_flip_pairs": _rate(flip_sensitivity),
            "directional_flip_accuracy": _rate(directional_flip),
            "stability_on_gold_same_pairs": _rate(same_stability),
            "exact_accuracy_on_gold_same_pairs": _rate(same_exact),
            "intended_relation_matches_adjudicated_gold": _rate(intended_relation_match),
        }
        _write_csv(
            output_dir / f"{model_name}_contrast_predictions.csv",
            ("pair_id", "side", "item_id", "variant_type", "text", "gold", "prediction"),
            flat_rows,
        )

    result = {
        "frozen_pair_content_sha256": manifest["canonical_pair_content_sha256"],
        "pairs_total": len(by_pair),
        "pairs_resolved": len(resolved_pairs),
        "items_unresolved_as_ambiguous": unresolved,
        "agreement": {
            "three_class_sentiment_cohen_kappa": kappa_sentiment,
            "same_vs_flip_relation_cohen_kappa": kappa_relation,
        },
        "models": model_results,
    }
    with (output_dir / "contrast_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def _named_paths(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        parsed[name] = Path(raw_path)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and score human minimal-pair sentiment data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--pairs", type=Path, required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    freeze_parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    freeze_parser.add_argument("--config", action="append", type=Path, default=[])
    freeze_parser.add_argument("--min-pairs", type=int, default=100)

    draft_parser = subparsers.add_parser("prepare-draft-review")
    draft_parser.add_argument("--drafts", type=Path, required=True)
    draft_parser.add_argument("--output", type=Path, required=True)

    finalize_parser = subparsers.add_parser("finalize-draft-review")
    finalize_parser.add_argument("--review", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.add_argument("--min-pairs", type=int, default=100)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--frozen-dir", type=Path, required=True)

    adjudicate_parser = subparsers.add_parser("prepare-adjudication")
    adjudicate_parser.add_argument("--frozen-dir", type=Path, required=True)
    adjudicate_parser.add_argument("--annotator-a", type=Path, required=True)
    adjudicate_parser.add_argument("--annotator-b", type=Path, required=True)
    adjudicate_parser.add_argument("--output", type=Path, required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--frozen-dir", type=Path, required=True)
    score_parser.add_argument("--annotator-a", type=Path, required=True)
    score_parser.add_argument("--annotator-b", type=Path, required=True)
    score_parser.add_argument("--adjudicator", type=Path, required=True)
    score_parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    score_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-draft-review":
        print(f"drafts={prepare_draft_review(args.drafts, args.output)}")
    elif args.command == "finalize-draft-review":
        print(f"accepted={finalize_draft_review(args.review, args.output, args.min_pairs)}")
    elif args.command == "freeze":
        manifest = freeze_pairs(
            args.pairs,
            args.output_dir,
            _named_paths(args.model),
            args.config,
            args.min_pairs,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    elif args.command == "verify":
        print(json.dumps(verify_freeze(args.frozen_dir), indent=2))
    elif args.command == "prepare-adjudication":
        count = prepare_adjudication(
            args.frozen_dir, args.annotator_a, args.annotator_b, args.output
        )
        print(f"disagreements={count}")
    elif args.command == "score":
        result = score_contrast_set(
            args.frozen_dir,
            args.annotator_a,
            args.annotator_b,
            args.adjudicator,
            _named_paths(args.model),
            args.output_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
