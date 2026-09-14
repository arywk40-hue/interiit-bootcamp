from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rinlu.evaluation.contrast_set import (
    finalize_draft_review,
    freeze_pairs,
    prepare_adjudication,
    prepare_draft_review,
    score_contrast_set,
    verify_freeze,
)


class ContrastSetTests(unittest.TestCase):
    def test_model_drafts_require_human_review_before_freezing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); drafts = root / "drafts.jsonl"
            values = [
                ("e1", "emoji_flip", "bahut accha 😄", "bahut accha 😒"),
                ("p1", "punctuation_context_flip", "great!", "great..."),
                ("c1", "punctuation_context_flip", "plan accha tha.",
                 "plan accha tha, lekin teen ghante late."),
            ]
            rows = [{"pair_id": pair_id, "category": category, "text_a": left,
                     "text_b": right, "surface_variation": "draft",
                     "provenance": "model_drafted_pending_human_review",
                     "sentiment_a": "positive", "sentiment_b": "negative",
                     "annotator_1_label": None, "annotator_2_label": None,
                     "adjudicated_label": None}
                    for pair_id, category, left, right in values]
            drafts.write_text("\n".join(json.dumps(row) for row in rows))
            review = root / "review.csv"
            self.assertEqual(prepare_draft_review(drafts, review), 3)
            with review.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle); review_rows = list(reader)
                self.assertNotIn("sentiment_a", reader.fieldnames)
            for row in review_rows:
                row["decision"] = "accept"; row["human_reviewer_id"] = "human-1"
            with review.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=review_rows[0].keys())
                writer.writeheader(); writer.writerows(review_rows)
            pairs = root / "pairs.csv"
            pairs.write_text("existing human data\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires at least 4 pairs"):
                finalize_draft_review(review, pairs, min_pairs=4)
            self.assertEqual(pairs.read_text(encoding="utf-8"), "existing human data\n")
            self.assertEqual(finalize_draft_review(review, pairs, min_pairs=3), 3)
            with pairs.open(newline="", encoding="utf-8") as handle:
                variants = {row["variant_type"] for row in csv.DictReader(handle)}
            self.assertEqual(variants, {"emoji_flip", "punctuation_flip", "context_flip"})

    def _write_pairs(self, path: Path):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "pair_id",
                    "variant_type",
                    "text_a",
                    "text_b",
                    "writer_id",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "pair_id": "p001",
                    "variant_type": "emoji_flip",
                    "text_a": "bahut accha",
                    "text_b": "bahut accha 😒",
                    "writer_id": "w1",
                    "notes": "",
                }
            )
            writer.writerow(
                {
                    "pair_id": "p002",
                    "variant_type": "spelling_same",
                    "text_a": "bahut accha!",
                    "text_b": "bohot acha!",
                    "writer_id": "w2",
                    "notes": "",
                }
            )

    def test_freeze_hashes_content_and_hides_pair_metadata_from_annotators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.csv"
            self._write_pairs(pairs)
            frozen = root / "frozen"
            manifest = freeze_pairs(pairs, frozen, {}, [], min_pairs=2)
            self.assertEqual(manifest["pair_count"], 2)
            self.assertTrue(all(verify_freeze(frozen).values()))
            with (frozen / "annotator_a.csv").open(encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header, ["item_id", "text", "label"])

    def test_tampering_breaks_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.csv"
            self._write_pairs(pairs)
            frozen = root / "frozen"
            freeze_pairs(pairs, frozen, {}, [], min_pairs=2)
            frozen_pairs = frozen / "pairs_unlabeled_frozen.csv"
            frozen_pairs.write_text(
                frozen_pairs.read_text(encoding="utf-8").replace("bahut accha", "badal diya", 1),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                verify_freeze(frozen)

    def test_blind_annotations_are_scored_as_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.csv"
            self._write_pairs(pairs)
            model = Pipeline(
                [
                    ("vectorizer", CountVectorizer()),
                    ("classifier", LogisticRegression(max_iter=100)),
                ]
            )
            model.fit(
                ["bad service", "very bad", "okay", "fine", "good service", "very good"],
                ["negative", "negative", "neutral", "neutral", "positive", "positive"],
            )
            model_path = root / "model.joblib"
            joblib.dump({"task": "sentiment", "model": model}, model_path)
            frozen = root / "frozen"
            freeze_pairs(pairs, frozen, {"model": model_path}, [], min_pairs=2)

            labels_by_text = {
                "bahut accha": "positive",
                "bahut accha 😒": "negative",
                "bahut accha!": "positive",
                "bohot acha!": "positive",
            }
            for filename in ("annotator_a.csv", "annotator_b.csv"):
                path = frozen / filename
                with path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    row["label"] = labels_by_text[row["text"]]
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["item_id", "text", "label"])
                    writer.writeheader()
                    writer.writerows(rows)

            adjudicator = frozen / "adjudicator.csv"
            self.assertEqual(
                prepare_adjudication(
                    frozen,
                    frozen / "annotator_a.csv",
                    frozen / "annotator_b.csv",
                    adjudicator,
                ),
                0,
            )
            result = score_contrast_set(
                frozen,
                frozen / "annotator_a.csv",
                frozen / "annotator_b.csv",
                adjudicator,
                {"model": model_path},
                root / "results",
            )
            self.assertEqual(result["agreement"]["three_class_sentiment_cohen_kappa"], 1.0)
            self.assertEqual(result["pairs_resolved"], 2)
            self.assertIn("flip_sensitivity_on_gold_flip_pairs", result["models"]["model"])


if __name__ == "__main__":
    unittest.main()
