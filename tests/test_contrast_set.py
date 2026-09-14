from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rinlu.evaluation.contrast_set import (
    freeze_pairs,
    prepare_adjudication,
    score_contrast_set,
    verify_freeze,
)


class ContrastSetTests(unittest.TestCase):
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
