from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rinlu.data.conll import (
    attach_labels,
    duplicate_key,
    parse_conll,
    read_test_labels,
    repair_mojibake,
)
from rinlu.data.prepare import prepare_dataset


LABELLED = """meta\t1\tpositive
bahut\tHin
good\tEng
😄\tO
meta\t2\tnegative
nahi\tHin
bad\tEng
"""

UNLABELLED = """meta\t3
bahut\tHin
good\tEng
😄\tO
"""


class DataTests(unittest.TestCase):
    def test_parse_and_attach(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.conll"
            path.write_text(LABELLED, encoding="utf-8")
            rows = parse_conll(path, "train", require_labels=True)
            self.assertEqual([row.uid for row in rows], ["1", "2"])
            self.assertEqual(rows[0].text, "bahut good 😄")
            self.assertEqual(rows[0].language_tags, ("Hin", "Eng", "O"))

            test_path = root / "test.conll"
            test_path.write_text(UNLABELLED, encoding="utf-8")
            labels_path = root / "labels.csv"
            labels_path.write_text("Uid,Sentiment\n3,positive\n", encoding="utf-8")
            test_rows = parse_conll(test_path, "test", require_labels=False)
            attached = attach_labels(test_rows, read_test_labels(labels_path))
            self.assertEqual(attached[0].label, "positive")

    def test_duplicate_key_is_case_and_space_insensitive(self):
        self.assertEqual(duplicate_key("  KYA   ho  "), duplicate_key("kya ho"))

    def test_repairs_utf8_decoded_as_windows_1252(self):
        self.assertEqual(repair_mojibake("â€¦"), "…")
        self.assertEqual(repair_mojibake("ðŸ˜…"), "😅")
        self.assertEqual(repair_mojibake("à¤¬à¤¹à¥à¤¤"), "बहुत")
        self.assertEqual(repair_mojibake("à¨¦à¨¾à¨¤à¨¾"), "ਦਾਤਾ")
        self.assertEqual(repair_mojibake("aap busy ho?"), "aap busy ho?")

    def test_prepare_removes_train_test_leakage_and_train_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.conll"
            train.write_text(
                LABELLED
                + "meta\t4\tpositive\nbahut\tHin\ngood\tEng\n😄\tO\n",
                encoding="utf-8",
            )
            dev = root / "dev.conll"
            dev.write_text("meta\t5\tneutral\nok\tEng\n", encoding="utf-8")
            test = root / "test.conll"
            test.write_text(UNLABELLED, encoding="utf-8")
            labels = root / "labels.csv"
            with labels.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Uid", "Sentiment"])
                writer.writerow(["3", "positive"])
            manifest = prepare_dataset(
                train, dev, test, labels, root / "out", enforce_expected_sizes=False
            )
            self.assertEqual(manifest["counts_after_cleaning"]["train"], 1)
            self.assertEqual(len(manifest["removed_from_train"]), 2)


if __name__ == "__main__":
    unittest.main()
