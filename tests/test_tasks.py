"""Small invented examples here test code, never serve as benchmark evidence."""
import json
import tempfile
import unittest
from pathlib import Path

from rinlu.data.tasks import clean_splits, load_qa, load_summary
from rinlu.evaluation.metrics import rouge_scores, token_f1
from rinlu.evaluation.tasks import benchmark, parameter_count
from rinlu.qa import SpanReader
from rinlu.sentiment.features import build_model
from rinlu.summarization import ExtractiveSummarizer
from rinlu.neural import ByteMultiTaskModel, collate_bytes, encode_bytes


class TaskTests(unittest.TestCase):
    def test_random_byte_model_preserves_symbols_and_runs_all_heads(self):
        positive = encode_bytes("service acchi hai 😄!", "sentiment", max_length=64)
        self.assertIn(5, positive["symbol_ids"])
        batch = collate_bytes([positive])
        model = ByteMultiTaskModel()
        self.assertLess(model.parameter_report()["unique_parameters"], 500_000_000)
        self.assertEqual(tuple(model("sentiment", **batch)["logits"].shape), (1, 3))
        self.assertEqual(tuple(model("intent", **batch)["logits"].shape), (1, 57))
        self.assertEqual(tuple(model("summarization", **batch)["logits"].shape), (1,))
        qa = collate_bytes([encode_bytes("kab?", "qa", max_length=64,
                                        context="delivery kal hogi")])
        output = model("qa", **qa)
        self.assertEqual(tuple(output["start_logits"].shape), tuple(qa["input_ids"].shape))
        self.assertTrue((output["start_logits"][qa["segment_ids"] == 0] < -1000).all())

    def test_duplicate_text_never_crosses_splits(self):
        splits = {"train": [{"text": "  KAL meeting"}, {"text": "unique train"}],
                  "dev": [{"text": "kal meeting"}], "test": [{"text": "KAL MEETING"}]}
        cleaned, removed = clean_splits(splits)
        self.assertEqual(cleaned["train"], [{"text": "unique train"}])
        self.assertEqual(cleaned["dev"], [])
        self.assertEqual(removed, {"test": 0, "dev": 1, "train": 1})

    def test_qa_contexts_do_not_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "raw/qa"
            folder.mkdir(parents=True)
            rows = [dict(id=f"{i}:{j}", context=f"City {i} ka name Delhi hai.",
                         query=f"city {i} ka naam kya hai {j}?", answer="Delhi", language="Hindi")
                    for i in range(20) for j in range(2)]
            rows.append(dict(id="bad", context="general", query="kya hai?", answer="image", language="Hindi"))
            (folder / "code_mixed_qa_train.json").write_text(json.dumps({"questions": rows}))
            splits, audit = load_qa(root)
            groups = {s: {r["group"] for r in values} for s, values in splits.items()}
            self.assertFalse(groups["train"] & groups["dev"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["dev"] & groups["test"])
            self.assertEqual(audit["excluded"]["no_text_context"], 1)

    def test_summary_copies_source_and_preserves_order(self):
        text = "A: order late hai 😒!\nB: kal deliver hoga.\nA: refund chahiye!!!"
        model = ExtractiveSummarizer(max_sentences=2)
        result = model.predict(text)
        self.assertEqual(result["spans"], sorted(result["spans"]))
        self.assertEqual(result["summary"], " ".join(text[a:b] for a, b in result["spans"]))
        self.assertFalse(result["trained"])
        self.assertLessEqual(len(result["spans"]), 2)

    def test_summary_training_and_word_budget(self):
        text = "A: order late hai. B: kal deliver hoga. A: theek hai."
        model = ExtractiveSummarizer(max_words=4).fit([{"text": text, "summary": "kal deliver hoga"}])
        result = model.predict(text)
        self.assertTrue(result["trained"])
        self.assertLessEqual(len(result["summary"].split()), 4)
        self.assertEqual(parameter_count(model)["total_learned_scalars"], 7)
        self.assertEqual(model.predict("")["summary"], "")

    def test_summary_keeps_trailing_emoji_with_its_turn(self):
        text = "A: bahut accha! 😒\nB: refund milega."
        result = ExtractiveSummarizer(max_sentences=2).predict(text)
        self.assertIn("A: bahut accha! 😒", result["summary"])
        self.assertEqual(len(result["spans"]), 2)

    def test_summary_rejects_unaligned_reference_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "raw/summarization"
            folder.mkdir(parents=True)
            (folder / "train.source").write_text("a\nb\n")
            (folder / "train.target").write_text("a\n")
            with self.assertRaises(ValueError):
                load_summary(Path(directory))

    def test_qa_returns_only_real_context_offsets(self):
        rows = [{"text": "delivery kab hogi?", "context": "Delivery kal hogi.", "answer": "kal"},
                {"text": "kitne items hain?", "context": "Total 5 items hain.", "answer": "5"}]
        model = SpanReader().fit(rows)
        result = model.predict("delivery kab hogi?", "Delivery parso hogi.")
        self.assertFalse(result["no_answer"])
        self.assertEqual(result["answer"], "Delivery parso hogi."[result["start"]:result["end"]])
        self.assertTrue(model.predict("xyzxyz", "abcabc")["no_answer"])
        self.assertTrue(model.predict("kya?", "")["no_answer"])
        with self.assertRaises(ValueError):
            model.predict("kya?", "a" * 12001)

    def test_metrics_handle_repetitions_and_empty_answers(self):
        self.assertAlmostEqual(token_f1("good good", "good"), 2 / 3)
        self.assertEqual(token_f1("", ""), 1)
        self.assertEqual(token_f1("", "good"), 0)
        self.assertEqual(rouge_scores("a b c", "a b c"),
                         {"rouge1_f1": 1, "rouge2_f1": 1, "rougeL_f1": 1})

    def test_configuration_changes_actual_model(self):
        config = {"classifier": {"C": 0.5, "solver": "liblinear", "max_iter": 100, "class_weight": "balanced"},
                  "character_features": {"max_features": 25, "ngram_range": [1, 3], "min_df": 1},
                  "symbol_features": {"weight": 3, "max_features": 10}}
        model = build_model("char_tfidf", config=config)
        self.assertEqual(model.named_steps["features"].max_features, 25)
        self.assertEqual(model.named_steps["classifier"].estimator.C, .5)
        model.fit(["accha 😄", "bahut accha 😄", "bura 😒", "bahut bura 😒"],
                  ["positive", "positive", "negative", "negative"])
        self.assertGreater(parameter_count(model)["idf_values"], 0)

    def test_benchmark_rejects_zero_runs(self):
        with self.assertRaises(ValueError):
            benchmark("summarization", ExtractiveSummarizer(), [{"text": "hello"}], runs=0)


if __name__ == "__main__":
    unittest.main()
