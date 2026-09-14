"""Small invented examples here test code, never serve as benchmark evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from rinlu.data.tasks import clean_splits, load_qa, load_summary
from rinlu.evaluation.metrics import rouge_scores, token_f1
from rinlu.evaluation.tasks import benchmark, parameter_count
from rinlu.qa import SpanReader
from rinlu.sentiment.features import build_model
from rinlu.summarization import ExtractiveSummarizer
from rinlu.neural import (ByteMultiTaskConfig, ByteMultiTaskModel, collate_bytes, encode_bytes,
                          mask_unicode_characters, unicode_byte_groups)
from scripts.train_from_scratch import build_pretraining_corpus


class TaskTests(unittest.TestCase):
    def test_byte_model_parameter_count_is_frozen(self):
        self.assertEqual(ByteMultiTaskModel().parameter_report()["unique_parameters"], 4_306_511)

    def test_masking_selects_complete_unicode_characters(self):
        encoded = encode_bytes("a😄ह", "masked_byte", max_length=32)
        original_groups = unicode_byte_groups(encoded["input_ids"])
        corrupted, labels = mask_unicode_characters(
            encoded, torch.Generator().manual_seed(7), probability=1
        )
        for group in original_groups:
            selected = [labels[position] != -100 for position in group]
            self.assertTrue(all(selected) or not any(selected))
        self.assertEqual([labels[p] for g in original_groups for p in g],
                         [encoded["input_ids"][p] for g in original_groups for p in g])
        self.assertEqual(len(corrupted), len(encoded["input_ids"]))

    def test_pretraining_builder_excludes_heldout_and_is_deterministic(self):
        rows = [{"text": "train one", "source": "a"},
                {"text": "DEV SECRET", "source": "a"},
                {"text": "train two", "source": "b"}]
        heldout = {"dev secret"}
        first = build_pretraining_corpus(rows, [], heldout, 2, 17)
        second = build_pretraining_corpus(rows, [], heldout, 2, 17)
        self.assertEqual(first, second)
        self.assertNotIn("DEV SECRET", [row["text"] for row in first])

    def test_tiny_model_has_finite_gradients_and_learns(self):
        torch.manual_seed(11)
        config = ByteMultiTaskConfig(d_model=32, layers=1, heads=4,
                                     feed_forward=64, dropout=0, max_positions=32,
                                     intent_labels=2, downsample_stages=1)
        model = ByteMultiTaskModel(config)
        batch = collate_bytes([encode_bytes(text, "sentiment", 32)
                               for text in ("accha", "bura", "accha", "bura")])
        labels = torch.tensor([2, 0, 2, 0])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        initial = float(F.cross_entropy(model("sentiment", **batch)["logits"], labels).detach())
        for _ in range(8):
            loss = F.cross_entropy(model("sentiment", **batch)["logits"], labels)
            optimizer.zero_grad(); loss.backward()
            self.assertTrue(all(torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()
                                if parameter.grad is not None))
            optimizer.step()
        final = float(F.cross_entropy(model("sentiment", **batch)["logits"], labels).detach())
        self.assertLess(final, initial)

    def test_checkpoint_round_trip_preserves_logits(self):
        torch.manual_seed(5)
        model = ByteMultiTaskModel().eval()
        batch = collate_bytes([encode_bytes("service acchi hai 😄", "sentiment", 48)])
        expected = model("sentiment", **batch)["logits"].detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            restored = ByteMultiTaskModel().eval()
            restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
            actual = restored("sentiment", **batch)["logits"].detach()
        self.assertTrue(torch.equal(expected, actual))

    def test_random_byte_model_preserves_symbols_and_runs_all_heads(self):
        positive = encode_bytes("service acchi hai 😄!", "sentiment", max_length=64)
        self.assertIn(5, positive["symbol_ids"])
        batch = collate_bytes([positive])
        model = ByteMultiTaskModel()
        self.assertLess(model.parameter_report()["unique_parameters"], 500_000_000)
        self.assertEqual(tuple(model("sentiment", **batch)["logits"].shape), (1, 3))
        self.assertEqual(tuple(model("intent", **batch)["logits"].shape), (1, 64))
        self.assertEqual(tuple(model("summarization", **batch)["logits"].shape), (1,))
        qa = collate_bytes([encode_bytes("kab?", "qa", max_length=64,
                                        context="delivery kal hogi")])
        output = model("qa", **qa)
        self.assertEqual(tuple(output["start_logits"].shape), tuple(qa["input_ids"].shape))
        self.assertTrue((output["start_logits"][qa["segment_ids"] == 0] < -1000).all())
        self.assertEqual(set(model.forward_all(**qa)), {
            "sentiment_logits", "intent_logits", "summary_logits", "start_logits",
            "end_logits", "answerable_logits"})

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
            (folder / "human_expansion.jsonl").write_text(json.dumps({
                "uid": "human-1", "text": "parcel kab aaya?", "context": "Parcel kal aaya.",
                "answer": "kal", "group": "delivery-note-1", "split": "test",
                "reviewer_ids": ["r1", "r2"], "consent_to_use": True}) + "\n")
            splits, audit = load_qa(root)
            groups = {s: {r["group"] for r in values} for s, values in splits.items()}
            self.assertFalse(groups["train"] & groups["dev"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["dev"] & groups["test"])
            self.assertEqual(audit["excluded"]["no_text_context"], 1)
            self.assertEqual(audit["human_expansion_rows"], 1)
            self.assertIn("human-1", [row["uid"] for row in splits["test"]])

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
