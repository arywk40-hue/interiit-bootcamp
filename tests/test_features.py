import unittest

import numpy as np

from rinlu.sentiment.features import conservative_canonicalize, symbol_view
from rinlu.sentiment.infer import predict
from rinlu.sentiment.train import select_candidate
from rinlu.evaluation.robustness import drop_internal_vowel, remove_emoji, repeat_letter


class FeatureTests(unittest.TestCase):
    def test_legacy_inference_accepts_current_model_bundle(self):
        class Classifier:
            classes_ = np.asarray(["negative", "neutral", "positive"])

            def predict_proba(self, texts):
                self.last_texts = texts
                return np.asarray([[0.1, 0.2, 0.7]])

        classifier = Classifier()
        result = predict({"task": "sentiment", "model": classifier}, "accha hai")
        self.assertEqual(result["label"], "positive")
        self.assertEqual(classifier.last_texts, ["accha hai"])

    def test_canonicalizer_preserves_emoji_and_punctuation(self):
        actual = conservative_canonicalize("  BOHOOOOT   badhiya 😒!!! ")
        self.assertEqual(actual, "bohoot badhiya 😒!!!")

    def test_perturbations_are_deterministic(self):
        text = "mera delivery abhi tak nahi aya"
        self.assertEqual(drop_internal_vowel(text, "10"), drop_internal_vowel(text, "10"))
        self.assertEqual(repeat_letter(text, "10"), repeat_letter(text, "10"))
        self.assertNotEqual(drop_internal_vowel(text, "10"), text)
        self.assertNotEqual(repeat_letter(text, "10"), text)

    def test_remove_emoji_only_removes_emoji(self):
        self.assertEqual(remove_emoji("badhiya 😒!!!"), "badhiya !!!")

    def test_selection_prefers_latency_inside_accuracy_tolerance(self):
        results = {
            "fast": {"dev": {"macro_f1": 0.6105}, "latency": {"p95_ms": 0.7}},
            "slow": {"dev": {"macro_f1": 0.6106}, "latency": {"p95_ms": 1.3}},
            "weak": {"dev": {"macro_f1": 0.59}, "latency": {"p95_ms": 0.2}},
        }
        selected, eligible = select_candidate(results, macro_f1_tolerance=0.002)
        self.assertEqual(selected, "fast")
        self.assertEqual(set(eligible), {"fast", "slow"})

    def test_symbol_view_keeps_emoji_and_punctuation(self):
        self.assertEqual(symbol_view("Bohot badhiya 😒!!! ₹500"), "😒 ! ! ! ₹")


if __name__ == "__main__":
    unittest.main()
