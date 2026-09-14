"""Extractive Hinglish summaries: rank turns, avoid repetition, keep source order.

No decoder or translation is involved. Selected text is copied exactly, including
speaker names, emoji and punctuation. Extraction can still omit necessary context.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

from rinlu.evaluation.metrics import token_f1


def sentence_spans(text):
    """Return source offsets; punctuation is retained in each sentence."""
    spans, start = [], 0
    # In dialogues, keep entire turns together so a trailing emoji is not detached.
    boundary = r"\n+|\s*<(?:br|sep)>\s*" if re.search(r"\n|<(?:br|sep)>", text) else r"(?<=[.!?])\s+(?=\w)"
    for match in re.finditer(boundary, text):
        end = match.start()
        if text[start:end].strip():
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((left, end))
        start = match.end()
    if text[start:].strip():
        left = start + len(text[start:]) - len(text[start:].lstrip())
        spans.append((left, len(text)))
    return spans


class ExtractiveSummarizer:
    def __init__(self, max_sentences=3, max_words=80, redundancy=0.4):
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.redundancy = redundancy
        self.vectorizer = HashingVectorizer(analyzer="char", ngram_range=(3, 5),
                                            n_features=16384, alternate_sign=False, norm="l2")
        self.ranker = None

    def _features(self, text):
        if len(text) > 12000:
            raise ValueError("Summarizer supports at most 12,000 characters per request")
        spans = sentence_spans(text)
        if len(spans) > 128:
            raise ValueError("Summarizer supports at most 128 turns/sentences per request")
        chunks = [text[a:b] for a, b in spans]
        if not chunks:
            return spans, np.empty((0, 6)), np.empty((0, 0))
        vectors = self.vectorizer.transform(chunks)
        similarity = (vectors @ vectors.T).toarray()
        features = []
        for i, chunk in enumerate(chunks):
            features.append([similarity[i].mean(), min(len(chunk.split()), 80) / 80,
                             i / max(len(chunks) - 1, 1), float(i == 0),
                             float(i == len(chunks) - 1), float("?" in chunk)])
        return spans, np.asarray(features), similarity

    def fit(self, rows):
        features, targets, weights = [], [], []
        for row in rows:
            spans, matrix, _ = self._features(row["text"])
            features.extend(matrix)
            targets.extend(token_f1(row["text"][a:b], row["summary"]) for a, b in spans)
            weights.extend([1 / len(spans)] * len(spans))
        if not features:
            raise ValueError("No summary training sentences")
        # ROUGE-1 overlap supplies an extractive training target from human summaries.
        self.ranker = Ridge(alpha=1.0).fit(features, targets, sample_weight=weights)
        return self

    def predict(self, text):
        if self.max_sentences < 1 or self.max_words < 1:
            raise ValueError("Summary limits must be positive")
        if len(text) > 12000:
            raise ValueError("Summarizer supports at most 12,000 characters per request")
        spans, features, similarity = self._features(text)
        if not spans:
            return {"summary": "", "spans": [], "mode": "extractive"}
        scores = self.ranker.predict(features) if self.ranker is not None else features[:, 0]
        selected, words = [], 0
        remaining = list(range(len(spans)))
        while remaining and len(selected) < self.max_sentences:
            best = max(remaining, key=lambda i: scores[i] - self.redundancy * (
                max(similarity[i, j] for j in selected) if selected else 0))
            remaining.remove(best)
            a, b = spans[best]
            length = len(text[a:b].split())
            if words + length <= self.max_words:
                selected.append(best)
                words += length
        # A single long sentence cannot be shortened without cutting its meaning.
        # Return a contiguous prefix only as an explicitly marked fallback.
        truncated = False
        chosen_spans = [spans[i] for i in sorted(selected)]
        if not chosen_spans:
            a, b = spans[int(np.argmax(scores))]
            tokens = list(re.finditer(r"\S+", text[a:b]))
            b = a + tokens[min(self.max_words, len(tokens)) - 1].end()
            chosen_spans, truncated = [(a, b)], True
        return {"summary": " ".join(text[a:b] for a, b in chosen_spans),
                "spans": chosen_spans, "mode": "extractive",
                "trained": self.ranker is not None, "truncated_sentence": truncated}
