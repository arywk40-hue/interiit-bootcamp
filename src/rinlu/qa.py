"""A small learned span ranker, with transparent lexical features.

Retrieve three sentences with character overlap, enumerate short answer spans,
then score them with a linear model. No answers are stored in the model.
"""
import re

import numpy as np
from sklearn.linear_model import Ridge

from rinlu.evaluation.metrics import answer_tokens, token_f1
from rinlu.summarization import sentence_spans


STOP = set("a an the is are of to in and ka ki ke hai hain h kya what which who when where how".split())


def content_words(text):
    return set(answer_tokens(text)) - STOP


def chargrams(text):
    text = " ".join(sorted(content_words(text)))
    return {text[i:i + 3] for i in range(len(text) - 2)}


class SpanReader:
    def __init__(self, max_answer_words=8, top_sentences=3, threshold=0.0):
        self.max_answer_words = max_answer_words
        self.top_sentences = top_sentences
        self.threshold = threshold
        self.ranker = None

    def candidates(self, question, context):
        if len(context) > 12000 or len(question) > 1000:
            raise ValueError("QA limits: 12,000 context characters and 1,000 question characters")
        if self.max_answer_words < 1 or self.top_sentences < 1:
            raise ValueError("QA candidate limits must be positive")
        qwords, qgrams = content_words(question), chargrams(question)
        sentences = sentence_spans(context)
        ranked = []
        for a, b in sentences:
            words, grams = content_words(context[a:b]), chargrams(context[a:b])
            overlap = len(qwords & words) / max(1, len(qwords))
            char_overlap = len(qgrams & grams) / max(1, len(qgrams))
            ranked.append((overlap + char_overlap, overlap, char_overlap, a, b))
        ranked.sort(reverse=True)
        q = question.casefold()
        types = [bool(re.search(pattern, q)) for pattern in
                 (r"\b(kitn\w*|how many|how much)\b", r"\b(kab|when)\b",
                  r"\b(kaun|kisne|who)\b", r"\b(kahan|kaha|where)\b",
                  r"\b(abbreviation|full form)\b")]
        spans, features = [], []
        for _, overlap, char_overlap, a, b in ranked[:self.top_sentences]:
            tokens = list(re.finditer(r"\b\w+(?:[,.]\d+)*\b", context[a:b]))
            if not tokens:
                continue
            # Compute token properties once, then all span sums with prefix differences.
            words = [answer_tokens(t.group()) for t in tokens]
            properties = [[len(w), any(c.isdigit() for c in t.group()),
                           any(c.islower() for c in t.group()), any(c.isupper() for c in t.group()),
                           *[word in w for word in sorted(qwords)]] for t, w in zip(tokens, words)]
            prefix = np.vstack([np.zeros(len(properties[0])), np.cumsum(properties, axis=0)])
            starts, offsets = np.indices((len(tokens), self.max_answer_words))
            ends = starts + offsets + 1
            valid = ends <= len(tokens)
            starts, ends = starts[valid], ends[valid]
            sums = prefix[ends] - prefix[starts]
            lengths = (ends - starts) / self.max_answer_words
            numeric = (sums[:, 1] > 0).astype(float)
            capital = np.asarray([t.group()[0].isupper() for t in tokens])[starts]
            acronym = (sums[:, 2] == 0) & (sums[:, 3] > 0)
            repeated = (sums[:, 4:] > 0).sum(axis=1) / np.maximum(1, sums[:, 0])
            base = np.column_stack([np.full(len(starts), overlap), np.full(len(starts), char_overlap),
                                    lengths, numeric, capital, acronym, repeated, starts / len(tokens),
                                    np.asarray([w[0] in STOP for w in words])[starts],
                                    np.asarray([w[-1] in STOP for w in words])[ends - 1]])
            interactions = (np.asarray(types)[None, :, None] *
                            np.column_stack([numeric, capital, acronym, lengths])[:, None, :])
            features.append(np.column_stack([base, interactions.reshape(len(starts), -1)]))
            spans.extend((a + tokens[i].start(), a + tokens[j - 1].end()) for i, j in zip(starts, ends))
        return spans, np.vstack(features).astype(np.float32) if features else np.empty((0, 30), dtype=np.float32)

    def fit(self, rows):
        features, targets, weights = [], [], []
        self.training_candidate_hits = 0
        rng = np.random.default_rng(42)
        for row in rows:
            spans, matrix = self.candidates(row["text"], row["context"])
            scores = np.asarray([token_f1(row["context"][a:b], row["answer"]) for a, b in spans])
            if not len(scores):
                continue
            self.training_candidate_hits += int(scores.max() == 1)
            positive, negative = np.flatnonzero(scores > 0), np.flatnonzero(scores == 0)
            # Keep overlapping answers and a fixed-size sample of distractors per question.
            sample = rng.choice(negative, min(100, len(negative)), replace=False)
            selected = np.concatenate([positive, sample])
            features.extend(matrix[selected])
            targets.extend(scores[selected])
            weights.extend([1 / len(selected)] * len(selected))
        if not features:
            raise ValueError("No trainable QA spans")
        self.ranker = Ridge(alpha=1.0).fit(features, targets, sample_weight=weights)
        return self

    def predict(self, question, context):
        if self.ranker is None:
            raise ValueError("Train the QA model first")
        if len(context) > 12000 or len(question) > 1000:
            raise ValueError("QA limits: 12,000 context characters and 1,000 question characters")
        spans, features = self.candidates(question, context)
        if not spans or not question.strip() or features[:, :2].max() == 0:
            return {"answer": "", "start": None, "end": None, "no_answer": True, "score": 0.0}
        scores = self.ranker.predict(features)
        best = int(np.argmax(scores))
        if scores[best] < self.threshold:
            return {"answer": "", "start": None, "end": None, "no_answer": True, "score": float(scores[best])}
        a, b = spans[best]
        return {"answer": context[a:b], "start": a, "end": b,
                "no_answer": False, "score": float(scores[best])}
