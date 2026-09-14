from __future__ import annotations

import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


_REPEATED_CHARACTER = re.compile(r"([^\W\d_])\1{2,}", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def conservative_canonicalize(text: str) -> str:
    """Create an auxiliary view while retaining the untouched raw view."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'").replace("‘", "'")
    text = _REPEATED_CHARACTER.sub(r"\1\1", text)
    return _WHITESPACE.sub(" ", text).strip()


class CanonicalTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [conservative_canonicalize(text) for text in X]


def symbol_view(text: str) -> str:
    """Keep punctuation and symbol code points, including emoji, in order."""
    return " ".join(
        character
        for character in text
        if unicodedata.category(character).startswith(("P", "S"))
    )


class SymbolTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [symbol_view(text) for text in X]


def _classifier(random_state: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    return OneVsRestClassifier(
        LogisticRegression(
            C=4.0,
            solver="liblinear",
            max_iter=500,
            class_weight="balanced",
            random_state=random_state,
        )
    )


def _char_vectorizer(max_features: int = 180_000):
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_model(name: str, random_state: int = 42, config: dict | None = None):
    if name == "char_tfidf":
        features = _char_vectorizer()
    elif name == "word_char_tfidf":
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=80_000,
                        sublinear_tf=True,
                        token_pattern=r"(?u)\b\w+\b",
                        dtype=np.float32,
                    ),
                ),
                ("char", _char_vectorizer()),
            ]
        )
    elif name == "raw_canonical_char_tfidf":
        features = FeatureUnion(
            [
                ("raw_char", _char_vectorizer(120_000)),
                (
                    "canonical_char",
                    Pipeline(
                        [
                            ("canonicalize", CanonicalTextTransformer()),
                            ("tfidf", _char_vectorizer(120_000)),
                        ]
                    ),
                ),
            ]
        )
    elif name == "char_symbol_tfidf":
        features = FeatureUnion(
            [
                ("char", _char_vectorizer(180_000)),
                (
                    "symbol",
                    Pipeline(
                        [
                            ("symbols", SymbolTextTransformer()),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    analyzer="char",
                                    ngram_range=(1, 3),
                                    min_df=2,
                                    max_features=10_000,
                                    sublinear_tf=True,
                                    dtype=np.float32,
                                ),
                            ),
                        ]
                    ),
                ),
            ],
            transformer_weights={"char": 1.0, "symbol": 2.0},
        )
    else:
        raise ValueError(f"unknown model {name!r}")
    model = Pipeline([("features", features), ("classifier", _classifier(random_state))])
    if config is not None:
        params = {}
        classifier = config["classifier"]
        for key in ("C", "solver", "max_iter", "class_weight"):
            params[f"classifier__estimator__{key}"] = classifier[key]
        # Apply the same configured character representation to each candidate view.
        for path, value in model.get_params().items():
            if isinstance(value, TfidfVectorizer) and value.analyzer == "char":
                channel = "symbol_features" if "symbol" in path else "character_features"
                for key in ("ngram_range", "min_df", "max_features", "sublinear_tf"):
                    if key in config[channel]:
                        setting = config[channel][key]
                        params[f"{path}__{key}"] = tuple(setting) if key == "ngram_range" else setting
        if name == "char_symbol_tfidf":
            params["features__transformer_weights"] = {"char": 1.0, "symbol": config["symbol_features"]["weight"]}
        model.set_params(**params)
    return model


MODEL_NAMES = (
    "char_tfidf",
    "word_char_tfidf",
    "raw_canonical_char_tfidf",
    "char_symbol_tfidf",
)
