# Roman Indic NLU — Hinglish Sentiment

This repository currently implements one complete downstream task: a compact,
leakage-audited Hinglish sentiment classifier. It predicts `negative`,
`neutral`, or `positive` while preserving emoji and punctuation.

The selected model is a character 3–5 gram TF-IDF representation with a
class-balanced one-vs-rest logistic-regression classifier. It was selected by
the lowest batch-one p95 latency among models within 0.002 macro-F1 of the best
development score.

## Current result

- Test macro-F1: **0.6857**
- Test accuracy: **0.6820**
- CPU batch-one latency: **0.59 ms p50 / 0.83 ms p95**
- Sequential throughput: **1,635 predictions/s**
- Saved model size: **5.59 MB**
- Linear coefficients and intercepts: **540,003**

These latency numbers were measured in the current Linux x86_64 environment
with one BLAS/OpenMP thread. They must be rerun on the final presentation
hardware.

## Repository layout

```text
src/rinlu/data/             CoNLL parser, encoding repair and leakage checks
src/rinlu/sentiment/        features, model training and inference CLI
src/rinlu/evaluation/       token coverage, robustness and latency metrics
data/processed/sentiment/   generated leakage-clean JSONL splits
models/sentiment/           trained candidate and final model artifacts
reports/sentiment/          aggregate audits, robustness results and report
tests/                      deterministic parser and transformation tests
```

## Reproduce

From the repository root:

```bash
python -m pip install -e .

PYTHONPATH=src python -m unittest discover -s tests -v

PYTHONPATH=src python -m rinlu.data.prepare \
  --train ../upload/Hinglish_train_14k_split_conll.txt \
  --dev ../upload/Hinglish_dev_3k_split_conll.txt \
  --test ../upload/Hinglish_test_unalbelled_conll_updated.txt \
  --test-labels ../upload/Hinglish_test_labels.txt \
  --output-dir data/processed/sentiment

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python -m rinlu.sentiment.train \
  --data-dir data/processed/sentiment \
  --output-dir models/sentiment \
  --latency-runs 500 \
  --selection-tolerance 0.002 \
  --final-evaluate

PYTHONPATH=src python -m rinlu.evaluation.audit \
  --data-dir data/processed/sentiment \
  --output reports/sentiment/tokenization_audit.json

PYTHONPATH=src python -m rinlu.evaluation.robustness \
  --model models/sentiment/final_model.joblib \
  --data data/processed/sentiment/test_labeled.jsonl \
  --output-dir reports/sentiment
```

Run inference:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python -m rinlu.sentiment.infer \
  --model models/sentiment/final_model.joblib \
  --text "item delivered bol rha h but mila hi nhi 😒"
```

The repository intentionally excludes generated `*.jsonl` datasets and
`*.joblib` model binaries. Run the preparation and training commands above to
recreate them locally. This avoids redistributing source text or model
artifacts until the benchmark licence and derivative-artifact terms have been
verified.

## Data integrity

The supplied files contain 14,000 train, 3,000 development and 3,000 test
records. Preparation performs the following documented corrections:

- removes one duplicate inside train;
- removes two exact train/test text leaks from train;
- skips 17 empty source-token rows with no recoverable text;
- repairs 2,777 mojibake-corrupted test token rows;
- verifies that every test UID has exactly one supplied label.

The source dataset's provenance and redistribution licence still need to be
confirmed. Raw files are therefore not copied into this repository.

## Important limitation

The deterministic spelling perturbations are synthetic diagnostics. Emoji
removal is reported only as an input ablation; it is not treated as a
meaning-preserving transformation. A human-written emoji-contrast set is still
required before claiming robustness to pragmatic sentiment flips.
