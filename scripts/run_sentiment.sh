#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m unittest discover -s tests -v

python -m rinlu.data.prepare \
  --train ../upload/Hinglish_train_14k_split_conll.txt \
  --dev ../upload/Hinglish_dev_3k_split_conll.txt \
  --test ../upload/Hinglish_test_unalbelled_conll_updated.txt \
  --test-labels ../upload/Hinglish_test_labels.txt \
  --output-dir data/processed/sentiment

python -m rinlu.sentiment.train \
  --data-dir data/processed/sentiment \
  --output-dir models/sentiment \
  --latency-runs 500 \
  --selection-tolerance 0.002 \
  --final-evaluate

python -m rinlu.evaluation.audit \
  --data-dir data/processed/sentiment \
  --output reports/sentiment/tokenization_audit.json

python -m rinlu.evaluation.robustness \
  --model models/sentiment/final_model.joblib \
  --data data/processed/sentiment/test_labeled.jsonl \
  --output-dir reports/sentiment

