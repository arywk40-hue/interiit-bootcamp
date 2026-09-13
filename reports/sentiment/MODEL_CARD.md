# Model Card — Character TF-IDF Hinglish Sentiment Classifier

## Model

- Character TF-IDF with 3–5 character n-grams
- Minimum document frequency: 2
- Feature cap: 180,000
- Class-balanced one-vs-rest logistic regression
- 540,003 learned coefficients and intercepts
- Serialized size: 5.59 MB

The model consumes reconstructed raw token text. It does not transliterate into
Devanagari, remove emoji or strip punctuation.

## Selection

Three candidates were trained using seed 42. A candidate was eligible when its
development macro-F1 was within 0.002 of the best result. The eligible model
with the lowest measured batch-one p95 CPU latency was selected.

The character-only and word+character candidates were accuracy-equivalent by
this rule. Character-only was selected because its p95 latency was lower.

## Final performance

| Metric | Value |
|---|---:|
| Test macro-F1 | 0.6857 |
| Test accuracy | 0.6820 |
| Negative F1 | 0.7207 |
| Neutral F1 | 0.6007 |
| Positive F1 | 0.7357 |
| Calibration error, 10 bins | 0.0793 |
| Batch-one p50 | 0.59 ms |
| Batch-one p95 | 0.83 ms |

Latency includes vectorization and prediction. It was measured with one
OpenMP/BLAS thread on the current Linux x86_64 environment and is not a promise
for other hardware.

## Intended use

- research baselines;
- latency and robustness experiments;
- demonstrations using similar informal Hinglish text.

## Not intended for

- autonomous moderation or customer penalties;
- safety-critical decisions;
- languages or domains not evaluated here;
- deployment without privacy, bias and domain-shift review.

## Known limitations

- Neutral is the weakest class.
- Character n-grams can memorize topic, profanity, usernames and URL fragments.
- Emoji ablation indicates only modest sensitivity; pragmatic emoji flips have
  not been validated with human-written contrast pairs.
- Synthetic spelling perturbations are diagnostics, not evidence of complete
  invariance to natural Romanization drift.
- No comparison against a compact neural encoder or Indic/multilingual
  subword model has been completed yet.
