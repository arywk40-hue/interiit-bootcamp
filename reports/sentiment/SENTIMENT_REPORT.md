# Hinglish Sentiment Milestone Report

Date: 2026-09-13  
Status: reproducible linear baseline complete

## Research question

Can a character-based classifier handle Romanized Hinglish variation while
remaining comfortably below the problem statement's footprint and latency
limits?

## Dataset audit

The supplied CoNLL corpus contains 14,000 train, 3,000 development and 3,000
test messages. Each message has a UID, sentence-level sentiment and token-level
language tags (`Hin`, `Eng`, `O`, `EMT`). Test labels are stored separately and
joined by UID only for the final evaluation.

Cleaning produced 13,997 training messages:

- train UID `709` was a duplicate inside train;
- train UID `13159` exactly duplicated a test message;
- train UID `6553` exactly duplicated test UID `32866`;
- 13 train and four test rows contained a language tag but no token text and
  were skipped;
- 2,777 test token rows contained recoverable UTF-8/Windows-1252 mojibake and
  were repaired before duplicate checking and evaluation.

All corrections and source hashes are recorded in `manifest.json`.

## Representation evidence

Against the cleaned training vocabulary:

| Evaluation split | Exact word-token OOV | Character OOV |
|---|---:|---:|
| Development | 9.85% | 0.036% |
| Test | 9.51% | 0.054% |

The much lower character OOV rate supports a character representation for
open-ended Romanized spellings. This is a corpus-coverage result, not yet a
comparison against a trained BPE or WordPiece tokenizer.

## Candidate selection

All candidates use class-balanced one-vs-rest logistic regression and seed 42.

| Candidate | Dev macro-F1 | p95 latency | Artifact |
|---|---:|---:|---:|
| Character TF-IDF | 0.6105 | 0.72 ms | 5.54 MB |
| Word + character TF-IDF | 0.6106 | 1.49 ms | 6.90 MB |
| Raw + canonical character TF-IDF | 0.5991 | 1.92 ms | 7.49 MB |
| Character + explicit symbol channel | 0.5967 | 1.41 ms | 5.61 MB |

The word+character score exceeds character-only by only 0.00005 macro-F1.
Under the frozen selection rule, models within 0.002 macro-F1 of the best score
are treated as accuracy-equivalent and the lowest-p95 model is selected.
Therefore, the character-only model is the final architecture.

The proposed auxiliary canonicalized view was rejected: it reduced dev
macro-F1 and increased latency. An explicit 1–3 gram emoji/punctuation channel
also reduced dev macro-F1. Both negative ablations are retained as results.

After training the explicit-symbol candidate on the same cleaned train+dev
data, it obtained 0.6672 test macro-F1 with 1.36 ms p95 latency, compared with
0.6857 and 0.83 ms for character-only. It is therefore not the clean-set
winner. Its exact artifact is preregistered as a secondary model for the human
minimal-pair test, where flip sensitivity—not aggregate accuracy—is the target.

## Final test result

The selected architecture was retrained on cleaned train plus development data
and evaluated once on the repaired, labelled test split.

| Metric | Result |
|---|---:|
| Accuracy | 0.6820 |
| Macro-F1 | 0.6857 |
| Weighted-F1 | 0.6817 |
| Calibration error, 10 bins | 0.0793 |
| Model size | 5.59 MB |
| Learned coefficients + intercepts | 540,003 |

Per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.707 | 0.736 | 0.721 | 900 |
| Neutral | 0.605 | 0.596 | 0.601 | 1,100 |
| Positive | 0.744 | 0.728 | 0.736 | 1,000 |

Confusion matrix, rows = gold and columns = prediction:

| Gold \\ Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 662 | 194 | 44 |
| Neutral | 237 | 656 | 207 |
| Positive | 38 | 234 | 728 |

Neutral is the weakest class and is confused in both directions. This is the
main target for manual error review.

## Robustness and semantic ablations

| Condition | Examples | Macro-F1 | Change |
|---|---:|---:|---:|
| Clean test | 3,000 | 0.6857 | — |
| Deterministic internal-vowel deletion | 2,998 changed | 0.6826 | −0.0032 |
| Deterministic repeated-letter noise | 3,000 changed | 0.6777 | −0.0080 |
| Natural emoji-containing slice | 599 | 0.6960 | slice, not delta |
| Emoji removed from the same slice | 599 | 0.6896 | −0.0064 |
| Punctuation removed | 2,966 changed | 0.6825 | −0.0032 |

The model is relatively stable under the two synthetic spelling perturbations.
However, removing emoji changes only 3.2% of predictions and lowers macro-F1 by
0.0064 on emoji-containing messages. The small effect suggests that the model
uses some emoji signal but may still underuse pragmatic cues. It does not prove
correct handling of sentiment-flipping emoji pairs.

## Throughput trade-off

End-to-end batch-one timing includes TF-IDF transformation and classification:

| Metric | Result |
|---|---:|
| p50 | 0.59 ms |
| p95 | 0.83 ms |
| p99 | 1.04 ms |
| Mean | 0.61 ms |
| Sequential throughput | 1,635 predictions/s |

Measurement environment: Linux x86_64, Python 3.12, one OpenMP/BLAS thread,
500 timed requests following 30 warm-up requests. These are environment-specific
measurements and must be repeated on the final deployment machine.

## Error-analysis artifacts

- `test_predictions.csv` contains every test prediction and class probability.
- `high_confidence_errors.csv` is generated locally and ranks incorrect test
  predictions by confidence. It is excluded from the public repository because
  it contains source text.
- `robustness_results.json` contains complete slice metrics, confusion matrices
  and explicit limitations.

## Claims we can and cannot make

Supported:

- character features avoid most exact word-OOV failures;
- the selected model satisfies the latency and footprint limits in this test
  environment;
- deterministic spelling drift causes a small but measurable performance drop;
- the handcrafted canonical view does not improve this baseline.

Not yet supported:

- superiority over IndicBERT, multilingual BPE or a compact neural encoder;
- invariance to naturally occurring human spelling variation;
- correct interpretation of emoji-driven pragmatic flips;
- generalization from social-media sentiment to customer-support traffic.

The next defensible experiment is a small human-written contrast set containing
natural shorthand pairs and emoji-driven label flips.
