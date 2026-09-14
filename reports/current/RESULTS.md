# Current measured results

Run date: 2026-09-13. macOS ARM64, Python 3.13.0, one numerical-library thread.
All fitted models use train only. Test is held out from candidate selection.

| Task | Test examples | Quality | p95, ms | Artifact, MB | Learned scalars |
|---|---:|---|---:|---:|---:|
| sentiment | 3000 | macro-F1 0.6805 | 0.73 | 5.515 | 720,003 |
| intent | 6390 | macro-F1 0.4988 | 3.82 | 8.827 | 1,394,725 |
| qa | 7 | token-F1 0.3175; EM 0.2857 | 1.67 | 0.001 | 31 |
| summarization | — | No real-data evaluation; GupShup unavailable | — | — | 0 unfitted / 7 fitted |

## Frozen mBERT and weighted ensemble

`rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment` was evaluated
without changing its weights. Its model card reports accuracy 0.588889 and F1
0.582678 on SAIL 2017; those figures are not directly comparable with SentiMix.
On the complete 3,000-example SentiMix test it obtained macro-F1 0.4262 and p95
47.76 ms. It has 177,855,747 parameters.

The ensemble searched mBERT weights 0.00–1.00 in steps of 0.05 on all 3,000
development examples. The selected weights were **1.00 character model and 0.00
mBERT**; every nonzero mBERT weight lowered development macro-F1. The frozen test
result therefore remains macro-F1 0.6805 and p95 0.73 ms. The deployment pipeline
does not load mBERT. The decision and artifact hashes are fixed in
`configs/sentiment_ensemble.json`.

## Private WhatsApp domain-adaptation ablation

Five cleaned chat exports supplied locally were deduplicated into 47,046 messages.
The messages have no reviewed task labels, so they were used only to fit the
character TF-IDF vocabulary and IDF values. Classifier weights were fitted only
from the original labelled training splits. URLs, phone numbers and email addresses
are replaced before fitting, and no raw messages are copied into this repository.

| Task | Original test macro-F1 | Adapted test macro-F1 | Adapted p95, ms | Decision |
|---|---:|---:|---:|---|
| sentiment | 0.6805 | 0.6788 | 0.69 | keep original |
| intent | 0.4988 | 0.4876 | 3.29 | keep original |

Unlabelled vocabulary adaptation did not improve either held-out result. The
adapted artifacts remain local because their vocabularies contain character
n-grams derived from private chats. Source hashes and redaction counts are stored
in the ignored `reports/domain_adapted/` reports so the experiment is reproducible
without committing private content.

The summarizer is functional in unsupervised mode. Its training/test CLI was checked
with explicitly invented temporary fixtures, which are not benchmark evidence.

## Timing and footprint

300 warm-model timed calls, spread across each held-out split. Includes input
processing, scoring and output construction; excludes CLI startup, model loading,
networking and file I/O. These are sequential timings, not concurrent service load.
All three measured fitted tasks meet p95 <10 ms on this corpus and machine.
This does not establish worst-case latency or summarization latency.

QA originally measured 14.39 ms p95. Prefix sums and NumPy array operations reduced
that to the value above. All candidate feature matrices and predictions remained
exactly equal across all 102 usable questions; weights were unchanged.

## Error analysis

**Sentiment**: accuracy 0.6773; selected `char_tfidf`.
Weakest supported labels: neutral (F1 0.596, n=1100), negative (F1 0.706, n=900), positive (F1 0.740, n=1000).

**Intent**: accuracy 0.8241; selected `word_char_tfidf`.
Weakest supported labels: GET_EVENT_ORGANIZER (F1 0.000, n=1), GET_INFO_ROUTE (F1 0.000, n=1), GET_LOCATION (F1 0.000, n=4), GET_REMINDER_LOCATION (F1 0.000, n=10), HELP_REMINDER (F1 0.000, n=1).

**QA**: 74 train / 21 dev / 7 test questions from 25 distinct normalized contexts.
Only two of seven test answers exactly match. This sample is far too small for a
reliable performance estimate. Span ranking and answer boundaries remain weak.
The tiny linear reader is useful as an explainable baseline, not a reliable QA system.
No-answer quality has not been measured against human-labelled unanswerable cases.

## Illustrative CLI failures

These short developer-written checks are not human benchmark annotations.
`service acchi hai 😄` was predicted as neutral. For question `delivery kab hogi?`
and context `Delivery kal hogi.`, QA returned `Delivery kal`, including an extra
word. These illustrate weak sentiment generalization and imprecise span boundaries;
we did not hardcode corrections for the demonstrations.

## Robustness diagnostics

| Task | Edit | Changed examples | Metric delta on the same examples |
|---|---|---:|---:|
| sentiment | vowel_drop | 2998 | -0.0013 |
| sentiment | repeat_letter | 3000 | -0.0053 |
| sentiment | emoji_ablation | 609 | -0.0014 |
| sentiment | punctuation_ablation | 2966 | -0.0041 |
| intent | vowel_drop | 6305 | -0.0562 |
| intent | repeat_letter | 6390 | -0.0231 |
| intent | punctuation_ablation | 1924 | -0.0009 |
| qa | vowel_drop | 7 | +0.0000 |
| qa | repeat_letter | 7 | -0.0857 |
| qa | punctuation_ablation | 6 | +0.0000 |

Vowel deletion and repetition are synthetic edits. Emoji/punctuation removal are
semantic ablations, not guaranteed meaning-preserving noise. The small aggregate
effect of emoji removal is not evidence of correct emoji-flip interpretation.

## What prevents a complete PS claim

- The new 4,304,712-parameter shared byte model is implemented but not yet trained.
- GupShup access is needed for real summary training and evaluation.
- Natural spelling and pragmatic emoji-flip pairs still require human annotation.
- QA needs substantially more context-grounded human data and stronger accuracy.
- Intent covers TOP tasks, not the proposed customer-support taxonomy.
- Measurements are corpus/hardware specific; no concurrency or worst-case guarantee.

Raw answer-containing prediction outputs are generated locally and ignored by Git.
The historical sentiment results are a separate train+dev experiment and remain
unchanged. Dataset URLs, revisions and hashes are in `data/source_manifest.json`.
The final test suite contains 23 passing tests.
