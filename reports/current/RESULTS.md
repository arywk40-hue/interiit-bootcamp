# Current measured results

Run date: 2026-09-14. macOS ARM64, Python 3.13.0, one numerical-library thread.
All fitted models use train only. Test is held out from candidate selection.

## From-scratch shared byte model

The 4,306,511-parameter model was initialized randomly. Masked-byte pretraining
used 10,000 balanced, deduplicated train-only messages from SentiMix,
Hinglish-TOP, PHINC, CMQA, public anonymized Gupshup chat and five redacted local
chat sources. The normalized overlap with downstream development/test text was
zero. Masked-byte loss was 3.4819 after one epoch. Joint sentiment, intent and QA
training then ran through epoch 9; the checkpoint was selected by the mean of the
three development metrics. Total measured CPU training time was 1,788 seconds
across the initial and resumed runs.

| Neural int8 task | Held-out examples actually read | Quality | End-to-end p95 | End-to-end p99 |
|---|---:|---:|---:|---:|
| sentiment, 160 bytes | 3,000 | macro-F1 0.4162; accuracy 0.4200 | 5.55 ms | 9.66 ms |
| intent, 160 bytes | 6,390 | macro-F1 0.0562; accuracy 0.2704 | 5.18 ms | 7.92 ms |
| QA, 256 bytes | 2 of 7 | token-F1 0.0000 | 8.81 ms | 9.79 ms |
| summary candidate score, 160 bytes | — | untrained; no h2h labels | 5.62 ms | 6.65 ms |

These are the final dynamic-int8 test measurements, at least 300 warm sequential
calls per graph with one ONNX Runtime thread. End-to-end timing includes byte encoding,
inference and output construction, and excludes graph loading and file I/O. The
summary number times one candidate score, not a complete multi-turn summary.
The unified int8 artifact is 5,566,115 bytes and contains one encoder copy plus
all task heads. FP32 ONNX errors versus PyTorch were at most `2.39e-6`; task-level
int8 maximum absolute differences were 0.0020–0.0229 on the export inputs.

The trained byte model meets the size and measured p95 gates for the bounded
graphs, but its quality is substantially below the compact baselines. It should
remain an experimental architecture. For the positive demonstration
`service bahut acchi hai 😄`, int8 inference now returns `positive`; this single
developer-written example is not evaluation evidence. For `delivery kab hogi?`
with `Order processing complete hai. Delivery kal hogi.`, it returned only `.`,
which exposes the reader's weak boundary learning.

### Neural robustness diagnostics

| Task | Edit | Changed examples | Macro-F1 delta | Prediction stability |
|---|---|---:|---:|---:|
| sentiment | vowel deletion | 2,998 | -0.0002 | 0.7091 |
| sentiment | letter repetition | 3,000 | +0.0051 | 0.7213 |
| sentiment | emoji removal | 609 | -0.0369 | 0.6125 |
| sentiment | punctuation removal | 2,966 | -0.0618 | 0.5091 |
| intent | vowel deletion | 6,305 | -0.0041 | 0.6090 |
| intent | letter repetition | 6,390 | -0.0058 | 0.5900 |
| intent | punctuation removal | 1,924 | -0.0020 | 0.7994 |

Vowel deletion and repetition are synthetic diagnostics. Emoji/punctuation
removal changes meaning in some messages. Their larger sentiment effects show
that the byte model uses those channels; they do not establish correct pragmatic
flips without the preregistered human contrast set.

### Model-drafted contrast diagnostic (not official evaluation)

The supplied 120 pairs contain model-proposed labels which have not been checked
by people. Testing against those draft labels is useful for finding failures, but
the resulting numbers are not human-grounded quality evidence and do not replace
the freeze, blind annotation and adjudication protocol. The drafts contain 122
positive and 118 negative message labels, with no neutral examples, so this is
also not a balanced three-class evaluation.

| Model | Draft-label macro-F1 | Message accuracy | Pair exact accuracy | Relation accuracy | Online p95 |
|---|---:|---:|---:|---:|---:|
| deployed character baseline | 0.1756 | 0.1708 | 0.0667 | 0.4250 | 0.36 ms |
| trained byte int8 | 0.2671 | 0.4667 | 0.2417 | 0.5000 | 2.54 ms |

Relation accuracy measures whether predictions preserve or flip sentiment as the
draft proposes. On spelling and shorthand pairs, it was 0.8667/0.7667 for the
baseline and 0.8333/0.8000 for the byte model. On proposed emoji flips, it was
0.0000 and 0.0333; on punctuation/context flips, 0.0667 and 0.3333. The models
are therefore often invariant to the very cues this challenge targets. Full
per-category results, artifact hashes and failure examples are in
`draft_contrast_diagnostic.json`.

## Sustained concurrent throughput

The focused sentiment setting issued 1,000 in-process requests to one already-loaded
shared model. The secondary-task settings used 500 requests each.
The measurement includes preprocessing, inference and output construction. It
excludes process startup and file I/O. Numerical libraries and each ONNX session
used one thread; concurrency came from request workers.

| Sentiment implementation | Workers | Requests/s | p95 latency |
|---|---:|---:|---:|
| deployed character baseline | 1 | 1,238 | 0.87 ms |
| deployed character baseline | 2 | 1,204 | 2.49 ms |
| deployed character baseline | 4 | 1,073 | 13.90 ms |
| deployed character baseline | 8 | 1,150 | 13.84 ms |
| neural int8 | 1 | 210 | 4.98 ms |
| neural int8 | 2 | 400 | 5.42 ms |
| neural int8 | 4 | 686 | 8.76 ms |
| neural int8 | 8 | 622 | 20.30 ms |

The best latency-compliant sentiment setting is the one-worker character model.
Four neural workers improve throughput while retaining single-digit p95. Eight
workers raise contention and violate the latency gate;
extra baseline threads add contention without a reliable throughput gain. Full
sentiment, intent and QA measurements are in `throughput.json`. These are local
threaded measurements, not distributed-server capacity claims.

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

- The trained shared byte model is fast but does not beat the compact baselines.
- GupShup access is needed for real summary training and evaluation.
- A 120-pair model-drafted spelling, shorthand, emoji and punctuation/context
  candidate bank now has a label-free human editing form. It still requires human
  review, freezing, two-person blind annotation and adjudication before scoring.
- QA has only 102 usable examples; only two held-out answers fit the bounded reader,
  and its final token-F1 is zero.
- Intent covers TOP tasks, not the proposed customer-support taxonomy.
- Measurements are corpus/hardware specific; no concurrency or worst-case guarantee.

Raw answer-containing prediction outputs are generated locally and ignored by Git.
The historical sentiment results are a separate train+dev experiment and remain
unchanged. Dataset URLs, revisions and hashes are in `data/source_manifest.json`.
The final test suite contains 29 passing tests.
