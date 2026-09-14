# Hinglish NLU: four explainable baselines

This implements sentiment, intent classification, extractive summarization, and
context-based extractive QA for Problem Statement 4. The current models use
character features and small linear models. There is no Transformer, LLM router,
or hidden API call at inference.

**Status:** sentiment, intent, and QA train and run locally. Summarization runs in
an explicitly untrained, unsupervised mode; supervised training and evaluation
need the GupShup dataset. Human emoji-flip evaluation is also pending. This is a
working four-task baseline, not evidence that every PS requirement is solved.

Read [architecture.md](architecture.md) for the technical explanation, equations,
limitations, and how to present the project. Current measurements and error analysis are in
[reports/current/RESULTS.md](reports/current/RESULTS.md). The older `reports/sentiment/` and `models/sentiment/` results
belong to the previous sentiment experiment and are retained as historical results.

## Run it

Use Python 3.12 or newer; this run used Python 3.13.0. Commands below assume the
`interiit-bootcamp` directory. Dependencies are pinned in `pyproject.toml`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/fetch_data.py
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m rinlu.run train --task all
PYTHONPATH=src python -m rinlu.run test --task all --runs 300
```

`all` completes available tasks and exits with code **2** if any required dataset
or model is missing. The reason is saved in `reports/current/pending.json`.
This is deliberate: missing evaluation must not look like a successful full run.
Train and test are separate commands. Select models on development data; run the
held-out test after that decision. Do not tune against the resulting test report.

Individual commands are available with `--task sentiment`, `--task intent`,
`--task qa`, or `--task summarization`. `--help` lists paths and options.

```bash
PYTHONPATH=src python -m rinlu.run predict --task sentiment --text 'service acchi hai 😄'
PYTHONPATH=src python -m rinlu.run predict --task intent --text 'kal subah saat baje alarm laga do'
PYTHONPATH=src python -m rinlu.run predict --task qa --text 'delivery kab hogi?' --context 'Delivery kal hogi.'
PYTHONPATH=src python -m rinlu.run predict --task summarization --text-file conversation.txt
```

These are interface demonstrations, not benchmark examples or guaranteed correct
predictions. Create `conversation.txt` with one speaker turn per line. The summary
copies selected source turns in order. It does **not** translate into English.
For long QA contexts use `--context-file context.txt`.

Models are local `models/current/*.joblib` bundles. A bundle contains the fitted
model, task, configuration, and hashes of the data splits. Inference loads it once
per CLI process. Reported latency assumes an already-loaded model, so CLI startup
and disk loading are not included. Production callers should keep the model loaded.

## Final from-scratch architecture

The final neural candidate is `ByteMultiTaskModel` in `src/rinlu/neural.py`:

- 4,304,712 parameters, initialized randomly;
- raw UTF-8 byte input plus explicit Unicode symbol classes;
- one shared convolution-downsampled Transformer encoder;
- classification heads for sentiment and intent;
- extractive span/turn heads for QA and summarization;
- no existing model weights and no autoregressive decoding.

The architecture, training objectives, latency gates, and audited Hugging Face
dataset shortlist are documented in `architecture.md`. The compact fitted models
remain baselines until this neural candidate has trained quality measurements.

The SAIL-2017 mBERT checkpoint can be evaluated as a frozen external comparison:

```bash
PYTHONPATH=src python scripts/evaluate_external_sentiment.py --ensemble --limit 300
```

This command never updates checkpoint weights and never reads private WhatsApp
text. It searches external-model weights from 0 to 1 in steps of 0.05 on the
development split, freezes the best value, and reports test performance. Remove
`--limit 300` for the complete 3,000-example SentiMix test. Its model-card score
is from SAIL 2017, so only this local evaluation is comparable with our held-out
result.

## Private WhatsApp domain adaptation

Real chats without human task labels are used only to fit character vocabulary
and IDF. Sentiment/intent labels still come from the public training datasets:

```bash
PYTHONPATH=src python scripts/train_domain_adapted.py --task sentiment \
  --chat /Users/ariyanbhakat/Downloads/files-3/whatsapp_training_corpus.txt \
  --chat /Users/ariyanbhakat/Downloads/files-2/IIT_Mandi_Freshers_2026_clean.txt \
  --chat /Users/ariyanbhakat/Downloads/files-2/Robotronics_Club_Core_Team_26_27_clean.txt \
  --chat /Users/ariyanbhakat/Downloads/files-2/PRAYAS_4_0_Participants_clean.txt \
  --chat /Users/ariyanbhakat/Downloads/files-2/Fresher_s_THEBOYS_2025_clean.txt
```

The script deduplicates normalized messages, redacts URLs/phones/emails in memory,
stores only source hashes and counts in its report, and never copies raw chats.
The fitted TF-IDF vocabulary can still contain rare private n-grams, so the model
artifact must remain private unless chat participants consent to publication.

## Datasets and access

| Task | Dataset | Use and access |
|---|---|---|
| Sentiment | [SentiMix 2020](https://aclanthology.org/2020.semeval-1.100/) | Public [mirror](https://huggingface.co/datasets/RTT1/SentiMix); matches the prior corpus after line-ending conversion. Original source terms still need confirmation before redistribution. |
| Intent | [Google Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset) | Human-only train/validation/test TSVs, Apache-2.0 repository license. Generated augmentation is excluded. |
| QA | [CMQA challenge](https://github.com/khyathiraghavi/code_switched_QA) | Hindi-English questions with real text contexts and contiguous answers. The source COPYING notice permits use with attribution and marked modifications. |
| Summarization | [GupShup](https://github.com/midas-research/gupshup) | Request from the authors. Use **h2h**, Hinglish dialogue to Hinglish summary, for this extractive baseline. |

The downloader uses pinned revisions and verifies SHA-256 against
`data/source_manifest.json`. Dataset text remains in ignored `data/raw/`.
It never executes source-repository code. The SentiMix mirror's short OpenRAIL
metadata is not treated as proof of the original dataset's redistribution terms.

Use the [official access form](https://docs.google.com/forms/d/1zvUk7WcldVF3RCoHdWzQPzPprtSJClrnHoIOYbzaJEI/viewform).
It requires attribution and prohibits public redistribution of the corpus. The
repository README documents h2e/e2e; explicitly request the Hinglish reference
summaries described in the paper for this h2h baseline.

Once received, format the aligned **h2h** files at:

```text
data/raw/summarization/train.source   data/raw/summarization/train.target
data/raw/summarization/dev.source     data/raw/summarization/dev.target
data/raw/summarization/test.source    data/raw/summarization/test.target
```

Each line contains one complete dialogue or its reference summary. Rename the
publisher's validation split to `dev` if necessary, keeping pairs aligned. Preserve
turn separators such as `<br>`, `<sep>`, or real newlines inside input at inference.
Do not substitute English reference summaries and describe that as h2h evaluation.

## Read the code in this order

Only six new runtime/download modules were needed. Each is 150 lines or fewer.
The remaining machinery reuses the existing repository.

| File | What you need to explain |
|---|---|
| `scripts/fetch_data.py` | Pinned URLs, local downloads, source notices, checksum verification. |
| `src/rinlu/data/conll.py` | Existing CoNLL parsing, Unicode repair, label joins, normalized duplicate keys. |
| `src/rinlu/data/prepare.py` | Existing sentiment audit; now removes train/dev and dev/test leaks too. |
| `src/rinlu/data/tasks.py` | Four source adapters; real intent labels; context-grouped QA split; aligned summary pairs. |
| `src/rinlu/sentiment/features.py` | Character/word/symbol TF-IDF variants and one-vs-rest logistic regression. Shared by sentiment and intent. |
| `src/rinlu/qa.py` | Candidate sentence retrieval, short span features, learned ranking, source offsets. |
| `src/rinlu/summarization.py` | Dialogue segmentation, character similarity, sentence ranking, redundancy penalty. |
| `src/rinlu/evaluation/metrics.py` | Existing sentiment metrics plus explicitly defined token-F1 and ROUGE-1/2/L F1. |
| `src/rinlu/evaluation/tasks.py` | Task metrics, learned scalar counts, real timing, synthetic spelling diagnostics and ablations. |
| `src/rinlu/run.py` | The common train/test/predict CLI; saves models, configurations, data hashes, and reports. |
| `configs/sentiment.json` | Historical filename; the new CLI reads it for both classifiers and QA/summary settings. |
| `src/rinlu/evaluation/contrast_set.py` | Existing blind human evaluation: freeze hashes, annotation, adjudication, pair accuracy and bootstrap intervals. Supports current model bundles. |
| `tests/test_tasks.py` | New data integrity, extraction, configuration, and metric checks using tiny test-only examples. |
| `tests/test_data.py`, `test_features.py`, `test_contrast_set.py` | Existing parser/feature tests and the human evaluation workflow. |

The `__init__.py` files mark Python packages. The old `sentiment/train.py`,
`infer.py`, `export_candidate.py`, `evaluation/audit.py`, and
`evaluation/robustness.py` reproduce the historical sentiment experiment;
`scripts/run_sentiment.sh` still expects its original `../upload/` inputs.
Use `rinlu.run` for current work. The drawing scripts in `tools/` and PNGs are
presentation assets, not model code; the original broad architecture diagram is
historical. `.gitignore` excludes raw data, binaries, and predictions containing
source extracts. `pyproject.toml` controls installation and exact dependencies.

## The remaining human work

Use [data/challenge/README.md](data/challenge/README.md) to collect and independently
label natural spelling and emoji-flip pairs. Tests written by an AI cannot establish
human pragmatic accuracy. The existing contrast CSV is empty and must stay an
honest record of what has actually been collected.

After collecting pairs, freeze the **current** `models/current/sentiment.joblib`
and current configuration before annotation. The old preregistration hashes refer
to the previous experiment and cannot certify these new artifacts.
