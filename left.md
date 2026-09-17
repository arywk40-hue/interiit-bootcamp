# PS4 Completion Plan — What's Done vs What's Left

**Last updated:** 2026-09-15  
**Status:** Baselines production-ready; human evaluation & data gaps remain

---

## ✅ COMPLETED — Engineering & Baselines

### Core Models (all meet ≤500M params, p95 <10ms)
| Task | Model | Quality (held-out) | Latency p95 | Artifact | Learned Scalars |
|------|-------|-------------------|-------------|----------|-----------------|
| Sentiment | `char_tfidf` (TF-IDF 3-5g + OvR LogReg) | macro-F1 **0.6805** | **0.54 ms** | 5.5 MB | 720,003 |
| Intent | `word_char_tfidf` (TF-IDF 1-2g word + 3-5g char + OvR LogReg) | macro-F1 **0.4988** | **3.84 ms** | 8.8 MB | 1,394,725 |
| QA | `linear_span_ranker` (Ridge on 30 span features) | token-F1 **0.3175** (7 test) | **1.58 ms** | 1 KB | 31 |
| Summarization | `linear_extractive_summarizer` (Ridge on 6 features) | **No real eval** (GupShup missing) | — | — | 0 unfitted / 7 fitted |

### Neural Experiment (experimental, not selected)
- `ByteMultiTaskModel` — 4,306,511 params, random init, UTF-8 bytes + Unicode symbol classes
- Masked-byte pretraining (10k balanced messages, zero dev/test overlap)
- Joint supervision (sentiment + intent + QA + MLM)
- ONNX dynamic int8 export: 5.57 MB unified graph, FP32→ONNX max diff 2.39e-6
- Meets p95 gates (sentiment 5.55ms, intent 5.18ms, QA 8.81ms) but **quality below baselines** (sentiment 0.4162 F1)

### Infrastructure & Reproducibility
- [x] CLI: `python -m rinlu.run train/test/predict --task all`
- [x] Pinned dataset fetching (`scripts/fetch_data.py`) with SHA-256 verification in `data/source_manifest.json`
- [x] Split hashing, code fingerprinting, parameter counting in every report
- [x] Train/dev/test split isolation (no leakage; normalized duplicate removal)
- [x] Robustness diagnostics: vowel deletion, letter repetition, emoji/punctuation ablation
- [x] Throughput benchmarking at 1/2/4/8 workers (`scripts/benchmark_throughput.py`)
- [x] External mBERT comparison (frozen, ensemble search on dev → selected 1.0 char model)
- [x] Private WhatsApp domain adaptation (vocab/IDF only, no label fitting) — no improvement
- [x] Technical whitepaper: `architecture.md` (tokenization, architecture, error analysis, throughput trade-offs)

### Deliverables from PS4
| Deliverable | Status |
|-------------|--------|
| 1. Reproducible training/inference codebase | ✅ Complete |
| 2. Technical whitepaper | ✅ `architecture.md` |
| At least one downstream NLP task | ✅ Sentiment + Intent + QA (3 tasks) |

---

## ❌ NOT DONE — Human Evaluation & Data Gaps

### 1. Human Contrast Set Evaluation (Critical for "emoji/punctuation as semantic modulators" claim)
**Location:** `data/challenge/`
- `model_drafts_pending_review.jsonl` — 120 model-drafted pairs (spelling/shorthand/emoji/punctuation flips)
- `draft_human_review.csv` — Label-free editing form for human reviewers

**Required workflow (per `data/challenge/README.md`):**
- [ ] Human reviewers edit/validate the 120 candidate pairs
- [ ] Freeze **current** `models/current/sentiment.joblib` + config hashes **before** annotation
- [ ] Two independent blind annotators label each pair
- [ ] Third-person adjudication on disagreements
- [ ] Compute: pair accuracy, flip sensitivity, directional flip accuracy, bootstrap CIs
- [ ] Only then can you claim "emoji/punctuation affect semantics" with human evidence

**Current diagnostic (NOT official):**  
Byte model relation accuracy on draft labels: spelling 0.833/0.800, emoji flips 0.033, punctuation 0.333 — models often **invariant** to targeted cues.

---

### 2. Summarization Training & Evaluation (Blocked on Data)
**Required:** GupShup h2h (Hinglish dialogue → Hinglish summary)  
**Access:** [Official request form](https://docs.google.com/forms/d/1zvUk7WcldVF3RCoHdWzQPzPprtSJClrnHoIOYbzaJEI/viewform) — requires attribution, prohibits public redistribution

**File layout once received:**
```
data/raw/summarization/
├── train.source   train.target
├── dev.source     dev.target
├── test.source    test.target
```
- [ ] Request access & receive data
- [ ] Format aligned pairs (preserve `<br>`/`<sep>`/newlines as turn separators)
- [ ] Run `PYTHONPATH=src python -m rinlu.run train --task summarization`
- [ ] Run `PYTHONPATH=src python -m rinlu.run test --task summarization`
- [ ] Report ROUGE-1/2/L F1 + small human factuality check

**Current state:** Unsupervised centrality baseline only; Ridge head exists but untrained.

---

### 3. QA Expansion (Tiny Test Set)
**Current:** 102 usable CMQA examples → 74/21/7 train/dev/test (group-shuffled)  
Only **2 of 7** test answers fit 256-byte reader; token-F1 = 0.0000 on neural, 0.3175 on linear

**Required:** Independently reviewed human QA pairs
- [ ] Add to `data/raw/qa/human_expansion.jsonl` (one JSONL per line):
  ```json
  {"uid": "...", "text": "question", "context": "...", "answer": "exact span", "group": "doc_id", "split": "train|dev|test", "reviewer_ids": ["r1","r2"], "consent_to_use": true}
  ```
- [ ] Each group stays in one split; ≥2 anonymous reviewer IDs; answer = exact context span
- [ ] Loader validates and reports accepted count

---

### 4. Intent Domain Mismatch
**Current:** Google Hinglish-TOP (task-oriented: CREATE_ALARM, SEND_MESSAGE, etc.)  
**PS4 context:** Customer support chats (delivery, orders, complaints)  
- [ ] Either: map TOP labels to support taxonomy, or
- [ ] Document limitation explicitly in whitepaper/submission

---

### 5. Latency Guarantees Documentation
**Current:** Corpus-specific measurements on this hardware (macOS ARM64, 1 BLAS thread)  
- [ ] Document input limits (2000 char sentiment, 1000/12000 QA, 12000/128 summary) are **not** worst-case guarantees
- [ ] Note: "Measured sub-10ms on benchmark ≠ guaranteed on all allowed inputs/hardware"

---

## 📋 QUICK REFERENCE — Commands to Run Next

```bash
# 1. Human contrast set — after freeze & annotation
# (no CLI command yet; requires human process per data/challenge/README.md)

# 2. Summarization — once GupShup data placed
PYTHONPATH=src python -m rinlu.run train --task summarization
PYTHONPATH=src python -m rinlu.run test --task summarization --runs 300

# 3. QA expansion — once human_expansion.jsonl added
PYTHONPATH=src python -m rinlu.run train --task qa
PYTHONPATH=src python -m rinlu.run test --task qa --runs 300

# 4. Full re-test after any data/model change
PYTHONPATH=src python -m rinlu.run test --task all --runs 300

# 5. Neural export/benchmark (if revisiting)
PYTHONPATH=src python scripts/train_from_scratch.py --private /path/to/chat1.txt ...
PYTHONPATH=src python scripts/export_byte_model.py --runs 300
PYTHONPATH=src python scripts/benchmark_throughput.py --requests 500 --workers 1,2,4,8
```

---

## 🎯 SUBMISSION READINESS CHECKLIST

| PS4 Constraint | Met? | Evidence |
|----------------|------|----------|
| Romanized/code-mixed Indic text | ✅ | UTF-8 bytes + char TF-IDF preserve all variants |
| ≤500M parameters | ✅ | Baselines: 0.7M–1.4M; Neural: 4.3M |
| Single-digit ms latency | ✅ | Sentiment 0.54ms, Intent 3.84ms, QA 1.58ms p95 |
| Spelling/slang robustness | ⚠️ | Synthetic diagnostics only; **needs human contrast set** |
| Emoji/punctuation as semantic modulators | ⚠️ | Features retain them; **needs human flip evaluation** |
| At least one downstream NLP task | ✅ | Sentiment (primary), Intent, QA — all working |
| Reproducible codebase | ✅ | CLI, pinned deps, split hashes, code hashes |
| Technical whitepaper | ✅ | `architecture.md` |

**Bottom line:** Engineering is done. The submission blocker is **human-annotated evaluation data** (contrast set, GupShup summaries, expanded QA). No code changes needed — only data collection and human review per the documented protocols.