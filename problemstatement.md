Here is **Problem Statement 4 extracted line by line from the visible PDF text on pages 4–5**.

## 4: The Polyglot’s Shorthand (450pts)

### Background & Context
Every second, millions of customer support chats, reviews, and delivery queries flood consumer platforms across South Asia.
The text arriving in these ingestion queues looks nothing like standard benchmarks.
Users rarely write in formal native scripts or pure English; instead, streams are saturated with informal, Latinized/Romanized code-mixing:
> “bhai order cancel krdo please, urgent meeting h,”
or
> “item delivered bol rha h but mila hi nhi 🤷‍♂️.”

When routed into existing NLP infrastructure, the pipeline collapses.

Native Indic models (like IndicBERT) expect native scripts and shatter Romanized words into out-of-vocabulary character noise.

English models view the syntax as gibberish.

Massive frontier LLMs can parse the meaning, but their inference costs and multi-hundred-millisecond latencies make real-time, high-throughput routing impossible. 

### Compounding this is the chaos of conversational shorthand:

**Phonetic volatility & typos:**
The same phrase appears in endless variants:
> “kya kar rahe ho”
> “kya kr rhe ho”
> “kya krre ho”

**Intra-sentence mixing:**
English loanwords blend fluidly into Indic syntax:
> “aap busy ho?”

**Pragmatic flips via emojis:**
Subtle cues invert meaning entirely.
> “Bohot badhiya service”
is praise, but
> “Bohot badhiya service 😒”
is a blistering complaint. 

---

# The Challenge
The goal is to build an **ultra-efficient linguistic engine** that treats **Latinized, code-mixed text as a first-class citizen**.

It must power downstream tasks:
* Sentiment analysis
* Intent classification
* Summarization
* Question answering

while adhering to strict deployment realities. 

### 1. Footprint
**Maximum 500M parameters** with **single-digit millisecond latency** and **high throughput**.

### 2. Robustness

Invariant to:

* phonetic spelling drift
* slang
* typos

### 3. Multimodal text semantics

Seamless handling of **emojis and punctuation as core semantic modulators rather than stripped noise.** 
---
# Key Deliverables

**1.** Reproducible training and inference codebase across downstream tasks.

**2.** Technical whitepaper justifying your:

* tokenization choices
* architecture choices
* error analyses
* throughput trade-offs 

### Note

You may choose **any one/multiple downstream NLP tasks** like:

* Sentiment Analysis
* Text Classification

and **create a suitable dataset for testing/training as needed.** 

So the central constraints of **PS4** are: **Romanized/code-mixed Indic text + ≤500M parameters + single-digit-ms latency + spelling/slang robustness + emojis/punctuation must affect semantics + at least one downstream NLP task.**
it