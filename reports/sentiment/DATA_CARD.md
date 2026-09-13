# Data Card — Supplied Hinglish Sentiment Corpus

## Status

The files were supplied by the project team. Their original publication,
licence and redistribution terms have not yet been verified. They must not be
published with the repository until that verification is complete.

## Task and schema

Each CoNLL record contains:

- a `meta` row with UID and sentence-level sentiment;
- token rows with one of `Hin`, `Eng`, `O`, or `EMT`;
- one label: `negative`, `neutral`, or `positive`.

The test CoNLL file is unlabelled; labels are supplied in a separate UID-indexed
CSV.

## Splits

| Split | Supplied | After cleaning |
|---|---:|---:|
| Train | 14,000 | 13,997 |
| Development | 3,000 | 3,000 |
| Test | 3,000 | 3,000 |

Cleaned training labels: 4,101 negative, 5,263 neutral and 4,633 positive.

## Corrections

- One exact duplicate inside train was removed.
- Two exact train/test text leaks were removed from train.
- Seventeen empty token rows with no recoverable text were skipped.
- 2,777 mojibake-corrupted test token rows were repaired through a guarded
  Windows-1252/UTF-8 reversal.
- File hashes, affected UIDs and correction counts are stored in
  `data/processed/sentiment/manifest.json`.

## Observed properties

- Messages average approximately 26 source tokens and 127 characters.
- Every test message contains both `Hin` and `Eng` tags according to the source
  annotations.
- The test split contains 599 messages with detected emoji.
- Exact word-token OOV against cleaned train is 9.51% on test; character OOV is
  0.054%.

## Intended use

Research evaluation of compact sentiment representations for informal
Romanized Hindi-English social-media text.

## Limitations and risks

- Social-media language may not represent customer-support conversations.
- The data contains profanity, political content and potentially abusive text.
- Writer/source identifiers are unavailable, so writer-disjoint splitting
  cannot be verified.
- Sentence-level sentiment may be ambiguous, especially for neutral and
  sarcastic examples.
- Token-level language tags are source annotations and were not re-annotated.
- Encoding repair is deterministic and audited but cannot recover information
  absent from the original file.
