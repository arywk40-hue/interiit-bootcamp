# Human Sentiment Contrast Set Protocol

`human_contrasts.csv` intentionally contains no model-written evaluation rows.
The separately named `model_drafts_pending_review.jsonl` contains 120 supplied
model drafts. They are candidate prompts for human editing and cannot be scored,
frozen, or described as human evidence in their current form.

## Target

Collect at least 100 pairs, with approximately equal coverage of:

1. Romanization/spelling variants that preserve sentiment.
2. Natural shorthand variants that preserve sentiment.
3. Emoji changes that genuinely flip sentiment.
4. Punctuation changes that preserve or flip sentiment according to context.

Each pair should sound like a message a person would naturally send. Do not
start from one template and mechanically replace words.

## Annotation

- Record an anonymous `writer_id`; do not collect names or contact details.
- Two people independently assign `label_a` and `label_b`.
- A third person resolves disagreements.
- Labels are `negative`, `neutral`, or `positive`.
- `meaning_relation` is `same_sentiment` or `sentiment_flip`.
- Keep this set evaluation-only. Do not train on it.

## Quality checks

- Both messages must be understandable without private context.
- A spelling pair should change form, not meaning.
- An emoji-flip pair must genuinely change the perceived sentiment.
- Reject forced, unnatural or ambiguous pairs.
- Report agreement before adjudication.

Enter the unlabelled pair text in `human_contrasts.csv`. Do not put labels in
that file.

## Review the supplied model drafts

`draft_human_review.csv` omits every model-proposed sentiment and relation label.
A consenting human editor must fill `decision` with `accept`, `rewrite`, or
`reject`, supply an anonymous `human_reviewer_id`, and explain ambiguous cases in
`review_notes`. For `rewrite`, both `reviewed_text_a` and `reviewed_text_b` are
required. Reject forced ellipsis sarcasm or emoji changes whose meaning is unclear.

The blank review form is reproducible with:

```bash
PYTHONPATH=src python -m rinlu.evaluation.contrast_set prepare-draft-review \
  --drafts data/challenge/model_drafts_pending_review.jsonl \
  --output data/challenge/draft_human_review.csv
```

After all 120 decisions are complete, validate and convert the accepted rows:

```bash
PYTHONPATH=src python -m rinlu.evaluation.contrast_set finalize-draft-review \
  --review data/challenge/draft_human_review.csv \
  --output data/challenge/human_contrasts.csv \
  --min-pairs 100
```

Accepted rows retain `human-reviewed model draft` provenance in their notes.
They must be reported as human-reviewed model drafts. Rows a person fully rewrites
can additionally be counted as human-edited. This editing stage does not supply
gold sentiment labels; the two blind annotators below still label each message.

## Freeze and blind the set

After at least 100 pairs are written, run this before anyone labels them:

```bash
PYTHONPATH=src python -m rinlu.evaluation.contrast_set freeze \
  --pairs data/challenge/human_contrasts.csv \
  --output-dir data/challenge/frozen_v1 \
  --model current_sentiment=models/current/sentiment.joblib \
  --config configs/sentiment.json
```

This writes:

- an immutable copy of the complete unlabelled pair file;
- content, ID, configuration and model SHA-256 hashes;
- two independently shuffled annotation forms;
- a private mapping from anonymous item IDs back to pairs.

This creates a new freeze for the current model bundle. The historical
`reports/sentiment/contrast_preregistration.json` refers to older model and
configuration hashes. Do not reuse it to certify the current experiment.

Commit and push the frozen directory before labeling. A local timestamp alone
is not externally verifiable.

Give `annotator_a.csv` and `annotator_b.csv` to different people. They fill
only the `label` column and must not see each other's file.

After both forms are complete:

```bash
PYTHONPATH=src python -m rinlu.evaluation.contrast_set prepare-adjudication \
  --frozen-dir data/challenge/frozen_v1 \
  --annotator-a data/challenge/frozen_v1/annotator_a.csv \
  --annotator-b data/challenge/frozen_v1/annotator_b.csv \
  --output data/challenge/frozen_v1/adjudicator.csv
```

The third annotator labels only the disagreements, without seeing the first two
labels. Majority agreement becomes gold. If all three choose different labels,
that item and its pair are excluded as ambiguous and reported.

Finally, run the locked scoring command documented by
`python -m rinlu.evaluation.contrast_set score --help`.
