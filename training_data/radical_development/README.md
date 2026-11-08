# Radical development data

`cases.json` contains 24 single-step cases, 12 each from the RC and RS labels in
FlowER's `flower_new_dataset/train.txt`. Rows are sorted by content-derived ID;
the sample uses the first 12 per label. `extraction_report.json` records the
source hash and complete extraction counts: 4,606 rows, 206 exact duplicates,
4,400 unique cases, no parse/radical-detection rejections.

These are **train-derived development cases**, not held-out evaluation or
complete reaction pathways. No arrows or pathway grouping are fabricated.
The reaction reference is recorded separately from the model input. The custom
case import API accepts their starting materials/products; keep provenance in
this source artifact when using an importer that omits extra fields.

The authored deterministic arrow/chain contract lives in
`tests/fixtures/radical_contract.json`. Its successful replay is implementation
evidence, not an LLM quality measurement and not an approved prompt trace.

Reproduce the complete extraction without opening validation/test files:

```sh
PYTHONPATH=. python scripts/extract_radical_training.py \
  --train ../FlowER/data/flower_new_dataset/train.txt \
  --output /tmp/wiggum-radical-training.json
```

The extractor accepts only `train.txt`, rejects a leaderboard_holdout path,
retains source-line references and hashes, deduplicates exact labelled reactions,
and does not mutate the existing numeric curriculum or any evaluation tier.

Source: Joung et al., *Electron flow matching for generative reaction mechanism
prediction*, Nature 645 (2025), DOI 10.1038/s41586-025-09426-9.
Dataset: https://figshare.com/articles/dataset/28359407/3 (MIT-labelled release).
The user authorized this local noncommercial testing use on 2026-11-07.
