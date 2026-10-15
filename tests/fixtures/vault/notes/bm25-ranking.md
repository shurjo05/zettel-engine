---
id: 20261015100000
title: BM25 Ranking
type: concept
tags: [retrieval, keyword-search]
aliases: [Okapi BM25]
sources:
  - external:robertson-zaragoza-2009
created: 2026-10-15
updated: 2026-10-15
---
# BM25 Ranking

BM25 scores a document for a query by summing term weights that grow with term frequency, saturate, and are normalized by document length.

## Details

BM25 improves on TF-IDF with two parameters: `k1` controls saturation and `b` controls length normalization.

## Related

- [[reciprocal-rank-fusion]]: fuses BM25 results with vector results
