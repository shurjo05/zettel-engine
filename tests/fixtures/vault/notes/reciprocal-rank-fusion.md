---
id: 20261015100100
title: Reciprocal Rank Fusion
type: pattern
tags: [retrieval, hybrid-search]
aliases: [RRF]
sources:
  - external:cormack-2009
created: 2026-10-15
updated: 2026-10-15
---
# Reciprocal Rank Fusion

Reciprocal rank fusion merges several ranked lists by summing 1 / (k + rank) for each document, so it needs ranks but not comparable scores.

## Related

- [[BM25-Ranking|BM25]]: one of the lists being fused
- [[embeddings#Details]]: the other list comes from vector search
