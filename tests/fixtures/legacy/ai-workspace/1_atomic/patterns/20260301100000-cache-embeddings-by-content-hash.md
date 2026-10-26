---
id: 20260301100000-cache-embeddings-by-content-hash
title: Cache Embeddings by Content Hash
type: pattern
domain: retrieval
tags: [embeddings, caching]
aliases: []
source: conversation:2026-03-01
created: 2026-03-01
updated: 2026-03-02
status: seed
confidence: high
related: []
---

# Cache Embeddings by Content Hash

## Summary
Key each stored vector by a hash of the note's text, so a renamed note keeps its vector and an edited one gets a new vector.

## Core Idea
The hash changes exactly when the text changes. Paths change on rename, so a path key would re-embed renamed notes for nothing.

## Example
A rename from `a.md` to `b.md` costs no model call.

## Links
[[event-handlers-in-c-sharp]]
- [[20260302110000-batch-model-calls]] - embedding many notes at once

## Source Note
Derived from: conversation:2026-03-01
