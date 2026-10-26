---
id: "20260506140001"
title: "Event Handlers in C#"
type: "snippet"
domain: ["coding"]
tags: ["csharp", "events"]
source: "local-repo:demo-app"
confidence: 0.9
created: "2026-05-06"
---

## Summary
An event handler in C# subscribes with += and must unsubscribe with -= to avoid keeping the publisher's subscriber alive.

## Core Idea
The publisher holds a reference to every handler, so a forgotten -= leaks the subscriber.

## Links
- [[cache-embeddings-by-content-hash]] - unrelated, but linked in the old vault
