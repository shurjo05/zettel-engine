# Processing the inbox

Paste this into Claude Code or Codex, or save it as a Claude Code slash command
(`~/.claude/commands/zk-inbox.md`). It needs the zk MCP server started with
`--allow-writes`.

---

Process my Zettelkasten inbox with the zk tools.

1. Call `list_inbox`. For each capture, oldest first, call `read_capture`.
2. Find the ideas in it worth keeping: things I would want to find again in a month, stated so they
   make sense without the conversation. Most captures hold zero to three. Skip chit-chat, one-off
   debugging and anything already obvious.
3. For each idea, call `search_notes` and `check_duplicates`. If an existing note already covers it,
   tell me which one instead of proposing a new note.
4. Otherwise call `propose_note` with one idea per note, a one-to-three-sentence summary, the
   details in `body`, `sources`, `links` to related notes you found, and `capture` set to the
   capture's name.
5. Show me each proposal (the rendered note and its duplicate check) and wait for my answer.
   Call `apply_proposal` only for the ones I approve, and `discard` the ones I reject.
6. When a capture has nothing worth keeping, or all its proposals are handled, `discard` it with
   `kind: "capture"`.
