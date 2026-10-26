---
description: Turn the captures in my zk inbox into notes
---

Ingest my Zettelkasten inbox with the zk tools. Don't ask me anything along the way. I only want to
hear about what was written and about any warnings.

1. Call `list_inbox`. For each capture, oldest first, call `read_capture`.
2. Find the ideas in it worth keeping: things I'd want to find again in a month, stated so they make
   sense without the conversation. Most captures hold zero to three. Skip chit-chat, one-off
   debugging and anything already obvious.
3. For each idea, call `search_notes` and `check_duplicates`. Leave out ideas an existing note
   already covers, and link the rest to the related notes you found.
4. Call `ingest_notes` once per capture, with all of its notes and `capture` set to the capture's
   name. Give each note one to three test questions, asked the way I'd ask them, not copied from
   the note.
5. When a capture has nothing worth keeping, call `discard` with `kind: "capture"`.
6. Finish with one line per capture (the notes written and any held as duplicates), followed by
   every warning `ingest_notes` returned.
