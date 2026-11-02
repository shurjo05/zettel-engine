---
description: Decide each note held as a likely duplicate in my zk vault
---

Resolve the notes held in my Zettelkasten as likely duplicates, with the zk tools. Don't ask me
anything along the way. I only want to hear what you decided for each one and why.

1. Call `list_held`. For each held note, call `get_held_pair`.
2. Read the held note and the closest existing note side by side, and decide:
   - **Same idea**: the held note restates the existing one, or is a narrower or reworded version
     of it. Call `merge_held` with `into` set to the existing note's path and a merged note that
     keeps everything the existing note says and adds only what the held note says beyond it.
     Keep the existing note's title unless the held one is clearly better.
   - **Related, not the same**: the two share a topic but make different claims, or one is a
     method and the other its use. Call `apply_held`.
   When the closest note is the wrong match but one of the others is the same idea, merge into
   that one. Give a one-sentence `reason` either way.
3. Finish with one line per held note: merged into which note, or released, and the reason.
