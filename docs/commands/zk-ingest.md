---
description: Save what's worth keeping from this conversation as zk notes
---

Turn this conversation into Zettelkasten notes with the zk tools, without asking me to review them.

1. Find the ideas from this conversation worth finding again in a month: decisions and their
   reasons, things learned, fixes that will come up again. State each one so it makes sense on its
   own. Most conversations hold zero to three. Skip chit-chat and anything already obvious.
   $ARGUMENTS
2. For each idea, call `search_notes` and `check_duplicates`. Leave out ideas an existing note
   already covers, and link the rest to the related notes you found.
3. Call `ingest_notes` once with all the notes. Give each note
   `sources: ["conversation:<today's date> <project or topic>"]` and one to three test questions,
   asked the way I'd ask them, not copied from the note.
4. Reply with the titles written, any held as likely duplicates, and every warning returned.
   If nothing is worth keeping, say so and don't call `ingest_notes`.
