# Data flow

Vault content is private. This file lists every place it goes. Update it whenever a feature
sends vault content somewhere new.

| Feature | Reads | Writes | Leaves this machine? |
| --- | --- | --- | --- |
| `zk index`, `zk check` | Markdown files in the vault | SQLite index outside the vault | No |

Planned features and their expected flow (to be confirmed when built):

- Local embeddings (M4): note text goes to a model running on this machine. Does not leave it.
- MCP server (M3): note excerpts are returned to the MCP client (Claude Code, Codex), which
  sends them to its model provider as part of the conversation.
- Optional API features (`zk ask`, API embeddings): opt-in only, documented here before use.

The repository itself contains no vault content; tests use the synthetic vault in
`tests/fixtures/vault`.
