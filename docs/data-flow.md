# Data flow

Vault content is private. This file lists every place it goes. Update it whenever a feature
sends vault content somewhere new.

| Feature | Reads | Writes | Leaves this machine? |
| --- | --- | --- | --- |
| `zk index`, `zk check`, `zk search`, `zk show`, `zk eval` | Markdown files in the vault | SQLite index outside the vault | No |
| Semantic search, `zk index --embed` (M4) | Note text from the index | Vectors in the SQLite index | No: the model runs locally. Only the model weights are downloaded, from huggingface.co, into the zk data folder |
| `zk serve` (MCP, M3) | The vault (to refresh the index) and the index | SQLite index outside the vault | Yes, the parts an agent asks for: see below |

### MCP server

The server makes no network calls. It answers the MCP client that started it (Claude Code,
Codex) over stdin/stdout. The client puts each tool result into the conversation, and the
conversation is sent to that client's model provider (Anthropic, OpenAI) under the client's own
data terms. What leaves the machine is therefore what tools return:

- `search_notes`: up to 25 hits, each with path, id, title, type, summary and a short snippet.
- `get_note`: one note's metadata and body (capped at 20,000 characters) and the titles of linked notes.
- `get_related`: up to 50 neighbouring notes' paths, titles and summaries.

The whole vault is never sent at once. Notes you would not want sent to a model provider should be
kept out of the served vault, for example with `--exclude`.

Planned features and their expected flow (to be confirmed when built):

- Optional API features (`zk ask`, API embeddings): opt-in only, documented here before use.

The repository itself contains no vault content; tests use the synthetic vault in
`tests/fixtures/vault`.
