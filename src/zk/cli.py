"""Command-line interface: index, check, search, show, eval, capture and serve.

The CLI holds no logic of its own; it parses arguments, calls the library and
prints the result. The MCP server (`zk serve`, zk/server.py) is a second door
into the same functions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from zk import capture, evaluate, lookup, vault
from zk.embed import DEFAULT_MODEL, FastEmbedder, update_embeddings
from zk.index import DEFAULT_EXCLUDES, build_index, default_db_path, health_report, open_db
from zk.search import DEFAULT_MODE, MODES, search

# Builds the embedder for --model; tests swap in a stand-in that needs no download.
make_embedder = FastEmbedder


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("vault", type=Path, help="path to the vault folder")
    parser.add_argument("--db", type=Path, help="index file (default: outside the vault)")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="skip files matching this vault-relative glob (repeatable); "
        f"always skipped: hidden folders, {', '.join(DEFAULT_EXCLUDES)}",
    )


def _db(args: argparse.Namespace) -> Path:
    if not args.vault.is_dir():
        sys.exit(f"zk: {args.vault} is not a folder")
    return args.db or default_db_path(args.vault)


def _open(args: argparse.Namespace):
    vault: Path = args.vault
    db = _db(args)
    conn = open_db(db)
    stats = build_index(conn, vault, (*DEFAULT_EXCLUDES, *args.exclude))
    print(
        f"indexed {stats.total} notes "
        f"({stats.added} added, {stats.updated} updated, {stats.removed} removed, "
        f"{stats.unchanged} unchanged) -> {db}",
        file=sys.stderr,
    )
    return conn


def _model_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"embedding model (default {DEFAULT_MODEL})"
    )


def _embedder(args: argparse.Namespace, conn):
    """The embedder for --model, with every note's vector brought up to date."""
    embedder = make_embedder(args.model)
    stats = update_embeddings(conn, embedder)
    print(
        f"embeddings ({embedder.model}): {stats.embedded} new, {stats.reused} reused, "
        f"{stats.removed} removed",
        file=sys.stderr,
    )
    return embedder


def cmd_index(args: argparse.Namespace) -> int:
    conn = _open(args)
    if args.embed:
        _embedder(args, conn)
    conn.close()
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    conn = _open(args)
    report = health_report(conn)
    conn.close()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report, limit=args.limit))
    return 1 if report.errors else 0


def cmd_search(args: argparse.Namespace) -> int:
    conn = _open(args)
    embedder = _embedder(args, conn) if args.mode != "keyword" else None
    hits = search(
        conn, args.query, k=args.k, type=args.type, tag=args.tag, mode=args.mode, embedder=embedder
    )
    conn.close()
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2))
        return 0
    if not hits:
        print("no matches")
    for h in hits:
        print(f"{h.rank:>2}. {h.title}  ({h.type or '-'}, score {h.score:.2f})")
        print(f"    {h.path}")
        if h.snippet:
            print(f"    {h.snippet}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    conn = _open(args)
    try:
        note = lookup.get_note(conn, args.note)
    except LookupError as exc:
        sys.exit(f"zk: {exc}")
    finally:
        conn.close()
    if args.json:
        print(json.dumps(note, indent=2))
        return 0
    print(f"{note['title']}  ({note['type'] or '-'}, id {note['id'] or '-'})")
    print(f"path: {note['path']}")
    if note["tags"]:
        print(f"tags: {', '.join(note['tags'])}")
    print(f"links to: {', '.join(n['title'] for n in note['links_out']) or '-'}")
    print(f"linked from: {', '.join(n['title'] for n in note['links_in']) or '-'}")
    if note["broken_links"]:
        print(f"broken links: {', '.join(note['broken_links'])}")
    print()
    print(note["body"].rstrip())
    if note["truncated"]:
        print("... (truncated)")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    """Save a session or a piece of text into the vault inbox. Never fails a hook."""
    if not args.vault.is_dir():
        sys.exit(f"zk: {args.vault} is not a folder")
    now = datetime.now().astimezone()
    captured = now.isoformat(timespec="seconds")
    try:
        if args.text is not None:
            source = re.sub(r"[^a-z0-9]+", "-", (args.source or "manual").lower()).strip("-")
            text = capture.render_text_capture(args.text, source or "manual", captured)
            name = f"inbox/{now:%Y-%m-%d-%H%M}-{source or 'manual'}.md"
        else:
            if args.hook:
                event = json.load(sys.stdin)
                transcript = Path(event["transcript_path"])
                session = re.sub(
                    r"[^A-Za-z0-9-]", "", str(event.get("session_id") or transcript.stem)
                )
                project = Path(event["cwd"]).name if event.get("cwd") else None
            else:
                transcript, session, project = args.transcript, args.transcript.stem, None
            with transcript.open(encoding="utf-8") as f:
                turns = capture.parse_transcript(f)
            keep, why = capture.worth_keeping(turns)
            if not keep:
                print(f"zk capture: skipped ({why})", file=sys.stderr)
                return 0
            text = capture.render_capture(turns, f"claude-code:{session}", captured, project)
            name = f"inbox/{now:%Y-%m-%d-%H%M}-claude-{session[:8]}.md"
        path = vault.write_new(args.vault, vault.unique_path(args.vault, name), text)
    except Exception as exc:  # a broken capture must never break the session that triggered it
        print(f"zk capture: failed: {exc}", file=sys.stderr)
        return 0 if args.hook else 1
    print(f"zk capture: saved {path}", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from zk.server import create_server  # imported here so other commands start fast

    db = _db(args)
    print(f"zk: serving {args.vault.resolve()} over stdio (read-only), index {db}", file=sys.stderr)
    excludes = (*DEFAULT_EXCLUDES, *args.exclude)
    create_server(args.vault, db, excludes, embedder=make_embedder(args.model)).run("stdio")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    queries = evaluate.load_queries(args.queries)
    modes = list(dict.fromkeys(args.mode or [DEFAULT_MODE]))
    conn = _open(args)
    embedder = _embedder(args, conn) if set(modes) - {"keyword"} else None
    reports = {
        m: evaluate.run_eval(conn, queries, depth=10, mode=m, embedder=embedder) for m in modes
    }
    conn.close()
    if args.json:
        out = {
            m: {
                "summary": r.summary(),
                "by_kind": {k: r.summary(k) for k in r.kinds()},
                "results": [q.__dict__ for q in r.results],
            }
            for m, r in reports.items()
        }
        print(json.dumps(out if len(modes) > 1 else out[modes[0]], indent=2))
    else:
        print("\n\n".join(f"== {m} ==\n{evaluate.format_report(r)}" for m, r in reports.items()))
    return 0


def format_report(report, limit: int = 20) -> str:
    def section(title: str, rows: list[str]) -> list[str]:
        if not rows:
            return []
        shown = rows if limit <= 0 else rows[:limit]
        out = ["", f"{title} ({len(rows)})"] + [f"  {r}" for r in shown]
        if len(shown) < len(rows):
            out.append(f"  ... {len(rows) - len(shown)} more (use --limit 0 for all)")
        return out

    types = ", ".join(f"{t} {n}" for t, n in report.by_type.items()) or "none"
    lines = [
        f"notes: {report.notes} ({types})",
        f"links: {report.links} ({len(report.broken_links)} broken, "
        f"{len(report.ambiguous_links)} ambiguous)",
        f"orphans (no inbound links): {len(report.orphans)}",
        f"no outgoing links: {len(report.no_outgoing)}",
        f"errors: {report.errors}   warnings: {report.warnings}",
    ]
    by_code: dict[str, int] = {}
    for issue in report.issues:
        key = f"{issue['level']}: {issue['code']}"
        by_code[key] = by_code.get(key, 0) + 1
    lines += section("issues by kind", [f"{n:>4}  {k}" for k, n in sorted(by_code.items())])
    lines += section(
        "duplicate ids", [f"{i}: {', '.join(paths)}" for i, paths in report.duplicate_ids.items()]
    )
    lines += section(
        "errors",
        [f"{i['path']}: {i['message']}" for i in report.issues if i["level"] == "error"],
    )
    lines += section(
        "broken links", [f"{b['path']} -> [[{b['target']}]]" for b in report.broken_links]
    )
    lines += section(
        "ambiguous links", [f"{b['path']} -> [[{b['target']}]]" for b in report.ambiguous_links]
    )
    lines += section("orphans", report.orphans)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zk", description="Zettelkasten index and tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build or refresh the index (reads the vault only)")
    _common(p_index)
    p_index.add_argument("--embed", action="store_true", help="also compute note embeddings")
    _model_option(p_index)
    p_index.set_defaults(func=cmd_index)

    p_check = sub.add_parser("check", help="index, then report the vault's health")
    _common(p_check)
    p_check.add_argument("--json", action="store_true", help="machine-readable output")
    p_check.add_argument("--limit", type=int, default=20, help="rows per list, 0 for all")
    p_check.set_defaults(func=cmd_check)

    p_search = sub.add_parser("search", help="keyword (BM25) or semantic search over the vault")
    _common(p_search)
    p_search.add_argument("query", help="free text")
    p_search.add_argument(
        "--mode", choices=MODES, default=DEFAULT_MODE, help=f"default {DEFAULT_MODE}"
    )
    _model_option(p_search)
    p_search.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    p_search.add_argument("--type", help="only notes of this type")
    p_search.add_argument("--tag", help="only notes with this tag")
    p_search.add_argument("--json", action="store_true", help="machine-readable output")
    p_search.set_defaults(func=cmd_search)

    p_show = sub.add_parser("show", help="print one note with its links (what get_note returns)")
    _common(p_show)
    p_show.add_argument("note", help="id, file name, vault path or title")
    p_show.add_argument("--json", action="store_true", help="machine-readable output")
    p_show.set_defaults(func=cmd_show)

    p_eval = sub.add_parser("eval", help="measure retrieval quality on a labelled query set")
    _common(p_eval)
    p_eval.add_argument("queries", type=Path, help="YAML query set (see zk/evaluate.py)")
    p_eval.add_argument(
        "--mode", action="append", choices=MODES, help="search mode to evaluate (repeatable)"
    )
    _model_option(p_eval)
    p_eval.add_argument("--json", action="store_true", help="machine-readable output")
    p_eval.set_defaults(func=cmd_eval)

    p_capture = sub.add_parser("capture", help="save a session or text into the vault inbox")
    p_capture.add_argument("vault", type=Path, help="path to the vault folder")
    src = p_capture.add_mutually_exclusive_group(required=True)
    src.add_argument("--hook", action="store_true", help="read a Claude Code hook event on stdin")
    src.add_argument("--transcript", type=Path, help="a Claude Code transcript (.jsonl)")
    src.add_argument("--text", help="free text to capture, e.g. from Codex or by hand")
    p_capture.add_argument("--source", help="label for --text captures (default manual)")
    p_capture.set_defaults(func=cmd_capture)

    p_serve = sub.add_parser("serve", help="run the read-only MCP server over stdio")
    _common(p_serve)
    _model_option(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
