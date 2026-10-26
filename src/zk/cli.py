"""Command-line interface: index, check, search, show, eval, capture and serve.

The CLI holds no logic of its own; it parses arguments, calls the library and
prints the result. The MCP server (`zk serve`, zk/server.py) is a second door
into the same functions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from zk import capture, checks, evaluate, lookup, migrate, vault
from zk.embed import DEFAULT_MODEL, FastEmbedder, update_embeddings
from zk.index import (
    DEFAULT_EXCLUDES,
    build_index,
    data_dir,
    default_db_path,
    health_report,
    open_db,
)
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
    long = checks.long_notes(conn)
    pairs = checks.near_duplicates(conn, _embedder(args, conn)) if args.duplicates else None
    conn.close()
    if args.json:
        out = {**report.to_dict(), "long_notes": [{"path": p, "tokens": n} for p, n in long]}
        if pairs is not None:
            out["near_duplicates"] = [{"a": a, "b": b, "similarity": s} for a, b, s in pairs]
        print(json.dumps(out, indent=2))
    else:
        print(format_report(report, limit=args.limit, long=long, pairs=pairs))
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
    if args.hook and os.environ.get(INGESTING):
        print("zk capture: skipped (this is zk's own ingest session)", file=sys.stderr)
        return 0
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
    if args.then:
        log = _start_ingest(args.then, args.vault, now)
        print(f"zk capture: started ingest, log {log}", file=sys.stderr)
    return 0


INGESTING = "ZK_INGESTING"  # set for the agent `--then` starts, so its own session isn't captured


def _start_ingest(command: str, vault_dir: Path, now: datetime) -> Path:
    """Run `command` in the background, detached from the session that is ending."""
    logs = data_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"ingest-{now:%Y-%m-%d-%H%M%S}.log"
    detach = (
        {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with log.open("ab") as out:
        subprocess.Popen(  # the hook's own command line, from the user's settings
            command,
            shell=True,
            cwd=vault_dir,
            env={**os.environ, INGESTING: "1"},
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            **detach,
        )
    return log


def cmd_import(args: argparse.Namespace) -> int:
    """Import a legacy vault's notes. Reads the legacy folder only; --dry-run writes nothing."""
    if not args.legacy.is_dir():
        sys.exit(f"zk: {args.legacy} is not a folder")
    if args.dry_run:
        _db(args)  # checks the vault folder exists
        report = migrate.dry_run(args.legacy, args.vault, make_embedder(args.model), args.queries)
        print(migrate.format_report(report, args.legacy))
        print("dry run: nothing was written", file=sys.stderr)
        return 0
    conn = _open(args)
    embedder = _embedder(args, conn)
    try:
        report = migrate.import_legacy(args.legacy, args.vault, conn, embedder, args.queries)
    except (migrate.InvalidProposal, ValueError) as exc:
        sys.exit(f"zk: import stopped, nothing written: {exc}")
    finally:
        conn.close()
    print(migrate.summary(report))
    print(f"report: {migrate.REPORTS_DIR}/ in the vault; commit: {report.commit or '-'}")
    if report.commit_error:
        print(f"zk: {report.commit_error}", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from zk.server import create_server  # imported here so other commands start fast

    db = _db(args)
    access = "with ingest tools" if args.allow_writes else "read-only"
    print(f"zk: serving {args.vault.resolve()} over stdio ({access}), index {db}", file=sys.stderr)
    excludes = (*DEFAULT_EXCLUDES, *args.exclude)
    server = create_server(
        args.vault,
        db,
        excludes,
        embedder=make_embedder(args.model),
        allow_writes=args.allow_writes,
    )
    server.run("stdio")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.queries is not None:
        queries = evaluate.load_queries(args.queries)
    else:
        queries = checks.load_vault_queries(args.vault)
        if not queries:
            sys.exit(f"zk: no query file given and no test questions in {args.vault / 'eval'}")
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


def format_report(report, limit: int = 20, long=(), pairs=None) -> str:
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
    lines += section(
        f"over about {checks.MAX_TOKENS} tokens (the end is not embedded)",
        [f"{p} (~{n})" for p, n in long],
    )
    if pairs is not None:
        lines += section(
            f"near-duplicates (similarity >= {checks.DUPLICATE_AT})",
            [f"{s:.2f}  {a}  ~  {b}" for a, b, s in pairs],
        )
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
    p_check.add_argument(
        "--duplicates", action="store_true", help="also list near-duplicate notes (embeds them)"
    )
    _model_option(p_check)
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
    p_eval.add_argument(
        "queries",
        type=Path,
        nargs="?",
        help="YAML query set (see zk/evaluate.py); default: the questions saved in VAULT/eval/",
    )
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
    p_capture.add_argument(
        "--then",
        metavar="COMMAND",
        help="after saving a capture, run COMMAND in the background (e.g. a headless agent "
        "that ingests the inbox); it runs with ZK_INGESTING=1 so its own session is not captured",
    )
    p_capture.set_defaults(func=cmd_capture)

    p_import = sub.add_parser("import", help="import a legacy vault's notes (reads it only)")
    p_import.add_argument(
        "legacy", type=Path, help="the legacy vault folder, which is never changed"
    )
    _common(p_import)
    _model_option(p_import)
    p_import.add_argument("--queries", type=Path, help="a legacy query set to carry over to eval/")
    p_import.add_argument(
        "--dry-run", action="store_true", help="import into a throwaway copy and print the report"
    )
    p_import.set_defaults(func=cmd_import)

    p_serve = sub.add_parser("serve", help="run the read-only MCP server over stdio")
    _common(p_serve)
    _model_option(p_serve)
    p_serve.add_argument(
        "--allow-writes",
        action="store_true",
        help="also offer the ingest tools, which create files in the vault (never overwrite)",
    )
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
