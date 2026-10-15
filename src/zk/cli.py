"""Command-line interface: `zk index`, `zk check`, `zk search`, `zk eval`.

The CLI holds no logic of its own; it parses arguments, calls the library and
prints the result. The MCP server (milestone 3) will be a second door into the
same functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zk import evaluate
from zk.index import DEFAULT_EXCLUDES, build_index, default_db_path, health_report, open_db
from zk.search import search


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


def _open(args: argparse.Namespace):
    vault: Path = args.vault
    if not vault.is_dir():
        sys.exit(f"zk: {vault} is not a folder")
    db = args.db or default_db_path(vault)
    conn = open_db(db)
    stats = build_index(conn, vault, (*DEFAULT_EXCLUDES, *args.exclude))
    print(
        f"indexed {stats.total} notes "
        f"({stats.added} added, {stats.updated} updated, {stats.removed} removed, "
        f"{stats.unchanged} unchanged) -> {db}",
        file=sys.stderr,
    )
    return conn


def cmd_index(args: argparse.Namespace) -> int:
    _open(args).close()
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
    hits = search(conn, args.query, k=args.k, type=args.type, tag=args.tag)
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


def cmd_eval(args: argparse.Namespace) -> int:
    queries = evaluate.load_queries(args.queries)
    conn = _open(args)
    report = evaluate.run_eval(conn, queries, depth=10)
    conn.close()
    if args.json:
        print(
            json.dumps(
                {
                    "summary": report.summary(),
                    "by_kind": {k: report.summary(k) for k in report.kinds()},
                    "results": [r.__dict__ for r in report.results],
                },
                indent=2,
            )
        )
    else:
        print(evaluate.format_report(report))
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
    p_index.set_defaults(func=cmd_index)

    p_check = sub.add_parser("check", help="index, then report the vault's health")
    _common(p_check)
    p_check.add_argument("--json", action="store_true", help="machine-readable output")
    p_check.add_argument("--limit", type=int, default=20, help="rows per list, 0 for all")
    p_check.set_defaults(func=cmd_check)

    p_search = sub.add_parser("search", help="keyword search (BM25) over the vault")
    _common(p_search)
    p_search.add_argument("query", help="free text; words are OR-ed and ranked by BM25")
    p_search.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    p_search.add_argument("--type", help="only notes of this type")
    p_search.add_argument("--tag", help="only notes with this tag")
    p_search.add_argument("--json", action="store_true", help="machine-readable output")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="measure retrieval quality on a labelled query set")
    _common(p_eval)
    p_eval.add_argument("queries", type=Path, help="YAML query set (see zk/evaluate.py)")
    p_eval.add_argument("--json", action="store_true", help="machine-readable output")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
