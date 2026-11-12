"""Command-line interface: index, check, search, eval, capture, process, hooks and serve.

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

from zk import (
    agent,
    capture,
    checks,
    evaluate,
    hooks,
    lookup,
    maintain,
    migrate,
    process,
    sessions,
    vault,
)
from zk.embed import DEFAULT_MODEL, FastEmbedder, update_embeddings
from zk.hook import read_event
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
# Builds the headless model for answering and judging; tests swap in a stand-in.
make_ask = agent.claude


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
    report.waiting_links, report.broken_links = checks.split_waiting(
        args.vault, report.broken_links
    )
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


def cmd_suggest(args: argparse.Namespace) -> int:
    conn = _open(args)
    embedder = _embedder(args, conn)
    report = {
        "links": maintain.link_suggestions(conn, embedder),
        "orphans": maintain.orphan_report(conn, embedder),
        "maps": [c for c in maintain.clusters(conn, embedder) if c["map"] is None],
    }
    conn.close()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    limit = args.limit

    def section(title: str, rows: list[str]) -> None:
        shown = rows if limit <= 0 else rows[:limit]
        print(f"\n{title} ({len(rows)})")
        for r in shown:
            print(f"  {r}")
        if len(shown) < len(rows):
            print(f"  ... {len(rows) - len(shown)} more (use --limit 0 for all)")

    section(
        "links to consider (similar, not linked)",
        [f"{x['similarity']:.2f}  {x['a']}  <->  {x['b']}" for x in report["links"]],
    )
    section(
        "orphans (nothing links to them) and notes that might",
        [
            f"{o['path']}  <-  "
            + (
                ", ".join(f"{c['path']} ({c['similarity']:.2f})" for c in o["candidates"])
                or "no likely parent"
            )
            for o in report["orphans"]
        ],
    )
    section(
        "clusters with no map note",
        [
            f"{len(c['notes'])} notes"
            + (f" [{', '.join(c['shared_tags'])}]" if c["shared_tags"] else "")
            + f", cohesion {c['cohesion']:.2f}: {', '.join(c['notes'])}"
            + (
                f" (closest to a map: {c['closest_to_map']['path']}, links to "
                f"{c['closest_to_map']['links_to']})"
                if c["closest_to_map"]["links_to"]
                else ""
            )
            for c in report["maps"]
        ],
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    conn = _open(args)
    embedder = _embedder(args, conn) if args.mode != "keyword" else None
    hits = search(
        conn,
        args.query,
        k=args.k,
        type=args.type,
        tag=args.tag,
        mode=args.mode,
        embedder=embedder,
        project=args.project,
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
    if args.catch_up:
        found = sessions.catch_up(args.vault, args.days, dry_run=args.dry_run)
        for f in found:
            if f["saved"] or args.verbose:
                print(f"{f['saved'] or 'skipped'}  {f['session']}  ({f['why']})")
        saved = sum(1 for f in found if f["saved"])
        verb = "would capture" if args.dry_run else "captured"
        print(f"zk capture: {verb} {saved} of {len(found)} recent sessions", file=sys.stderr)
        return 0
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
            path = vault.write_new(args.vault, vault.unique_path(args.vault, name), text)
        else:
            if args.hook:
                saved, why = sessions.from_hook(args.vault, read_event(), now)
            else:
                saved, why = sessions.capture_session(
                    args.vault, sessions.describe(args.transcript), now
                )
            if not saved:
                print(f"zk capture: skipped ({why})", file=sys.stderr)
                return 0
            path = saved
    except Exception as exc:  # a broken capture must never break the session that triggered it
        print(f"zk capture: failed: {exc}", file=sys.stderr)
        return 0 if args.hook else 1
    print(f"zk capture: saved {path}", file=sys.stderr)
    if args.then:
        log = _start_ingest(args.then, args.vault, now)
        print(f"zk capture: started ingest, log {log}", file=sys.stderr)
    return 0


INGESTING = sessions.INGESTING  # set for the agent `--then` starts, so its session isn't captured


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


def cmd_process(args: argparse.Namespace) -> int:
    """Ingest the inbox and resolve held notes with headless agents, when there is work."""
    if not args.vault.is_dir():
        sys.exit(f"zk: {args.vault} is not a folder")
    commands = process.default_commands(args.resolve_model, args.vault)
    if args.ingest_command:
        commands["ingest"] = args.ingest_command
    if args.resolve_command:
        commands["resolve"] = args.resolve_command
    try:
        summary = process.run(args.vault, args.catch_up, args.days, commands)
    except process.Busy as exc:
        print(f"zk process: skipped, {exc}; it will pick up new captures", file=sys.stderr)
        return 0
    print(
        f"zk process: {summary['caught_up']} caught up, {summary['ingest_runs']} ingest and "
        f"{summary['resolve_runs']} resolve run(s); log {summary['log']}",
        file=sys.stderr,
    )
    if summary.get("reason") == "usage limit":
        print("zk process: stopped at Claude's usage limit; the next run picks up", file=sys.stderr)
        return 1
    if summary["stalled"]:
        print(
            f"zk process: the ingest agent processed none of {len(summary['stalled'])} "
            "capture(s); see the log",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_hooks(args: argparse.Namespace) -> int:
    """Install or show zk's Claude Code hooks and rules file."""
    if args.action == "status":
        return _hooks_status(args.vault)
    if args.vault is None or not args.vault.is_dir():
        sys.exit("zk hooks install: give the vault folder")
    try:
        result = hooks.install(args.vault, not args.no_process, args.dry_run, not args.no_recall)
    except ValueError as exc:
        print(f"zk hooks: {exc}", file=sys.stderr)
        return 1
    print(f"claude: {result['action']} in {result['path']}")
    for event in result["added"]:
        print(f"  {'would add' if args.dry_run else 'added'} the {event} hook")
    for event in result["existing"]:
        print(f"  {event}: zk's hook is already there")
    if result.get("rules"):
        print(f"  {'would add' if args.dry_run else 'added'} the rules file {result['rules']}")
    if args.dry_run and result.get("diff"):
        print(result["diff"])
    if result.get("backup"):
        print(f"  backup: {result['backup']}")
    _warn_if_no_server(args.vault)
    return 0


def _hooks_status(vault_dir: Path | None) -> int:
    try:
        found = hooks.installed()
    except ValueError as exc:
        found, note = {}, str(exc)
    else:
        note = "installed" if found else "not installed"
    print(f"claude hooks: {note} ({hooks.settings_path()})")
    for event, commands in found.items():
        for command in commands:
            print(f"  {event}: {command}")
    rules = hooks.rules_path()
    print(f"rules file: {'installed' if rules.exists() else 'not installed'} ({rules})")
    for name, label in ((hooks.TASK_NAME, "daily catch-up"), (hooks.REVIEW_TASK, "weekly review")):
        print(f"{label}: {'installed' if hooks.schedule_status(name) else 'not installed'}")
    ready, detail = hooks.server_ready()
    print(f"zk server for everyday sessions: {detail}")
    record = sessions.load_record()
    captures = sorted(
        (v.get("checked", ""), v["capture"]) for v in record.values() if v.get("capture")
    )
    print(f"sessions seen: {len(record)}, captured: {len(captures)}")
    if captures:
        print(f"  last capture: {captures[-1][1]} at {captures[-1][0]}")
    logs = sorted((data_dir() / "logs").glob("process-*.log"))
    if logs:
        print(f"last process run: {logs[-1]}")
        tail = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-1:]
        for line in tail:
            print(f"  {line}")
    if process.lock_path().exists():
        print(f"running now (lock {process.lock_path()})")
    if vault_dir and vault_dir.is_dir():
        from zk.ingest import list_held, list_inbox

        print(f"inbox: {len(list_inbox(vault_dir))} capture(s), {len(list_held(vault_dir))} held")
    return 0


def _warn_if_no_server(vault_dir: Path) -> None:
    ready, detail = hooks.server_ready()
    if ready is False:
        print(f"  warning: {detail}. Register a read-only one for everyday sessions with:")
        if "--allow-writes" in detail:
            print("  claude mcp remove zk --scope user")
        print(f"  {hooks.server_command(vault_dir)}")


def cmd_schedule(args: argparse.Namespace) -> int:
    """Install, show or remove the daily catch-up task (Windows Task Scheduler)."""
    if args.action == "status":
        for name in (hooks.TASK_NAME, hooks.REVIEW_TASK):
            print(hooks.schedule_status(name) or f"zk schedule: {name!r} not installed")
        return 0
    if args.action == "remove":
        removed = hooks.remove_schedule()
        print(f"zk schedule: {'removed' if removed else 'nothing to remove'}", file=sys.stderr)
        return 0
    if args.vault is None or not args.vault.is_dir():
        sys.exit("zk schedule install: give the vault folder")
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", args.at):
        sys.exit("zk schedule install: --at takes a time like 21:00")
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", args.review_at):
        sys.exit("zk schedule install: --review-at takes a time like 20:00")
    review_on = None if args.review_day == "never" else args.review_day
    result = hooks.install_schedule(
        args.vault, args.at, data_dir(), args.dry_run, review_on, args.review_at
    )
    if result["action"] == "added":
        print(f"zk schedule: added the task {result['task']!r}, daily at {result['at']}")
        if result["review"]:
            print(f"zk schedule: added the task {hooks.REVIEW_TASK!r}, {result['review']}")
    elif result["action"] == "would add":
        print(result["xml"])
    else:
        print("zk schedule: Task Scheduler is Windows only. A cron line that does the same:")
        print(result["cron"])
    return 0


def cmd_maps(args: argparse.Namespace) -> int:
    """Write map notes for clusters and projects; rewrite them when their notes change."""
    from zk import maps

    conn = _open(args)
    embedder = _embedder(args, conn)
    try:
        result = maps.update(args.vault, conn, embedder, dry_run=args.dry_run)
        if not args.dry_run:
            build_index(conn, args.vault, (*DEFAULT_EXCLUDES, *args.exclude))
    finally:
        conn.close()
    print(maps.format_update(result))
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    """Print the session-start brief, and roughly what it costs."""
    from zk import recall

    conn = _open(args)
    try:
        text = recall.brief(conn, args.cwd)
    finally:
        conn.close()
    print(text or "(nothing: the vault has no notes)")
    print(f"\nabout {checks.estimate_tokens(text)} tokens", file=sys.stderr)
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Read each project's structure into an inbox capture, for the next ingest."""
    from zk import learn

    if not args.vault.is_dir():
        sys.exit(f"zk: {args.vault} is not a folder")
    for folder in args.folders:
        if args.skip:
            learn.skip(folder)
            print(f"zk learn: {folder} will not be read", file=sys.stderr)
            continue
        try:
            result = learn.learn(args.vault, folder, dry_run=args.dry_run)
        except FileNotFoundError as exc:
            print(f"zk learn: {exc}", file=sys.stderr)
            continue
        if args.dry_run and result.get("text"):
            print(result["text"])
        print(
            f"zk learn: {result['project']}: {result['saved'] or 'nothing saved'} "
            f"({result['why']})",
            file=sys.stderr,
        )
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    """Copy a project's docs into the vault, one note per section; re-sync what changed."""
    from zk import docs

    if not args.vault.is_dir():
        sys.exit(f"zk: {args.vault} is not a folder")
    if args.forget:
        found = docs.forget(args.folder)
        print(f"zk docs: {'stopped syncing' if found else 'was not syncing'} {args.folder}")
        return 0
    try:
        result = docs.import_docs(
            args.vault,
            args.folder,
            include=args.only or None,
            exclude=args.skip if args.skip else None,
            pdftotext=args.pdftotext,
            project=args.project,
            dry_run=args.dry_run,
            code=False if args.no_code else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"zk docs: {exc}", file=sys.stderr)
        return 1
    if not args.dry_run and (result["new"] or result["updated"] or result["retired"]):
        conn = _open(args)  # index and embed the sections now, so search finds them
        try:
            _embedder(args, conn)
        finally:
            conn.close()
    if not args.quiet:
        print(docs.format_result(result))
        if args.dry_run:
            for rel in result["new"] + result["updated"]:
                print(f"  {rel}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """The health review: write it to inbox/reports/ (or just print it)."""
    from zk import review

    conn = _open(args)
    embedder = _embedder(args, conn)
    try:
        result = review.run(args.vault, conn, embedder, write=not args.print)
    finally:
        conn.close()
    print(result["text"])
    if result.get("file"):
        print(f"zk review: saved {result['file']}", file=sys.stderr)
    return 0


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
    from zk.server import run_stdio

    run_stdio(server)
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
    if args.answers:
        from zk.answer import format_answers, run_answers

        picked = (
            queries[:: max(1, len(queries) // args.limit)][: args.limit] if args.limit else queries
        )
        mode = modes[0]
        embedder = embedder or (_embedder(args, conn) if mode != "keyword" else None)
        try:
            report = run_answers(conn, picked, make_ask(args.claude_model), embedder, mode=mode)
        except agent.AgentError as exc:
            sys.exit(f"zk eval --answers: {exc}")
        finally:
            conn.close()
        print(json.dumps(report, indent=2) if args.json else format_answers(report))
        return 0
    if args.context:
        from zk.context import format_context, run_context

        found = {m: run_context(conn, queries, m, embedder) for m in modes}
        conn.close()
        if args.json:
            print(json.dumps(found if len(modes) > 1 else found[modes[0]], indent=2))
        else:
            print("\n\n".join(f"== {m} ==\n{format_context(r)}" for m, r in found.items()))
        return 0
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


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer one question from the notes (sends it and the top notes to Anthropic)."""
    from zk.answer import ask

    conn = _open(args)
    embedder = _embedder(args, conn)
    try:
        answer = ask(conn, args.question, make_ask(args.claude_model), embedder, k=args.k)
    except agent.AgentError as exc:
        sys.exit(f"zk ask: {exc}")
    finally:
        conn.close()
    print(answer.text)
    print(
        f"\n(from {', '.join(answer.given) or 'no notes'}; {answer.context_tokens} tokens)",
        file=sys.stderr,
    )
    if not answer.grounded:
        print("zk ask: warning, it cited notes it wasn't given", file=sys.stderr)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Judge note pairs near each similarity cut and report precision per band."""
    from zk import calibrate

    conn = _open(args)
    embedder = _embedder(args, conn)
    try:
        result = calibrate.run(
            args.vault,
            conn,
            embedder,
            None if args.no_judge else make_ask(args.claude_model),
            per_band=args.per_band,
        )
    except agent.AgentError as exc:
        sys.exit(f"zk calibrate: {exc}")
    finally:
        conn.close()
    print(json.dumps(result, indent=2) if args.json else calibrate.format_calibration(result))
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
        f"{len(report.ambiguous_links)} ambiguous"
        + (f", {len(report.waiting_links)} waiting on a held note" if report.waiting_links else "")
        + ")",
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
        "waiting on a held note (they resolve if apply_held releases it)",
        [f"{b['path']} -> [[{b['target']}]]" for b in report.waiting_links],
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

    p_suggest = sub.add_parser(
        "suggest", help="suggest links, parents for orphans, and map notes (changes nothing)"
    )
    _common(p_suggest)
    p_suggest.add_argument("--json", action="store_true", help="machine-readable output")
    p_suggest.add_argument("--limit", type=int, default=20, help="rows per list, 0 for all")
    _model_option(p_suggest)
    p_suggest.set_defaults(func=cmd_suggest)

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
    p_search.add_argument("--project", help="only notes recorded from this project")
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
    p_eval.add_argument(
        "--context",
        action="store_true",
        help="measure context tokens per question (zk tools vs naive RAG vs the whole vault)",
    )
    p_eval.add_argument(
        "--answers",
        action="store_true",
        help="answer questions from the notes with headless Claude and score the citations "
        "(sends the questions and retrieved notes to Anthropic)",
    )
    p_eval.add_argument(
        "--limit", type=int, default=20, help="--answers: questions to ask, spread evenly (0 = all)"
    )
    p_eval.add_argument("--claude-model", help="--answers: Claude model (default: your default)")
    p_eval.add_argument("--json", action="store_true", help="machine-readable output")
    p_eval.set_defaults(func=cmd_eval)

    p_ask = sub.add_parser("ask", help="answer a question from the notes, citing them")
    _common(p_ask)
    _model_option(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=5, help="notes to give the model (default 5)")
    p_ask.add_argument("--claude-model", help="Claude model (default: your default)")
    p_ask.set_defaults(func=cmd_ask)

    p_cal = sub.add_parser(
        "calibrate", help="re-check the similarity thresholds on pairs judged by Claude"
    )
    _common(p_cal)
    _model_option(p_cal)
    p_cal.add_argument(
        "--per-band", type=int, default=6, help="pairs per band on each side of a cut (default 6)"
    )
    p_cal.add_argument("--claude-model", help="Claude model for judging (default: your default)")
    p_cal.add_argument(
        "--no-judge", action="store_true", help="only reuse saved labels; send nothing"
    )
    p_cal.add_argument("--json", action="store_true", help="machine-readable output")
    p_cal.set_defaults(func=cmd_calibrate)

    p_capture = sub.add_parser("capture", help="save a session or text into the vault inbox")
    p_capture.add_argument("vault", type=Path, help="path to the vault folder")
    src = p_capture.add_mutually_exclusive_group(required=True)
    src.add_argument("--hook", action="store_true", help="read a Claude Code hook event on stdin")
    src.add_argument("--transcript", type=Path, help="a Claude Code transcript (.jsonl)")
    src.add_argument("--text", help="free text to capture, e.g. from Codex or by hand")
    src.add_argument(
        "--catch-up",
        action="store_true",
        help="capture recent Claude Code sessions that no hook captured",
    )
    p_capture.add_argument("--source", help="label for --text captures (default manual)")
    p_capture.add_argument(
        "--days", type=float, default=3, help="--catch-up: how far back to look (default 3)"
    )
    p_capture.add_argument(
        "--dry-run", action="store_true", help="--catch-up: list what would be captured"
    )
    p_capture.add_argument(
        "--verbose", action="store_true", help="--catch-up: also list skipped sessions"
    )
    p_capture.add_argument(
        "--then",
        metavar="COMMAND",
        help="after saving a capture, run COMMAND in the background (e.g. a headless agent "
        "that ingests the inbox); it runs with ZK_INGESTING=1 so its own session is not captured",
    )
    p_capture.set_defaults(func=cmd_capture)

    p_process = sub.add_parser(
        "process", help="ingest the inbox, then resolve held notes, with headless agents"
    )
    p_process.add_argument("vault", type=Path, help="path to the vault folder")
    p_process.add_argument(
        "--catch-up", action="store_true", help="first capture sessions no hook captured"
    )
    p_process.add_argument("--days", type=float, default=3, help="--catch-up window (default 3)")
    p_process.add_argument(
        "--resolve-model", default="opus", help="model for resolving held notes (default opus)"
    )
    p_process.add_argument("--ingest-command", help="run this instead of the default ingest agent")
    p_process.add_argument(
        "--resolve-command", help="run this instead of the default resolve agent"
    )
    p_process.set_defaults(func=cmd_process)

    p_hooks = sub.add_parser("hooks", help="install or show zk's Claude Code hooks")
    p_hooks.add_argument("action", choices=["install", "status"])
    p_hooks.add_argument("vault", type=Path, nargs="?", help="path to the vault folder")
    p_hooks.add_argument(
        "--no-process",
        action="store_true",
        help="only capture; don't start `zk process` after each session",
    )
    p_hooks.add_argument(
        "--no-recall",
        action="store_true",
        help="only the session-end capture: no brief, nudge, error notes or rules file",
    )
    p_hooks.add_argument("--dry-run", action="store_true", help="show the change, make none")
    p_hooks.set_defaults(func=cmd_hooks)

    p_schedule = sub.add_parser(
        "schedule", help="the daily catch-up and weekly review tasks (Windows)"
    )
    p_schedule.add_argument("action", choices=["install", "status", "remove"])
    p_schedule.add_argument("vault", type=Path, nargs="?", help="path to the vault folder")
    p_schedule.add_argument("--at", default="21:00", help="time of day, HH:MM (default 21:00)")
    p_schedule.add_argument(
        "--review-day",
        default="fri",
        choices=[*hooks.DAYS, "never"],
        help="day of the weekly health review (default fri; never: no review task)",
    )
    p_schedule.add_argument(
        "--review-at", default="12:00", help="time of the weekly review, HH:MM (default 12:00)"
    )
    p_schedule.add_argument("--dry-run", action="store_true", help="print the task, add nothing")
    p_schedule.set_defaults(func=cmd_schedule)

    p_maps = sub.add_parser("maps", help="write or update the map notes for topics and projects")
    _common(p_maps)
    _model_option(p_maps)
    p_maps.add_argument("--dry-run", action="store_true", help="list the changes, write nothing")
    p_maps.set_defaults(func=cmd_maps)

    p_brief = sub.add_parser("brief", help="print what a session is told about the vault")
    _common(p_brief)
    p_brief.add_argument("--cwd", type=Path, help="as if the session started in this folder")
    p_brief.set_defaults(func=cmd_brief)

    p_learn = sub.add_parser(
        "learn", help="read a project's structure into an inbox capture (no model)"
    )
    p_learn.add_argument("vault", type=Path, help="path to the vault folder")
    p_learn.add_argument("folders", type=Path, nargs="+", help="project folders to read")
    p_learn.add_argument("--dry-run", action="store_true", help="print the capture, save nothing")
    p_learn.add_argument(
        "--skip", action="store_true", help="never read these projects (also not automatically)"
    )
    p_learn.set_defaults(func=cmd_learn)

    p_docs = sub.add_parser(
        "docs", help="copy a project's docs and code maps into the vault (no model)"
    )
    _common(p_docs)
    _model_option(p_docs)
    p_docs.add_argument("folder", type=Path, help="the project folder")
    p_docs.add_argument(
        "--only", action="append", default=[], help="doc files to read (glob; default *.md, *.pdf)"
    )
    p_docs.add_argument(
        "--skip",
        action="append",
        default=[],
        help="doc files to leave out (glob; with a /, it matches the path: docs/uml/*.pdf)",
    )
    p_docs.add_argument("--pdftotext", help="path to Poppler's pdftotext, to read PDFs")
    p_docs.add_argument("--project", help="project name (default: the folder's name)")
    p_docs.add_argument("--dry-run", action="store_true", help="list the changes, write nothing")
    p_docs.add_argument(
        "--no-code", action="store_true", help="no code maps of the source files, docs only"
    )
    p_docs.add_argument("--forget", action="store_true", help="stop syncing this project's docs")
    p_docs.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    p_docs.set_defaults(func=cmd_docs)

    p_review = sub.add_parser("review", help="the vault health review (no model)")
    _common(p_review)
    _model_option(p_review)
    p_review.add_argument("--print", action="store_true", help="print it, write no report")
    p_review.set_defaults(func=cmd_review)

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
