"""Read one of the person's own projects for free, so its notes cost one ingest: `zk learn`.

Pass 1, here, uses no model. It reads the folder tree, the README's opening,
the manifests (what the project is built with), recent commit subjects (the
problems it hit), and, for each source file, the names of its classes and
functions (Python through its syntax tree, other languages by their declaration
keywords). It never sends file contents beyond those names, and the text is
redacted like any capture. Each file's hash is remembered, so a later run lists
only what changed, and an unchanged project writes nothing.

The result is a capture in inbox/ (source repo:<name>, project <name>). Pass 2 is
the ordinary ingest: the next `zk process`, which a session's end already
starts, turns it into notes on how the project is built. So learning a project
adds no model call of its own.

A session that starts in a repository zk hasn't seen starts this in the
background (see hook.py). `zk learn --skip FOLDER` keeps a project out for good.
The record of what was read lives in the zk data folder, never in the project.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from zk import capture, vault
from zk.index import data_dir
from zk.search import project_slug

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "bin", "obj", "build", "dist", "out", "target",
    ".venv", "venv", "env", "__pycache__", ".idea", ".vs", ".vscode", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "packages", "vendor", "third_party", ".next", ".gradle",
    "coverage", "site-packages", "Debug", "Release", "x64", "logs", ".tox", "install", "log",
}  # fmt: skip
# Never read, even for their names' sake: likely to hold secrets.
SECRET_FILES = (
    ".env*", "*.pem", "*.key", "*.pfx", "*.p12", "id_rsa*", "id_ed25519*", "*credentials*",
    "*secret*", "*.keystore", "appsettings*.json", "*.tfvars",
)  # fmt: skip
MANIFESTS = (
    "pyproject.toml", "requirements*.txt", "setup.py", "setup.cfg", "environment.yml",
    "package.json", "Cargo.toml", "go.mod", "CMakeLists.txt", "package.xml", "pom.xml",
    "build.gradle*", "Gemfile", "*.csproj", "*.sln", "Dockerfile", "docker-compose*.yml",
    "Makefile", "*.launch.py",
)  # fmt: skip
CODE = {
    ".py": "python", ".cs": "c", ".java": "c", ".kt": "c", ".ts": "c", ".tsx": "c",
    ".js": "c", ".jsx": "c", ".cpp": "c", ".cc": "c", ".c": "c", ".h": "c", ".hpp": "c",
    ".swift": "c", ".go": "go", ".rs": "rust", ".ipynb": "notebook",
}  # fmt: skip
README_CHARS = 3000
MANIFEST_CHARS = 1500
MAX_FILES = 4000  # files walked
MAX_LISTED = 300  # source files listed with their names
NAMES_PER_FILE = 15
MAX_FILE_BYTES = 1_000_000
COMMITS = 30

_C_TYPES = re.compile(r"\b(?:class|interface|struct|enum|record|trait)\s+([A-Z]\w*)")
_C_FUNCS = re.compile(r"\bfunction\s+([A-Za-z_]\w*)|\b(?:def|fun|func)\s+([A-Za-z_]\w*)\s*\(")
_GO = re.compile(r"^\s*(?:func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)|type\s+([A-Z]\w*))", re.M)
_RUST = re.compile(r"\b(?:fn|struct|enum|trait)\s+([A-Za-z_]\w*)")


def record_path() -> Path:
    return data_dir() / "learn.json"


def load_record() -> dict:
    try:
        data = json.loads(record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("repos", {})
    data.setdefault("skip", [])
    return data


def save_record(record: dict) -> None:
    path = record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _key(root: Path) -> str:
    return str(root.resolve()).lower() if os.name == "nt" else str(root.resolve())


def is_known(root: Path) -> bool:
    """Whether a project was learned already, or kept out with --skip."""
    record = load_record()
    key = _key(root)
    return key in record["repos"] or key in record["skip"]


def skip(root: Path) -> None:
    record = load_record()
    record["skip"] = sorted({*record["skip"], _key(root)})
    save_record(record)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns)


def walk(root: Path) -> list[Path]:
    """The project's files, without build output, dependencies or likely secrets."""
    out: list[Path] = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if _matches(name, SECRET_FILES):
                continue
            out.append(Path(folder) / name)
            if len(out) >= MAX_FILES:
                return out
    return out


def names_in(path: Path, text: str) -> list[str]:
    """Class and function names declared in a source file."""
    kind = CODE.get(path.suffix.lower())
    found: list[str] = []
    if kind == "python":
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                found.append(f"class {node.name}")
                found += [
                    f"{node.name}.{n.name}"
                    for n in node.body
                    if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                    and not n.name.startswith("__")
                ]
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                found.append(f"def {node.name}")
    elif kind == "c":
        found = [f"type {m}" for m in _C_TYPES.findall(text)]
        found += [a or b for a, b in _C_FUNCS.findall(text)]
    elif kind == "go":
        found = [a or f"type {b}" for a, b in _GO.findall(text)]
    elif kind == "rust":
        found = _RUST.findall(text)
    elif kind == "notebook":
        try:
            cells = json.loads(text).get("cells", [])
        except ValueError:
            return []
        source = "\n".join(
            "".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"
        )
        return names_in(path.with_suffix(".py"), source)
    return list(dict.fromkeys(found))[:NAMES_PER_FILE]


def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _tree(root: Path, files: list[Path], depth: int = 2) -> str:
    """Folders to `depth`, with how many files each holds."""
    counts: dict[str, int] = {}
    for f in files:
        parts = f.relative_to(root).parts[:-1][:depth]
        for i in range(1, len(parts) + 1):
            key = "/".join(parts[:i])
            counts[key] = counts.get(key, 0) + 1
    return "\n".join(
        f"{'  ' * k.count('/')}{k.rsplit('/', 1)[-1]}/ ({n})" for k, n in sorted(counts.items())
    )


def _commits(root: Path) -> list[str]:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "log", f"-n{COMMITS}", "--format=%s"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [s for s in done.stdout.splitlines() if s.strip()] if done.returncode == 0 else []


def scan(root: Path, previous: dict[str, str] | None = None) -> tuple[str, dict[str, str], int]:
    """The structural summary of a project, its file hashes, and how many files changed.

    With `previous` hashes, only changed and new source files are listed.
    """
    root = root.resolve()
    files = walk(root)
    hashes: dict[str, str] = {}
    texts: dict[str, str] = {}
    for f in files:
        text = _read(f)
        if text is None:
            continue
        rel = f.relative_to(root).as_posix()
        hashes[rel] = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        texts[rel] = text
    changed = [r for r in hashes if (previous or {}).get(r) != hashes[r]]
    removed = sorted(set(previous or {}) - set(hashes))
    parts = [f"Project: {root.name}"]
    if previous:
        parts.append(f"Changes since zk last read it: {len(changed)} files changed or added")
        if removed:
            parts.append("Removed: " + ", ".join(removed[:50]))
    else:
        parts.append(f"Folders (files in each):\n{_tree(root, files)}")
        readme = next(
            (r for r in texts if r.lower() in ("readme.md", "readme.rst", "readme.txt", "readme")),
            None,
        )
        if readme:
            parts.append(f"README (start):\n{texts[readme][:README_CHARS].strip()}")
    manifests = [r for r in changed if _matches(r.rsplit("/", 1)[-1], MANIFESTS)]
    for rel in manifests[:12]:
        parts.append(f"Manifest {rel}:\n{texts[rel][:MANIFEST_CHARS].strip()}")
    listed = []
    for rel in changed:
        if Path(rel).suffix.lower() not in CODE:
            continue
        names = names_in(Path(rel), texts[rel])
        if names:
            listed.append(f"- {rel}: {', '.join(names)}")
        if len(listed) >= MAX_LISTED:
            break
    if listed:
        parts.append("Source files and what they declare:\n" + "\n".join(listed))
    commits = _commits(root)
    if commits and not previous:
        parts.append("Recent commit subjects:\n" + "\n".join(f"- {c}" for c in commits))
    return "\n\n".join(parts), hashes, len(changed) + len(removed)


def learn(vault_dir: Path, root: Path, now: datetime | None = None, dry_run: bool = False) -> dict:
    """Write a capture of one project's structure (or of what changed). Returns what it did."""
    now = now or datetime.now().astimezone()
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a folder")
    if root == vault_dir.resolve() or vault_dir.resolve() in root.parents:
        return {"project": root.name, "saved": None, "why": "this is the vault itself"}
    record = load_record()
    key = _key(root)
    if key in record["skip"]:
        return {"project": root.name, "saved": None, "why": "kept out with --skip"}
    seen = record["repos"].get(key, {})
    text, hashes, changes = scan(root, seen.get("files"))
    slug = project_slug(root.name)
    if seen and not changes:
        return {"project": slug, "saved": None, "why": "unchanged since last time"}
    if dry_run:
        return {"project": slug, "saved": "(dry run)", "why": f"{changes} files", "text": text}
    body = capture.render_capture(
        [capture.Turn("user", text)], f"repo:{slug}", now.isoformat(timespec="seconds"), slug
    )
    name = vault.unique_path(vault_dir, f"inbox/{now:%Y-%m-%d-%H%M}-learn-{slug[:40]}.md")
    saved = vault.write_new(vault_dir, name, body)
    record = load_record()  # re-read: another run may have finished meanwhile
    record["repos"][key] = {
        "project": slug,
        "learned": now.isoformat(timespec="seconds"),
        "files": hashes,
        "capture": name,
    }
    save_record(record)
    return {"project": slug, "saved": str(saved), "why": f"{changes} files read"}


def start(vault_dir: Path, root: Path) -> None:
    """Run `zk learn` detached, so the session that is starting doesn't wait for it."""
    from zk.sessions import INGESTING

    flags = (
        {
            "creationflags": subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        }
        if os.name == "nt"
        else {"start_new_session": True}
    )
    subprocess.Popen(
        [sys.executable, "-m", "zk", "learn", str(vault_dir), str(root)],
        cwd=vault_dir,
        env={**os.environ, INGESTING: "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **flags,
    )
