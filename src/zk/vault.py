"""The only module that writes to the vault.

Writes create a new file or move one, and never delete. The content appears all
at once (written to a temporary file, then hard-linked or renamed into place),
so a crash cannot leave half a note behind. The one write that changes an
existing file, `replace`, does so only if the file still has the content the
caller read, so an edit made in the meantime is never lost; git keeps the old
version.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


class VaultWriteError(Exception):
    pass


def _inside(vault: Path, target: Path) -> Path:
    vault = vault.resolve()
    target = (vault / target).resolve()
    if vault != target and vault not in target.parents:
        raise VaultWriteError(f"{target} is outside the vault {vault}")
    return target


def unique_path(vault: Path, rel: str) -> str:
    """`rel`, or `rel` with -2, -3, ... before the extension if that name is taken."""
    path = Path(rel)
    candidate, n = path, 1
    while (vault / candidate).exists():
        n += 1
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
    return candidate.as_posix()


def write_new(vault: Path, rel: str, text: str) -> Path:
    """Create vault/rel with `text`. Fails if the file already exists; never overwrites."""
    target = _inside(vault, Path(rel))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".zk-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        try:
            try:
                os.link(tmp, target)  # atomic, and fails if target exists
            except FileExistsError:
                raise
            except OSError:
                if os.name != "nt":
                    raise
                os.rename(tmp, target)  # no hard links here; on Windows rename never replaces
        except FileExistsError:
            raise VaultWriteError(f"{rel} already exists; zk never overwrites a file") from None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def replace(vault: Path, rel: str, text: str, expected_hash: str) -> Path:
    """Replace the existing file vault/rel with `text`, atomically.

    Refuses unless the file's current SHA-256 is `expected_hash` (what the
    caller read), so a change made since then, say in Obsidian, is not lost.
    """
    target = _inside(vault, Path(rel))
    if not target.is_file():
        raise VaultWriteError(f"{rel} does not exist; replace only changes an existing file")
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
        raise VaultWriteError(f"{rel} changed since it was read; read it again")
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".zk-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def move_new(vault: Path, src: str, dst: str) -> Path:
    """Move vault/src to vault/dst. Fails if dst exists; the content is never lost or replaced."""
    source = _inside(vault, Path(src))
    target = _inside(vault, Path(dst))
    if not source.is_file():
        raise VaultWriteError(f"{src} does not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            os.link(source, target)
        except FileExistsError:
            raise
        except OSError:
            if os.name != "nt":
                raise
            os.rename(source, target)  # Windows rename never replaces an existing file
            return target
    except FileExistsError:
        raise VaultWriteError(f"{dst} already exists; zk never overwrites a file") from None
    os.unlink(source)  # the content now lives at target
    return target


def git_commit(vault: Path, paths: list[str], message: str) -> str | None:
    """Commit exactly `paths` (added, changed or removed) if the vault is a git repo.

    Anything else you have staged or changed is left alone. Returns the commit
    hash, or None when the vault is not a repository.
    """
    if not (vault / ".git").exists():
        return None

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(vault), *args], capture_output=True, text=True, check=check
        )

    known = [
        p
        for p in paths
        if (vault / p).exists()
        or git("ls-files", "--error-unmatch", "--", p, check=False).returncode == 0
    ]
    if not known:
        return None
    git("add", "-A", "--", *known)
    git("commit", "--quiet", "-m", message, "--only", "--", *known)
    return git("rev-parse", "HEAD").stdout.strip()
