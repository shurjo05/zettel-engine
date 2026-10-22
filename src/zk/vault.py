"""The only module that writes to the vault.

Every write creates a new file and never replaces one: the target name must not
exist yet, and the content appears all at once (written to a temporary file,
then hard-linked into place), so a crash cannot leave half a note behind.
"""

from __future__ import annotations

import os
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
