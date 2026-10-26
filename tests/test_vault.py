import pytest

from zk.vault import VaultWriteError, unique_path, write_new


def test_write_new_creates_and_never_overwrites(tmp_path):
    path = write_new(tmp_path, "inbox/a.md", "first\n")
    assert path.read_text(encoding="utf-8") == "first\n"
    with pytest.raises(VaultWriteError, match="never overwrites"):
        write_new(tmp_path, "inbox/a.md", "second\n")
    assert path.read_text(encoding="utf-8") == "first\n"
    assert sorted(p.name for p in (tmp_path / "inbox").iterdir()) == ["a.md"]  # no temp files


def test_unique_path_adds_a_counter(tmp_path):
    assert unique_path(tmp_path, "inbox/a.md") == "inbox/a.md"
    write_new(tmp_path, "inbox/a.md", "x")
    assert unique_path(tmp_path, "inbox/a.md") == "inbox/a-2.md"
    write_new(tmp_path, "inbox/a-2.md", "x")
    assert unique_path(tmp_path, "inbox/a.md") == "inbox/a-3.md"


def test_writes_outside_the_vault_are_refused(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    with pytest.raises(VaultWriteError, match="outside the vault"):
        write_new(vault, "../escape.md", "x")
    assert not (tmp_path / "escape.md").exists()
