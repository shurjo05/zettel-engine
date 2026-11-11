import re
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest

from zk.embed import normalize

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A private copy of the fixture vault, so tests can edit it freely."""
    copy = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, copy)
    return copy


class WordEmbedder:
    """A stand-in for a real embedding model: a hashed bag of crudely stemmed words.

    It knows nothing about meaning, but it is deterministic, instant and needs no
    download, which is enough to test how vectors are stored, cached and ranked.
    """

    def __init__(self, model: str = "test/word-hash", dim: int = 64):
        self.model = model
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vector(self, text: str):
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            for suffix in ("ing", "ed", "es", "s"):
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[: -len(suffix)]
                    break
            vec[zlib.crc32(word.encode()) % self.dim] += 1
        return normalize(vec)

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def embedder() -> WordEmbedder:
    return WordEmbedder()


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch):
    """The CLI's default (hybrid) search needs an embedder; tests never fetch a real model."""
    from zk import cli

    monkeypatch.setattr(cli, "make_embedder", lambda model: WordEmbedder(model))


@pytest.fixture(autouse=True)
def isolated_homes(tmp_path_factory, monkeypatch):
    """zk's data folder and Claude Code's config folder point at throwaway folders."""
    base = tmp_path_factory.mktemp("homes")
    for var, name in [
        ("ZK_DATA_DIR", "zk"),
        ("CLAUDE_CONFIG_DIR", "claude"),
    ]:
        monkeypatch.setenv(var, str(base / name))
    return base


@pytest.fixture(autouse=True)
def no_claude_calls(monkeypatch):
    """Tests never ask a real Claude Code install about its MCP servers."""
    from zk import hooks

    monkeypatch.setattr(hooks, "server_ready", lambda: (None, "not checked in tests"))


@pytest.fixture(autouse=True)
def no_real_claude(monkeypatch):
    """Tests never start a real headless Claude; a test that needs answers sets its own."""
    from zk import cli

    def refuse(model=None):
        def ask(prompt):
            raise AssertionError("a test tried to call Claude")

        return ask

    monkeypatch.setattr(cli, "make_ask", refuse)


@pytest.fixture(autouse=True)
def no_background_work(monkeypatch):
    """Tests never load a real embedding model for maps, or start a detached zk learn."""
    from zk import learn, process

    started = []
    monkeypatch.setattr(process, "refresh_maps", lambda vault_dir, log: None)
    monkeypatch.setattr(learn, "start", lambda vault_dir, root: started.append(root))
    return started
