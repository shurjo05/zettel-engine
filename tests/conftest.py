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
