import shutil
from pathlib import Path

import pytest

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A private copy of the fixture vault, so tests can edit it freely."""
    copy = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, copy)
    return copy
