from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from contextd.config import get_settings

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_contextd_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CONTEXTD_HOME", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
