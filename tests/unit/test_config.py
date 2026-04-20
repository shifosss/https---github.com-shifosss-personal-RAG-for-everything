from pathlib import Path

import pytest
from pydantic import ValidationError

from contextd.config import Settings, get_settings


def test_default_data_root_uses_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXTD_HOME", raising=False)
    s = Settings()
    assert s.data_root == Path.home() / ".contextd"


def test_env_override_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONTEXTD_HOME", str(tmp_path))
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.data_root == tmp_path
    finally:
        get_settings.cache_clear()


def test_settings_is_frozen() -> None:
    s = Settings()
    # pydantic v2 BaseSettings with frozen=True raises ValidationError on mutation.
    # Note: storage DTOs use @dataclass(frozen=True), which raises FrozenInstanceError instead —
    # don't conflate the two across the pydantic/dataclass boundary.
    with pytest.raises(ValidationError):
        s.data_root = Path("/nope")  # type: ignore[misc]
