import logging

import pytest

from contextd.logging_ import configure_logging, get_logger


def test_default_level_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXTD_LOG_LEVEL", raising=False)
    configure_logging()
    assert logging.getLogger("contextd").level == logging.INFO


def test_debug_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTD_LOG_LEVEL", "DEBUG")
    configure_logging()
    assert logging.getLogger("contextd").level == logging.DEBUG


def test_get_logger_is_structlog() -> None:
    configure_logging()
    log = get_logger("test")
    # structlog BoundLogger has a `bind` method; stdlib Logger does not.
    assert hasattr(log, "bind")
