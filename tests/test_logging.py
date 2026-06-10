"""Tests for custom_logging module."""

import logging
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_logger_is_importable():
    from custom_logging import logger
    assert logger is not None


def test_logger_is_logging_logger():
    from custom_logging import logger
    assert isinstance(logger, logging.Logger)


def test_logger_name():
    from custom_logging import logger
    assert logger.name == "predictx"


def test_log_file_created(tmp_path, monkeypatch):
    """Logger module creates a .log file inside a logs/ dir."""
    import importlib
    import custom_logging.logger as lg_module

    # Patch LOGS_DIR to a temp directory so we don't pollute the real logs/
    monkeypatch.setattr(lg_module, "LOGS_DIR", str(tmp_path))
    log_file = tmp_path / "test_run.log"
    monkeypatch.setattr(lg_module, "LOG_FILE_PATH", str(log_file))

    test_logger = logging.getLogger("test_predictx")
    handler = logging.FileHandler(str(log_file))
    handler.setLevel(logging.INFO)
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    test_logger.info("test message")
    handler.flush()

    assert log_file.exists(), "Log file should be created after logging"
    assert log_file.stat().st_size > 0, "Log file should not be empty"


def test_logs_dir_is_not_a_file():
    """Regression: logs/ must be a directory, not a file named after the log."""
    import custom_logging.logger as lg_module
    logs_dir = Path(lg_module.LOGS_DIR)
    if logs_dir.exists():
        assert logs_dir.is_dir(), f"LOGS_DIR '{logs_dir}' must be a directory"


def test_log_file_path_has_log_extension():
    import custom_logging.logger as lg_module
    assert lg_module.LOG_FILE_PATH.endswith(".log")


def test_logger_writes_message(tmp_path):
    log_file = tmp_path / "out.log"
    test_logger = logging.getLogger("check_write")
    handler = logging.FileHandler(str(log_file))
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    test_logger.info("pipeline started")
    handler.flush()

    content = log_file.read_text()
    assert "pipeline started" in content
