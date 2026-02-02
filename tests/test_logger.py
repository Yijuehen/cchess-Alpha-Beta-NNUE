"""
Tests for logging module.
"""

import pytest
import logging
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import ChessEngineLogger, get_logger, setup_logging


def test_logger_creation():
    """Test logger can be created."""
    logger_mgr = ChessEngineLogger("test")
    assert logger_mgr.logger is not None
    assert logger_mgr.logger.name == "test"


def test_get_logger():
    """Test get_logger function."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_log_levels():
    """Test different log levels."""
    logger = get_logger("test_levels")

    # These should not raise exceptions
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")


def test_file_logging(tmp_path):
    """Test file logging creates files."""
    logger_mgr = ChessEngineLogger(
        "test_file",
        log_dir=str(tmp_path),
        console=False,
        file=True
    )

    logger = logger_mgr.get_logger()
    logger.info("Test message")

    # Check file was created
    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) > 0

    # Check log contains message
    with open(log_files[0], 'r') as f:
        content = f.read()
        assert "Test message" in content


def test_json_formatting():
    """Test JSON formatter."""
    import json

    logger_mgr = ChessEngineLogger(
        "test_json",
        json_format=True,
        file=False,
        console=False
    )

    formatter = logger_mgr._get_formatter(json_format=True)

    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data['level'] == 'INFO'
    assert data['message'] == 'Test message'
    assert 'timestamp' in data


def test_setup_logging():
    """Test setup_logging function."""
    logger_mgr = setup_logging(
        name="test_setup",
        level="DEBUG"
    )

    assert logger_mgr.logger.name == "test_setup"
    assert logger_mgr.logger.level == logging.DEBUG


def test_set_level():
    """Test set_level method."""
    logger_mgr = ChessEngineLogger("test_level")

    logger_mgr.set_level("WARNING")
    assert logger_mgr.logger.level == logging.WARNING

    logger_mgr.set_level("ERROR")
    assert logger_mgr.logger.level == logging.ERROR
