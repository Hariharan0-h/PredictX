"""Tests for exception/Custom_exception.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from exception import CustomException


def test_custom_exception_is_exception_subclass():
    assert issubclass(CustomException, Exception)


def test_custom_exception_captures_message():
    try:
        raise ValueError("bad input")
    except Exception as e:
        exc = CustomException(e, sys)
    assert "bad input" in str(exc)


def test_custom_exception_captures_filename():
    try:
        raise RuntimeError("oops")
    except Exception as e:
        exc = CustomException(e, sys)
    assert exc.file_name != "Unknown"
    assert ".py" in exc.file_name


def test_custom_exception_captures_lineno():
    try:
        raise RuntimeError("oops")
    except Exception as e:
        exc = CustomException(e, sys)
    assert isinstance(exc.lineno, int)
    assert exc.lineno > 0


def test_custom_exception_str_format():
    try:
        raise ZeroDivisionError("division by zero")
    except Exception as e:
        exc = CustomException(e, sys)
    s = str(exc)
    assert "Error occurred in python script name" in s
    assert "line number" in s
    assert "error message" in s


def test_custom_exception_no_active_traceback():
    """When constructed outside an except block, lineno/file_name fall back gracefully."""
    exc = CustomException("standalone error", sys)
    assert exc.lineno == "Unknown"
    assert exc.file_name == "Unknown"


def test_custom_exception_can_be_raised_and_caught():
    with pytest.raises(CustomException):
        try:
            x = 1 / 0
        except Exception as e:
            raise CustomException(e, sys)
