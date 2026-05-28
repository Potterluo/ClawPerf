"""Tests for error classification and runner utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clawperf.runner import classify_error


def test_classify_http_4xx():
    bd = MagicMock()
    bd.success = False
    bd.status_code = 400
    bd.error = "Bad request"
    assert classify_error(bd) == "http_4xx"


def test_classify_http_5xx():
    bd = MagicMock()
    bd.success = False
    bd.status_code = 500
    bd.error = "Internal server error"
    assert classify_error(bd) == "http_5xx"


def test_classify_timeout():
    bd = MagicMock()
    bd.success = False
    bd.status_code = None
    bd.error = "TimeoutError: request timed out"
    assert classify_error(bd) == "timeout"


def test_classify_timeout_lowercase():
    bd = MagicMock()
    bd.success = False
    bd.status_code = None
    bd.error = "asyncio timeout"
    assert classify_error(bd) == "timeout"


def test_classify_network():
    bd = MagicMock()
    bd.success = False
    bd.status_code = None
    bd.error = "Connection refused"
    assert classify_error(bd) == "network"


def test_classify_none():
    assert classify_error(None) == "network"