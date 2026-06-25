"""Tests for logging configuration and third-party logger quieting."""

from __future__ import annotations

import logging

from clawperf.logging_setup import setup_logging


def test_setup_logging_attaches_handler():
    lg = setup_logging(verbose=False)
    assert lg.level == logging.INFO
    assert any(getattr(h, "_clawperf", False) for h in lg.handlers)


def test_verbose_level():
    lg = setup_logging(verbose=True)
    assert lg.level == logging.DEBUG


def test_non_verbose_quiets_third_party_loggers():
    """evalscope ERROR tracebacks break the progress bar; quiet them in non-verbose."""
    setup_logging(verbose=False)
    assert logging.getLogger("evalscope").level == logging.CRITICAL
    assert logging.getLogger("aiohttp").level == logging.CRITICAL


def test_verbose_keeps_third_party_audible():
    setup_logging(verbose=True)
    assert logging.getLogger("evalscope").level == logging.INFO


def test_clawperf_logger_does_not_propagate():
    """No double-print via the root logger."""
    lg = setup_logging(verbose=False)
    assert lg.propagate is False
