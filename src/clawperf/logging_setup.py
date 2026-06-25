"""Centralized logging configuration for ClawPerf.

Without this, ``logging.getLogger("clawperf")`` resolves to the root logger's
default level (WARNING) with no handlers, so every ``logger.info(...)`` call in
the runner is silently dropped — making long setup phases (tokenizer download,
content generation) look like the process has frozen.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "clawperf"

# Libraries whose ERROR/traceback output breaks the tqdm progress bar.
_NOISY_LOGGERS = ("evalscope", "aiohttp", "aiohttp.access", "urllib3", "asyncio")


def quiet_third_party(verbose: bool = False) -> None:
    """Quiet noisy third-party loggers.

    evalscope calls ``logging.basicConfig(level=INFO, force=True)`` at import
    time (after our setup_logging), so this must be (re)applied once evalscope
    has been imported — otherwise its ERROR tracebacks break the progress bar.
    In verbose mode the libraries are left audible for debugging.
    """
    level = logging.INFO if verbose else logging.CRITICAL
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        if not verbose:
            # Don't double-print through the root handler evalscope installed.
            lg.propagate = False


class TqdmLogHandler(logging.Handler):
    """Emit log records through ``tqdm.write()`` so messages never break a
    live progress bar. Falls back to ``logging.StreamHandler`` behavior when
    tqdm is unavailable."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                from tqdm import tqdm

                tqdm.write(msg)
            except Exception:
                print(msg)
        except Exception:
            self.handleError(record)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure the ``clawperf`` logger exactly once.

    - Always attaches a handler so INFO messages are visible.
    - Routes output through ``tqdm.write`` to coexist with progress bars.
    - Stops propagation so we don't double-print via the root logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    desired_level = logging.DEBUG if verbose else logging.INFO

    # Avoid stacking duplicate handlers on re-entry (e.g. CLI re-import).
    if not any(getattr(h, "_clawperf", False) for h in logger.handlers):
        handler = TqdmLogHandler()
        handler.setLevel(logging.DEBUG)  # let the logger level be the gate
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        handler._clawperf = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    logger.setLevel(desired_level)
    logger.propagate = False

    # Apply third-party quieting (re-applied later after evalscope import).
    quiet_third_party(verbose)
    return logger
