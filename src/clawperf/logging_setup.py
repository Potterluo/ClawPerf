"""Centralized logging configuration for ClawPerf.

Without this, ``logging.getLogger("clawperf")`` resolves to the root logger's
default level (WARNING) with no handlers, so every ``logger.info(...)`` call in
the runner is silently dropped — making long setup phases (tokenizer download,
content generation) look like the process has frozen.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "clawperf"


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

    # Third-party libraries (evalscope, aiohttp) log ERROR tracebacks that break
    # the tqdm progress bar. In non-verbose mode, quiet them to CRITICAL so only
    # the live `err=N` counter and the final error distribution surface failures.
    # Errors are still recorded per-turn (BenchmarkData.success=False). In verbose
    # mode, leave them audible for debugging.
    third_party = ("evalscope", "aiohttp", "aiohttp.access", "urllib3", "asyncio")
    tp_level = logging.INFO if verbose else logging.CRITICAL
    for name in third_party:
        logging.getLogger(name).setLevel(tp_level)

    return logger
