"""Utility helpers for configuring logging across API runtimes."""

from __future__ import annotations

import logging
import os

DEFAULT_LOG_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def configure_logging() -> None:
    """Configure root logging handlers based on the current environment.

    In production-like environments (``ENVIRONMENT`` set to ``production`` or
    ``staging``), this attempts to forward logs to Google Cloud Logging. For
    other environments it keeps the standard logging configuration while still
    honouring ``LOG_LEVEL`` and ``LOG_FORMAT`` overrides.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", DEFAULT_LOG_FORMAT)
    environment = os.getenv("ENVIRONMENT", "local").lower()

    level = getattr(logging, log_level_name, logging.INFO)
    root_logger = logging.getLogger()
    logging.captureWarnings(True)

    if environment in {"production", "staging"}:
        try:
            from google.cloud import logging as cloud_logging  # type: ignore

            # TODO: Set up notification channels in Cloud Monitoring and replace CHANNEL_ID_PLACEHOLDER
            # in monitoring policy files with actual channel IDs
            cloud_logging.Client().setup_logging()
            root_logger.setLevel(level)
            logging.getLogger(__name__).info(
                "Configured Cloud Logging handler (environment=%s)", environment
            )
            return
        except Exception as exc:  # pragma: no cover - defensive guard for CI runs
            logging.basicConfig(level=level, format=log_format)
            logging.getLogger(__name__).warning(
                (
                    "Unable to initialize Cloud Logging (environment=%s): %s; "
                    "using standard logging."
                ),
                environment,
                exc,
            )
            return

    if not root_logger.handlers:
        logging.basicConfig(level=level, format=log_format)
    else:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)


__all__ = ["configure_logging"]
