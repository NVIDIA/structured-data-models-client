# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

_ENV_KUMO_RELATIONAL_LOG = 'KUMO_RELATIONAL_LOG'

_HANDLER_ATTR = '_sdm_relational_engine_handler'


def _install_handler(logger: logging.Logger) -> None:
    r"""Attach our formatter only to the ``sdm_relational_engine`` logger.

    A library must not call :func:`logging.basicConfig`: that attaches a
    handler to the *root* logger and silently reconfigures the host
    application's logging. Handling our own records here and not propagating
    them keeps the previous console output without touching anyone else's.

    An application that has already configured logging owns the output format,
    so nothing is installed in that case and records are left to propagate to
    its handlers. Installing ours unconditionally meant a host doing structured
    logging could neither capture these records (``propagate`` was off) nor
    format them.
    """
    if any(getattr(h, _HANDLER_ATTR, False) for h in logger.handlers):
        return
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt=(
                '[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] '
                '%(message)s'
            ),
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    )
    setattr(handler, _HANDLER_ATTR, True)
    logger.addHandler(handler)
    logger.propagate = False


def initialize_logging() -> None:
    r"""Initialize Kumo Relational logging.

    This touches only the ``sdm_relational_engine`` logger. It once raised
    ``matplotlib``, ``urllib3`` and ``snowflake`` to ``ERROR``. Importing the
    package then suppressed three unrelated libraries for the whole process,
    including applications that never predict and had configured them
    otherwise. Applications, not libraries, decide whether to quiet noisy
    dependencies.
    """
    logger: logging.Logger = logging.getLogger('sdm_relational_engine')

    _install_handler(logger)

    default_level = os.getenv(_ENV_KUMO_RELATIONAL_LOG) or 'INFO'
    try:
        logger.setLevel(default_level)
    except (TypeError, ValueError):
        logger.setLevel(logging.INFO)
        logger.warning(
            'Logging level %s could not be properly parsed. '
            'Defaulting to INFO log level.',
            default_level,
        )
