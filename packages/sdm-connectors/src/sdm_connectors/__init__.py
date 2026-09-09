# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm_connectors._version import __version__
from sdm_connectors.backends import connect
from sdm_connectors.reader import read, read_table
from sdm_connectors.sql import (
    ConnectorError,
    MissingBackendError,
    quote_ident,
    resolve_sql,
)

__all__ = [
    'ConnectorError',
    'MissingBackendError',
    '__version__',
    'connect',
    'quote_ident',
    'read',
    'read_table',
    'resolve_sql',
]
