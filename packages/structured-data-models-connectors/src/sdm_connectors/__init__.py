# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from sdm_connectors._version import __version__
from sdm_connectors.source import (
    DatabricksSource,
    DuckDBSource,
    PostgresSource,
    SemanticModel,
    SnowflakeSource,
    Source,
    SQLiteSource,
)
from sdm_connectors.sql import (
    ConnectorError,
    MissingBackendError,
    quote_ident,
    resolve_sql,
)

if TYPE_CHECKING:
    from sdm_connectors.reader import read, read_table


def __getattr__(name: str) -> Any:
    # The readers, which need pandas, are imported when first named, so a
    # module of this package (``sdm_connectors.providers``, which a provider
    # imports) loads where pandas cannot be imported.
    if name in ('read', 'read_table'):
        from sdm_connectors import reader

        return getattr(reader, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted({*globals(), 'read', 'read_table'})


__all__ = [
    'ConnectorError',
    'DatabricksSource',
    'DuckDBSource',
    'MissingBackendError',
    'PostgresSource',
    'SQLiteSource',
    'SemanticModel',
    'SnowflakeSource',
    'Source',
    '__version__',
    'quote_ident',
    'read',
    'read_table',
    'resolve_sql',
]
