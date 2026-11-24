# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python host's data providers: each answers the fetch steps of a core
run, one request at a time, with an object that exports Arrow through
``__arrow_c_stream__``.

A provider reads no pandas and converts nothing: the driver's Arrow, or,
from a driver that gives rows (psycopg), the Arrow pyarrow builds of its
values, is what crosses to the core, where admission and the schema checks
hold it (design, "Core, host and binding"). A test holds every module here
to that, by its imports and by running one with pandas unimportable.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    'DatabricksSparkProvider',
    'DatabricksSqlProvider',
    'DuckDBProvider',
    'PostgresProvider',
    'SnowflakeProvider',
    'SqliteProvider',
]

#: The module each provider class is written in. Every one of them imports the
#: warehouse package, which is an optional dependency, so a module is imported
#: when a built-in source first names its provider rather than when this one
#: is. A route over frames names none of them and needs none of the package.
_WRITTEN_IN: dict[str, str] = {
    'DatabricksSparkProvider': 'databricks',
    'DatabricksSqlProvider': 'databricks',
    'DuckDBProvider': 'duckdb',
    'PostgresProvider': 'postgres',
    'SnowflakeProvider': 'snowflake',
    'SqliteProvider': 'sqlite',
}


def __getattr__(name: str) -> Any:
    r"""A provider class, imported on each read of its name.

    The class read is not bound here. A name bound on the first read would
    outlive the module it was read from, and a run that follows a test which
    swapped a driver out and back would answer with the class the swap made
    rather than the one the module holds.
    """
    if name not in _WRITTEN_IN:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module = importlib.import_module(
        f'sdm_client._rust.providers.{_WRITTEN_IN[name]}'
    )
    return getattr(module, name)
