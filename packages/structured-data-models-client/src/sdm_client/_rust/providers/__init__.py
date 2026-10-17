# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python host's data providers: each answers the fetch steps of a core
run, one request at a time, with an object that exports Arrow through
``__arrow_c_stream__``.

A provider reads no pandas and converts nothing: the driver's Arrow is what
crosses to the core, where admission and the schema checks hold it
(design, "Core, host and binding"). A test holds every module here to that,
by its imports and by running one with pandas unimportable.
"""

from typing import Any


def for_source(source: Any) -> tuple[dict[str, Any], Any]:
    r"""The provider choice a core run over ``source`` names, and the
    provider answering its fetches: the built-in SQL dialect of its backend,
    run on its connection (DuckDB's, from T2.2 SQLite's).

    Raises:
        ValueError: ``source`` is of a backend the core does not read, which
            the routing seam never sends here.
    """
    backend = getattr(source, 'backend', None)
    if backend == 'duckdb':
        from sdm_client._rust.providers.duckdb import DuckDBProvider

        provider: Any = DuckDBProvider(source.connection)
    elif backend == 'sqlite':
        from sdm_client._rust.providers.sqlite import SqliteProvider

        provider = SqliteProvider(source.connection)
    else:
        raise ValueError('the core reads no source of this backend')
    return {'provider': 'sql', 'dialect': backend}, provider
