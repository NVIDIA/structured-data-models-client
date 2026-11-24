# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A built-in source's way to the core, on a driver a test hands it.

The core names every source by what it declares (the host choice) and
asks it typed requests; a built-in source answers each through its SQL
kit, which writes the statements and hands each fetch (its SQL plan and
request) with its binds to the driver. A test that drives a driver
through the core reaches it this way: the kit opened for the source's SQL
choice (``{"provider": "sql", "dialect": ...}``), as
``sdm_client.sources`` opens it, over the test's driver, which may record
or alter what it is handed.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from sdm_client import provider as p
from sdm_client._rust import host_providers
from sdm_client._rust.sql_kit import KitRun


class _KitSource(p.Source):
    r"""A built-in source's answers, its kit's statements run by ``driver``."""

    def __init__(self, sql: dict[str, Any], driver: Any) -> None:
        self._run = KitRun(sql, driver)

    def capabilities(self) -> p.Capabilities:
        return self._run.capabilities()

    def answer(
        self, request: p.DataRequest, expected: pa.Schema | None
    ) -> p.ArrowStream:
        return self._run.answer(request)

    def failure(self, error: Exception) -> p.ProviderFailure:
        return self._run.failure(error)


def chosen(sql: dict[str, Any], driver: Any) -> tuple[dict[str, Any], Any]:
    r"""The core's provider choice for the built-in source of SQL choice
    ``sql`` on ``driver``, and the run loop's provider answering its
    fetches, as :func:`sdm_client._rust.host_providers.chosen` makes them."""
    return host_providers.chosen(_KitSource(sql, driver))


class Driven:
    r"""A driver, each fetch the kit hands it passed to ``seen`` first
    (``seen(fetch, binds)``), and what it reads of its failures kept."""

    def __init__(self, driver: Any, seen: Any) -> None:
        self._driver = driver
        self._seen = seen

    def __call__(self, fetch: dict[str, Any], binds: Any = None) -> Any:
        self._seen(fetch, binds)
        return self._driver(fetch, binds)

    def error_fields(self, error: BaseException) -> dict[str, Any] | None:
        read = getattr(self._driver, 'error_fields', None)
        return read(error) if callable(read) else None
