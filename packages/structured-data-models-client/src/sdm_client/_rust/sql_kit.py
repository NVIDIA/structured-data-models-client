# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A built-in source answered through its SQL kit: the core sends it the
typed requests it sends every source, and the kit, the engine's SQL written
once in the core's library, runs each on the caller's driver.

The driver's provider (``sdm_client._rust.providers``) runs the statements
the kit hands it, as it ran the core's; it writes no SQL and decides
nothing. One kit answers one run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

__all__ = ['KitProvider', 'chosen']


class KitProvider:
    r"""The run loop's provider for a built-in source: each fetch's typed
    request answered by ``kit`` over ``driver``, the provider that runs the
    statements on the caller's connection."""

    def __init__(self, kit: Any, driver: Any) -> None:
        self._kit = kit
        self._driver = driver
        read = getattr(driver, 'error_fields', None)
        if callable(read):
            self.error_fields = read

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        def run(plan: bytes, request: bytes, binds: dict[int, Any]) -> Any:
            fetch = {'plan': json.loads(plan), 'request': json.loads(request)}
            return self._driver(fetch, binds)

        request = json.dumps(step['request']).encode()
        return self._kit.answer(request, dict(keys or {}), run)

    def failure_facts(self, provided: dict[str, Any]) -> dict[str, Any]:
        r"""What the driver's failure ``provided`` (the contract's
        ``ProviderError``) means, as the engine reads it."""
        return json.loads(self._kit.failure(json.dumps(provided).encode()))


def chosen(choice: dict[str, Any], driver: Any) -> tuple[dict[str, Any], Any]:
    r"""The provider choice and provider of the built-in source whose SQL
    choice is ``choice``, its statements run by ``driver``: a source
    declaring what its kit declares, answered through the kit."""
    from sdm_client._rust.run import native

    kit = native().SqlKit(json.dumps(choice).encode())
    capabilities = json.loads(kit.capabilities())
    return {'provider': 'host', 'capabilities': capabilities}, KitProvider(
        kit, driver
    )
