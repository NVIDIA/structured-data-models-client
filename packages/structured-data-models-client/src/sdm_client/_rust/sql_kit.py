# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""One operation's answers of a built-in source: its SQL kit, the
engine's SQL written once in the core's library, answering each typed
request on the source's driver.

The kit is one per operation: it settles the namespace a source leaves to
its connection once, and reads an identical catalog read once, for that
operation only. A built-in source opens one where every source's
operation starts, its :meth:`~sdm_client.provider.Source.capabilities`.

The driver's provider (``sdm_client._rust.providers``) runs the statements
the kit hands it on the caller's connection; it writes no SQL and decides
nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from sdm_client import provider as p
from sdm_client._rust import errors, host_providers
from sdm_client._rust.run import native, provider_error

__all__ = ['KitRun']


class KitRun:
    r"""One operation of the built-in source whose SQL choice is ``choice``
    (the contract's ``SqlChoice``), its statements run by ``driver``."""

    def __init__(self, choice: Mapping[str, Any], driver: Any) -> None:
        self._kit = native().SqlKit(json.dumps(choice).encode())
        self._driver = driver

    def capabilities(self) -> p.Capabilities:
        r"""What the source answers, as its kit declares it."""
        return host_providers.read_capabilities(
            json.loads(self._kit.capabilities())
        )

    def answer(self, request: p.DataRequest) -> p.ArrowStream:
        r"""The answer to ``request``, in the contract's shape.

        Raises:
            StructuredDataError: The kit refused the request by name, the
                error the operation then fails with.
            Exception: The driver's failure, as it raised it.
        """
        wire, keys = host_providers.write_request(request)
        core = native()
        try:
            answer: p.ArrowStream = self._kit.answer(
                json.dumps(wire).encode(), keys, self._run
            )
        except core.CoreError as refused:
            raise errors.refusal(refused.control, refused.attachments) from None
        return answer

    def failure(self, error: Exception) -> p.ProviderFailure:
        r"""What the driver's failure ``error`` tells the core: what the
        driver reads of it, and what it means, as the kit reads it."""
        provided = provider_error(error, self._driver)
        facts = json.loads(self._kit.failure(json.dumps(provided).encode()))
        return host_providers.read_failure({**provided, 'facts': facts})

    def _run(self, plan: bytes, request: bytes, binds: dict[int, Any]) -> Any:
        fetch = {'plan': json.loads(plan), 'request': json.loads(request)}
        return self._driver(fetch, binds)
