# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The size limits this client holds a request to before it is sent.

Each mirrors a limit the serving layer enforces, so the client refuses or
batches first and says which limit, rather than sending a request the
endpoint refuses without saying how to meet it. A deployment sets its own
(the serving layer's ``TFM_MAX_TRAIN_ROWS``, ``TFM_MAX_TEST_ROWS`` and
``TFM_MAX_PAYLOAD_BYTES``), so each is a setting with the serving layer's
default. Raise one to match a deployment configured to take more:

>>> limits = Limits(max_tabular_context_rows=200_000,
...                 max_request_bytes=512 * 1024 * 1024)
>>> client = SDMClient(url, limits=limits)
>>> request = query.materialize(indices=keys, limits=limits)

or set its environment variable, ``SDM_CLIENT_`` and the setting's name in
capitals, such as ``SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS=200000``. A value
passed to :class:`Limits` comes before the environment's, which comes before
the default. The names, variables and defaults are the core's, the one table
every host reads, and the core resolves them.
"""

from __future__ import annotations

import contextvars
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any

from sdm_client.errors import StructuredDataError


@dataclass(frozen=True)
class Limits:
    r"""The size limits a client holds requests to, each ``None`` to take
    its environment variable, else its default.

    Attributes:
        max_relational_context_rows: The most context rows a relational
            request carries; ``num_context_samples`` asks for at most this
            many. ``SDM_CLIENT_MAX_RELATIONAL_CONTEXT_ROWS``, 10,000.
        max_relational_predict_rows: The most rows one relational call
            scores; more are asked in several calls.
            ``SDM_CLIENT_MAX_RELATIONAL_PREDICT_ROWS``, 10,000.
        max_temporal_link_predict_rows: The most rows one temporal link
            prediction call scores.
            ``SDM_CLIENT_MAX_TEMPORAL_LINK_PREDICT_ROWS``, 200.
        max_tabular_context_rows: The most context rows a single-table
            request carries; ``materialize()`` samples down to it.
            ``SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS``, 10,000.
        max_request_bytes: The most bytes one call's body takes. A call
            past it that scores several rows is split into smaller calls; one
            whose context alone is past it is refused.
            ``SDM_CLIENT_MAX_REQUEST_BYTES``, 31,457,280 (30 MiB). A
            Databricks serving endpoint also holds each call to its own
            ``max_request_bytes``.

    Raises:
        StructuredDataError: With code ``INVALID_CONFIGURATION``, if a value
            is not a whole number of at least 1.
    """

    max_relational_context_rows: int | None = None
    max_relational_predict_rows: int | None = None
    max_temporal_link_predict_rows: int | None = None
    max_tabular_context_rows: int | None = None
    max_request_bytes: int | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise StructuredDataError(
                    f'{field.name} takes a whole number of at least 1, '
                    f'got {type(value).__name__} {value!r}',
                    code='INVALID_CONFIGURATION',
                )
            if value < 1:
                raise StructuredDataError(
                    f'{field.name} is {value:,}, and it takes a whole number '
                    f'of at least 1',
                    code='INVALID_CONFIGURATION',
                )

    def _given(self) -> dict[str, Any]:
        r"""What a run's start carries of these limits: the values set here,
        and the text of each setting's environment variable that is set,
        read now, which the core resolves (contract ``GivenLimits``)."""
        from sdm_client._rust.run import native

        given = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        }
        environment = {
            env: os.environ[env]
            for _, env, _ in native().LIMITS
            if env in os.environ
        }
        start: dict[str, Any] = {}
        if given:
            start['given'] = given
        if environment:
            start['environment'] = environment
        return start


def defaults() -> dict[str, int]:
    r"""Each setting's default, by name: the core's one table of them."""
    from sdm_client._rust.run import native

    return {name: default for name, _, default in native().LIMITS}


_IN_EFFECT: contextvars.ContextVar[Limits | None] = contextvars.ContextVar(
    'sdm_client_limits', default=None
)


@contextmanager
def limits_in_effect(limits: Limits | None) -> Iterator[None]:
    r"""Hold every run started inside to ``limits``; ``None`` keeps the ones
    already in effect. A client's calls and ``materialize(limits=...)`` set
    them for their own runs only."""
    if limits is None:
        yield
        return
    token = _IN_EFFECT.set(limits)
    try:
        yield
    finally:
        _IN_EFFECT.reset(token)


def in_effect() -> Limits:
    r"""The limits a run started now is held to: those set by the call it is
    part of, else the environment's and the defaults."""
    return _IN_EFFECT.get() or Limits()


def check(limits: Any) -> Limits | None:
    r"""``limits`` as an argument takes it: a :class:`Limits`, or ``None``."""
    if limits is None or isinstance(limits, Limits):
        return limits
    raise StructuredDataError(
        f'limits takes an sdm_client.Limits, got {type(limits).__name__}',
        code='INVALID_CONFIGURATION',
    )
