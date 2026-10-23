# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: the shadow check is deleted in Phase 5, with the old
# validator; the core route stays.

r"""S0: validating a predictive query on the core.

``PredictiveQuery._resolve`` comes here on the core route. The core's
``validate_query`` decides validity; a refusal is the caller's
``INVALID_REQUEST``, and a refusal of the core's own (a local zone it cannot
read, say) raises as its code maps, never as a refusal of the query. An accepted query finishes through the old
materializer, which needs the old validator's resolved query; until S1 is
switched, that is also the shadow check, and a disagreement is
``INTERNAL_ERROR`` (design, "Routing seam").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdm_client._rust import errors, results
from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

#: The core's warnings the old validator gives on this route in the
#: reference's own words when the core accepts the query: a day-first date
#: literal (D4), which the shadow check's validator reads with
#: ``pd.to_datetime``. They are held while the core validates and shown
#: only when it refuses, since a refused query meets no shadow check.
#: Bridge code: it goes with the shadow check in Phase 5, and the core's
#: warning is shown then.
SHADOWED = frozenset({'DATE_LITERAL_DAY_FIRST'})

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

    from sdm_client._rust.results import QueryFacts


#: The language's named profiles, which the core validates a query for.
PROFILES = ('FINE_TUNING', 'RELATIONAL', 'TABULAR', 'TIMESERIES', 'ANOMALY')


def _profile(profile: str) -> dict[str, Any]:
    r"""A profile, named, as the contract's ``ProfileSpec``: one of the
    named five, with no object of the old PQL package (T3.16, rule 11)."""
    if profile in PROFILES:
        return {'name': profile, 'model': None}
    raise StructuredDataError(
        'the core validates a query for one of the named profiles only',
        code='INTERNAL_ERROR',
    )


def validate(
    text: str, schema: dict[str, Any], profile: str
) -> tuple[bool, list[str], list[dict[str, Any]], QueryFacts | None]:
    r"""The core's answer over ``schema``, the contract's ``QuerySchema``
    (:func:`~sdm_client._rust.spec.query_schema`, written with no object of
    the old PQL package, T3.16): whether it accepts ``text``, its reasons if
    not, the warnings it held of an accepted query (:data:`SHADOWED`), and
    what it resolved of an accepted one (``validate_query``'s ``query``,
    contract v28: its text as the core writes it back, its forecast's
    steps).

    The query text and the schema travel as attachments, so neither the
    query's length nor the schema's width meets the control cap
    (decision H5). The shadow check's validator says an accepted query's
    day-first literal itself; a caller that runs no shadow check (a query
    the core gathers alone, T3.8) shows the held warnings
    (``errors.warn_held``).
    """
    from sdm_client._rust import codec

    operation = {
        'op': 'validate_query',
        'query': 1,
        'schema': 2,
        'profile': _profile(profile),
    }
    attachments = {
        1: codec.attachment(text),
        2: codec.attachment(schema),
    }
    kept: list[dict[str, Any]] = []
    try:
        result = results.expect(
            run(operation, attachments, quiet=SHADOWED, kept=kept).result,
            results.Validated,
        )
    except StructuredDataError:
        # A run that ends with an error after reading a day-first literal
        # (a packed literal after it, say) meets no shadow check either:
        # the literal's warning is shown before the error (carried row U1).
        errors.warn_held(kept)
        raise
    if not result.accepted:
        # No shadow check follows a refusal: the core's warnings are the
        # only ones the caller gets.
        errors.warn_held(kept)
        kept = []
    reasons = [
        diagnostic.message
        for diagnostic in result.diagnostics
        if diagnostic.severity == 'error'
    ]
    return result.accepted, reasons, kept, result.query


def shadow(
    text: str, definition: Any, profile: str
) -> ValidatedPredictiveQuery:
    r"""The old validator's resolved query for a query the core accepted.

    The old materializer needs it. A query the old validator refuses is a
    disagreement between the two, a defect: ``INTERNAL_ERROR``.
    """
    from predictive_query import validate as old

    from sdm_client.base import language_profile

    resolved, response = old(text, definition, language_profile(profile))
    if resolved is None or not response.ok:
        raise StructuredDataError(
            'the core accepted a query the old validator refuses; this is a '
            'defect, not a problem with the query',
            code='INTERNAL_ERROR',
            details={
                'codes': [error.code for error in response.errors],
            },
        )
    return resolved
