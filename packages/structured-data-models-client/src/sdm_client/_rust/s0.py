# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate predictive queries through the shared Rust implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdm_client._rust import errors, results
from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

# Emit literal warnings once, after validation or before propagating a refusal.
DEFERRED_WARNINGS = frozenset(
    {'DATE_LITERAL_DAY_FIRST', 'INVALID_STRING_ESCAPE'}
)

if TYPE_CHECKING:
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
    """Return validity, diagnostics, deferred warnings and resolved query facts.

    Query text and schema travel as attachments rather than control messages.
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
            run(
                operation, attachments, quiet=DEFERRED_WARNINGS, kept=kept
            ).result,
            results.Validated,
        )
    except StructuredDataError:
        # Preserve warnings emitted before a later validation error.
        errors.warn_held(kept)
        raise
    if not result.accepted:
        # A refused query has no later result on which to emit warnings.
        errors.warn_held(kept)
        kept = []
    reasons = [
        diagnostic.message
        for diagnostic in result.diagnostics
        if diagnostic.severity == 'error'
    ]
    return result.accepted, reasons, kept, result.query
