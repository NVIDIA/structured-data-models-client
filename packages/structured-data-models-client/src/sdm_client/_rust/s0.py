# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: the shadow check is deleted in Phase 5, with the old
# validator; the core route stays.

r"""S0: validating a predictive query on the core.

``PredictiveQuery._resolve`` comes here on the core route. The core's
``validate_query`` decides validity; a refusal is the caller's
``INVALID_REQUEST``, and a refusal of the core's own (``UNSUPPORTED_REQUEST``
for a date literal it does not read yet) raises as its code maps, never as
a refusal of the query. An accepted query finishes through the old
materializer, which needs the old validator's resolved query; until S1 is
switched, that is also the shadow check, and a disagreement is
``INTERNAL_ERROR`` (design, "Routing seam").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from predictive_query import Profile, ValidatedPredictiveQuery


def _profile(profile: Profile) -> dict[str, Any]:
    r"""A profile as the contract's ``ProfileSpec``: one of the named five."""
    import predictive_query.profiles as named

    for name in (
        'FINE_TUNING',
        'RELATIONAL',
        'TABULAR',
        'TIMESERIES',
        'ANOMALY',
    ):
        if getattr(named, name) == profile:
            return {'name': name, 'model': None}
    raise StructuredDataError(
        'the core validates a query for one of the named profiles only',
        code='INTERNAL_ERROR',
    )


def _schema(definition: Any) -> dict[str, Any]:
    r"""A PQL schema definition as the contract's ``QuerySchema``."""
    tables = [
        {
            'name': name,
            'columns': [
                {
                    'name': column.name,
                    'dtype': column.dtype.value,
                    'stype': column.stype.value,
                }
                for column in table.cols
            ],
            'primary_key': table.pkey,
            'time_column': table.time_col,
            'end_time_column': table.end_time_col,
        }
        for name, table in definition.tables.items()
    ]
    links = [
        [
            {'table': key.table_name, 'column': key.col_name}
            for key in group.columns
        ]
        for group in definition.col_groups
    ]
    return {'tables': tables, 'links': links}


def validate(
    text: str, definition: Any, profile: Profile
) -> tuple[bool, list[str]]:
    r"""The core's answer: whether it accepts ``text``, and its reasons if not.

    The query text and the schema travel as attachments, so neither the
    query's length nor the schema's width meets the control cap
    (decision H5).
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
        2: codec.attachment(_schema(definition)),
    }
    result = run(operation, attachments).result
    reasons = [
        diagnostic['message']
        for diagnostic in result['diagnostics']
        if diagnostic['severity'] == 'error'
    ]
    return bool(result['accepted']), reasons


def shadow(
    text: str, definition: Any, profile: Profile
) -> ValidatedPredictiveQuery:
    r"""The old validator's resolved query for a query the core accepted.

    The old materializer needs it. A query the old validator refuses is a
    disagreement between the two, a defect: ``INTERNAL_ERROR``.
    """
    from predictive_query import validate as old

    resolved, response = old(text, definition, profile)
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
