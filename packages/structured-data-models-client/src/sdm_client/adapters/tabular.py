# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular endpoint diagnostics and Python value observations."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

import pandas as pd

from sdm_client.base import ModelSpec
from sdm_client.errors import StructuredDataError, UnsupportedRequestError
from sdm_client.models import KUMO_TABULAR
from sdm_client.request import Request
from sdm_client.wire import (
    infer_tfm_dtype,
)

# Output fields to ask for when the caller named none, by task kind. Asking for
# everything would make a regression request carry probabilities it cannot
# have, which the endpoint rejects.
_DEFAULT_OUTPUTS = {
    'classification': ('prediction', 'probabilities'),
    'regression': ('prediction',),
}

# A session pins model, task, schema and context; only the rows being scored
# travel with each later call. ``metadata`` rides along on both halves because
# its request id is what correlates the call with the endpoint's logs.
_SESSION_PINNED = ('model', 'task', 'schema', 'context')
_SESSION_SCORING = ('predict', 'output', 'inference', 'metadata')

# What the contract allows a `PredictionItem.id` to be.


def _request_id() -> str:
    return f'sdm-{uuid.uuid4().hex[:12]}'


class TabularAdapter:
    r"""Shapes and reads ``kumo-tabular`` requests."""

    spec: ModelSpec = KUMO_TABULAR

    def explain_error(
        self, error: Exception, payload: dict[str, Any]
    ) -> Exception:
        r"""Nothing to add: this endpoint's refusals already name the column
        they are about."""
        return error

    def close(self) -> None:
        r"""Nothing to release: this adapter sends through the client's own
        transport and opens nothing of its own."""


def _raw_dtype_of(series: pd.Series, declared: str | None = None) -> str:
    r"""Like :func:`_dtype_of`, but keeping UNTYPED as the answer.

    A column with no values contributes nothing to a widening. Turning it into
    ``'string'`` first is what made it decide the outcome.

    A `float64` that is float only because pandas has no integer that holds a
    null is reported as the integer it was declared to be. Widening on it sent
    an id through a float: `9007199254740993` left as `9007199254740992.0`,
    which is the corruption the integer rules exist to prevent, arriving by the
    one route that looks like an ordinary type difference. The values decide,
    not the dtype. A genuine float column of whole numbers is still a float,
    because nothing declared it otherwise.
    """
    if (
        declared
        and declared.startswith('int')
        and pd.api.types.is_float_dtype(series.dtype)
        and series.isna().any()
    ):
        present = series.dropna()
        if present.empty or bool((present == present.astype('int64')).all()):
            return declared
    return infer_tfm_dtype(series)


def _a_row_the_pinned_schema_cannot_hold(
    request: Request,
    declared: Mapping[str, str] | None,
    error: Exception,
) -> Exception:
    r"""Name the session, for a failure about a dtype the caller never wrote.

    The scoring rows are written to the schema the session pinned, so a value
    this package cannot write that way fails naming a type the caller did not
    choose, or where pandas raises first, as a bare `ValueError` from
    inside. Which column and which two types is what makes it actionable.
    """
    if declared is None:
        if isinstance(error, (StructuredDataError, UnsupportedRequestError)):
            return error
        return StructuredDataError(str(error), code='INVALID_REQUEST')
    try:
        predict = request.predict.instance_table
    except Exception:  # noqa: BLE001 - already reporting a failure
        return StructuredDataError(str(error), code='INVALID_REQUEST')
    differing = [
        f'{name} was pinned as {declared[name]} and these rows hold '
        f'{predict[name].dtype}'
        for name in predict.columns
        if name in declared and str(predict[name].dtype) != declared[name]
    ]
    if not differing:
        return error
    return StructuredDataError(
        f'these rows cannot be written to the schema this session pinned: '
        f'{"; ".join(differing)}. A scoring call does not send a schema. It '
        f'was settled by fit(), so the rows have to be writable as it '
        f'describes them. Give the column the type it was pinned with, or '
        f'fit() again on a context that has the type you want. ({error})',
        code='INVALID_REQUEST',
    )


#: What a dtype means when nothing else says. Only reached for a column whose
#: meaning was never claimed, so it is the dtype's own reading rather than a
#: guess about the data.
