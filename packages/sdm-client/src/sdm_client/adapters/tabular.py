# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Sending a single-table request to ``kumo-tabular``.

This model does no heavy client-side work: there is no graph to build and
nothing to sample, so the adapter shapes the envelope, sends it, and reads the
answer back. That is why the tabular path ships in the base install and runs
anywhere, while the relational one needs a native extra.

Two rules this adapter holds to, both of which the endpoint enforces and would
otherwise report as an opaque 422.

The **target's declared dtype must equal the schema's dtype for that column**.
They are taken from one place here so they cannot drift.

The **task kind is never narrowed**. The endpoint accepts only the coarse
``classification`` and decides binary versus multiclass itself from the
context, so a client that narrowed it would both be guessing and be rejected.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pandas as pd
from sdm_api.generated.tfm_api import (
    PredictionResponse,
    ResponseContractError,
)

from sdm_client.base import ModelSpec
from sdm_client.errors import StructuredDataError
from sdm_client.models import KUMO_TABULAR
from sdm_client.request import Request
from sdm_client.result import PredictionResult
from sdm_client.wire import encode_table, widen_tfm_dtype

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
_SESSION_SCORING = ('predict', 'output', 'metadata')


def _request_id() -> str:
    return f'sdm-{uuid.uuid4().hex[:12]}'


class TabularAdapter:
    r"""Shapes and reads ``kumo-tabular`` requests."""

    spec: ModelSpec = KUMO_TABULAR

    def default_output(self, request: Request) -> tuple[str, ...]:
        return _DEFAULT_OUTPUTS.get(request.task.kind, ('prediction',))

    def prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterator[dict[str, Any]]:
        r"""One payload. The endpoint caps nothing this model has to split on."""
        yield self._envelope(request, outputs)

    def session_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        envelope = self._envelope(request, outputs)
        pinned = {
            key: envelope[key] for key in _SESSION_PINNED if key in envelope
        }
        pinned['metadata'] = envelope['metadata']
        return pinned

    def session_prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterator[dict[str, Any]]:
        envelope = self._envelope(request, outputs)
        yield {
            key: envelope[key] for key in _SESSION_SCORING if key in envelope
        }

    def normalize(
        self,
        response: dict[str, Any],
        outputs: tuple[str, ...],
        payload: dict[str, Any] | None = None,
    ) -> PredictionResult:
        try:
            parsed = PredictionResponse.from_dict(response)
        except (ResponseContractError, TypeError, ValueError) as error:
            # A 200 whose body does not match the contract is still a failure,
            # and it has to arrive as one of this client's errors rather than
            # as a binding-layer exception nobody is catching.
            raise StructuredDataError(
                f'{self.spec.model_id} returned a response that does not '
                f'match the contract: {error}',
                code='INVALID_RESPONSE',
            ) from error
        return PredictionResult(
            model=parsed.model,
            request_id=parsed.id,
            items=parsed.predictions,
            metadata=parsed.metadata,
        )

    def close(self) -> None:
        r"""Nothing to release: this adapter sends through the client's own
        transport and opens nothing of its own."""

    def _envelope(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        table = request.schema[request.entity_table]
        context = request.context.instance_table
        predict = request.predict.instance_table

        # A column's dtype has to describe both halves. Taking it from the
        # context alone would call an all-null context column 'string' and then
        # send integers under it from the predict half.
        dtypes = {name: column.dtype for name, column in table.columns.items()}
        for name in predict.columns:
            if name in dtypes:
                dtypes[name] = widen_tfm_dtype(
                    (dtypes[name], _dtype_of(predict[name]))
                )

        target = request.task.target.column_name
        missing = [name for name in context.columns if name not in dtypes]
        if missing:
            raise StructuredDataError(
                f'schema does not describe context column(s) {missing}',
                code='INVALID_REQUEST',
            )

        schema_columns = {
            name: {
                'dtype': dtypes[name],
                'stype': self._stype_for(request, table[name].stype, name),
            }
            for name in context.columns
        }
        instance_schema: dict[str, Any] = {'columns': schema_columns}
        key = table.primary_key
        if key:
            instance_schema['primary_key'] = (
                [key] if isinstance(key, str) else list(key)
            )

        envelope: dict[str, Any] = {
            'model': self.spec.model_id,
            'task': {
                'kind': request.task.kind,
                'target': {
                    'column_name': target,
                    # The one place this dtype is decided, so it cannot differ
                    # from the schema's answer for the same column.
                    'dtype': dtypes[target],
                },
            },
            'schema': {'instance_table': instance_schema},
            'context': {'instance_table': encode_table(context, dtypes)},
            'predict': {'instance_table': encode_table(predict, dtypes)},
            'output': {'fields': list(outputs)},
            'metadata': {'request_id': _request_id()},
        }
        if request.inference:
            envelope['inference'] = dict(request.inference)
        return envelope

    @staticmethod
    def _stype_for(request: Request, inferred: str, column: str) -> str:
        r"""The semantic type to declare for one column.

        Schema inference is task-agnostic, so an integer label column comes
        back ``numerical`` -- which is right for a regression target and wrong
        for a classification one. The task is what settles it, and only the
        adapter knows the task, so the correction happens here rather than
        being baked into the schema a caller may reuse for another question.
        """
        if column != request.task.target.column_name:
            return inferred
        if request.task.is_classification:
            return 'categorical'
        return 'numerical' if inferred == 'categorical' else inferred


def _dtype_of(series: pd.Series) -> str:
    from sdm_client.wire import UNTYPED, infer_tfm_dtype

    dtype = infer_tfm_dtype(series)
    return dtype if dtype != UNTYPED else 'string'
