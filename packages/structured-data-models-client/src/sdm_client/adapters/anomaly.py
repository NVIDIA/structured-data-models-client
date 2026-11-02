# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Shaping a request for ``kumo-anomaly``.

Short, because this model is asked the least of any here. There is no target
column to type, no context to agree with the rows being scored, and no session
to pin, so what is left is the schema the readings are written against.

The envelope still carries an empty ``context``. The contract requires that
section, and a model with nothing to learn from has a context with no rows
rather than no context, which is a thing the envelope can say.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from sdm_api.generated.tfm_api import (
    PredictionItem,
    PredictionResponse,
    ResponseContractError,
)

from sdm_client.base import ModelSpec
from sdm_client.errors import StructuredDataError, UnsupportedRequestError
from sdm_client.models import KUMO_ANOMALY
from sdm_client.request import Request
from sdm_client.result import PredictionResult
from sdm_client.wire import (
    UNTYPED,
    encode_table,
    infer_tfm_dtype,
    json_ready,
    widen_tfm_dtype,
)
from sdm_client.wire.inference import typed_inference

_DEFAULT_OUTPUTS = ('prediction', 'probabilities')


def _request_id() -> str:
    return f'sdm-{uuid.uuid4().hex[:12]}'


class AnomalyAdapter:
    r"""Shapes and reads ``kumo-anomaly`` requests."""

    spec: ModelSpec = KUMO_ANOMALY

    def default_output(self, request: Request) -> tuple[str, ...]:
        return _DEFAULT_OUTPUTS

    def prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterator[dict[str, Any]]:
        r"""One payload. The score for a row depends on the rows around it,
        so splitting a series into batches would change the answer."""
        yield self._envelope(request, outputs)

    def session_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        raise UnsupportedRequestError(
            f'{self.spec.model_id} holds no session: it is shown no labelled '
            f'rows, so there is no fitted context to pin and score against. '
            f'Call predict(), which sends the readings with every call'
        )

    def session_prediction_plan(
        self,
        request: Request,
        outputs: tuple[str, ...],
        pinned: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        raise UnsupportedRequestError(f'{self.spec.model_id} holds no session')

    def normalize(
        self,
        response: dict[str, Any],
        outputs: tuple[str, ...],
        payload: dict[str, Any] | None = None,
    ) -> PredictionResult:
        try:
            parsed = PredictionResponse.from_dict(response)
        except (ResponseContractError, TypeError, ValueError) as error:
            raise StructuredDataError(
                f'{self.spec.model_id} returned a response that does not '
                f'match the contract: {error}',
                code='INVALID_RESPONSE',
            ) from error
        return self.answered(
            parsed.predictions,
            outputs,
            payload,
            model=parsed.model,
            request_id=parsed.id,
            metadata=parsed.metadata,
        )

    def answered(
        self,
        items: Sequence[PredictionItem],
        outputs: tuple[str, ...],
        payload: dict[str, Any] | None,
        *,
        model: str,
        request_id: str = '',
        metadata: Mapping[str, Any] | None = None,
    ) -> PredictionResult:
        return PredictionResult(
            model=model,
            request_id=request_id,
            items=items,
            metadata=dict(metadata or {}),
            outputs=outputs,
        )

    def explain_error(
        self, error: Exception, payload: dict[str, Any]
    ) -> Exception:
        r"""Nothing to add: a refusal here is about the readings, which the
        caller handed over whole."""
        return error

    def close(self) -> None:
        r"""Nothing to release."""

    def _envelope(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        if request.payloads is not None:
            raise UnsupportedRequestError(
                f'this request was gathered for another model and carries '
                f'the sampled payloads that model reads; '
                f'{self.spec.model_id} is asked with Request.anomaly(...)'
            )
        request.schema.check_consistent()

        table = request.schema[request.entity_table]
        readings = request.predict.instance_table

        dtypes = {
            name: (UNTYPED if column.untyped else column.dtype)
            for name, column in table.columns.items()
        }
        for name in readings.columns:
            if name in dtypes:
                dtypes[name] = widen_tfm_dtype(
                    (dtypes[name], infer_tfm_dtype(readings[name]))
                )
        dtypes = {name: dtype or 'string' for name, dtype in dtypes.items()}

        missing = [name for name in readings.columns if name not in dtypes]
        if missing:
            raise StructuredDataError(
                f'schema does not describe column(s) {missing}',
                code='INVALID_REQUEST',
            )

        instance_schema: dict[str, Any] = {
            'columns': {
                name: {'dtype': dtypes[name], 'stype': table[name].stype}
                for name in readings.columns
            }
        }
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
                    'column_name': request.task.target.column_name,
                    'dtype': request.task.target.dtype,
                },
            },
            'schema': {'instance_table': instance_schema},
            'context': {
                'instance_table': encode_table(readings.iloc[:0], dtypes)
            },
            'predict': {'instance_table': encode_table(readings, dtypes)},
            'output': {'fields': list(outputs)},
            'metadata': {'request_id': _request_id()},
        }
        if request.inference:
            envelope['inference'] = typed_inference(
                json_ready(request.inference)
            )
        return envelope
