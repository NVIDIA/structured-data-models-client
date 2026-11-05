# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Shaping a request for ``kumo-forecast``.

The history travels as the context, and the horizon as rows to score that
carry no columns. That is the model's own shape: it reads nothing from those
rows, and the count of them is the number of steps asked for.

Nothing known about the future goes out, because nothing known about the
future is read. The model discards a future covariate with a warning nobody
is obliged to see, so `Request.forecast` offers no way to supply one rather
than accepting a column and dropping it.
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

from sdm_client.adapters.single_table import (
    meaning_a_single_meaning_model_reads,
)
from sdm_client.base import ModelSpec
from sdm_client.errors import StructuredDataError, UnsupportedRequestError
from sdm_client.models import KUMO_FORECAST
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

_DEFAULT_OUTPUTS = ('prediction',)


def _request_id() -> str:
    return f'sdm-{uuid.uuid4().hex[:12]}'


class ForecastAdapter:
    r"""Shapes and reads ``kumo-forecast`` requests."""

    spec: ModelSpec = KUMO_FORECAST

    def default_output(self, request: Request) -> tuple[str, ...]:
        return _DEFAULT_OUTPUTS

    def prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterator[dict[str, Any]]:
        r"""One payload. Each step is forecast from the ones before it, so
        splitting the horizon would ask a different question of each part."""
        yield self._envelope(request, outputs)

    def session_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        raise UnsupportedRequestError(
            f'{self.spec.model_id} holds no session: it fits nothing, so '
            f'there is no context to pin. Call predict(), which sends the '
            f'history with every call'
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
        unstepped = [
            item.row_index for item in items if item.forecast_step is None
        ]
        if unstepped:
            raise StructuredDataError(
                f'a forecast answer for row(s) {unstepped} names no step, so '
                f'it cannot be put in order',
                code='INVALID_RESPONSE',
            )
        result = PredictionResult(
            model=model,
            request_id=request_id,
            items=items,
            metadata=dict(metadata or {}),
            outputs=outputs,
        )
        result._names_its_rows = False
        result._names_its_step = True
        return result

    def explain_error(
        self, error: Exception, payload: dict[str, Any]
    ) -> Exception:
        r"""Nothing to add: a refusal here is about the history, which the
        caller handed over whole."""
        return error

    def close(self) -> None:
        r"""Nothing to release."""

    def _envelope(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        r"""The request on the wire, refused where it carries no horizon.

        A request without one was built by the constructor that takes a half
        to score. Those rows are the future, and this model reads nothing
        about the future: it would discard every value in them and answer
        from the count alone, which is a different question answered
        quietly.
        """
        if request.payloads is not None:
            raise UnsupportedRequestError(
                f'this request was gathered for another model and carries '
                f'the sampled payloads that model reads; '
                f'{self.spec.model_id} is asked with Request.forecast(...)'
            )
        horizon = getattr(request, 'horizon', None)
        if horizon is None:
            raise UnsupportedRequestError(
                f'{self.spec.model_id} forecasts from a history and a number '
                f'of steps, and reads nothing about the steps themselves. '
                f'Build the request with '
                f'Request.forecast(history, target=..., horizon=...)'
            )
        request.schema.check_consistent()

        table = request.schema[request.entity_table]
        history = request.context.instance_table
        target = request.task.target.column_name

        dtypes = {
            name: (UNTYPED if column.untyped else column.dtype)
            for name, column in table.columns.items()
        }
        for name in history.columns:
            if name in dtypes:
                dtypes[name] = widen_tfm_dtype(
                    (dtypes[name], infer_tfm_dtype(history[name]))
                )
        dtypes = {name: dtype or 'string' for name, dtype in dtypes.items()}

        missing = [name for name in history.columns if name not in dtypes]
        if missing:
            raise StructuredDataError(
                f'schema does not describe column(s) {missing}',
                code='INVALID_REQUEST',
            )

        instance_schema: dict[str, Any] = {
            'columns': {
                name: {
                    'dtype': dtypes[name],
                    'stype': meaning_a_single_meaning_model_reads(
                        table[name], dtypes[name]
                    ),
                }
                for name in history.columns
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
                    'column_name': target,
                    'dtype': dtypes[target],
                },
                'num_forecasts': horizon,
            },
            'schema': {'instance_table': instance_schema},
            'context': {'instance_table': encode_table(history, dtypes)},
            'predict': {
                'instance_table': encode_table(
                    request.predict.instance_table, dtypes
                )
            },
            'output': {'fields': list(outputs)},
            'metadata': {'request_id': _request_id()},
        }
        if request.inference:
            envelope['inference'] = typed_inference(
                json_ready(request.inference)
            )
        return envelope
