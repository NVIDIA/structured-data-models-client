# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Sending a relational request to ``kumo-relational``.

Unlike the tabular model, this one does heavy client-side work: building the
graph, resolving the query, sampling each entity's neighbourhood and encoding
the result. All of that happens in :meth:`PredictiveQuery.materialize`, which
runs in the engine, so by the time a request reaches this adapter its payloads
are already finished.

That split is why this adapter is short. It sends what was gathered, and reads
the answer back. It does not reimplement the gathering, which would duplicate
point-in-time correctness logic that already lives -- and is tested -- in the
engine.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

from sdm_api.generated.tfm_api import PredictionResponse, ResponseContractError

from sdm_client.base import ModelSpec
from sdm_client.errors import StructuredDataError, UnsupportedRequestError
from sdm_client.models import KUMO_RELATIONAL
from sdm_client.request import Request
from sdm_client.result import PredictionResult

# What to ask for when the caller named nothing, by task kind. Asking for
# everything would make a regression request carry probabilities it cannot
# have, which the endpoint rejects.
_DEFAULT_OUTPUTS = {
    'classification': ('prediction', 'probabilities'),
    'binary_classification': ('prediction', 'probabilities'),
    'multiclass_classification': ('prediction', 'probabilities'),
    'regression': ('prediction',),
    'forecasting': ('prediction',),
    'temporal_link_prediction': ('rankings',),
}

_SESSION_PINNED = ('model', 'task', 'schema', 'context')
_SESSION_SCORING = ('predict', 'output', 'metadata')


def _request_id() -> str:
    return f'sdm-{uuid.uuid4().hex[:12]}'


class RelationalAdapter:
    r"""Shapes and reads ``kumo-relational`` requests."""

    spec: ModelSpec = KUMO_RELATIONAL

    def default_output(self, request: Request) -> tuple[str, ...]:
        return _DEFAULT_OUTPUTS.get(request.task.kind, ('prediction',))

    def prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterator[dict[str, Any]]:
        payloads = self._payloads(request)
        self._preflight(payloads, outputs)
        for payload in payloads:
            envelope = dict(payload)
            envelope['model'] = self.spec.model_id
            envelope['output'] = {'fields': list(outputs)}
            envelope.setdefault('metadata', {})['request_id'] = _request_id()
            inference = self._inference(envelope.get('inference'), request)
            if inference is None:
                envelope.pop('inference', None)
            else:
                envelope['inference'] = inference
            yield envelope

    def _inference(
        self, inference: Any, request: Request
    ) -> dict[str, Any] | None:
        """Send only the inference options this deployment accepts.

        The engine was written against a different backend and fills in an
        inference config on every request, including keys this one refuses
        outright. Dropping them is safe because they are the engine's defaults
        rather than anything asked for -- but if the *caller* asked for one, it
        is named rather than silently discarded, because a dropped request for
        four estimators is a different prediction than the one that was wanted.
        """
        if not isinstance(inference, dict):
            return None
        config = inference.get('inference_config')
        if not isinstance(config, dict):
            return dict(inference)

        accepted = self.spec.runtime.inference_options
        unsupported = sorted(set(config) - accepted)
        asked_for = set(
            request.requested_inference.get('inference_config') or ()
        )
        refused = sorted(set(unsupported) & asked_for)
        if refused:
            raise UnsupportedRequestError(
                f'{self.spec.model_id} does not accept inference option(s) '
                f'{refused}; it accepts {sorted(accepted)}',
                details={
                    'model': self.spec.model_id,
                    'unsupported': refused,
                },
            )

        trimmed = dict(inference)
        trimmed['inference_config'] = {
            key: value for key, value in config.items() if key in accepted
        }
        return trimmed

    def session_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        payloads = self._payloads(request)
        if len(payloads) > 1:
            # A session pins one context. Several gathered parts each carry
            # their own, so pinning the first would score later batches
            # against the wrong one.
            raise UnsupportedRequestError(
                f'this request gathered {len(payloads)} batches, and a session '
                f'pins one context; use predict() instead',
                details={'batches': len(payloads)},
            )
        envelope = next(iter(self.prediction_plan(request, outputs)))
        pinned = {
            key: envelope[key] for key in _SESSION_PINNED if key in envelope
        }
        pinned['metadata'] = envelope['metadata']
        return pinned

    def session_prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterator[dict[str, Any]]:
        for envelope in self.prediction_plan(request, outputs):
            yield {
                key: envelope[key]
                for key in _SESSION_SCORING
                if key in envelope
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
            raise StructuredDataError(
                f'{self.spec.model_id} returned a response that does not '
                f'match the contract: {error}',
                code='INVALID_RESPONSE',
            ) from error

        explanation = next(
            (
                item.explanation
                for item in parsed.predictions
                if item.explanation
            ),
            None,
        )
        return PredictionResult(
            model=parsed.model,
            request_id=parsed.id,
            items=_named_entities(parsed.predictions, payload),
            metadata=parsed.metadata,
            explanation=explanation,
        )

    def close(self) -> None:
        r"""Nothing to release: gathering happens in the query, which owns
        whatever it opened, and sending goes through the client's transport."""

    def _preflight(
        self, payloads: tuple[dict[str, Any], ...], outputs: tuple[str, ...]
    ) -> None:
        """Refuse what the endpoint enforces, before spending the round trip.

        An explanation is the narrow case: the model explains one entity at a
        time, and only within the smaller context budget. Sending more gets a
        500 that names neither limit, so both are checked here.
        """
        if 'explanation' not in outputs:
            return

        limits = self.spec.runtime.limits
        scored = sum(
            len(
                payload.get('predict', {})
                .get('instance_table', {})
                .get('rows', [])
            )
            for payload in payloads
        )
        allowed = limits['explanation_predict_rows']
        if scored != allowed:
            raise UnsupportedRequestError(
                f'an explanation covers exactly {allowed} entity, and this '
                f'request scores {scored}; ask for one at a time',
                details={'model': self.spec.model_id, 'scored': scored},
            )

        context = sum(
            len(
                payload.get('context', {})
                .get('instance_table', {})
                .get('rows', [])
            )
            for payload in payloads
        )
        cap = limits['context_rows_fast']
        if context > cap:
            raise UnsupportedRequestError(
                f'an explanation is only produced within the smaller context '
                f'budget of {cap:,} rows, and this request gathered {context:,}. '
                f'Re-materialize with num_context_samples={cap} or fewer',
                details={'model': self.spec.model_id, 'context_rows': context},
            )

    @staticmethod
    def _payloads(request: Request) -> tuple[dict[str, Any], ...]:
        payloads = request.payloads
        if not payloads:
            raise UnsupportedRequestError(
                'a relational request has to be gathered first. Build it with '
                'PredictiveQuery(...).materialize(indices=[...]) rather than '
                'with Request(context=..., predict=...), which is the '
                'single-table form',
                details={'model': KUMO_RELATIONAL.model_id},
            )
        return payloads


def _named_entities(
    items: tuple[Any, ...], payload: dict[str, Any] | None
) -> list[Any]:
    """Put the caller's entity back on each row, and check it belongs there.

    Entity values never travel: the engine sends an opaque reference and an
    internal instance id, so a response's ``id`` is the transport key rather
    than the user's key. The reference column carries the entity and the
    instance-id column carries the key the endpoint answers under, so the two
    together restore the mapping. It is read from the payload that produced
    this response, so a run split into several calls maps each batch against
    its own rows.

    **A mismatch is an error, not a silent pass.** Rewriting an id is exactly
    the operation that can misattribute an answer, so every row is checked
    against the batch it claims to belong to before its id is replaced.

    Forecasting is the one task that answers a row more than once, and only
    then: a row may repeat across steps, but not within one.
    """
    import dataclasses

    if payload is None:
        return list(items)
    table = payload.get('predict', {}).get('instance_table', {})
    columns = table.get('columns') or []
    rows = table.get('rows') or []
    reference = next(
        (name for name in columns if name.startswith('__kumo_entity_ref')),
        None,
    )
    if reference is None:
        return list(items)

    entities = [row[columns.index(reference)] for row in rows]
    instances = (
        [row[columns.index('instance_id')] for row in rows]
        if 'instance_id' in columns
        else None
    )
    forecasting = _forecast_task(payload)

    named: list[Any] = []
    seen: set[tuple[int, int | None]] = set()
    for item in items:
        index = item.row_index
        if index is None:
            raise StructuredDataError(
                'the response is missing a row_index, so its answer cannot be '
                'attributed to an entity',
                code='INVALID_RESPONSE',
            )
        if not 0 <= index < len(entities):
            raise StructuredDataError(
                f'the response names row_index {index}, and this batch asked '
                f'about {len(entities)} rows',
                code='INVALID_RESPONSE',
            )

        step = item.forecast_step
        if forecasting:
            if step is None:
                raise StructuredDataError(
                    f'a forecast answer for row {index} names no step, so its '
                    f'place in the horizon is unknown',
                    code='INVALID_RESPONSE',
                )
        elif step is not None:
            raise StructuredDataError(
                f'a {payload.get("task", {}).get("kind")!r} answer carries a '
                f'forecast step, which only a forecast has',
                code='INVALID_RESPONSE',
            )

        key = (index, step)
        if key in seen:
            where = (
                f'row {index} step {step}' if forecasting else f'row {index}'
            )
            raise StructuredDataError(
                f'the response answers {where} more than once',
                code='INVALID_RESPONSE',
            )
        seen.add(key)

        if instances is not None:
            if item.id is None:
                raise StructuredDataError(
                    f'the answer for row {index} carries no id, and this batch '
                    f'asked under {instances[index]!r}, so the two cannot be '
                    f'matched',
                    code='INVALID_RESPONSE',
                )
            if str(item.id) != str(instances[index]):
                raise StructuredDataError(
                    f'the response pairs row_index {index} with id '
                    f'{item.id!r}, and this batch asked about '
                    f'{instances[index]!r} there',
                    code='INVALID_RESPONSE',
                )
        named.append(dataclasses.replace(item, id=str(entities[index])))

    answered = {index for index, _ in seen}
    if answered != set(range(len(entities))):
        missing = sorted(set(range(len(entities))) - answered)
        raise StructuredDataError(
            f'the response left row(s) {missing} unanswered',
            code='INVALID_RESPONSE',
        )
    return named


def _forecast_task(payload: dict[str, Any]) -> bool:
    """Whether this batch asked a forecast.

    Read from the request rather than from the shape of the answer. Deciding
    it from the answer let a single item carrying a step switch off the
    duplicate guard for every other row in the response.
    """
    return payload.get('task', {}).get('kind') == 'forecasting'
