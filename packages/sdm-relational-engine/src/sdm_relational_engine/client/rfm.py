# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from sdm_api.endpoints import Endpoint, HTTPMethod
from sdm_api.generated.tfm_api import (
    PredictionItem,
    PredictionResponse,
    TFMOperations,
)

from sdm_relational_engine.client.transport import RFMTransport
from sdm_relational_engine.client.utils import raise_on_error
from sdm_relational_engine.exceptions import InvalidResponseError
from sdm_relational_engine.rfm.inference.requests import RFMPredictResponse


def _path_segment(value: str) -> str:
    r"""Escape a server-supplied ID for use in a URL path.

    The session id comes back from the NIM, so it is not ours to trust: an
    id carrying a slash would otherwise redirect the call to a different
    route. ``quote`` does not escape dots because they are unreserved. A
    resolver removes ``.`` and ``..``, so those two values name no session and
    are refused. ``...`` is an ordinary segment and is escaped like any other.
    An empty ID is refused because it contributes no segment.
    """
    if value in {'', '.', '..'}:
        raise InvalidResponseError(
            f'{value!r} is not a session id: it does not survive as a path '
            f'segment when the URL is resolved, so the call would be made '
            f'against a different route than the one it names'
        )
    return quote(value, safe='')


class RFMAPI:
    r"""Define the typed KumoRelational API."""

    def __init__(self, client: RFMTransport) -> None:
        self._client = client

    def predict(
        self,
        request: Mapping[str, Any],
        *,
        entity_ids: Sequence[Any],
        instance_ids: Sequence[Any],
        anchor_times: Sequence[Any] | None = None,
    ) -> RFMPredictResponse:
        r"""Make predictions using the RFM model.

        Args:
            request: The predict request as a universal TFM JSON envelope.
            entity_ids: Private, batch-local entity values ordered like the
                predict instance table. Response row indexes are correlated
                back to these values.
            instance_ids: Generated transport keys ordered like the predict
                instance table, used only to validate response correlation.
            anchor_times: Anchor timestamps ordered like the predict instance
                table. The NIM response does not echo them back, so they are
                re-attached here to keep the ``ANCHOR_TIMESTAMP`` column of
                the prediction table.

        Returns:
            RFMPredictResponse containing the predictions
        """
        response = self._client._request(
            TFMOperations.run_prediction.endpoint,
            json=request,
            headers={'Content-Type': 'application/json'},
        )
        raise_on_error(response)
        return self._parse_predict_response(
            response,
            entity_ids=entity_ids,
            instance_ids=instance_ids,
            anchor_times=anchor_times,
            requested_model=request.get('model'),
        )

    def create_session(self, request: Mapping[str, Any]) -> str:
        r"""Create a persistent inference session that pins the context.

        The context (``model`` + ``task`` + ``schema`` + ``context``) is
        uploaded once; :meth:`session_predict` then references it by the
        returned ``session_id`` so subsequent batches send only their
        prediction rows.

        A ``200`` with no ``session_id`` is an ``InvalidResponseError``, like a
        malformed prediction body: the request that produced it is the client's
        own generated payload, so the caller cannot have influenced whether the
        server echoes an id and must not be told they did.
        """
        response = self._client._request(
            TFMOperations.create_session.endpoint,
            json=request,
            headers={'Content-Type': 'application/json'},
        )
        raise_on_error(response)
        try:
            body = response.json()
        except ValueError as error:
            raise InvalidResponseError(
                'Create-session response was not valid JSON.'
            ) from error
        session_id = body.get('session_id') if isinstance(body, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise InvalidResponseError(
                'Create-session response did not include a session_id.'
            )
        return session_id

    def session_predict(
        self,
        session_id: str,
        request: Mapping[str, Any],
        *,
        entity_ids: Sequence[Any],
        instance_ids: Sequence[Any],
        anchor_times: Sequence[Any] | None = None,
    ) -> RFMPredictResponse:
        r"""Predict within an existing session.

        ``request`` carries only ``predict`` / ``output`` / ``inference``; the
        session supplies the pinned context. Correlation of the response back
        to entities works exactly as in :meth:`predict`.
        """
        response = self._client._request(
            Endpoint(
                path=f'/v1/sessions/{_path_segment(session_id)}/predictions',
                method=HTTPMethod.POST,
            ),
            json=request,
            headers={'Content-Type': 'application/json'},
        )
        raise_on_error(response)
        return self._parse_predict_response(
            response,
            entity_ids=entity_ids,
            instance_ids=instance_ids,
            anchor_times=anchor_times,
        )

    def delete_session(self, session_id: str) -> None:
        r"""Delete a session. Idempotent server-side (204 whether present or
        already gone); callers treat this as best-effort cleanup.
        """
        self._client._request(
            Endpoint(
                path=f'/v1/sessions/{_path_segment(session_id)}',
                method=HTTPMethod.DELETE,
            )
        )

    @staticmethod
    def _parse_predict_response(
        response: Any,
        *,
        entity_ids: Sequence[Any],
        instance_ids: Sequence[Any],
        anchor_times: Sequence[Any] | None,
        requested_model: str | None = None,
    ) -> RFMPredictResponse:
        # The identity mappings describe the request we sent, so a mismatch
        # among them is a caller-side contract error and must not be reported
        # as a malformed response. Checked before the guard below, which turns
        # everything it catches into an InvalidResponseError.
        _validate_identity_mappings(entity_ids, instance_ids, anchor_times)
        try:
            prediction_response = PredictionResponse.from_dict(response.json())
            _warn_on_unexpected_model(prediction_response, requested_model)
            return _prediction_response_to_rfm(
                prediction_response,
                entity_ids=entity_ids,
                instance_ids=instance_ids,
                anchor_times=anchor_times,
            )
        except InvalidResponseError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InvalidResponseError(
                f'The KumoRelational NIM returned a prediction response that does '
                f'not match the contract '
                f'({type(error).__name__}: {error or "no detail"})'
            ) from error


def _prediction_response_to_rfm(
    response: PredictionResponse,
    *,
    entity_ids: Sequence[Any],
    instance_ids: Sequence[Any],
    anchor_times: Sequence[Any] | None = None,
) -> RFMPredictResponse:
    rows = _correlated_prediction_rows(
        response,
        entity_ids=entity_ids,
        instance_ids=instance_ids,
        anchor_times=anchor_times,
    )

    columns: list[str] = []
    for row in rows:
        for column_name in row:
            if column_name not in columns:
                columns.append(column_name)
    return RFMPredictResponse(
        prediction={
            'columns': columns,
            'data': [
                [row.get(column_name) for column_name in columns]
                for row in rows
            ],
        }
    )


def _prediction_item_to_row(
    item: PredictionItem,
    *,
    entity_id: Any,
    anchor_time: Any = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {'ENTITY': entity_id}
    if anchor_time is not None:
        row['ANCHOR_TIMESTAMP'] = anchor_time
    if item.prediction is not None:
        row['PREDICTION'] = item.prediction
    if item.probabilities is not None:
        for name, value in item.probabilities.items():
            row[f'{name.upper()}_PROB'] = value
    if item.scores is not None:
        row['SCORES'] = list(item.scores)
    if item.rankings is not None:
        row['RANKINGS'] = [dict(ranking) for ranking in item.rankings]
    if item.embeddings is not None:
        row['EMBEDDINGS'] = list(item.embeddings)
    if item.quantiles is not None:
        for name, value in item.quantiles.items():
            row[f'Q_{name.upper()}'] = value
    if item.explanation is not None:
        row['EXPLANATION'] = item.explanation
    return row


def _prediction_item_to_ranking_rows(
    item: PredictionItem,
    *,
    entity_id: Any,
    anchor_time: Any = None,
) -> list[dict[str, Any]]:
    if item.rankings is None:
        raise ValueError(
            'Kumo Relational ranking response is missing rankings.'
        )
    if item.prediction is not None:
        raise ValueError(
            'Kumo Relational ranking response must not include prediction.'
        )

    rows: list[dict[str, Any]] = []
    for ranking in item.rankings:
        if 'id' not in ranking:
            raise ValueError('Kumo Relational ranking item is missing id.')
        if 'score' not in ranking:
            raise ValueError('Kumo Relational ranking item is missing score.')
        extra_fields = set(ranking) - {'id', 'score'}
        if extra_fields:
            unexpected = ', '.join(sorted(extra_fields))
            raise ValueError(
                'Kumo Relational ranking item includes unexpected fields: '
                f'{unexpected}.'
            )
        ranking_id = ranking['id']
        if not isinstance(ranking_id, str):
            raise ValueError(
                'Kumo Relational ranking item id must be a string.'
            )
        score = float(ranking['score'])
        if not math.isfinite(score):
            raise ValueError(
                'Kumo Relational ranking item score must be finite.'
            )

        row: dict[str, Any] = {'ENTITY': entity_id}
        if anchor_time is not None:
            row['ANCHOR_TIMESTAMP'] = anchor_time
        row['CLASS'] = ranking_id
        row['SCORE'] = score
        if item.explanation is not None:
            row['EXPLANATION'] = item.explanation
        rows.append(row)
    return rows


def _warn_on_unexpected_model(
    prediction_response: Any,
    requested_model: str | None,
) -> None:
    r"""Say so when a NIM answers as a model we did not ask for.

    The contract requires the response to carry a ``model`` but does not
    require it to echo the request, so a mismatch is not a protocol violation
    and must not fail the call. It is still worth surfacing: predictions that
    came from a different model than the caller selected otherwise look
    indistinguishable from the ones they wanted.
    """
    served = getattr(prediction_response, 'model', None)
    if not requested_model or not served or served == requested_model:
        return
    logging.getLogger('sdm_relational_engine').warning(
        'Requested model %r but the deployment answered as %r; the '
        'predictions came from the model it names.',
        requested_model,
        served,
    )


def _validate_identity_mappings(
    entity_ids: Sequence[Any],
    instance_ids: Sequence[Any],
    anchor_times: Sequence[Any] | None,
) -> None:
    r"""Check the parallel request mappings against each other.

    These describe the request, not the response, so a mismatch is a caller
    contract error and stays a plain ``ValueError``.
    """
    expected_count = len(entity_ids)
    if len(instance_ids) != expected_count:
        raise ValueError(
            'Kumo Relational request identity mappings have different lengths: '
            f'{expected_count} entities and {len(instance_ids)} instances.'
        )
    if anchor_times is not None and len(anchor_times) != expected_count:
        raise ValueError(
            'Kumo Relational request identity mappings have different lengths: '
            f'{expected_count} entities and {len(anchor_times)} anchor times.'
        )


def _correlated_prediction_rows(
    response: PredictionResponse,
    *,
    entity_ids: Sequence[Any],
    instance_ids: Sequence[Any],
    anchor_times: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    _validate_identity_mappings(entity_ids, instance_ids, anchor_times)
    entities = list(entity_ids)
    instances = list(instance_ids)
    anchors = list(anchor_times) if anchor_times is not None else None
    expected_count = len(entities)
    is_forecast = _is_forecast_response(response)
    forecast_steps_by_index: dict[int, set[int]] = {}
    if not is_forecast and len(response.predictions) != expected_count:
        raise ValueError(
            'Kumo Relational prediction response count does not match the request: '
            f'expected {expected_count}, got {len(response.predictions)}.'
        )

    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    for item in response.predictions:
        row_index = item.row_index
        if row_index is None:
            raise ValueError(
                'Kumo Relational prediction response is missing row_index.'
            )
        if row_index < 0 or row_index >= expected_count:
            raise ValueError(
                'Kumo Relational prediction response row_index is out of range: '
                f'{row_index}.'
            )
        if is_forecast:
            if item.forecast_step is None:
                raise ValueError(
                    'Kumo Relational forecasting response is missing forecast_step.'
                )
            if item.forecast_step <= 0:
                raise ValueError(
                    'Kumo Relational forecasting response forecast_step must be positive.'
                )
            forecast_steps = forecast_steps_by_index.setdefault(
                row_index, set()
            )
            if item.forecast_step in forecast_steps:
                raise ValueError(
                    'Kumo Relational forecasting response contains duplicate '
                    f'forecast_step {item.forecast_step} for row_index {row_index}.'
                )
            forecast_steps.add(item.forecast_step)
        elif row_index in rows_by_index:
            raise ValueError(
                'Kumo Relational prediction response contains duplicate row_index: '
                f'{row_index}.'
            )

        expected_id = str(instances[row_index])
        if item.id != expected_id:
            raise ValueError(
                'Kumo Relational prediction response id does not match request '
                f'instance_id at row_index {row_index}: expected '
                f'{expected_id!r}, got {item.id!r}.'
            )
        rows = _prediction_item_rows_for_response(
            response,
            item,
            entity_id=entities[row_index],
            anchor_time=anchors[row_index] if anchors is not None else None,
        )
        rows_by_index.setdefault(row_index, []).extend(rows)

    missing = [
        index for index in range(expected_count) if index not in rows_by_index
    ]
    if missing:
        raise ValueError(
            'Kumo Relational prediction response is missing row_index values: '
            f'{missing}.'
        )

    return [
        row for index in range(expected_count) for row in rows_by_index[index]
    ]


def _prediction_item_rows_for_response(
    response: PredictionResponse,
    item: PredictionItem,
    *,
    entity_id: Any,
    anchor_time: Any = None,
) -> list[dict[str, Any]]:
    if _is_ranking_response(response):
        return _prediction_item_to_ranking_rows(
            item, entity_id=entity_id, anchor_time=anchor_time
        )
    if _is_multiclass_response(response) and item.probabilities is not None:
        return _prediction_item_to_multiclass_rows(
            item, entity_id=entity_id, anchor_time=anchor_time
        )
    row = _prediction_item_to_row(
        item, entity_id=entity_id, anchor_time=anchor_time
    )
    if _is_forecast_response(response):
        row['FORECAST_STEP'] = item.forecast_step
    return [row]


def _prediction_item_to_multiclass_rows(
    item: PredictionItem,
    *,
    entity_id: Any,
    anchor_time: Any = None,
) -> list[dict[str, Any]]:
    r"""Return one row per class, sorted by descending score.

    Each row contains (CLASS, SCORE, PREDICTED), matching the long, per-class
    layout users had for multi-class classification on the SaaS client. This is
    easier to read than a wide column-per-class table.
    """
    predicted = None if item.prediction is None else str(item.prediction)
    ranked = sorted(
        item.probabilities.items(), key=lambda kv: kv[1], reverse=True
    )
    rows: list[dict[str, Any]] = []
    for name, value in ranked:
        row: dict[str, Any] = {'ENTITY': entity_id}
        if anchor_time is not None:
            row['ANCHOR_TIMESTAMP'] = anchor_time
        row['CLASS'] = name
        row['SCORE'] = value
        row['PREDICTED'] = str(name) == predicted
        if item.explanation is not None:
            row['EXPLANATION'] = item.explanation
        rows.append(row)
    return rows


def _is_ranking_response(response: PredictionResponse) -> bool:
    task_kind = str(response.metadata.get('task_kind', '')).lower()
    output_type = str(response.metadata.get('output_type', '')).lower()
    return (
        task_kind in {'ranking', 'temporal_link_prediction'}
        or output_type == 'rankings'
    )


def _is_forecast_response(response: PredictionResponse) -> bool:
    task_kind = str(response.metadata.get('task_kind', '')).lower()
    output_type = str(response.metadata.get('output_type', '')).lower()
    return task_kind in {'forecast', 'forecasting'} or output_type == 'forecast'


def _is_multiclass_response(response: PredictionResponse) -> bool:
    task_kind = str(response.metadata.get('task_kind', '')).lower()
    return task_kind in {'multiclass', 'multiclass_classification'}
