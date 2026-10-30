# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public numeric-effects journeys shared by frozen recording and acceptance."""

from __future__ import annotations

import json
import warnings
from decimal import InvalidOperation
from typing import Any

import numpy as np
import pandas as pd

from sdm_client import KumoTabular, PredictiveQuery, SDMClient

_ACTIONS = (
    'call',
    'raise',
    'ignore',
    'switch_raise',
    'mutate_zero',
    'mutate_nan',
    'type_error',
    'value_error',
    'invalid_operation',
    'overflow_error',
    'control',
    'signed_overflow',
    'unsigned_overflow',
) + (('promotion_legacy',) if hasattr(np, '_set_promotion_state') else ())
CASES = tuple(
    f'{action}/{direction}/indices'
    for action in _ACTIONS
    for direction in ('forward', 'reverse')
) + tuple(
    f'mutate_nan/{direction}/predict' for direction in ('forward', 'reverse')
)


class CallbackStop(BaseException):
    """A process-control-shaped exception whose identity must survive cleanup."""


class Endpoint:
    url = 'http://localhost'

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body)
        return {
            'model': 'kumo-tabular',
            'id': 'numeric-effects',
            'predictions': [{'row_index': 0, 'prediction': 0.0}],
            'metadata': {},
        }

    def predict_body(self, body: bytes) -> bytes:
        return json.dumps(self.predict(json.loads(body))).encode()

    def close(self) -> None:
        pass


def observe_case(
    name: str, *, value_error: type[ValueError] = ValueError
) -> dict[str, Any]:
    action, direction, entrypoint = name.split('/')
    n, wanted = (81, 40) if entrypoint == 'indices' else (10002, 10000)
    sampled: list[pd.DataFrame] = []

    class Frame(pd.DataFrame):
        @property
        def _constructor(self) -> type[Frame]:
            return Frame

        def take(
            self, indices: Any, axis: int = 0, **kwargs: Any
        ) -> pd.DataFrame:
            result = super().take(indices, axis=axis, **kwargs)
            if axis == 0 and len(self) == n - 1 and len(result) == wanted:
                sampled.append(result)
            return result

    owners = (
        [np.int8(1), 128]
        if action == 'signed_overflow'
        else [np.uint8(1), -1]
        if action == 'unsigned_overflow'
        else [np.float32(1), 1e100]
    )
    if direction == 'reverse':
        owners.reverse()
    labels = np.empty(n, dtype=object)
    for at in range(n):
        labels[at] = owners[at % 2]
    frame = Frame(
        {
            'id': range(n),
            'feature': np.arange(n) * 0.5,
            'target': np.arange(n) + 100.0,
        },
        index=pd.Index(labels, dtype=object),
    )
    query = PredictiveQuery(
        'PREDICT data.target FOR EACH data.id', {'data': frame}
    )
    events: list[list[Any]] = []
    failure: BaseException | None = None

    def handler(error: str, flag: int) -> None:
        nonlocal failure
        events.append([error, flag])
        if len(events) != 1:
            return
        if action == 'switch_raise':
            np.seterr(over='raise')
        elif action in ('mutate_zero', 'mutate_nan'):
            assert len(sampled) == 1
            sampled[0].index._values[:] = (
                np.float32(0) if action == 'mutate_zero' else float('nan')
            )
        elif action == 'promotion_legacy':
            assert set_promotion is not None
            set_promotion('legacy')
        elif action in (
            'type_error',
            'value_error',
            'invalid_operation',
            'overflow_error',
            'control',
        ):
            kind = {
                'type_error': TypeError,
                'value_error': value_error,
                'invalid_operation': InvalidOperation,
                'overflow_error': OverflowError,
                'control': CallbackStop,
            }[action]
            failure = kind('numeric handler refusal')
            raise failure

    get_promotion = getattr(np, '_get_promotion_state', None)
    set_promotion = getattr(np, '_set_promotion_state', None)
    previous_promotion = None if get_promotion is None else get_promotion()
    previous_errors = np.geterr()
    previous_handler = np.geterrcall()
    answer: dict[str, Any] = {}
    try:
        if set_promotion is not None:
            set_promotion('weak')
        np.seterr(over=action if action in ('raise', 'ignore') else 'call')
        np.seterrcall(handler)
        with warnings.catch_warnings(record=True) as caught:
            try:
                request = query.materialize(
                    **(
                        {'indices': [n - 1], 'num_context_samples': wanted}
                        if entrypoint == 'indices'
                        else {
                            'predict': frame.iloc[[-1]].drop(columns=['target'])
                        }
                    ),
                    task='regression',
                )
                answer['context_ids'] = request.context.instance_table[
                    'id'
                ].tolist()
                endpoint = Endpoint()
                client = SDMClient.__new__(SDMClient)
                client._start(endpoint)
                try:
                    result = client.predict(request, model=KumoTabular())
                    answer['encoded_ids'] = [
                        row[0]
                        for row in endpoint.calls[0]['context'][
                            'instance_table'
                        ]['rows']
                    ]
                    answer['result'] = result.to_pandas().to_dict(
                        orient='records'
                    )
                finally:
                    client.close()
            except (Exception, CallbackStop) as error:  # noqa: BLE001 - public observation
                answer['error'] = {
                    'type': type(error).__name__,
                    'message': str(error),
                    'original_callback_exception': error is failure,
                }
            answer['warnings'] = [
                {'type': type(w.message).__name__, 'message': str(w.message)}
                for w in caught
            ]
            answer['shared_warnings'] = [
                {'type': type(w.message).__name__, 'message': str(w.message)}
                for w in caught
                if not w.category.__module__.startswith('pandas.')
            ]
    finally:
        np.seterr(**previous_errors)
        np.seterrcall(previous_handler)
        if set_promotion is not None:
            set_promotion(previous_promotion)
    answer['events'] = events
    answer['sampled_frames'] = len(sampled)
    return answer
