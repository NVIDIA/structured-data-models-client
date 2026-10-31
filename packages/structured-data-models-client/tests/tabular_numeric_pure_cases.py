# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary numeric public journeys, without user callbacks or frame subclasses."""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CASES = [
    f'{width}/{entry}/{mode}/default/{direction}'
    for width in ('float16', 'float32', 'multi_float32')
    for entry in ('indices', 'predict')
    for mode in ('ignore', 'raise', 'warn')
    for direction in ('forward', 'reverse')
]
CASES += [
    f'float32/indices/warn/{action}/{direction}'
    for action in ('always', 'ignore', 'error')
    for direction in ('forward', 'reverse')
]
if np.__version__.startswith('1.26.'):
    CASES += [
        f'{width}/{entry}/ignore/default/{direction}'
        for width in ('int8', 'uint8')
        for entry in ('indices', 'predict')
        for direction in ('forward', 'reverse')
    ]


class Endpoint:
    url = 'http://localhost'

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body)
        return {
            'model': 'kumo-tabular',
            'id': 'numeric-pure-reference',
            'predictions': [{'row_index': 0, 'prediction': 0.0}],
            'metadata': {},
        }

    def predict_body(self, body: bytes) -> bytes:
        return json.dumps(self.predict(json.loads(body))).encode()

    def close(self) -> None:
        pass


def observe_case(name: str) -> dict[str, Any]:
    from sdm_client import KumoTabular, PredictiveQuery, SDMClient

    width, entry, mode, action, direction = name.split('/')
    n = 81 if entry == 'indices' else 10002
    first = getattr(np, width.removeprefix('multi_'))(1)
    second = 128 if width == 'int8' else -1 if width == 'uint8' else 1e100
    pair = [first, second]
    if direction == 'reverse':
        pair.reverse()
    labels = np.empty(n, dtype=object)
    for position in range(n):
        labels[position] = pair[position % 2]
    index = pd.Index(labels, dtype=object)
    if width.startswith('multi_'):
        level = np.empty(2, dtype=object)
        level[:] = pair
        index = pd.MultiIndex(
            levels=[pd.Index(level, dtype=object), pd.Index([0])],
            codes=[np.arange(n) % 2, np.zeros(n, dtype=np.intp)],
            verify_integrity=False,
        )
    frame = pd.DataFrame(
        {
            'id': range(n),
            'feature': np.arange(n) * 0.5,
            'target': np.arange(n) + 100.0,
        },
        index=index,
    )
    getmode = getattr(np, '_get_promotion_state', None)
    setmode = getattr(np, '_set_promotion_state', None)
    previous = None if getmode is None else getmode()
    if setmode is not None:
        setmode('weak')
    outcome: dict[str, Any] = {}
    endpoint = Endpoint()
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    try:
        with (
            np.errstate(over=mode),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter(action)
            try:
                query = PredictiveQuery(
                    'PREDICT data.target FOR EACH data.id', {'data': frame}
                )
                arguments = (
                    {'indices': [n - 1], 'num_context_samples': 40}
                    if entry == 'indices'
                    else {'predict': frame.iloc[[-1]].drop(columns=['target'])}
                )
                request = query.materialize(**arguments, task='regression')
                outcome['context_ids'] = request.context.instance_table[
                    'id'
                ].tolist()
                result = client.predict(request, model=KumoTabular())
                outcome['encoded_ids'] = [
                    row[0]
                    for row in endpoint.calls[0]['context']['instance_table'][
                        'rows'
                    ]
                ]
                outcome['predictions'] = result.to_pandas().to_dict(
                    orient='records'
                )
            except Exception as error:  # noqa: BLE001 - public differential outcome
                outcome['error'] = {
                    'type': type(error).__name__,
                    'message': str(error),
                }
            runs: list[dict[str, Any]] = []
            for warning in caught:
                if warning.category.__module__.startswith('pandas'):
                    continue
                site = None
                if warning.category is RuntimeWarning:
                    filename = Path(warning.filename)
                    site = {
                        'file': filename.relative_to(
                            Path(pd.__file__).parent
                        ).as_posix(),
                        'line': warning.lineno,
                    }
                event = {
                    'type': warning.category.__name__,
                    'message': str(warning.message),
                    'site': site,
                }
                if runs and runs[-1]['event'] == event:
                    runs[-1]['count'] += 1
                else:
                    runs.append({'event': event, 'count': 1})
            outcome['warnings'] = runs
    finally:
        client.close()
        if setmode is not None:
            setmode(previous)
    return outcome
