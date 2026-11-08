# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public count/task journeys shared by the frozen recorder and candidate tests."""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd

CASES = (
    'cap_default',
    'cap_explicit',
    'over_default',
    'over_explicit',
    'empty_all',
    'empty_missing',
    'task_indices',
    'task_frame',
    'count_owner',
    'warning_record',
    'warning_raise',
)


class StopWarning(Exception):
    """A user warning handler deliberately stops materialization."""


def observe_case(name: str) -> dict[str, Any]:
    from sdm_client import KumoTabular, PredictiveQuery, SDMClient
    from sdm_client.models import TABULAR_LIMITS

    assert name in CASES
    cap = TABULAR_LIMITS['context_rows_max']
    events: list[Any] = []
    recorded: list[dict[str, str]] = []
    outcome: dict[str, Any] = {}
    n = cap if name.startswith('cap_') else cap + 1
    if name in (
        'empty_all',
        'empty_missing',
        'task_indices',
        'task_frame',
        'count_owner',
    ):
        n = 12
    data = pd.DataFrame(
        {
            'id': range(n + 1),
            'feature': np.arange(n + 1, dtype=float),
            'target': np.arange(n + 1, dtype=float) + 100,
        }
    )
    if name == 'empty_missing':
        data.loc[: n - 1, 'target'] = np.nan
    if name.startswith('task_'):
        data['target'] = [i % 2 for i in range(n)] + [2]

    class Count(int):
        def __gt__(self, other: Any) -> bool:
            events.append(['count_gt', int(other)])
            return super().__gt__(other)

    class Endpoint:
        url = 'http://localhost'

        def predict(self, body: dict[str, Any]) -> dict[str, Any]:
            ids = [row[0] for row in body['context']['instance_table']['rows']]
            outcome['encoded_context_ids'] = ids
            outcome['encoded_task'] = body['task']['kind']
            events.append(['endpoint', len(ids)])
            return {
                'model': 'kumo-tabular',
                'id': 'count-reference',
                'predictions': [{'row_index': 0, 'prediction': 0.0}],
                'metadata': {},
            }

        def predict_body(self, body: bytes) -> bytes:
            return json.dumps(self.predict(json.loads(body))).encode()

        def close(self) -> None:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter('always')

        def show(
            message: Any,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            recorded.append(
                {'category': category.__name__, 'message': str(message)}
            )
            events.append(['warning', category.__name__, str(message)])
            if name == 'warning_raise' and 'were sampled' in str(message):
                raise StopWarning('warning handler stopped')

        warnings.showwarning = show
        client = None
        try:
            query = PredictiveQuery(
                'PREDICT data.target FOR EACH data.id', {'data': data}
            )
            arguments: dict[str, Any] = {
                'indices': list(range(n + 1)) if name == 'empty_all' else [n]
            }
            if name == 'task_frame':
                arguments = {'predict': data.iloc[[-1]].drop(columns='target')}
            if name.endswith('_explicit'):
                arguments['num_context_samples'] = cap
            if name == 'count_owner':
                arguments['num_context_samples'] = Count(4)
            if not name.startswith('task_'):
                arguments['task'] = 'regression'
            request = query.materialize(**arguments)
            outcome['context_ids'] = request.context.instance_table[
                'id'
            ].tolist()
            outcome['task'] = request.task.kind
            events.append(['materialized'])
            client = SDMClient.__new__(SDMClient)
            client._start(Endpoint())
            result = client.predict(request, model=KumoTabular()).to_pandas()
            outcome['prediction'] = {
                'columns': result.columns.tolist(),
                'rows': result.to_dict(orient='records'),
            }
            events.append(['predicted'])
        except Exception as error:
            outcome['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'code': getattr(error, 'code', None),
            }
        finally:
            if client is not None:
                client.close()
    return {
        'cap': cap,
        'outcome': outcome,
        'warnings': recorded,
        'events': events,
    }
