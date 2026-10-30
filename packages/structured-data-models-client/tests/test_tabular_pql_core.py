# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public PQL selection, model bytes and result journeys on both routes."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from sdm_client import KumoTabular, PredictiveQuery, SDMClient

ROUTES = ','.join(
    (
        'query.materialize/indices-binary/frames',
        'query.materialize/indices-multiclass/frames',
        'query.materialize/indices-regression/frames',
        'query.materialize/predict-frame/frames',
        'request.tabular/classification/frames',
        'request.tabular/regression/frames',
        'client.predict/tabular/frames',
    )
)


@pytest.fixture(params=['old', 'core'])
def route(request: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(
        'SDM_CORE_ROUTES', ROUTES if request.param == 'core' else ''
    )
    return request.param


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'id': range(11),
            'feature': [i * 0.5 for i in range(11)],
            'target': [100.0 + i for i in range(11)],
        },
        index=list(range(110, 99, -1)),
    )


def query(data: pd.DataFrame) -> PredictiveQuery:
    return PredictiveQuery(
        'PREDICT data.target FOR EACH data.id', {'data': data}
    )


class Endpoint:
    url = 'http://localhost'

    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body)
        return {
            'model': 'kumo-tabular',
            'id': 'pql-answer',
            'predictions': [
                {'row_index': i, 'prediction': float(i)}
                for i in range(self.rows)
            ],
            'metadata': {},
        }

    def predict_body(self, body: bytes) -> bytes:
        return json.dumps(self.predict(json.loads(body))).encode()

    def close(self) -> None:
        pass


def test_sampled_context_reaches_model_in_sorted_draw_order(route: str) -> None:
    request = query(frame()).materialize(
        indices=[10], num_context_samples=3, task='regression'
    )
    # RandomState42 chooses candidate offsets8,1,5. Descending source index
    # labels then order those sampled rows8,5,1, not source order1,5,8.
    assert request.context.instance_table['id'].tolist() == [8, 5, 1]
    endpoint = Endpoint(1)
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    result = client.predict(request, model=KumoTabular())
    assert [
        row[0] for row in endpoint.calls[0]['context']['instance_table']['rows']
    ] == [8, 5, 1]
    assert len(result.to_pandas()) == 1


def test_unreduced_context_retains_candidate_order(route: str) -> None:
    request = query(frame()).materialize(
        indices=[10], num_context_samples=10, task='regression'
    )
    assert request.context.instance_table['id'].tolist() == list(range(10))
    assert request.context.instance_table.index.tolist() == list(
        range(110, 100, -1)
    )


def test_repeated_keys_expand_every_matching_position(route: str) -> None:
    data = frame()
    asked = query(data)
    data.loc[109, 'id'] = 0
    request = asked.materialize(
        indices=[0, 10, 0], num_context_samples=10, task='regression'
    )
    assert request.predict.instance_table['id'].tolist() == [0, 0, 10, 0, 0]
    assert request.context.instance_table['id'].tolist() == list(range(2, 10))


def test_supplied_frame_excludes_matching_keys_and_unknown_targets(
    route: str,
) -> None:
    data = frame()
    data.loc[107, 'target'] = float('nan')
    scoring = data.iloc[[10, 0]].drop(columns=['target'])
    request = query(data).materialize(predict=scoring, task='regression')
    assert request.predict.instance_table is scoring
    assert request.context.instance_table['id'].tolist() == [
        1,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
    ]


def test_missing_key_diagnostic_keeps_repetitions(route: str) -> None:
    with pytest.raises(
        Exception, match='2 of the 3 indices are not in data.id:'
    ) as error:
        query(frame()).materialize(indices=[99, 1, 99], num_context_samples=3)
    assert '[99, 99]' in str(error.value)


INDEX_FAMILIES = (
    'range',
    'signed_duplicates',
    'unsigned',
    'floating_missing',
    'boolean',
    'datetime',
    'timezone',
    'timedelta',
    'period',
    'categorical',
    'multi',
    'string',
    'object_mixed',
    'nullable_integer',
    'arrow_integer',
    'arrow_string',
)


def indexed_frame(family: str) -> pd.DataFrame:
    n = 81
    data = pd.DataFrame(
        {
            'id': range(n),
            'feature': [i * 0.5 for i in range(n)],
            'target': [100.0 + i for i in range(n)],
        }
    )
    labels = [(i * 13) % 17 for i in range(n)]
    if family == 'range':
        index = pd.RangeIndex(n, 0, -1)
    elif family == 'signed_duplicates':
        index = pd.Index(labels, dtype='int64')
    elif family == 'unsigned':
        index = pd.Index([2**63 + value for value in labels], dtype='uint64')
    elif family == 'floating_missing':
        index = pd.Index(
            [
                float('nan') if i % 7 == 0 else float(value)
                for i, value in enumerate(labels)
            ]
        )
    elif family == 'boolean':
        index = pd.Index([bool(value % 2) for value in labels])
    elif family in ('datetime', 'timezone'):
        index = pd.DatetimeIndex(
            [
                pd.NaT
                if i % 7 == 0
                else pd.Timestamp('2020-01-01') + pd.Timedelta(days=value)
                for i, value in enumerate(labels)
            ]
        )
        if family == 'timezone':
            index = index.tz_localize('US/Pacific')
    elif family == 'timedelta':
        index = pd.TimedeltaIndex(
            [
                pd.NaT if i % 7 == 0 else pd.Timedelta(days=value)
                for i, value in enumerate(labels)
            ]
        )
    elif family == 'period':
        index = pd.PeriodIndex(
            [
                pd.NaT if i % 7 == 0 else pd.Period('2020-01', freq='M') + value
                for i, value in enumerate(labels)
            ],
            freq='M',
        )
    elif family == 'categorical':
        index = pd.CategoricalIndex(
            [
                None if i % 7 == 0 else str(value)
                for i, value in enumerate(labels)
            ],
            categories=[str(value) for value in range(16, -1, -1)],
            ordered=False,
        )
    elif family == 'multi':
        index = pd.MultiIndex.from_tuples(
            [
                (None if i % 7 == 0 else value % 3, str(value))
                for i, value in enumerate(labels)
            ],
            names=['a', 'b'],
        )
    elif family == 'string':
        index = pd.Index(
            [
                None if i % 7 == 0 else str(value)
                for i, value in enumerate(labels)
            ],
            dtype='str',
        )
    elif family == 'object_mixed':
        index = pd.Index(
            [
                str(value) if i % 3 == 0 else value
                for i, value in enumerate(labels)
            ],
            dtype=object,
        )
    elif family == 'nullable_integer':
        index = pd.Index(
            [None if i % 7 == 0 else value for i, value in enumerate(labels)],
            dtype='Int64',
        )
    elif family == 'arrow_integer':
        index = pd.Index(
            [None if i % 7 == 0 else value for i, value in enumerate(labels)],
            dtype='int64[pyarrow]',
        )
    elif family == 'arrow_string':
        index = pd.Index(
            [
                None if i % 7 == 0 else str(value)
                for i, value in enumerate(labels)
            ],
            dtype='string[pyarrow]',
        )
    else:
        raise AssertionError(family)
    data.index = index
    return data


@pytest.mark.parametrize('family', INDEX_FAMILIES)
def test_index_family_matches_frozen_public_context(
    route: str, family: str
) -> None:
    from pathlib import Path

    expected = json.loads(
        (
            Path(__file__).parent / 'data' / 'tabular_pql_index_reference.json'
        ).read_text()
    )[family]
    asked = query(indexed_frame(family))
    if 'error' in expected:
        with pytest.raises(TypeError) as error:
            asked.materialize(
                indices=[80], num_context_samples=40, task='regression'
            )
        assert type(error.value).__name__ == expected['error']
        assert str(error.value) == expected['message']
        return
    request = asked.materialize(
        indices=[80], num_context_samples=40, task='regression'
    )
    assert (
        request.context.instance_table['id'].tolist() == expected['context_ids']
    )
    endpoint = Endpoint(1)
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    result = client.predict(request, model=KumoTabular())
    assert [
        row[0] for row in endpoint.calls[0]['context']['instance_table']['rows']
    ] == expected['context_ids']
    assert len(result.to_pandas()) == 1


def test_forced_tabular_frame_routes_do_not_silently_run_legacy_relational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client import KumoRelational
    from sdm_client._routing import RouteError
    from sdm_client.materializers.relational import RelationalMaterializer

    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('legacy relational materialization was reached')

    monkeypatch.setattr(RelationalMaterializer, 'materialize', unexpected)
    with pytest.raises(
        RouteError, match='relational frames have no core route'
    ):
        query(frame()).materialize(
            indices=[10], num_context_samples=3, model=KumoRelational()
        )
