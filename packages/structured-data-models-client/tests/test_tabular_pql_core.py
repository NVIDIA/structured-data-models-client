# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public PQL selection, model bytes and result journeys on both routes."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tabular_pql_fixtures import (
    INDEX_FAMILIES,
    INDEX_SAMPLE_COUNTS,
    LABEL_FAMILIES,
    indexed_frame,
    label_frame,
)
from tabular_pql_reference import reference_case

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


@pytest.mark.parametrize('family', LABEL_FAMILIES)
@pytest.mark.parametrize('entrypoint', ['indices', 'predict'])
def test_exact_label_materialization_and_prediction(
    route: str,
    family: str,
    entrypoint: str,
) -> None:
    reference = reference_case('labels', family, entrypoint)
    data = label_frame(family)
    arguments = (
        {'indices': [80], 'num_context_samples': 40}
        if entrypoint == 'indices'
        else {'predict': data.iloc[[80]].drop(columns=['target'])}
    )
    if 'error' in reference:
        error_type = {'TypeError': TypeError, 'OverflowError': OverflowError}[
            reference['error']['type']
        ]
        with pytest.raises(error_type) as caught:
            query(data).materialize(
                **arguments,
                task='regression',
            )
        assert str(caught.value) == reference['error']['message']
        return
    request = query(data).materialize(
        **arguments,
        task='regression',
    )
    assert (
        request.context.instance_table['id'].tolist()
        == reference['context_ids']
    )
    endpoint = Endpoint(1)
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    try:
        result = client.predict(request, model=KumoTabular())
        assert [
            row[0]
            for row in endpoint.calls[0]['context']['instance_table']['rows']
        ] == reference['context_ids']
        assert (
            result.to_pandas().columns.tolist()
            == reference['prediction']['columns']
        )
        assert (
            result.to_pandas().to_dict(orient='records')
            == reference['prediction']['rows']
        )
    finally:
        client.close()


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


@pytest.mark.parametrize('family', INDEX_FAMILIES)
@pytest.mark.parametrize('entrypoint', ['indices', 'predict'])
def test_index_family_matches_frozen_public_context(
    route: str, family: str, entrypoint: str
) -> None:
    expected = reference_case('indices', family, entrypoint)
    data = indexed_frame(family)
    asked = query(data)
    arguments = (
        {
            'indices': [len(data) - 1],
            'num_context_samples': INDEX_SAMPLE_COUNTS.get(family, 40),
        }
        if entrypoint == 'indices'
        else {'predict': data.iloc[[-1]].drop(columns=['target'])}
    )
    if 'error' in expected:
        with pytest.raises(TypeError) as error:
            asked.materialize(**arguments, task='regression')
        assert type(error.value).__name__ == expected['error']['type']
        assert str(error.value) == expected['error']['message']
        return
    request = asked.materialize(**arguments, task='regression')
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
    assert (
        result.to_pandas().columns.tolist() == expected['prediction']['columns']
    )
    assert (
        result.to_pandas().to_dict(orient='records')
        == expected['prediction']['rows']
    )
    client.close()


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


def test_index_take_hook_precedes_target_validity(route: str) -> None:
    import numpy as np

    data = frame()
    asked = query(data)
    fired = []

    class TakingIndex(pd.Index):
        def take(self, *args: Any, **kwargs: Any) -> pd.Index:
            if not fired:
                fired.append(True)
                data.loc[109, 'target'] = float('nan')
            return super().take(*args, **kwargs)

    data.index = TakingIndex._simple_new(np.asarray(data.index))
    request = asked.materialize(
        indices=[10], num_context_samples=10, task='regression'
    )
    assert fired
    assert request.context.instance_table['id'].tolist() == [
        0,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]


def test_attrs_copy_effect_is_not_moved_after_target_filter(route: str) -> None:
    data = frame()
    scoring = data.iloc[[10]].drop(columns=['target'])
    asked = query(data)
    # A public write refreshes columns cached during schema construction on
    # older pandas, so the first attribute copy occurs on the entity read.
    data['target'] = data['target']
    fired = []

    class OnCopy:
        def __deepcopy__(self, memo: Any) -> Any:
            if not fired:
                fired.append(True)
                data.loc[109, 'target'] = float('nan')
            return self

    data.attrs['owner'] = OnCopy()
    request = asked.materialize(predict=scoring, task='regression')
    assert fired
    assert request.context.instance_table['id'].tolist() == [
        0,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]


@pytest.mark.parametrize(
    'family',
    [
        'unique_integer',
        'masked_nulls',
        'monotonic_ties',
        'date_unique',
        'arrow_ties',
        'multi',
    ],
)
def test_profile_independent_orders_do_not_read_numpy_producer(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client._rust import tabular_index

    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)

    def unexpected_profile() -> str:
        pytest.fail('this sampled order must not depend on a NumPy producer')

    monkeypatch.setattr(tabular_index, '_profile', unexpected_profile)
    indexes = {
        'unique_integer': pd.Index(range(110, 99, -1)),
        'masked_nulls': pd.Index(
            [None if i % 2 else float(i) for i in range(11)]
        ),
        'monotonic_ties': pd.Index([7] * 11),
        'date_unique': pd.Index(
            [date(2020, 1, 11 - i) for i in range(11)], dtype=object
        ),
        'arrow_ties': pd.Index(
            pd.array([i % 2 for i in range(11)], dtype='int64[pyarrow]')
        ),
        'multi': pd.MultiIndex.from_tuples([(i % 2, i % 3) for i in range(11)]),
    }
    data = frame()
    data.index = indexes[family]
    expected = json.loads(
        (
            Path(__file__).parent / 'data/tabular_pql_profile_reference.json'
        ).read_text()
    )[family]
    request = query(data).materialize(
        indices=[10],
        num_context_samples=3,
        task='regression',
    )
    assert request.context.instance_table['id'].tolist() == expected
    endpoint = Endpoint(1)
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    try:
        answer = client.predict(request, model=KumoTabular())
        assert [
            row[0]
            for row in endpoint.calls[0]['context']['instance_table']['rows']
        ] == expected
        assert len(answer.to_pandas()) == 1
    finally:
        client.close()


def test_tied_sort_binds_retained_facts_without_native_rereads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client._rust import tabular_index, tabular_materialize

    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)
    observations = []
    orders = []
    original_facts = tabular_index._facts
    original_call = tabular_materialize._call

    def facts(index: pd.Index, profile: str) -> dict[str, Any]:
        observations.append(index)
        return original_facts(index, profile)

    def call(operation: str, arguments: dict[str, Any]) -> Any:
        if operation == 'order_sampled_tabular_rows':
            orders.append(arguments['index'])
        return original_call(operation, arguments)

    monkeypatch.setattr(tabular_index, '_facts', facts)
    monkeypatch.setattr(tabular_materialize, '_call', call)
    request = query(indexed_frame('signed_duplicates')).materialize(
        indices=[80],
        num_context_samples=40,
        task='regression',
    )
    expected = reference_case('indices', 'signed_duplicates', 'indices')[
        'context_ids'
    ]
    assert request.context.instance_table['id'].tolist() == expected
    assert len(observations) == 1
    assert len(orders) == 2
    assert orders[0]['profile'] == 'unbound'
    assert orders[1]['profile'] != 'unbound'
    assert orders[0]['values'] is orders[1]['values']
    assert orders[0]['missing'] is orders[1]['missing']


def test_sampled_present_nan_is_reconstructed_instead_of_shallow_copied(
    route: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = frame().iloc[:4].copy()
    data.index = pd.Index(
        pd.arrays.FloatingArray(np.full(4, np.nan), np.zeros(4, dtype=bool))
    )
    sampled = []
    original_take = pd.DataFrame.take

    def take(
        self: pd.DataFrame, indices: Any, axis: int = 0, **kwargs: Any
    ) -> pd.DataFrame:
        result = original_take(self, indices, axis=axis, **kwargs)
        if axis == 0 and len(self) == 3 and len(indices) == 1:
            sampled.append(result)
        return result

    monkeypatch.setattr(pd.DataFrame, 'take', take)
    request = query(data).materialize(
        indices=[3],
        num_context_samples=1,
        task='regression',
    )
    assert len(sampled) == 1
    context = request.context.instance_table
    assert context.index is not sampled[0].index
    assert not np.shares_memory(
        context['id'].to_numpy(), sampled[0]['id'].to_numpy()
    )


def test_supplied_frame_refuses_context_cap_like_same_runtime_reference(
    route: str,
) -> None:
    from sdm_client.errors import StructuredDataError

    expected = reference_case('validation', 'supplied_frame_cap', 'predict')[
        'error'
    ]
    data = indexed_frame('range')
    with pytest.raises(StructuredDataError) as caught:
        query(data).materialize(
            predict=data.iloc[[80]].drop(columns=['target']),
            num_context_samples=40,
            task='regression',
        )
    assert type(caught.value).__name__ == expected['type']
    assert caught.value.code == expected['code']
    assert str(caught.value) == expected['message']


def test_legacy_dispatch_does_not_reuse_modern_avx2_or_avx512_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sdm_client._rust import tabular_index

    native = SimpleNamespace(
        __file__='producer.so',
        __cpu_features__={
            'AVX2': True,
            'AVX512_SKX': False,
            'X86_V3': True,
            'X86_V4': True,
        },
    )
    monkeypatch.setattr(
        tabular_index.importlib, 'import_module', lambda name: native
    )
    monkeypatch.setattr(
        tabular_index, '_producer', lambda path: 'avx512_skx_or_scalar'
    )
    assert tabular_index._profile() == 'scalar'
    native.__cpu_features__['AVX512_SKX'] = True
    assert tabular_index._profile() == 'avx512_gnu14_network64'


@pytest.mark.parametrize(
    'family,suffix',
    [('avx2_skx', ''), ('avx2_skx_unconditional', '_unconditional')],
)
def test_older_dispatch_reads_its_effective_groups(
    monkeypatch: pytest.MonkeyPatch, family: str, suffix: str
) -> None:
    from types import SimpleNamespace

    from sdm_client._rust import tabular_index

    features = {
        'X86_V3': True,
        'X86_V4': True,
        'AVX2': False,
        'AVX512_SKX': False,
    }
    native = SimpleNamespace(__file__='producer.so', __cpu_features__=features)
    monkeypatch.setattr(
        tabular_index.importlib, 'import_module', lambda name: native
    )
    monkeypatch.setattr(tabular_index, '_producer', lambda path: family)
    assert tabular_index._profile() == 'scalar'
    features['AVX2'] = True
    assert tabular_index._profile() == 'avx2_gnu14' + suffix
    features['AVX512_SKX'] = True
    assert tabular_index._profile() == 'avx512_gnu14' + suffix


def test_required_promotion_mode_is_observed_on_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sdm_client._routing import RouteError
    from sdm_client._rust import tabular_index

    monkeypatch.setattr(
        tabular_index.importlib,
        'import_module',
        lambda name: SimpleNamespace(__file__='producer.so'),
    )
    monkeypatch.setattr(
        tabular_index,
        '_producer_families',
        lambda path: ('avx2_skx_unconditional', 'weak_scalars', 'weak'),
    )
    state = ['weak']
    monkeypatch.setattr(
        np, '_get_promotion_state', lambda: state[0], raising=False
    )
    assert tabular_index._comparison_semantics() == 'weak_scalars'
    state[0] = 'legacy'
    with pytest.raises(RouteError, match='promotion mode'):
        tabular_index._comparison_semantics()
    state[0] = 'weak'
    assert tabular_index._comparison_semantics() == 'weak_scalars'
