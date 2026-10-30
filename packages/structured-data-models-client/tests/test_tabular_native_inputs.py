# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public parity for native-only direct options and retained response owners."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from test_tabular_direct_core import DIRECT, Endpoint, client

from sdm_client import KumoTabular, Request, StructuredDataError


def test_option_integer_coercion_is_separate_from_json_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client.adapters import tabular

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        events: list[str] = []

        class NativeInteger(int):
            def __int__(self, observed: list[str] = events) -> int:
                observed.append('integer')
                return 7

        def request_id(observed: list[str] = events) -> str:
            observed.append('request_id')
            return 'native-order'

        monkeypatch.setattr(tabular, '_request_id', request_id)
        request = Request(
            pd.DataFrame({'feature': [1, 2], 'target': [0, 1]}),
            pd.DataFrame({'feature': [3]}),
            target='target', task='classification',
            inference={'seed': NativeInteger(2**53 + 1)},
        )
        endpoint = Endpoint({
            'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
            'predictions': [{'row_index': 0, 'prediction': 0}],
        })
        client(endpoint).predict(request, model=KumoTabular())
        outcomes.append((events, endpoint.calls[0]['inference']))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0] == (['request_id', 'integer'], {'seed': 2**53 + 1})


@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('option', [object(), float('nan'), float('inf'), {('bad',): 1}])
def test_native_options_preserve_transport_refusal(
    closed: bool, option: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client.core.transport import Transport

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        transport = Transport('https://nim.example')
        if closed:
            transport.close()

        def unexpected_send(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('invalid options must fail before sending')

        monkeypatch.setattr(transport, '_post_raw', unexpected_send)
        request = Request(
            pd.DataFrame({'feature': [1, 2], 'target': [0, 1]}),
            pd.DataFrame({'feature': [3]}), target='target',
            task='classification', inference={'option': option},
        )
        try:
            with pytest.raises(StructuredDataError) as caught:
                client(transport).predict(request, model=KumoTabular())
            error = caught.value
            outcomes.append((type(error), error.code, str(error), type(error.__cause__)))
        finally:
            transport.close()
    assert outcomes[0] == outcomes[1]


def test_deferred_options_reach_custom_transport_without_json_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []
    retained = object()

    class NativeEndpoint:
        url = 'http://localhost'

        def __init__(self) -> None:
            self.seen: Any = None

        def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.seen = payload['inference']['native']
            return {
                'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
                'predictions': [{'row_index': 0, 'prediction': 1}],
            }

        def close(self) -> None:
            pass

    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        endpoint = NativeEndpoint()
        request = Request(
            pd.DataFrame({'feature': [1, 2], 'target': [0, 1]}),
            pd.DataFrame({'feature': [3]}), target='target',
            task='classification', inference={'native': retained},
        )
        result = client(endpoint).predict(request, model=KumoTabular())
        assert endpoint.seen is retained
        outcomes.append(result.items)
    assert outcomes[0] == outcomes[1]


def test_deferred_native_regression_response_retains_prediction_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = np.int64(7)

    class NativeEndpoint:
        url = 'http://localhost'

        def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
                'predictions': [{'row_index': 0, 'prediction': prediction}],
            }

        def close(self) -> None:
            pass

    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        request = Request(
            pd.DataFrame({'feature': [1, 2], 'target': [0.5, 1.5]}),
            pd.DataFrame({'feature': [3]}), target='target',
            task='regression', inference={'native': object()},
        )
        result = client(NativeEndpoint()).predict(request, model=KumoTabular())
        assert result.items[0].prediction is prediction



@pytest.mark.parametrize('inference', [{}, {'native': object()}])
def test_custom_transport_preserves_nested_metadata_and_response_owners(
    inference: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = np.int64(7)
    retained = object()
    nested = {'opaque': retained}

    class NativeEndpoint:
        url = 'http://localhost'

        def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                'model': 'kumo-tabular', 'id': 'answer', 'metadata': {'nested': nested},
                'predictions': [{'row_index': 0, 'prediction': prediction, 'metadata': {'nested': nested}}],
            }

        def predict_body(self, body: bytes) -> bytes:
            raise AssertionError('custom transport must retain its public predict behavior')

        def close(self) -> None:
            pass

    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        request = Request(
            pd.DataFrame({'feature': [1, 2], 'target': [0.5, 1.5]}),
            pd.DataFrame({'feature': [3]}), target='target',
            task='regression', inference=inference,
        )
        result = client(NativeEndpoint()).predict(request, model=KumoTabular())
        assert result.items[0].prediction is prediction
        assert result.metadata['nested'] is nested
        assert result.items[0].metadata['nested'] is nested


@pytest.mark.parametrize('probabilities', [{'0': 0.4, '1': 0.6}, {'0': 0.4, '00': 0.6}])
def test_native_class_restoration_and_requested_absence(
    probabilities: dict[str, float], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings
    from sdm_client import SDMWarning

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        endpoint = Endpoint({
            'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
            'predictions': [{'row_index': 0, 'prediction': '1', 'probabilities': probabilities}],
        })
        request = Request(
            pd.DataFrame({'feature': [1, 2], 'target': [0, 1]}),
            pd.DataFrame({'feature': [3]}), target='target', task='classification',
            inference={1: 'native-key'},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', SDMWarning)
            result = client(endpoint).predict(request, model=KumoTabular(), outputs=['prediction', 'probabilities', 'quantiles'])
        outcomes.append((result.items, [str(w.message) for w in caught if issubclass(w.category, SDMWarning)]))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('row_index', [0, 9])
def test_native_row_id_conversion_follows_position_validation(
    row_index: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []

    class NativeId(str):
        observed: list[str]

        def __new__(cls, observed: list[str]) -> NativeId:
            value = super().__new__(cls, '[999]')
            value.observed = observed
            return value

        def __str__(self) -> str:
            self.observed.append('identifier')
            return '[3]'

    class NativeEndpoint:
        url = 'http://localhost'

        def __init__(self, identifier: str) -> None:
            self.identifier = identifier

        def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
                'predictions': [{'row_index': row_index, 'id': self.identifier, 'prediction': 1}],
            }

        def close(self) -> None:
            pass

    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        observed: list[str] = []
        request = Request(
            pd.DataFrame({'row_id': [1, 2], 'target': [0, 1]}),
            pd.DataFrame({'row_id': [3]}), target='target', task='classification',
        )
        endpoint = NativeEndpoint(NativeId(observed))
        if row_index:
            with pytest.raises(StructuredDataError) as caught:
                client(endpoint).predict(request, model=KumoTabular())
            outcomes.append((observed, caught.value.code, str(caught.value)))
            assert observed == []
        else:
            result = client(endpoint).predict(request, model=KumoTabular())
            outcomes.append((observed, result.items))
            assert observed == ['identifier']
    assert outcomes[0] == outcomes[1]


def test_native_options_keep_databricks_wrapping_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from test_databricks_serving import opened

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        connected, endpoint = opened(answer={
            'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
            'predictions': [{'row_index': 0, 'prediction': 1}],
        })
        request = Request(
            pd.DataFrame({'feature': [1, 2], 'target': [0, 1]}),
            pd.DataFrame({'feature': [3]}), target='target', task='classification',
            inference={7: 'native-key'},
        )
        try:
            result = connected.predict(request, model=KumoTabular())
        finally:
            connected.close()
        assert len(endpoint.calls) == 1
        sent = json.loads(endpoint.calls[0]['dataframe_records'][0]['request_json'])
        sent['metadata'].pop('request_id')
        assert sent['inference'] == {'7': 'native-key'}
        outcomes.append((sent, result.items))
    assert outcomes[0] == outcomes[1]


def test_native_transport_observes_absent_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    class NativeEndpoint:
        url = 'http://localhost'

        def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen.append('inference' in payload)
            return {'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
                    'predictions': [{'row_index': 0, 'prediction': 1}]}

        def close(self) -> None:
            pass

    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        request = Request(pd.DataFrame({'feature': [1, 2], 'target': [0, 1]}),
                          pd.DataFrame({'feature': [3]}), target='target', task='classification')
        client(NativeEndpoint()).predict(request, model=KumoTabular())
    assert seen == [False, False]


def test_native_response_retains_unconsumed_unicode(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnconsumedId(str):
        def __str__(self) -> str:
            raise AssertionError('an unkeyed ID must not be converted')

    model = '\ud800'
    request_id = '\udfff'
    identifier = UnconsumedId('\ud900')
    outcomes = []

    class NativeEndpoint:
        url = 'http://localhost'

        def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {'model': model, 'id': request_id, 'metadata': {},
                    'predictions': [{'row_index': 0, 'id': identifier, 'prediction': 1}]}

        def close(self) -> None:
            pass

    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        request = Request(pd.DataFrame({'feature': [1, 2], 'target': [0, 1]}),
                          pd.DataFrame({'feature': [3]}), target='target', task='classification')
        result = client(NativeEndpoint()).predict(request, model=KumoTabular())
        assert result.items[0].id is identifier
        outcomes.append((result.model, result.request_id, result.items))
    assert outcomes[0] == outcomes[1]


def test_native_transport_payload_mutations_reach_layout_and_restoration(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = []

    class NativeEndpoint:
        url = 'http://localhost'

        def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
            payload['predict']['instance_table']['rows'][0][0] = 9
            payload['task']['target']['dtype'] = 'bool'
            return {'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
                    'predictions': [{'row_index': 0, 'id': '[9]', 'prediction': 'false'}]}

        def close(self) -> None:
            pass

    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        request = Request(pd.DataFrame({'row_id': [1, 2], 'target': [0, 1]}),
                          pd.DataFrame({'row_id': [3]}), target='target', task='classification',
                          inference={'opaque': object()})
        result = client(NativeEndpoint()).predict(request, model=KumoTabular())
        assert result.items[0].id == 9
        assert result.items[0].prediction is False
        outcomes.append(result.items)
    assert outcomes[0] == outcomes[1]
