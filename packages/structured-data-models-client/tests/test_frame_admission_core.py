# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public frame conversion and serialized native owner admission."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema, _routing
from sdm_client._rust import frame_materialize


@pytest.fixture(autouse=True)
def frame_routes(monkeypatch):
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(_routing.FRAME_ROUTES))


def query():
    frame = pd.DataFrame(
        {
            'id': range(20),
            'x': [1, 'a', None, True] * 5,
            'y': [i * 1.25 for i in range(20)],
        }
    )
    schema = Schema.from_data({'t': frame}, primary_keys={'t': 'id'})
    schema['t']['y'].stype = 'numerical'
    return PredictiveQuery('PREDICT t.y FOR EACH t.id', schema)


def test_mixed_object_text_keeps_values_and_missingness():
    request = query().materialize(
        indices=[1, 2], model=KumoRelational(), random_seed=7
    )
    table = request._payloads[0]['context']['related_tables']['t']
    position = table['columns'].index('x')
    assert {row[position] for row in table['rows']} == {'1', 'a', 'True', None}


def test_concurrent_first_and_unseeded_calls_share_one_serial_owner(
    monkeypatch,
):
    asked = query()
    original = frame_materialize.run
    monitor = threading.Lock()
    active = 0
    maximum = 0
    prepared = 0

    def monitored(operation, attachments):
        nonlocal active, maximum, prepared
        with monitor:
            active += 1
            maximum = max(maximum, active)
            prepared += operation['op'] == 'prepare_frame_graph'
        try:
            # Release the GIL while the other public call reaches admission.
            time.sleep(0.03)
            return original(operation, attachments)
        finally:
            with monitor:
                active -= 1

    monkeypatch.setattr(frame_materialize, 'run', monitored)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(2):
            start = threading.Barrier(2)

            def invoke(start=start):
                start.wait(timeout=5)
                return asked.materialize(
                    indices=[1, 2], model=KumoRelational(), random_seed=None
                )

            calls = [pool.submit(invoke) for _ in range(2)]
            assert all(call.result(timeout=30)._payloads for call in calls)
    assert prepared == 1
    assert maximum == 1


@pytest.mark.parametrize(
    'dates',
    [
        ['2020-01-01', '2020-01-02 00:00 +0100'],
        ['2020-01-01 00:00 +0100', '2020-01-02 00:00 -0800'],
        ['2020-01-01', '2020-01-02T00:00:00Z'],
    ],
)
def test_text_timestamps_match_native_utc_instants(dates):
    def payload(values):
        frame = pd.DataFrame(
            {
                'id': range(20),
                'time': values,
                'y': [i * 1.25 for i in range(20)],
            }
        )
        data = Schema.from_data(
            {'t': frame}, primary_keys={'t': 'id'}, time_columns={'t': 'time'}
        )
        data['t']['y'].stype = 'numerical'
        return (
            PredictiveQuery('PREDICT t.y FOR EACH t.id', data)
            .materialize(indices=[1, 2], model=KumoRelational(), random_seed=7)
            .payloads
        )

    text = dates * 10
    native = pd.to_datetime(text, format='mixed', utc=True)
    assert payload(text) == payload(native)
