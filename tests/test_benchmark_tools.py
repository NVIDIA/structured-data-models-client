# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check measurement math and the HTTP fixture without running client benchmarks."""

import importlib.util
import json
import subprocess
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

SPEC = importlib.util.spec_from_file_location(
    'client_cost',
    Path(__file__).resolve().parents[1] / 'scripts/benchmarks/client_cost.py',
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_rss_mean_weights_elapsed_time_and_peak_is_sampled():
    summary = bench.rss_summary([(0.0, 100), (1.0, 300), (4.0, 100)])
    assert summary['time_weighted_mean_bytes'] == 200
    assert summary['sampled_peak_bytes'] == 300
    assert summary['observed_seconds'] == 4
    with pytest.raises(ValueError, match='distinct RSS'):
        bench.rss_summary([(0.0, 100)])
    # Unequal spacing must not become an arithmetic sample mean.
    assert (
        bench.rss_summary([(0.0, 0), (1.0, 100), (10.0, 100)])[
            'time_weighted_mean_bytes'
        ]
        == 95
    )


def test_independent_response_preserves_relational_identity_and_gradient_shape():
    payload = {
        'predict': {
            'instance_table': {
                'columns': ['instance_id', 'x0', 'x1'],
                'rows': [['[12]', 4.0, 5.0], ['[17]', 8.0, 9.0]],
            },
            'related_tables': {
                'users': {
                    'columns': ['instance_id', 'id', 'x0', 'x1'],
                    'rows': [
                        ['[17]', 17, 8.0, 9.0],
                        ['[12]', 12, 4.0, 5.0],
                        ['[99]', 99, 1.0, 2.0],
                        ['[12]', 13, 6.0, 7.0],
                    ],
                }
            },
        }
    }
    plain = bench.answer(payload, 'kumo-relational', False)
    assert [row['id'] for row in plain['predictions']] == ['[12]', '[17]']
    assert all('explanation' not in row for row in plain['predictions'])
    explained = bench.answer(payload, 'kumo-relational', True)
    assert [row['prediction'] for row in explained['predictions']] == [1.0, 1.0]
    details = explained['predictions'][1]['explanation']['details']
    assert details['outputs'] == [{'kind': 'prediction', 'statistic': 'mean'}]
    assert details['instance_table'] == {
        'row_indices': [1],
        'columns': [],
        'gradients': [[[]]],
    }
    assert details['related_tables']['users'] == {
        'row_indices': [0],
        'columns': ['x0', 'x1'],
        'gradients': [[[0.25, 0.25]]],
    }
    first = explained['predictions'][0]['explanation']['details'][
        'related_tables'
    ]['users']
    assert first['row_indices'] == [1, 3]
    assert first['gradients'] == [[[0.25, 0.25], [0.25, 0.25]]]


def test_actual_http_session_fixture_and_counters():
    server = bench.Fixture(False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(
            Request(
                url + '/v1/sessions',
                json.dumps({'model': 'kumo-tabular'}).encode(),
            )
        ) as response:
            session = json.load(response)['session_id']
        payload = {
            'predict': {'instance_table': {'columns': ['x0'], 'rows': [[2.0]]}}
        }
        with urlopen(
            Request(
                url + f'/v1/sessions/{session}/predictions',
                json.dumps(payload).encode(),
            )
        ) as response:
            assert json.load(response)['predictions'] == [
                {'row_index': 0, 'prediction': 1.0}
            ]
        with urlopen(
            Request(url + f'/v1/sessions/{session}', method='DELETE')
        ) as response:
            assert response.status == 204
        assert {
            key: server.counts[key] for key in ('fit', 'predict', 'close')
        } == {'fit': 1, 'predict': 1, 'close': 1}
        assert server.counts['request_bytes'] > 0
        assert server.counts['response_bytes'] > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_source_identity_rejects_dirty_or_wrong_revision(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(
        [
            'git',
            '-C',
            str(tmp_path),
            '-c',
            'user.name=Fixture',
            '-c',
            'user.email=fixture@example.org',
            'commit',
            '-qm',
            'Fixture',
            '--allow-empty',
            '--signoff',
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True
    ).strip()
    assert bench.clean_revision(tmp_path, revision) == revision
    with pytest.raises(ValueError, match='clean and match'):
        bench.clean_revision(tmp_path, '0' * 40)
    (tmp_path / 'changed').write_text('changed')
    with pytest.raises(ValueError, match='clean and match'):
        bench.clean_revision(tmp_path, revision)


def test_wheel_proof_checks_installed_bytes_and_clean_revision(tmp_path):
    import zipfile

    installed = tmp_path / 'installed'
    installed.mkdir()
    (installed / '__init__.py').write_text('value = 1\n')
    wheel = tmp_path / 'client.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('sdm_client/__init__.py', 'value = 1\n')
    receipt = {
        'tree_clean': True,
        'commit': 'pinned',
        'sha256': bench.digest(wheel),
    }
    bench.verify_wheel(wheel, receipt, 'pinned', {'sdm_client': str(installed)})
    with pytest.raises(ValueError, match='clean pinned artifact'):
        bench.verify_wheel(
            wheel, receipt, 'different', {'sdm_client': str(installed)}
        )
    (installed / '__init__.py').write_text('value = 2\n')
    with pytest.raises(ValueError, match='installed package bytes differ'):
        bench.verify_wheel(
            wheel, receipt, 'pinned', {'sdm_client': str(installed)}
        )


def test_current_gradient_export_is_detected_with_instance_only_attribute():
    import sdm_client

    spec = importlib.util.spec_from_file_location(
        'python_case', Path(bench.__file__).with_name('python_case.py')
    )
    child = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(child)
    result = sdm_client.PredictionResult(
        model='kumo-tabular', request_id='fixture', items=[]
    )
    assert isinstance(
        result.explanations.gradients, sdm_client.GradientExplanations
    )
    assert child.gradient_capable(sdm_client)


def test_fixture_accepted_socket_disables_nagle():
    import socket

    server = bench.Fixture(False)
    try:
        with socket.create_connection(server.server_address):
            connection, _ = server.get_request()
            with connection:
                assert (
                    connection.getsockopt(
                        socket.IPPROTO_TCP, socket.TCP_NODELAY
                    )
                    == 1
                )
    finally:
        server.server_close()


@pytest.mark.parametrize('relational', [False, True])
def test_fixed_fixture_nonempty_gradients_through_public_client(relational):
    import pandas as pd
    from sdm_client import (
        KumoRelational,
        KumoTabular,
        PredictiveQuery,
        Schema,
        SDMClient,
    )
    from sdm_client import Request as ModelRequest

    users = pd.DataFrame(
        {
            'id': range(12),
            'x0': [float(i) for i in range(12)],
            'y': [float(i % 3) for i in range(12)],
        }
    )
    if relational:
        schema = Schema.from_data(
            {'users': users}, primary_keys={'users': 'id'}
        )
        schema['users']['y'].stype = 'numerical'
        schema['users']['x0'].stype = 'numerical'
        model = KumoRelational()
        request = PredictiveQuery(
            'PREDICT users.y FOR EACH users.id', schema
        ).materialize(
            model=model, indices=[10, 11], num_context_samples=4, random_seed=7
        )
    else:
        model = KumoTabular()
        request = ModelRequest(
            users.iloc[:-2],
            users.drop(columns=['y']).iloc[-2:],
            target='y',
            task='regression',
        )
    server = bench.Fixture(True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with SDMClient(url=f'http://127.0.0.1:{server.server_port}') as client:
            result = client.predict(
                request, model=model, outputs=['prediction', 'explanation']
            )
        frame = result.explanations.gradients.to_pandas()
        assert len(result.explanations.gradients) == 2
        assert len(frame) == 2
        assert (frame.gradient == 0.25).all()
        assert set(frame.column) == {'x0'}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_request_row_counts_keep_context_and_related_work_separate():
    payload = {
        'context': {
            'instance_table': {'rows': [[1], [2]]},
            'related_tables': {'users': {'rows': [[1], [2], [3]]}},
        },
        'predict': {
            'instance_table': {'rows': [[4]]},
            'related_tables': {
                'users': {'rows': [[4], [5]]},
                'events': {'rows': [[6]]},
            },
        },
    }
    assert bench.request_rows(payload, 'predict') == {
        'kind': 'predict',
        'context': {
            'present': True,
            'instance_rows': 2,
            'related_rows': {'users': 3},
        },
        'predict': {
            'present': True,
            'instance_rows': 1,
            'related_rows': {'users': 2, 'events': 1},
        },
    }
    assert bench.request_rows({}, 'fit')['predict'] == {
        'present': False,
        'instance_rows': 0,
        'related_rows': {},
    }


@pytest.mark.parametrize('fault', [None, 'value', 'missing', 'shape'])
def test_gradient_worker_keeps_correctness_without_duplicate_merge(
    monkeypatch, capsys, fault
):
    from types import SimpleNamespace

    from sdm_client import PredictionResult, StructuredDataError

    spec = importlib.util.spec_from_file_location(
        'gradient_worker', Path(bench.__file__).with_name('python_case.py')
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    answer = bench.answer

    def response(*args):
        value = answer(*args)
        for item in value['predictions']:
            if fault == 'missing':
                item.pop('explanation')
            elif fault == 'shape':
                item['explanation']['details']['instance_table'][
                    'gradients'
                ] = []
            elif fault == 'value':
                item['explanation']['details']['instance_table']['gradients'][
                    0
                ][0][0] = 0.5
        return value

    monkeypatch.setattr(bench, 'answer', response)

    def forbidden_merge(*args):
        raise AssertionError('unsupported duplicate merge was called')

    monkeypatch.setattr(PredictionResult, 'merge', forbidden_merge)
    server = bench.Fixture(True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        'SDM_BENCH_URL', f'http://127.0.0.1:{server.server_port}'
    )
    monkeypatch.setenv('SDM_BENCH_FIXED', '1')
    args = SimpleNamespace(
        workload='tabular-session',
        rows=12,
        scores=2,
        width=1,
        degree=1,
        warmups=0,
        loops=0,
        gradients=1,
    )
    try:
        if fault:
            with pytest.raises(
                (AssertionError, StructuredDataError)
            ) as refused:
                worker.run(args)
            assert 'unsupported duplicate merge was called' not in str(
                refused.value
            )
        else:
            worker.run(args)
            result = json.loads(capsys.readouterr().out)
            assert result['status'] == 'passed'
            assert result['iterations'][0]['gradient_cells'] == 2
            assert result['iterations'][0]['seconds']['result_merge'] is None
            assert result['unavailable_phases']['result_merge']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
