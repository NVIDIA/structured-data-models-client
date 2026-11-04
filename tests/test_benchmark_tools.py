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
            }
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
        'columns': ['x0', 'x1'],
        'gradients': [[[0.25, 0.25]]],
    }


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
