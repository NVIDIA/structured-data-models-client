# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current observation checks reject missing or corrupted native evidence."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / 'contract/runners/check_observations.py'


@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    folder = tmp_path_factory.mktemp('native-observations')
    path = folder / 'sqlite.jsonl'
    case = ROOT / 'contract/cases/s1/base.regression.json'
    subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            'python-core',
            '--database',
            'sqlite',
            '--out',
            str(path),
            str(case),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location(
        'native_observations', CHECKER
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    runner = checker.runner_module()
    expected = {
        'python-core': {
            'base.regression@sqlite': {
                'result': 'success',
                'stage': 'predict',
                'error': None,
                'code': None,
            }
        }
    }
    return (
        checker,
        runner,
        runner.load_cases([case]),
        expected,
        json.loads(path.read_text()),
    )


def test_capture_does_not_change_public_behavior(recorded, tmp_path):
    checker, _, _, _, captured = recorded
    path = tmp_path / 'without-capture.jsonl'
    subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            'python-core',
            '--database',
            'sqlite',
            '--no-capture',
            '--out',
            str(path),
            str(ROOT / 'contract/cases/s1/base.regression.json'),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    uncaptured = json.loads(path.read_text())
    assert uncaptured['driver']['capture'] is None
    assert captured['outcome'] == uncaptured['outcome']
    assert captured['host']['warnings'] == uncaptured['host']['warnings']
    for key in ('schema', 'result'):
        assert captured['public'][key] == uncaptured['public'][key]

    def payloads(item):
        return [
            (
                p['method'],
                p['path'],
                checker._payload(p['parsed'], (('metadata', 'request_id'),)),
            )
            for p in item['public']['payloads']
        ]

    assert payloads(captured) == payloads(uncaptured)
    for key in ('statements', 'bindings'):
        assert (
            captured['driver']['query_log'][key]
            == uncaptured['driver']['query_log'][key]
        )


def test_current_installed_client_observation(recorded, tmp_path):
    checker, runner, cases, expected, item = recorded
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps(item) + '\n')
    assert checker.check(path, cases, expected, 'sqlite', runner) == []


@pytest.mark.parametrize(
    'damage',
    [
        'no-native',
        'no-fetches',
        'no-prediction',
        'wrong-outcome',
        'wrong-input',
        'wrong-contract',
        'stale-commit',
        'wrong-cell',
        'no-operation',
        'missing-sql',
        'missing-query-log',
        'wrong-sql-count',
        'missing-endpoint-payload',
    ],
)
def test_corrupted_current_evidence_is_rejected(recorded, tmp_path, damage):
    checker, runner, cases, expected, original = recorded
    item = copy.deepcopy(original)
    if damage == 'no-native':
        item['identity']['native_loaded'] = False
    elif damage == 'no-fetches':
        item['core']['fetches'] = []
    elif damage == 'no-prediction':
        item['core']['predicted'] = None
    elif damage == 'wrong-outcome':
        item['outcome']['stage'] = 'materialize'
    elif damage == 'wrong-input':
        item['identity']['inputs']['case'] = '0' * 64
    elif damage == 'wrong-contract':
        item['identity']['contract'] = 1
    elif damage == 'stale-commit':
        item['identity']['commit']['sha'] = '0' * 40
    elif damage == 'wrong-cell':
        item['identity']['cell'] = 'ref-pandas3'
    elif damage == 'no-operation':
        item['identity']['operations'].remove('predict')
    elif damage == 'missing-sql':
        item.pop('driver')
    elif damage == 'missing-query-log':
        item['driver']['query_log'] = None
    elif damage == 'wrong-sql-count':
        item['driver']['query_log']['statements'] += 1
    elif damage == 'missing-endpoint-payload':
        item['public']['payloads'] = []
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps(item) + '\n')
    if damage == 'missing-query-log':
        with pytest.raises(ValueError, match='driver/query_log'):
            checker.check(path, cases, expected, 'sqlite', runner)
        return
    assert checker.check(
        path,
        cases,
        expected,
        'sqlite',
        runner,
        cell=original['identity']['cell']
        if damage != 'wrong-cell'
        else 'floor-pandas2.2',
    )


@pytest.mark.parametrize(
    'text',
    [
        '',
        '[',
        '[' * 100_000 + ']' * 100_000,
        '{"a":' * 100_000 + '1' + '}' * 100_000,
        json.dumps({f'k{i}': i for i in range(100_000)}),
        '{"format":"sdm-observation/1","format":"other"}',
        '{"number":NaN}',
    ],
)
def test_malformed_observations_fail_without_traceback(
    recorded, tmp_path, text
):
    checker, runner, _, _, _ = recorded
    path = tmp_path / 'malformed.jsonl'
    path.write_text(text)
    with pytest.raises(ValueError):
        checker.read_observations(path, runner)


def test_missing_expectation_is_not_inferred_from_observation(
    recorded, tmp_path
):
    checker, runner, cases, _, item = recorded
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps(item) + '\n')
    with pytest.raises(ValueError, match='explicit outcomes'):
        checker.check(path, cases, {}, 'sqlite', runner)


def test_duplicate_cases_are_not_silently_overwritten(recorded, tmp_path):
    checker, runner, _, _, item = recorded
    path = tmp_path / 'native.jsonl'
    path.write_text((json.dumps(item) + '\n') * 2)
    with pytest.raises(ValueError, match='duplicate case'):
        checker.read_observations(path, runner)


@pytest.mark.parametrize(
    ('left', 'right', 'equal'),
    [
        ('1e-05', '1e-5', True),
        ('1', '1.0', False),
        ('-0.0', '0.0', False),
        ('9007199254740992', '9007199254740993', False),
    ],
)
def test_numeric_tokens_keep_exact_kind_and_value(recorded, left, right, equal):
    checker = recorded[0]
    assert (
        checker._lossless({'number': left})
        == checker._lossless({'number': right})
    ) is equal


def test_request_id_mask_keeps_presence_and_type(recorded):
    checker = recorded[0]
    paths = (('metadata', 'request_id'),)
    original = {'metadata': {'request_id': 'one'}}
    assert checker._payload(original, paths) == checker._payload(
        {'metadata': {'request_id': 'two'}}, paths
    )
    for changed in ({'metadata': {}}, {'metadata': {'request_id': None}}):
        assert checker._payload(original, paths) != checker._payload(
            changed, paths
        )
    assert original['metadata']['request_id'] == 'one'


@pytest.mark.parametrize('field', ['native', 'versions', 'commit'])
def test_mixed_build_identities_fail(recorded, tmp_path, field):
    checker, runner, _, _, original = recorded
    other = copy.deepcopy(original)
    other['case'] = 'another-case'
    if field == 'native':
        other['identity'][field]['sha256'] = '0' * 64
    elif field == 'versions':
        other['identity'][field]['python'] = '3.14.0'
    else:
        other['identity'][field]['sha'] = '0' * 40
    path = tmp_path / 'mixed.jsonl'
    path.write_text(json.dumps(original) + '\n' + json.dumps(other) + '\n')
    with pytest.raises(ValueError, match='mixed build/runtime identity'):
        checker.read_observations(path, runner)


def test_native_hash_is_bound_to_installed_artifact(recorded, tmp_path):
    checker, runner, cases, expected, item = recorded
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps(item) + '\n')
    assert any(
        'native artifact hash differs' in failure
        for failure in checker.check(
            path, cases, expected, 'sqlite', runner, native_sha256='0' * 64
        )
    )
