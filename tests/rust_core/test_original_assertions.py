# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed defects at observation boundaries and require the original checks to fail."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for directory in [
    ROOT / 'tests/rust_core/rfm',
    ROOT / 'tests/rust_core',
    ROOT / 'tests/rust_core/sampling/driver',
    ROOT / 'scripts/parity',
]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def load(relative):
    spec = importlib.util.spec_from_file_location(
        'preserved_' + Path(relative).stem, ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pandas3_prediction(expected):
    import pandas as pd

    if int(pd.__version__.split('.')[0]) == 2:
        position = expected['prediction']['names'].index('ANCHOR_TIMESTAMP')
        column = expected['prediction']['columns'][position]
        assert column['type'] == 'timestamp[ns]'
        assert all(
            value is None or value % 1000 == 0 for value in column['values']
        )
        column['type'] = 'timestamp[us]'
        column['values'] = [
            None if value is None else value // 1000
            for value in column['values']
        ]
    return expected


def _positive_fixed_tasks(monkeypatch, module):
    original = module.task_table_want
    monkeypatch.setattr(
        module,
        'task_table_want',
        lambda call: _pandas3_prediction(original(call)),
    )


@pytest.mark.parametrize('seed', ['count', 'duplicate'])
def test_exact_forwarding_and_once_only_reject_corruption(monkeypatch, seed):
    from sdm_client import PredictiveQuery, _impl

    monkeypatch.setattr(_impl, 'OLD', False)
    module = load(
        'packages/structured-data-models-client/tests/test_pql_for_tabular.py'
    )
    real = PredictiveQuery.materialize

    def altered(self, **arguments):
        if seed == 'count':
            assert arguments['num_context_samples'] == 500
            arguments['num_context_samples'] = 501
        result = real(self, **arguments)
        if seed == 'duplicate':
            real(self, **arguments)
        return result

    def check():
        if seed == 'count':
            module.test_a_valid_context_count_reaches_the_tabular_materializer(
                500
            )
        else:
            module.test_relational_options_are_forwarded_to_the_relational_materializer(
                monkeypatch
            )

    check()
    monkeypatch.setattr(PredictiveQuery, 'materialize', altered)
    with pytest.raises(AssertionError):
        check()


@pytest.mark.parametrize('case', ['count', 'context-ceiling', 'entity-ceiling'])
def test_actual_uncached_gather_is_rejected(monkeypatch, case):
    from sdm_client import PredictiveQuery, _core, _impl, _schema_core
    from sdm_client._request_core import _native

    monkeypatch.setattr(_impl, 'OLD', False)
    filename = (
        'test_query_engine_boundary.py'
        if case == 'entity-ceiling'
        else 'test_context_samples.py'
    )
    module = load('packages/structured-data-models-client/tests/' + filename)
    real = PredictiveQuery.materialize
    completed = []

    def altered(self, **arguments):
        schema, frames = _schema_core.schema_state(self.schema)
        state = {
            'schema': schema,
            'query': self.text,
            'args': {
                'indices': [1],
                'num_context_samples': 1,
                'random_seed': 7,
            },
            'tags': frames.tags,
            'model': None,
            'coverage_warned': False,
        }
        answer, _ = _core.Driver(self.schema.source).drive(
            _native.materialize_run(json.dumps(state), frames.tables)
        )
        assert json.loads(answer)['model'] is not None
        completed.append(True)
        return real(self, **arguments)

    def check():
        if case == 'count':
            module.test_a_count_that_is_not_a_count_is_refused(0, 'at least 1')
        elif case == 'context-ceiling':
            module.test_the_relational_ceiling_is_refused_before_the_engine_is_reached(
                monkeypatch
            )
        else:
            module.test_the_ceiling_is_refused_before_the_engine_is_reached()

    check()
    monkeypatch.setattr(PredictiveQuery, 'materialize', altered)
    with pytest.raises(AssertionError):
        check()
    assert completed == [True]


@pytest.mark.parametrize('operation', ['sampler', 'runs'])
@pytest.mark.parametrize(
    'name',
    [
        'duckdb-a-pair-of-key-columns',
        'duckdb-a-time-of-day',
        'postgres-a-time-of-day',
        'duckdb-null-foreign-keys',
        'sqlite-null-foreign-keys',
        'postgres-null-foreign-keys',
        'sqlite-earliest-time-unreadable',
    ],
)
def test_fixed_cases_reject_unrelated_native_initialization(
    monkeypatch, operation, name
):
    import test_driver_fixtures as module
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', True)
    _positive_fixed_tasks(monkeypatch, module)
    case = next(
        case for case in module.CASES if case['id'].endswith('/' + name)
    )
    check = (
        module.test_the_core_stands_in_for_the_sampler
        if operation == 'sampler'
        else module.test_the_core_runs_what_the_engine_runs
    )
    check(case)
    reached = []

    def unavailable(*args):
        reached.append(True)
        raise AssertionError('unrelated native initialization')

    monkeypatch.setattr(module, 'native', unavailable)
    monkeypatch.setitem(module._compare.__globals__, 'native', unavailable)
    with pytest.raises(AssertionError, match='unrelated native initialization'):
        check(case)
    assert reached == [True]


@pytest.mark.parametrize(
    'name',
    [
        'duckdb-a-pair-of-key-columns',
        'duckdb-a-time-of-day',
        'postgres-a-time-of-day',
        'duckdb-null-foreign-keys',
        'sqlite-null-foreign-keys',
        'postgres-null-foreign-keys',
        'sqlite-earliest-time-unreadable',
    ],
)
@pytest.mark.parametrize(
    'seed', ['sql', 'params', 'task', 'warning', 'skip', 'malformed']
)
def test_corrected_references_reject_low_level_drift(monkeypatch, name, seed):
    import test_driver_fixtures as module
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', True)
    _positive_fixed_tasks(monkeypatch, module)
    case = next(
        case for case in module.CASES if case['id'].endswith('/' + name)
    )
    check = module.test_the_core_runs_what_the_engine_runs
    check(case)
    real_compare = module._compare
    real_drive = real_compare.__globals__['drive']
    reached = []

    def compare(method, boundary, want, got):
        if method == 'task_table':
            reached.append(True)
            assert len(boundary.calls) == 1
            if seed == 'skip':
                return 0, []
            if seed == 'sql':
                boundary.calls[0]['steps'][0]['sql'] += ' /* seeded drift */'
            if seed == 'params':
                boundary.calls[0]['steps'][0]['params'] = {'unexpected': 7}
        return real_compare(method, boundary, want, got)

    def drive(*arguments):
        value, asked = real_drive(*arguments)
        if isinstance(value, dict) and 'task_type' in value:
            if seed == 'task':
                value['num_forecasts'] += 1
            elif seed == 'warning':
                value['warnings'].append(
                    {'category': 'UserWarning', 'message': 'seeded drift'}
                )
            elif seed == 'malformed':
                return None, asked
        return value, asked

    monkeypatch.setattr(module, '_compare', compare)
    monkeypatch.setitem(real_compare.__globals__, 'drive', drive)
    with pytest.raises((AssertionError, TypeError)):
        check(case)
    assert reached == [True]


@pytest.mark.parametrize(
    'name',
    [
        'duckdb-a-pair-of-key-columns',
        'duckdb-a-time-of-day',
        'postgres-a-time-of-day',
        'duckdb-null-foreign-keys',
        'sqlite-null-foreign-keys',
        'postgres-null-foreign-keys',
        'sqlite-earliest-time-unreadable',
    ],
)
def test_missing_precise_correction_is_rejected(monkeypatch, name):
    import test_driver_fixtures as module
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', True)
    _positive_fixed_tasks(monkeypatch, module)
    case = next(
        case for case in module.CASES if case['id'].endswith('/' + name)
    )
    module.test_the_core_runs_what_the_engine_runs(case)
    monkeypatch.setattr(
        module, '_corrected_reference', lambda case: contextlib.nullcontext()
    )
    with pytest.raises(AssertionError):
        module.test_the_core_runs_what_the_engine_runs(case)


@pytest.mark.parametrize('backend', ['duckdb', 'sqlite', 'postgres'])
def test_incorrect_null_mask_is_rejected(monkeypatch, backend):
    import test_driver_fixtures as module
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', True)
    _positive_fixed_tasks(monkeypatch, module)
    case = next(
        case
        for case in module.CASES
        if case['id'].endswith('/' + backend + '-null-foreign-keys')
    )
    module.test_the_core_runs_what_the_engine_runs(case)
    original = module.ast.parse
    changed = []

    def wrong_mask(source, *args, **kwargs):
        if source == '(code >= 0) & _mask[code]':
            source = '(code >= 0) | _mask[code]'
            changed.append(True)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(module.ast, 'parse', wrong_mask)
    with pytest.raises(AssertionError):
        module.test_the_core_runs_what_the_engine_runs(case)
    assert changed == [True]


def test_corrected_reference_restores_instrumentation():
    import _fixture_ops
    import test_driver_fixtures as module
    from sdm_client.materializers.relational import RelationalMaterializer
    from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler
    from sdm_relational_engine.rfm.base import sql_sampler

    watched = [
        (sql_sampler, '_reject_reserved_columns'),
        (sql_sampler.SQLSampler, '_sample_subgraph'),
        (_fixture_ops, '_driver_schema'),
        (RelationalMaterializer, '_build_graph'),
        (SQLiteSampler, '_get_min_max_time_dict'),
    ]
    original = [getattr(owner, name) for owner, name in watched]
    for case in module.CASES:
        if module._fixed(case, module.FIXED_IN_THE_CORE):
            with pytest.raises(RuntimeError, match='seeded reference exit'):
                with module._corrected_reference(case):
                    raise RuntimeError('seeded reference exit')
            assert [getattr(owner, name) for owner, name in watched] == original


@pytest.mark.parametrize(
    'seed', ['old-error', 'malformed-success', 'skipped-native', 'warning']
)
def test_fixed_fixture_checks_complete_both_outcomes(monkeypatch, seed):
    import test_driver_fixtures as module
    from sdm_client import _impl

    case = next(
        case
        for case in module.CASES
        if case['id'].endswith('/duckdb-null-foreign-keys')
    )
    real = module._outcome

    def altered(given):
        if seed == 'old-error' and _impl.OLD:
            return 'expected_error', {'message': 'unrelated old error'}
        if seed == 'skipped-native' and not _impl.OLD:
            return module._recorded(given)
        result = real(given)
        if not _impl.OLD:
            if seed == 'malformed-success':
                return 'expected', None
            if seed == 'warning':
                result[1]['warnings'].append(
                    ['UserWarning', 'unrelated warning']
                )
        return result

    module._fixed_outcome(case)
    monkeypatch.setattr(module, '_outcome', altered)
    with pytest.raises(AssertionError):
        module._fixed_outcome(case)


@pytest.mark.parametrize('seed', ['send', 'warning', 'error', 'details'])
def test_fixed_refusal_preserves_warnings_and_exact_send_count(
    monkeypatch, seed
):
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', True)
    module = load('tests/rust_core/test_request_beyond_the_corpus.py')
    real = module._asked

    def altered(calls, when, change):
        (outcome, warned), sent = real(calls, when, change)
        if calls is module._Core:
            if seed == 'send':
                sent = sent + sent
            elif seed == 'warning':
                warned = warned + [('UserWarning', 'unrelated warning')]
            else:
                values = list(outcome[1])
                values[2 if seed == 'error' else 3] = (
                    'unrelated refusal' if seed == 'error' else {'extra': True}
                )
                outcome = ('error', tuple(values))
        return (outcome, warned), sent

    module.test_a_frame_changed_after_its_schema_read_it(
        'a-transport-key-added', 'between-gathers'
    )
    monkeypatch.setattr(module, '_asked', altered)
    with pytest.raises(AssertionError):
        module.test_a_frame_changed_after_its_schema_read_it(
            'a-transport-key-added', 'between-gathers'
        )


@pytest.mark.parametrize('seed', ['unrelated-replay', 'skipped-native'])
def test_fixed_replay_is_never_a_scenario_name_waiver(monkeypatch, seed):
    import _parity.corpus
    import test_decimal_indices
    from _fixed_contracts import fixed_replay
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', True)
    path = test_decimal_indices.SCENARIO
    reason = _parity.corpus.check(path)
    fixed_replay(path, reason)
    if seed == 'skipped-native':
        monkeypatch.setattr(_parity.corpus, 'check', lambda path: None)
    else:
        reason = 'unrelated replay failure'
    with pytest.raises(AssertionError):
        fixed_replay(path, reason)


@pytest.mark.parametrize('caller', ['all', 'decimal', 'capture', 'dialect'])
def test_every_replay_caller_rejects_unrelated_failures(
    monkeypatch, tmp_path, caller
):
    import _driver
    import _parity.corpus
    import test_decimal_indices as decimal
    import test_rfm_runs as module

    path = decimal.SCENARIO
    monkeypatch.setattr(
        _parity.corpus, 'check', lambda path: 'unrelated replay failure'
    )
    if caller == 'all':
        monkeypatch.setattr(module, 'transcripts', lambda: [path])
        boundary = type(
            'Boundary',
            (),
            {'installed': lambda self, describe: contextlib.nullcontext()},
        )()
        assert module._replay_all(boundary, None)
    elif caller == 'decimal':
        with pytest.raises(AssertionError):
            decimal._kept_call()
    elif caller == 'dialect':
        import capture_dialects

        monkeypatch.setattr(capture_dialects, 'TRANSCRIPTS', path.parent)
        capture = capture_dialects.Capture()
        with pytest.raises(AssertionError):
            capture_dialects.replay_corpus(capture)
        assert all(not calls for calls in capture.calls.values())
    else:
        monkeypatch.setattr(_driver, 'transcripts', lambda: [path])
        with pytest.raises(AssertionError):
            _driver.capture_corpus(
                tmp_path,
                _driver.capture_module().describe,
                frozenset({'duckdb/regression#decimal-entity-keys'}),
            )
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('seed', ['early-error', 'native-error'])
def test_dialect_recorder_validates_fixed_replay_and_preserves_prefix(
    monkeypatch, tmp_path, seed
):
    import capture_dialects as recorder
    from _parity import corpus
    from sdm_client import _impl, _request_core

    path = (
        ROOT
        / 'tests/parity/transcripts/duckdb'
        / 'regression--decimal-entity-keys/transcript.json'
    )
    monkeypatch.setattr(_impl, 'OLD', True)
    monkeypatch.setattr(recorder, 'TRANSCRIPTS', path.parent)
    prefix = recorder.Capture()
    with (
        recorder.installed(prefix),
        prefix.under('scenario:duckdb/regression#decimal-entity-keys'),
    ):
        assert corpus.check(path) is not None
    assert prefix.calls['duckdb']
    capture = recorder.Capture()
    with recorder.installed(capture):
        assert recorder.replay_corpus(capture) == 1
    assert capture.files() == prefix.files()
    assert capture.source is None
    reached = []

    def unavailable(*args, **kwargs):
        reached.append(True)
        raise RuntimeError('unrelated seeded replay failure')

    if seed == 'early-error':
        monkeypatch.setattr(corpus, 'run', unavailable)
    else:
        monkeypatch.setattr(
            _request_core._native, 'materialize_run', unavailable
        )
    with pytest.raises(AssertionError):
        recorder.main(['--out', str(tmp_path)])
    assert reached == [True]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    'seed', ['none', 'field', 'warning', 'sql', 'old-error', 'native-error']
)
def test_decimal_comparison_rejects_malformed_success_and_drift(
    monkeypatch, seed
):
    import copy

    import pandas as pd
    import test_decimal_indices as decimal
    import test_rfm_runs as module
    from sdm_client import _impl
    from sdm_relational_engine.rfm.rfm import KumoRelational

    monkeypatch.setattr(_impl, 'OLD', True)
    model, arguments, steps = decimal._kept_call()
    boundary = module.Boundary(
        KumoRelational, '_get_task_table', lambda model: model._sampler
    )
    with (
        decimal._answering(steps, None),
        boundary.installed(module._model_describer('_get_task_table')),
    ):
        with pytest.raises(RuntimeError):
            model._get_task_table(**arguments)
    expected = module._decimal_task_want(boundary.calls[0])
    if int(pd.__version__.split('.')[0]) == 2:
        position = expected['prediction']['names'].index('ANCHOR_TIMESTAMP')
        column = expected['prediction']['columns'][position]
        assert column['type'] == 'timestamp[ns]'
        assert all(
            value is None or value % 1000 == 0 for value in column['values']
        )
        column['type'] = 'timestamp[us]'
        column['values'] = [
            None if value is None else value // 1000
            for value in column['values']
        ]
    monkeypatch.setattr(
        module, '_decimal_task_want', lambda call: copy.deepcopy(expected)
    )
    count, problems = module._compare(
        'task_table', boundary, module.task_table_want, module.task_table_got
    )
    assert count == 1 and problems == []
    if seed == 'sql':
        boundary.calls[0]['steps'][0]['sql'] += ' /* seeded drift */'
    elif seed == 'old-error':
        boundary.calls[0]['error'] = RuntimeError('unrelated old refusal')
    else:
        real = module.drive

        def altered(*args):
            if seed == 'native-error':
                raise boundary.calls[0]['error']
            value, asked = real(*args)
            if seed == 'none':
                return None, asked
            if seed == 'field':
                value['num_forecasts'] += 1
            else:
                value['warnings'].append(
                    {'category': 'UserWarning', 'message': 'unrelated warning'}
                )
            return value, asked

        monkeypatch.setattr(module, 'drive', altered)
    if seed == 'none':
        with pytest.raises(TypeError):
            module._compare(
                'task_table',
                boundary,
                module.task_table_want,
                module.task_table_got,
            )
    else:
        _, problems = module._compare(
            'task_table',
            boundary,
            module.task_table_want,
            module.task_table_got,
        )
        assert problems
