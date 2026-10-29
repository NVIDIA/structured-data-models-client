# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual SQL materialization handovers feed the closed common journal."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path

import pandas as pd
import pytest
import test_t45_native_sql_labels as public
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema


@cache
def runner():
    name = 't45_actual_materialization_runner'
    path = Path(__file__).parents[3] / 'contract/runners/run_python.py'
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('case', ['ordinary', 'distinct', 'string-rhs'])
def test_public_source_and_native_label_products_have_actual_entry_and_late_read_proof(
    monkeypatch, case
):
    boundary = runner()._Boundary()
    try:
        result = public._observe(monkeypatch, case, True)
        capture = boundary.capture
        core = capture.core()
        observed = (
            runner()
            ._sibling('prediction_observation')
            .view(core, capture.operations)
        )
        [entry] = observed['materializations']
        assert entry['native'] is True
        assert 'materialize' in observed['operations']
        assert 'materialize' not in capture.operations
        stages = [
            record['input']['parsed']['stage']
            for record in core['materialization_preparations']
        ]
        assert stages[0] == 'task'
        assert 'population' in stages and 'label_inputs' in stages
        assert 'target_column' in stages
        assert any(
            record['owner_binds']
            for record in core['materialization_preparations']
        )
        task = core['materialization_preparations'][0]
        indices = task['input']['parsed']['request']['indices']
        assert (
            len(
                next(
                    item['scalars']
                    for item in task['input_reads']
                    if item['attachment'] == indices
                )
            )
            == 2
        )
        if result['outcome'] == 'success':
            assert entry['completion'] == 'returned'
            assert observed['materialized'] is not None
            terminal = next(
                record
                for record in core['materialization_preparations']
                if record['run'] == entry['terminal']['run']
            )
            assert terminal['output_reads']
            assert stages[-1] == 'request'
        else:
            assert result['stage'] == 'materialize'
            assert entry['completion'] == 'raised'
            assert entry['failure'] is not None
            assert core['materialization_guards']
        assert core['materialized'] is None
        assert capture._materialization_stack == []
        assert capture._materialization_owners == {}
        assert capture._materialization_presented == []
        assert all(
            owner.error_owner is None for owner in capture._runs.values()
        )
    finally:
        boundary.close()


def _display_failure(monkeypatch, captured):
    events = []
    display = RuntimeError('actual native display failed')

    class Unprintable(ValueError):
        def __str__(self):
            events.append('display')
            raise display

    native = Unprintable()
    actual = pd.Series.astype

    def converted(self, dtype, *args, **kwargs):
        if type(dtype) is str and dtype == 'int64':
            events.append('native-int-cast')
            raise native
        return actual(self, dtype, *args, **kwargs)

    with monkeypatch.context() as selected, public.connection([]) as db:
        public.route(selected, True)
        schema = Schema.from_source(DuckDBSource(db))
        boundary = runner()._Boundary() if captured else None
        try:
            selected.setattr(pd.Series, 'astype', converted)
            with pytest.raises(RuntimeError) as raised:
                PredictiveQuery(public.QUERY, schema).materialize(
                    anchor_time=pd.Timestamp('2023-03-01'), indices=[0, 1]
                )
            assert raised.value is display
            assert (
                BaseException.__getattribute__(display, '__context__') is native
            )
            assert BaseException.__getattribute__(display, '__cause__') is None
            if boundary is not None:
                core = boundary.capture.core()
                view = (
                    runner()
                    ._sibling('prediction_observation')
                    .view(core, boundary.capture.operations)
                )
                [entry] = view['materializations']
                assert entry['native'] and entry['completion'] == 'raised'
                assert entry['failure'] is None
                assert view['error'] is None and core['error'] is None
                assert core['materialization_guards']
                assert boundary.capture._materialization_owners == {}
        finally:
            if boundary is not None:
                boundary.close()
    return events


def test_actual_native_formatter_failure_is_retained_with_capture_and_without_it(
    monkeypatch,
):
    assert _display_failure(monkeypatch, False) == [
        'native-int-cast',
        'display',
    ]
    assert _display_failure(monkeypatch, True) == ['native-int-cast', 'display']
