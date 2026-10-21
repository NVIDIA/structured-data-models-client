# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The probes behind register entries DT-3 and ORD-1 (T1.21).

Run where the contract tests run (CI's ``engine`` job until T1.18):

    python -m pytest contract/tests -q

They run the candidate's Python engine on its default route, whose files
these probes reach (``rfm/backend/local/sampler.py``,
``rfm/base/sql_sampler.py``, ``rfm/base/sampler.py`` and
``rfm/inference/payload.py``) are byte for byte the reference's at
b413e9a; the same probes run against the reference checkout gave the
results the register's notes record. Each claim and what holds it:

- DT-3, the in-memory sampler: a graph with no time column is anchored at
  ``pd.Timestamp.min``; the sampler's seconds round trip
  (``rfm/backend/local/sampler.py:93`` and ``:160``) wraps it to
  2262-04-11T23:47:16.709551616, so the comparison at
  ``rfm/inference/payload.py:469-471`` no longer sees the sentinel and the
  payload's instance table gains an anchor time column holding that
  instant; the end-time check (``rfm/base/sampler.py:321-326``) compares
  against the same wrapped value and keeps every end time.
- DT-3, the SQL sampler: the same graph on DuckDB keeps the sentinel
  (``rfm/base/sql_sampler.py:573`` has no round trip), so the payload has
  no anchor time column, and the end-time check clears every end time.
- ORD-1: the unstable sort (``rfm/base/sampler.py:777-783``) runs only
  when ``sample_target`` is given the anchor ``'entity'``. The case
  format admits it from T2.1t (``anchor_time`` an ISO date or the literal
  ``entity``, for T2.11's ``anchor_entity`` cases), so whether the
  corpus reaches the sort is a matter of which anchors its cases name:
  running every case of the Phase 1b inventory and of ``contract/cases``
  on the default route calls ``sample_target`` with timestamps, but for
  T2.11's ``anchor_entity`` cases, whose users have distinct signup times
  (or none), so the sort meets no tie; the control shows the spy sees ``'entity'``
  when a public call passes it.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
CASES = ROOT / 'contract/cases'

# Timestamp.min's seconds, -9223372037, times 10**9, wrapped to int64.
WRAPPED = -9_223_372_037 * 10**9 + 2**64
WRAPPED_TIME = pd.Timestamp('2262-04-11 23:47:16.709551616')
END_TIMES = ['2020-01-01', None, '2030-01-01', '2021-05-05']


def _module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python')


# --- DT-3 -------------------------------------------------------------------------


def _in_memory_graph() -> Any:
    from sdm_relational_engine.rfm import Graph

    users = pd.DataFrame(
        {
            'user_id': range(4),
            'age': [20, 21, 22, 23],
            'until': pd.to_datetime(END_TIMES),
        }
    )
    graph = Graph.from_data(
        {'users': users}, edges=[], infer_metadata=False, verbose=False
    )
    graph['users'].primary_key = 'user_id'
    graph['users'].end_time_column = 'until'
    return graph


def _duckdb_graph(tmp_path: Path) -> Any:
    from sdm_relational_engine.rfm import Graph

    path = tmp_path / 'users.duckdb'
    connection = duckdb.connect(path)
    connection.execute(
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, '
        '(20 + i)::BIGINT AS age, '
        "[TIMESTAMP '2020-01-01', NULL, TIMESTAMP '2030-01-01', "
        "TIMESTAMP '2021-05-05'][i + 1] AS until FROM range(4) t(i)"
    )
    connection.close()
    return Graph.from_duckdb(
        path,
        tables=[
            dict(name='users', primary_key='user_id', end_time_column='until')
        ],
        verbose=False,
    )


def _sample(sampler: Any) -> tuple[Any, pd.Series]:
    r"""The subgraph of the four users at the graph's default anchor, as
    ``_get_context`` builds it for a task with no time column, and the
    anchor column the payload's instance table gets from it."""
    from sdm_relational_engine.rfm.inference import payload
    from sdm_relational_engine.rfm.inference.context import (
        PredictionContext,
        TaskType,
    )

    with warnings.catch_warnings():
        # DuckDB's missing-index advice, which says nothing about anchors.
        warnings.simplefilter('ignore')
        anchor = sampler.get_max_time()
        # No table has a time column, so the default anchor is the sentinel.
        assert anchor == pd.Timestamp.min
        subgraph = sampler.sample_subgraph(
            entity_table_names=('users',),
            entity_pkey=pd.Series([0, 1, 2, 3]),
            anchor_time=pd.Series(anchor).repeat(4).reset_index(drop=True),
            num_neighbors=[],
        )
    context = PredictionContext(
        task_type=TaskType.REGRESSION,
        entity_table_names=('users',),
        sampled_subgraph=subgraph,
        y_train=pd.Series([1.0, 2.0, 3.0], name='TARGET'),
        y_test=None,
    )
    frame, _, column = payload._instance_dataframe(context)
    anchors = frame[column] if column is not None else pd.Series([], dtype=str)
    return subgraph, anchors


def test_dt3_the_in_memory_round_trip_hides_the_sentinel() -> None:
    from sdm_relational_engine.rfm.backend.local import LocalSampler

    subgraph, anchors = _sample(LocalSampler(_in_memory_graph(), False))
    assert pd.Timestamp(WRAPPED) == WRAPPED_TIME
    assert subgraph.anchor_time.dtype == np.int64
    assert subgraph.anchor_time.tolist() == [WRAPPED] * 4
    # The comparison at payload.py:469-471 does not see the sentinel, so
    # the instance table carries the wrapped instant as its anchor time.
    assert anchors.name == '__kumo_anchor_time'
    assert anchors.tolist() == [WRAPPED_TIME] * 4
    # The end-time check compares against the wrapped value: none is
    # later, so every end time stays.
    until = subgraph.tables['users'].df['until']
    assert until.isna().tolist() == [False, True, False, False]


def test_dt3_the_sql_sampler_keeps_the_sentinel(tmp_path: Path) -> None:
    from sdm_relational_engine.rfm.backend.duckdb import DuckDBSampler

    # The graph holds the connection the sampler reads through.
    graph = _duckdb_graph(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sampler = DuckDBSampler(graph, False)
    subgraph, anchors = _sample(sampler)
    assert subgraph.anchor_time.dtype == np.int64
    assert subgraph.anchor_time.tolist() == [pd.Timestamp.min.value] * 4
    # The sentinel survives, so the instance table has no anchor time.
    assert anchors.empty
    # Every end time is after the sentinel, so every one is cleared.
    until = subgraph.tables['users'].df['until']
    assert len(until) == 4
    assert until.isna().all()


# --- ORD-1 ------------------------------------------------------------------------


class _Spy:
    r"""Records each ``sample_target`` call's case, query type and anchors,
    calling the engine's own ``sample_target`` with what it was given."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sdm_relational_engine.rfm.base.sampler import Sampler

        self.case = ''
        self.calls: list[tuple[str, str, Any, Any]] = []
        sample_target = Sampler.sample_target
        spy = self

        def recorded(
            sampler: Any, query: Any, *arguments: Any, **keywords: Any
        ) -> Any:
            spy.calls.append(
                (
                    spy.case,
                    str(query.query_type),
                    keywords['train_anchor_time'],
                    keywords['test_anchor_time'],
                )
            )
            return sample_target(sampler, query, *arguments, **keywords)

        monkeypatch.setattr(Sampler, 'sample_target', recorded)


def _every_case() -> list[dict[str, Any]]:
    inventory = run_python.load_inventory()
    ids = [entry['id'] for entry in inventory['cases']]
    assert len(ids) == 40
    # From T1.21e every inventory case is a committed case; one the
    # inventory holds and contract/cases does not is taken from it.
    committed = run_python.load_cases([CASES])
    have = {case['id'] for case, _ in committed}
    cases = committed + run_python.inventory_cases(
        [case_id for case_id in ids if case_id not in have]
    )
    return [case for case, _ in cases]


def test_ord1_the_case_format_admits_the_entity_anchor() -> None:
    # From T2.1t a case may name the entity anchor, as anchor_time or
    # context_anchor_time; any other text is still no anchor. Which anchors
    # the corpus exercises is test_ord1_no_case_reaches_the_tie_sort's.
    validate = run_python.validator('case')
    case = next(
        case
        for case in _every_case()
        if 'anchor_time' in case['input'].get('materialize', {})
    )
    assert run_python._invalid(validate, case) is None
    for key in ('anchor_time', 'context_anchor_time'):
        named = copy.deepcopy(case)
        named['input']['materialize'][key] = 'entity'
        assert run_python._invalid(validate, named) is None, key
        named['input']['materialize'][key] = 'entities'
        assert run_python._invalid(validate, named) is not None, key


#: T2.11's cases that pass the entity anchor and reach sample_target.
ENTITY_ANCHORED = frozenset(
    {
        'sample.anchor-entity',
        'sample.anchor-entity.regression',
        'sample.anchor-entity.context-entity',
        'sample.anchor-entity.null-time',
        'sample.anchor-entity.null-time.prediction-time',
    }
)


def test_ord1_no_case_reaches_the_tie_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    spy = _Spy(monkeypatch)
    bases = run_python.load_bases()
    cases = _every_case()
    for case in cases:
        if 'materialize' not in case['input']:
            continue
        spy.case = case['id']
        run_python._call(case, bases)
    calls = spy.calls
    reached = {case_id for case_id, _, _, _ in calls}
    # Not vacuous: the cases that reached sample_target on the reference
    # reach it here too, static queries among them.
    assert {
        'base.regression',
        'base.binary',
        'base.multiclass',
        's0.accept.sum-window',
        'time.timestamptz',
        'null.target-all-null',
    } <= reached
    assert len(reached) >= 31
    # The two static cases (base.multiclass, decimal.class-target-38-18)
    # predict for the shop base's users, which have no time column, so
    # neither has an entity time column for the sort to read either.
    assert any(kind == 'static' for _, kind, _, _ in calls)
    # From T2.11 the anchor_entity cases name the entity anchor on purpose,
    # and reach the sort; each gives its users distinct signup times (the
    # null-time cases one null, which count() leaves out), so the sort
    # meets no tie and ORD-1's instability cannot show.
    entity = {
        case_id
        for case_id, _, train, test in calls
        if 'entity' in (train, test)
    }
    assert entity == ENTITY_ANCHORED
    by_id = {case['id']: case for case in cases}
    for case_id in entity:
        connection = duckdb.connect()
        try:
            for statement in [
                *bases[by_id[case_id]['input']['base']],
                *by_id[case_id]['input'].get('setup', []),
            ]:
                connection.execute(statement)
            times, distinct = connection.execute(
                'SELECT count(signup), count(DISTINCT signup) FROM users'
            ).fetchone()
        finally:
            connection.close()
        assert times == distinct > 0, case_id
    for case_id, _, train, test in calls:
        if case_id in entity:
            continue
        assert isinstance(train, pd.Timestamp), case_id
        assert isinstance(test, pd.Timestamp), case_id


def test_ord1_the_spy_sees_the_entity_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    spy = _Spy(monkeypatch)
    spy.case = 'control'
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, '
        "['US', 'DE', 'IN'][1 + i % 3] AS country, "
        "TIMESTAMP '2024-01-01' + to_days((i % 2)::INTEGER) AS signed_up "
        'FROM range(40) t(i)'
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            DuckDBSource(connection), time_columns={'users': 'signed_up'}
        )
        PredictiveQuery(
            'PREDICT users.country FOR EACH users.user_id', schema
        ).materialize(indices=[0, 1], anchor_time='entity')
    connection.close()
    assert spy.calls == [('control', 'static', 'entity', 'entity')]
