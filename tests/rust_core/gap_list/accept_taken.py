# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P78.taken: each branch the Phase 7.7 verifier found an offline public
call reaches is taken by a recorded case or a cited test that drives the
core, and the core matches today's Python on the verifier's routes
(``CONTRACT.md``, "Taken").
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import coverage
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _routes  # noqa: E402
from _gap_list import (  # noqa: E402
    FALSE_CLAUSES,
    REACHABLE,
    REPO,
    Branch,
    Measured,
    check_lines,
    citations,
    corpus_missing,
    entry_of,
    executed,
    measure_tests,
    says_reachable_only,
)

ROUTE_IDS = [
    (name, backend)
    for name, routes in _routes.ROUTES.items()
    for backend in routes
]


# --- Measuring today's Python ----------------------------------------------------


@contextlib.contextmanager
def traced(branch: Branch) -> Iterator[list[bool]]:
    r"""Branch coverage of the branch's module while the block runs; the
    list yielded holds, afterwards, whether the branch was taken."""
    path = str((REPO / branch.module).resolve())
    scratch = Path(tempfile.mkdtemp(prefix='sdm-p78-traced-'))
    cov = coverage.Coverage(
        branch=True,
        include=[path],
        data_file=str(scratch / 'data'),
        config_file=False,
    )
    took: list[bool] = []
    cov.start()
    try:
        yield took
    finally:
        cov.stop()
        cov.save()
        out = scratch / 'report.json'
        cov.json_report(outfile=str(out))
        report = executed(json.loads(out.read_text('utf-8')))
        took.append(branch.pair in report.get(branch.module, set()))


def run_quietly(route: Callable[[], Any]) -> tuple[str, Any]:
    r"""``('value', what the route gave)`` or ``('error', what it raised)``."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return 'value', route()
        except Exception as error:  # noqa: BLE001 - kept, compared
            return 'error', error


# --- The branches ---------------------------------------------------------------


def test_the_branches_are_where_the_checks_say() -> None:
    check_lines(REACHABLE)


@pytest.mark.parametrize(
    ('name', 'backend'), ROUTE_IDS, ids=[f'{n}-{b}' for n, b in ROUTE_IDS]
)
def test_the_route_takes_its_branch(name: str, backend: str) -> None:
    r"""The verifier's route, through today's Python, takes the branch: the
    route is real, and the branch reachable offline from a public call."""
    check_lines(REACHABLE)
    branch = REACHABLE[name]
    with traced(branch) as took:
        run_quietly(_routes.ROUTES[name][backend])
    assert took == [True], f'{name} over {backend} did not take {branch}'


# --- Held by a recorded case or a cited test -------------------------------------


def _cited() -> dict[str, list[str]]:
    r"""Each still-listed branch's cited tests."""
    out = {}
    for name, branch in REACHABLE.items():
        entry = entry_of(branch)
        if entry is not None:
            out[name] = citations(entry.reason)
    return out


@cache
def _cited_run() -> Measured | None:
    r"""The tests the still-listed entries cite, measured in one run, each
    watched for calls into the binding."""
    tests = sorted({t for cited in _cited().values() for t in cited})
    if not tests:
        return None
    return measure_tests(tests, watch_core=tests)


@pytest.mark.parametrize('name', list(REACHABLE))
def test_held_by_a_recorded_case_or_a_cited_test(name: str) -> None:
    r"""Either the entry is gone and a transcript, fixture case or PQL case
    takes the branch (``tests/parity/coverage.json``), or its reason says
    "reachable only", no longer says "unreachable" or the false clause, and
    cites tests of which every one passes and at least one takes the branch
    and calls into ``sdm_core._native`` while it runs."""
    check_lines(REACHABLE)
    branch = REACHABLE[name]
    entry = entry_of(branch)
    if entry is None:
        missing = corpus_missing()
        assert branch.module in missing, (
            f'{branch.module} is not in tests/parity/coverage.json'
        )
        assert branch.pair not in missing[branch.module], (
            f'{branch} ({name}) is no longer listed, but no transcript, '
            'fixture case or PQL case takes it (tests/parity/coverage.json)'
        )
        return
    reason = entry.reason
    problems = []
    if not says_reachable_only(reason):
        problems.append('the reason does not say "reachable only"')
    if 'unreachable' in reason.lower():
        problems.append('the reason still says "unreachable"')
    for clause in FALSE_CLAUSES[name]:
        if clause.lower() in reason.lower():
            problems.append(f'the reason still says {clause!r}')
    cited = citations(reason)
    if not cited:
        problems.append('the reason cites no test by its node id, in backticks')
    assert not problems, f'{branch} ({name}): ' + '; '.join(problems)
    measured = _cited_run()
    assert measured is not None
    for test in cited:
        outcome = measured.outcomes.get(test)
        assert outcome is not None, (
            f'{branch} ({name}) cites {test}, which pytest did not run:\n'
            f'{measured.tail}'
        )
        assert outcome == 'passed', f'{test} {outcome}:\n{measured.tail}'
    taking = [
        t
        for t in cited
        if branch.pair in measured.taken_by(t).get(branch.module, set())
    ]
    assert taking, f'none of {cited} takes {branch} ({name})'
    driving = [t for t in taking if t in measured.core_calls]
    assert driving, (
        f'{taking} take {branch} ({name}), but none calls into '
        'sdm_core._native while it runs'
    )


# --- The core on the routes -------------------------------------------------------


def _rfm_runs() -> Any:
    import test_rfm_runs

    return test_rfm_runs


def _boundaries(*methods: str) -> list[Any]:
    from sdm_relational_engine.rfm.rfm import KumoRelational

    runs = _rfm_runs()
    return [
        runs.Boundary(KumoRelational, method, lambda model: model._sampler)
        for method in methods
    ]


@contextlib.contextmanager
def _installed(boundaries: list[Any]) -> Iterator[None]:
    runs = _rfm_runs()
    with contextlib.ExitStack() as stack:
        for boundary in boundaries:
            stack.enter_context(
                boundary.installed(runs._model_describer(boundary.method))
            )
        yield


def _compared(kind: str, boundary: Any) -> None:
    runs = _rfm_runs()
    want, got = {
        'task_table': (runs.task_table_want, runs.task_table_got),
        'context': (runs.context_want, runs.context_got),
    }[kind]
    count, problems = runs._compare(kind, boundary, want, got)
    assert count >= 1, f'today made no {kind} call'
    assert not problems, f'{kind}: ' + '\n'.join(problems[:10])


@pytest.mark.parametrize('backend', ['duckdb', 'postgres'])
def test_the_core_matches_on_a_list_column(backend: str) -> None:
    r"""The core's ``task_table`` and ``context`` runs, driven from today's
    calls on the route, ask today's statements and give today's outputs."""
    check_lines(REACHABLE)
    branch = REACHABLE['list-column-on-sql']
    boundaries = _boundaries('_get_task_table', '_get_context')
    with traced(branch) as took, _installed(boundaries):
        kind, got = run_quietly(_routes.ROUTES['list-column-on-sql'][backend])
    assert took == [True], f'today did not take {branch}'
    assert kind == 'value', f'today refused: {got!r}'
    for kind_, boundary in zip(
        ('task_table', 'context'), boundaries, strict=True
    ):
        _compared(kind_, boundary)


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
def test_the_core_matches_with_entity_times_at_max(backend: str) -> None:
    r"""The core's ``task_table`` run, driven from today's call on the
    route, raises what today's raised, with the warnings before it."""
    check_lines(REACHABLE)
    branch = REACHABLE['entity-times-at-max']
    (boundary,) = _boundaries('_get_task_table')
    with traced(branch) as took, _installed([boundary]):
        kind, got = run_quietly(_routes.ROUTES['entity-times-at-max'][backend])
    assert took == [True], f'today did not take {branch}'
    assert kind == 'error', f'today did not refuse: {got!r}'
    assert any(call['error'] is not None for call in boundary.calls), (
        'today refused, but not in _get_task_table'
    )
    _compared('task_table', boundary)


# --- Through the host ----------------------------------------------------------------


def _host() -> Any:
    from _sampling import native

    native('materialize_run')
    if 'request_host' in sys.modules:
        return sys.modules['request_host']
    spec = importlib.util.spec_from_file_location(
        'request_host', REPO / 'scripts' / 'parity' / 'request_host.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['request_host'] = module
    spec.loader.exec_module(module)
    return module


def _outcome(make: Callable[[], Any]) -> tuple[Any, list[Any]]:
    r"""What ``make`` gave, its payloads, or what it raised as class, code,
    message and details; with the warnings given on the way."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            got: Any = ('value', json.loads(json.dumps(make())))
        except Exception as error:  # noqa: BLE001 - compared
            cls = type(error)
            got = (
                'error',
                (
                    f'{cls.__module__}.{cls.__qualname__}',
                    getattr(error, 'code', None),
                    getattr(error, 'message', str(error)),
                    dict(getattr(error, 'details', None) or {}),
                ),
            )
    said = [
        (f'{w.category.__module__}.{w.category.__qualname__}', str(w.message))
        for w in caught
    ]
    return got, said


def _key_to_nan(table: str, key: str) -> Callable[[dict[str, Any]], None]:
    def change(tables: dict[str, Any]) -> None:
        import numpy as np

        tables[table][key] = np.nan

    return change


@pytest.mark.parametrize(
    ('table', 'key'), [('products', 'product_id'), ('users', 'user_id')]
)
def test_the_core_matches_without_a_valid_key(table: str, key: str) -> None:
    r"""A key set to NaN in place after ``Schema.from_data``, gathered
    through today's Python and through the core by the request host
    (``scripts/parity/request_host.py``): the same payloads, or the same
    refusal, and the same warnings."""
    check_lines(REACHABLE)
    branch = REACHABLE['no-valid-key']
    change = _key_to_nan(table, key)
    gather = {'indices': [1, 2], 'num_context_samples': 5, 'random_seed': 7}

    def today() -> Any:
        from sdm_client import PredictiveQuery, Schema

        tables = _routes.frames()
        schema = Schema.from_data(tables)
        change(tables)
        return list(
            PredictiveQuery(_routes.COUNT, schema)
            .materialize(**gather)
            .payloads
        )

    host = _host()

    def core() -> Any:
        tables = _routes.frames()
        schema = host.schema_from_data(tables)
        change(tables)
        query = host.predictive_query(_routes.COUNT, schema)
        return host.materialize(query, **gather).state['payloads']

    with traced(branch) as took:
        want = _outcome(today)
    assert took == [True], f'today did not take {branch}'
    got = _outcome(core)
    assert got == want


# --- The first unseeded entity read ---------------------------------------------------


def _sampler_class(backend: str) -> type:
    if backend == 'sqlite':
        from sdm_relational_engine.rfm.backend.sqlite.sampler import (
            SQLiteSampler,
        )

        return SQLiteSampler
    from sdm_relational_engine.rfm.backend.postgres.sampler import (
        PostgresSampler,
    )

    return PostgresSampler


def _drive_through(
    core: Any, run: Any, steps: list[dict[str, Any]], upto: int
) -> None:
    r"""Drive ``run``, answering its queries with ``steps``' results, until it
    has asked for the first ``upto`` of them; each must be the next step,
    its SQL byte for byte and its parameters encoded alike."""
    from _driver import Mismatch, encode_params, outcome_of

    position = 0
    while position < upto:
        step = run.next()
        kind = getattr(step, 'kind', None)
        if kind == 'done':
            raise Mismatch(
                f'the run finished after {position} statements, before '
                f'asking for statement {upto - 1}'
            )
        assert kind == 'run_queries', f'a step of kind {kind!r}'
        for query in list(step.queries):
            if position >= upto:
                return
            want = steps[position]
            assert query.sql == want['sql'], (
                f'statement {position}: {query.sql!r} instead of {want["sql"]!r}'
            )
            params = encode_params(query.params)
            assert json.dumps(params) == json.dumps(want['params']), (
                f'statement {position}: parameters {params} instead of '
                f'{want["params"]}'
            )
            run.submit(query.id, outcome_of(core, want, want['result']))
            position += 1


@pytest.mark.parametrize('backend', ['sqlite', 'postgres'])
def test_the_core_asks_today_s_first_unseeded_entity_read(backend: str) -> None:
    r"""With ``random_seed=None``, the core's ``task_table`` run, driven from
    today's call and answered with today's results, asks exactly today's
    statements up to the end of today's first entity draw
    (``_sample_entity_table``), the statement the branch writes among them.
    What comes after depends on draws from the system's entropy, and is not
    compared."""
    from _sampling import native

    name = f'{backend}-unseeded'
    check_lines(REACHABLE)
    branch = REACHABLE[name]
    (boundary,) = _boundaries('_get_task_table')
    cls = _sampler_class(backend)
    original = vars(cls)['_sample_entity_table']
    spans: list[tuple[int, int]] = []

    def noted(self: Any, *args: Any, **kwargs: Any) -> Any:
        steps = boundary._steps
        start = len(steps) if steps is not None else -1
        try:
            return original(self, *args, **kwargs)
        finally:
            steps = boundary._steps
            spans.append((start, len(steps) if steps is not None else -1))

    cls._sample_entity_table = noted
    try:
        with traced(branch) as took, _installed([boundary]):
            kind, got = run_quietly(_routes.ROUTES[name][backend])
    finally:
        cls._sample_entity_table = original
    assert took == [True], f'today did not take {branch}'
    assert kind == 'value', f'today refused: {got!r}'
    assert boundary.calls, 'today made no _get_task_table call'
    call = boundary.calls[0]
    assert call['error'] is None, call['error']
    assert spans and spans[0][0] >= 0, 'no entity draw inside _get_task_table'
    start, end = spans[0]
    assert end > start, 'the first entity draw ran no statement'
    core = native('task_table', 'Outcome')
    inputs_json, tables = call['described']
    _drive_through(
        core, core.task_table(inputs_json, tables), call['steps'], end
    )
