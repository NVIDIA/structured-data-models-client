# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core's sampling runs against today's Python on the driver fixtures.

``tests/parity/fixtures/driver.json`` (written by
``scripts/parity/make_fixtures.py``, its ``driver`` group) asks questions
through the public API of tables built for each case: keys, constraints and
indexes the corpus's tables never have, tables too large for one batch of
candidates, sequences stored as text, and questions the engine refuses.
``tests/parity/test_fixtures.py`` holds today's Python to each case's
payloads, warnings or error, under the coverage P75.coverage measures. Here
the core is held to the same cases, twice over:

- With the core standing in for every ``sample_target`` and
  ``sample_subgraph`` call, as P75.payload has it stand in
  (``_driver.Interceptor``), each case must give its expected payloads and
  warnings, or raise its error after the warnings recorded with it. Where
  the core's run raises, the warnings it carries are given before today's
  error is raised, as a host standing the core in will give them.
- Every call of ``KumoRelational._get_task_table`` and ``_get_context``, and
  of the DuckDB and SQLite checks for missing indexes, is watched as
  ``test_rfm_runs.py`` watches them. The core's run of each, driven from the
  call's inputs, must ask exactly the statements today's method ran and give
  what it gave, or raise an exception of the same class with the same text
  and the warnings today's method gave before it raised.

``tests/parity/test_fixtures.py`` holds today's Python to each refusal's
code and message; here it is also held to the warnings recorded with it.

For task 7b fixes, the original outcome is pinned to BASE's complete error
or payload digest and warnings. The fixed public materialization must
invoke the native run exactly once and match the governed fixture's entire
payload digest and warnings. This also covers fixes whose old execution
refuses before a sampler or task-table boundary exists.

Without the core or the engine installed, as in CI's test job, it skips.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import json
import sys
import textwrap
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('sdm_core')
pytest.importorskip('sdm_relational_engine')

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / 'sampling' / 'driver'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _driver import (  # noqa: E402
    METHODS,
    Interceptor,
    SwapFailed,
    _category,
    canon_warnings,
    capture_module,
    native,
)
from _fixture_ops import FixtureError, _driver_schema, run, same  # noqa: E402
from test_rfm_runs import (  # noqa: E402
    REPO,
    Boundary,
    _compare,
    _model_describer,
    context_got,
    context_want,
    duckdb_missing_got,
    duckdb_missing_want,
    sqlite_missing_got,
    sqlite_missing_want,
    task_table_got,
    task_table_want,
)

# Today's Python, which these hold the core to, runs public calls here.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _old_path import python_path  # noqa: E402,F401 - the fixture

pytestmark = pytest.mark.usefixtures('python_path')

FIXTURE = REPO / 'tests' / 'parity' / 'fixtures' / 'driver.json'
CASES: list[dict[str, Any]] = (
    json.loads(FIXTURE.read_text('utf-8'))['cases'] if FIXTURE.is_file() else []
)
IDS = [case['id'].rsplit('/', 1)[-1] for case in CASES]
REFUSED = [case for case in CASES if 'expected_error' in case]

#: The cases a Phase 9 task 7b fix answers otherwise than today's code, by
#: their ids, for each comparison they change: the core no longer stands in
#: for today's sampler there, by design (``CHANGELOG.md``; the shadow
#: comparison's ``fix_changes.json``).
NULL_KEY = 'a null foreign key joins no row'
TIME_OF_DAY = 'a TIME column is text, not the time column'
#: Each records the core's fixed answer, which today's code, run here, does
#: not give.
FIXED_IN_THE_CORE = {
    'duckdb-a-pair-of-key-columns': "a composite key's derived column is "
    'not refused as a reserved name',
    'duckdb-a-time-of-day': TIME_OF_DAY,
    'postgres-a-time-of-day': TIME_OF_DAY,
    'duckdb-null-foreign-keys': NULL_KEY,
    'sqlite-null-foreign-keys': NULL_KEY,
    'postgres-null-foreign-keys': NULL_KEY,
    'sqlite-earliest-time-unreadable': 'an unreadable earliest time is '
    'dropped, as the others are',
}
#: With the core's sampler standing in, today's engine joins no row for a
#: null key either, and gives what is recorded.
FIXED_IN_THE_SAMPLER = {
    case: why for case, why in FIXED_IN_THE_CORE.items() if why != NULL_KEY
}
FIXED_IN_THE_RUNS = FIXED_IN_THE_CORE


def _fixed(case: dict[str, Any], fixed: dict[str, str]) -> str | None:
    r"""Why ``case`` records an answer today's code does not give, if a
    fix in the core changed it."""
    return fixed.get(case['id'].rsplit('/', 1)[-1])


def _outcome(case: dict[str, Any]) -> tuple[str, Any]:
    r"""What the case's operation gives now: ``('expected', output)`` or
    ``('expected_error', described)``, the error as the fixture records it:
    its class, code, message, details keys and warnings."""
    try:
        got = run(case['operation'], case['input'])
    except FixtureError as error:
        return 'expected_error', error.described()
    return 'expected', json.loads(json.dumps(got))


def _recorded(case: dict[str, Any]) -> tuple[str, Any]:
    if 'expected' in case:
        return 'expected', case['expected']
    return 'expected_error', case['expected_error']


class _Kept:
    r"""The core's module, keeping the exception the last run started from it
    raised."""

    def __init__(self, core: Any) -> None:
        self._core = core
        self.raised: BaseException | None = None

    def __getattr__(self, name: str) -> Any:
        found = getattr(self._core, name)
        if name not in METHODS:
            return found

        def start(*args: Any) -> Any:
            return _KeptRun(found(*args), self)

        return start


class _KeptRun:
    def __init__(self, run: Any, kept: _Kept) -> None:
        self._run = run
        self._kept = kept

    def next(self) -> Any:
        try:
            return self._run.next()
        except Exception as raised:
            self._kept.raised = raised
            raise

    def submit(self, *args: Any) -> Any:
        return self._run.submit(*args)


class _GivingWarnings(Interceptor):
    r"""``_driver.Interceptor``, giving the warnings a run of the core raised
    with before today's error is raised, as a host standing the core in will
    give them."""

    def __init__(self, describe: Any, core: Any, to_python: Any) -> None:
        self._kept = _Kept(core)
        super().__init__(describe, self._kept, to_python)

    def _swap(self, method: str, *rest: Any) -> Any:
        self._kept.raised = None
        try:
            return super()._swap(method, *rest)
        except SwapFailed:
            raise
        except Exception:
            raised = self._kept.raised
            if raised is None or not hasattr(raised, 'warnings'):
                raise SwapFailed(
                    f'{method}: the run raised {raised!r} without warnings'
                ) from None
            for item in canon_warnings(raised.warnings, method):
                warnings.warn(
                    item['message'], _category(item['category']), stacklevel=2
                )
            raise


def test_the_driver_fixtures_exist() -> None:
    assert len(CASES) >= 40, f'{FIXTURE} holds {len(CASES)} cases'
    assert all('warnings' in case['expected_error'] for case in REFUSED)
    assert any(
        'Could not parse' in message
        for case in REFUSED
        for _, message in case['expected_error']['warnings']
    ), 'no refusal comes after a warning that times did not parse'


@pytest.mark.parametrize(
    'case', REFUSED, ids=[case['id'].rsplit('/', 1)[-1] for case in REFUSED]
)
def test_today_warns_before_its_errors_as_recorded(
    case: dict[str, Any],
) -> None:
    got = _outcome(case)
    want = _recorded(case)
    assert got[0] == want[0] and same(got[1], want[1]), (got, want)


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_the_core_stands_in_for_the_sampler(case: dict[str, Any]) -> None:
    fixed = _fixed(case, FIXED_IN_THE_SAMPLER)
    if fixed:
        _fixed_outcome(case)
        with _corrected_reference(case):
            _stands_in(case)
        return
    _stands_in(case)


def _stands_in(case: dict[str, Any]) -> None:
    core = native(*METHODS, 'Outcome')
    capture = capture_module()
    interceptor = _GivingWarnings(capture.describe, core, capture.to_python)
    with interceptor.installed():
        got = _outcome(case)
    want = _recorded(case)
    assert got[0] == want[0] and same(got[1], want[1]), (got, want)
    if want[0] == 'expected':
        assert interceptor.calls, 'no sampler call went to the core'
    if _fixed(case, FIXED_IN_THE_CORE):
        assert [call['method'] for call in interceptor.calls] == [
            'sample_target',
            'sample_subgraph',
        ]


def _missing_indices(method: str) -> Callable[[Any, dict[str, Any]], Any]:
    def describe(sampler: Any, arguments: dict[str, Any]) -> Any:
        return capture_module().describe(
            sampler, arguments['graph'], method, arguments
        )

    return describe


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_the_core_runs_what_the_engine_runs(case: dict[str, Any]) -> None:
    fixed = _fixed(case, FIXED_IN_THE_RUNS)
    if fixed:
        _fixed_outcome(case)
        with _current_corrected_reference(case):
            _runs_what_the_engine_runs(case)
        return
    _runs_what_the_engine_runs(case)


def _old_fixed_outcome(case: dict[str, Any]) -> tuple[str, Any]:
    """The seven outcomes in fd195c2's governed driver.json, unchanged."""
    from _fixed_contracts import missing_index_warning

    name = case['id'].rsplit('/', 1)[-1]
    indexed = [['builtins.UserWarning', missing_index_warning(7, 4)]]
    null_hashes = {
        'duckdb-null-foreign-keys': 'a987062ce5355d86477533c0f1ddc3ad107d752127186c807d437e9ac14fc320',
        'sqlite-null-foreign-keys': '81c8ea76aa555b5fddcbddd43ebad639fe509301da68d6850dcc5a22fca89b68',
        'postgres-null-foreign-keys': '2bd026d4a9ad51ea71b346ca479330318f6436c736d6a804fd8abf0b26d9c720',
    }
    if name in null_hashes:
        return 'expected', {
            'payloads': 1,
            'sha256': null_hashes[name],
            'warnings': [] if name.startswith('postgres-') else indexed,
        }
    if name == 'duckdb-a-pair-of-key-columns':
        message = (
            'stock.__kumo_key_73746f72655f636f6465_6974656d5f6964 uses a name '
            'reserved by the SQL backends: a column may not start with '
            "'__kumo_', because the generated joins already select columns "
            'of that name and the reference would be ambiguous. Rename the '
            'column, or project it under another name.'
        )
        warned = [
            [
                'sdm_client.core.warnings.SDMWarning',
                "table 'stock' has the composite primary key ['store_code', "
                "'item_id'], and a predictive query can only name a single key "
                "column, so this query is resolved against 'stock' as though it "
                'had no key. The request still carries the composite key; only '
                'the query does not see it. Name a single column as the primary '
                "key if the query needs to identify a row of 'stock'.",
            ]
        ]
    elif name in ('duckdb-a-time-of-day', 'postgres-a-time-of-day'):
        message = (
            'Failed to collect any context examples. Is your predictive '
            'query too restrictive?'
        )
        unparsable = [
            [
                'builtins.UserWarning',
                "Could not parse 8 of 8 value(s) of time column 'opens' "
                "(for example [datetime.time(10, 0)]); they were dropped as 'NaT'",
            ]
        ]
        warned = (
            unparsable
            + (indexed if name.startswith('duckdb-') else [])
            + unparsable
        )
    else:
        assert name == 'sqlite-earliest-time-unreadable'
        message = 'Expected pd.Timestamp, got NaT'
        warned = [['builtins.UserWarning', missing_index_warning(9, 5)]]
    return 'expected_error', {
        'class': 'sdm_client.errors.StructuredDataError',
        'code': 'INVALID_REQUEST',
        'message': message,
        'details_keys': [],
        'warnings': warned,
    }


def _current_original_fixed_outcome(case: dict[str, Any]) -> tuple[str, Any]:
    """Current upstream original observations; historical BASE pins stay separate."""
    name = case['id'].rsplit('/', 1)[-1]
    outcomes = {
        'duckdb-a-time-of-day': (
            'expected',
            {
                'payloads': 1,
                'sha256': '1f3694cdee0389939ea7d61b9e0a1a8bb11d8a2968f4c69bc76543bf0f5fc670',
                'warnings': [
                    [
                        'builtins.UserWarning',
                        'Could not parse 8 of 8 value(s) of '
                        "time column 'opens' (for example "
                        '[datetime.time(10, 0)]); they were '
                        "dropped as 'NaT'",
                    ],
                    [
                        'sdm_client.core.warnings.SDMWarning',
                        'Some time columns were inferred, '
                        'or refused, on a guess about what '
                        'a column holds:\n'
                        '- items.opens was not taken as the '
                        'time column because no row holds a '
                        'value. If it does say when a row '
                        'became true, name it: '
                        "time_columns={'items': 'opens'}.",
                    ],
                    [
                        'builtins.UserWarning',
                        'Missing 7 indices in 4 tables for '
                        'optimal database querying. For '
                        'improving runtime, we strongly '
                        'suggest to create indices for '
                        'primary and foreign keys, e.g., '
                        'automatically by instantiating '
                        'Kumo Relational via '
                        '`KumoRelational(graph, '
                        'optimize=True)`.',
                    ],
                    [
                        'builtins.UserWarning',
                        'Could not parse 32 of 32 value(s) '
                        "of time column 'opens' (for "
                        'example [datetime.time(10, 0)]); '
                        "they were dropped as 'NaT'",
                    ],
                ],
            },
        ),
        'postgres-a-time-of-day': (
            'expected',
            {
                'payloads': 1,
                'sha256': 'b4efb3e83f570b72bd5057952017f9a31f2b0df11f2bebeffe513d9f1e75a57c',
                'warnings': [
                    [
                        'builtins.UserWarning',
                        'Could not parse 8 of 8 value(s) '
                        "of time column 'opens' (for "
                        'example [datetime.time(10, 0)]); '
                        "they were dropped as 'NaT'",
                    ],
                    [
                        'sdm_client.core.warnings.SDMWarning',
                        'Some time columns were inferred, '
                        'or refused, on a guess about '
                        'what a column holds:\n'
                        '- items.opens was not taken as '
                        'the time column because no row '
                        'holds a value. If it does say '
                        'when a row became true, name it: '
                        "time_columns={'items': "
                        "'opens'}.",
                    ],
                    [
                        'builtins.UserWarning',
                        'Could not parse 32 of 32 '
                        "value(s) of time column 'opens' "
                        '(for example [datetime.time(10, '
                        '0)]); they were dropped as '
                        "'NaT'",
                    ],
                ],
            },
        ),
    }
    if name in outcomes:
        return outcomes[name]
    return _old_fixed_outcome(case)


def _fixed_outcome(case: dict[str, Any]) -> None:
    from sdm_client import _impl, _request_core

    listed = json.loads(
        (REPO / 'tests/rust_core/shadow/fix_changes.json').read_text()
    )
    assert case['id'] in listed
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_impl, 'OLD', True)
        old = _outcome(case)
    assert old == _current_original_fixed_outcome(case), old
    core = native('materialize_run', 'Outcome')
    real = core.materialize_run
    started = []

    def observed(*arguments):
        started.append(arguments)
        return real(*arguments)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_impl, 'OLD', False)
        patch.setattr(_request_core._native, 'materialize_run', observed)
        got = _outcome(case)
    assert len(started) == 1
    want = _recorded(case)
    assert got[0] == want[0] and same(got[1], want[1]), (got, want)


@contextlib.contextmanager
def _corrected_reference(case):
    from types import SimpleNamespace

    import pandas as pd
    from predictive_query.schema.typing import Dtype, Stype
    from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler
    from sdm_relational_engine.rfm.base import sql_sampler

    name = case['id'].rsplit('/', 1)[-1]
    with pytest.MonkeyPatch.context() as patch:
        if name == 'duckdb-a-pair-of-key-columns':
            original = sql_sampler._reject_reserved_columns
            derived = '__kumo_key_73746f72655f636f6465_6974656d5f6964'

            def check(graph):
                stock = graph.tables['stock']
                assert stock.primary_key.name == derived
                assert stock.primary_key_columns == ('store_code', 'item_id')
                assert stock[derived].expr is not None
                projected = {
                    table.name: SimpleNamespace(
                        name=table.name,
                        columns=[
                            column
                            for column in table.columns
                            if not (
                                table.name == 'stock' and column.name == derived
                            )
                        ],
                    )
                    for table in graph.tables.values()
                }
                original(SimpleNamespace(tables=projected))

            patch.setattr(sql_sampler, '_reject_reserved_columns', check)
        elif name in ('duckdb-a-time-of-day', 'postgres-a-time-of-day'):
            import _fixture_ops
            from sdm_client.materializers.relational import (
                RelationalMaterializer,
            )

            original_schema = _fixture_ops._driver_schema
            original_graph = RelationalMaterializer._build_graph

            @contextlib.contextmanager
            def schema(arguments):
                with contextlib.ExitStack() as stack:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        built = stack.enter_context(original_schema(arguments))
                    expected = (
                        "Could not parse 8 of 8 value(s) of time column 'opens' "
                        "(for example [datetime.time(10, 0)]); they were dropped as 'NaT'"
                    )
                    removed = 0
                    for warning in caught:
                        if (
                            warning.category is UserWarning
                            and str(warning.message) == expected
                        ):
                            removed += 1
                        else:
                            warnings.warn(
                                warning.message, warning.category, stacklevel=2
                            )
                    assert removed == 1
                    yield built

            def graph(materializer):
                built = original_graph(materializer)
                items = built.tables['items']
                assert items.time_column.name == 'opens'
                items.time_column = None
                items['opens']._dtype = Dtype.string
                items['opens'].stype = Stype.categorical
                return built

            patch.setattr(_fixture_ops, '_driver_schema', schema)
            patch.setattr(RelationalMaterializer, '_build_graph', graph)
        elif name == 'sqlite-earliest-time-unreadable':

            def bounds(sampler, names):
                from sdm_relational_engine.rfm.backend.sqlite import dialect

                tables = [
                    (
                        sampler.source_name_dict[table],
                        sampler.table_column_ref_dict[table][
                            sampler.time_column_dict[table]
                        ],
                    )
                    for table in names
                ]
                found = {
                    names[int(index)]: (
                        pd.Timestamp.max
                        if minimum is None
                        else pd.Timestamp(minimum),
                        pd.Timestamp.min
                        if maximum is None
                        else pd.Timestamp(maximum),
                    )
                    for index, minimum, maximum in sampler._rows(
                        dialect.time_ranges(tables)
                    )
                }
                for table, (minimum, maximum) in list(found.items()):
                    if not pd.isna(minimum):
                        continue
                    assert table == 'purchases'
                    assert sampler.time_column_dict[table] == 'purchased_at'
                    reference = sampler.table_column_ref_dict[table][
                        'purchased_at'
                    ]
                    source = sampler.source_name_dict[table]
                    rows = sampler._rows(
                        (
                            f'SELECT DISTINCT {reference} AS __kumo_time__ FROM {source} '
                            f'WHERE {reference} IS NOT NULL ORDER BY 1 ASC LIMIT 1000 OFFSET 0',
                            None,
                        )
                    )
                    assert len(rows) < 1000
                    parsed = [pd.Timestamp(row[0]) for row in rows]
                    readable = [value for value in parsed if not pd.isna(value)]
                    assert readable and pd.isna(parsed[0])
                    found[table] = (min(readable), maximum)
                return found

            patch.setattr(SQLiteSampler, '_get_min_max_time_dict', bounds)
        else:
            assert _fixed(case, FIXED_IN_THE_RUNS) == NULL_KEY
            original = sql_sampler.SQLSampler._sample_subgraph
            source = textwrap.dedent(inspect.getsource(original))
            assert hashlib.sha256(source.encode()).hexdigest() == (
                '857b4135df39b4f33fc359ba98c0a6227e4d0e4930747d9b6ea4634769455dfc'
            )
            tree = ast.parse(source)
            needle = ast.dump(ast.parse('_mask = _mask[code]').body[0])
            matches = [
                node for node in ast.walk(tree) if ast.dump(node) == needle
            ]
            assert len(matches) == 1
            matches[0].value = ast.parse(
                '(code >= 0) & _mask[code]', mode='eval'
            ).body
            ast.fix_missing_locations(tree)
            namespace = dict(original.__globals__)
            exec(
                compile(
                    tree,
                    '<corrected-null-foreign-key-reference>',
                    'exec',
                    dont_inherit=True,
                ),
                namespace,
            )
            corrected = namespace[original.__name__]
            assert inspect.signature(corrected) == inspect.signature(original)
            patch.setattr(sql_sampler.SQLSampler, '_sample_subgraph', corrected)
        yield


@contextlib.contextmanager
def _current_corrected_reference(case):
    name = case['id'].rsplit('/', 1)[-1]
    if name not in ('duckdb-a-time-of-day', 'postgres-a-time-of-day'):
        with _corrected_reference(case):
            yield
        return
    import _fixture_ops
    from predictive_query.schema.typing import Dtype, Stype
    from sdm_client.core.warnings import SDMWarning
    from sdm_client.materializers.relational import RelationalMaterializer

    original_schema = _fixture_ops._driver_schema
    original_graph = RelationalMaterializer._build_graph

    @contextlib.contextmanager
    def schema(arguments):
        with contextlib.ExitStack() as stack:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                built = stack.enter_context(original_schema(arguments))
            expected_parse = (
                "Could not parse 8 of 8 value(s) of time column 'opens' "
                "(for example [datetime.time(10, 0)]); they were dropped as 'NaT'"
            )
            expected_choice = (
                'Some time columns were inferred, or refused, on a guess about what a column holds:\n'
                '- items.opens was not taken as the time column because no row holds a value. '
                "If it does say when a row became true, name it: time_columns={'items': 'opens'}."
            )
            removed_parse = 0
            removed_choice = 0
            for warning in caught:
                if (
                    warning.category is UserWarning
                    and str(warning.message) == expected_parse
                ):
                    removed_parse += 1
                elif (
                    warning.category is SDMWarning
                    and str(warning.message) == expected_choice
                ):
                    removed_choice += 1
                else:
                    warnings.warn(
                        warning.message, warning.category, stacklevel=2
                    )
            assert removed_parse == 1
            assert removed_choice == 1
            yield built

    def graph(materializer):
        built = original_graph(materializer)
        items = built.tables['items']
        assert items.time_column is None
        items.time_column = None
        items['opens']._dtype = Dtype.string
        items['opens'].stype = Stype.categorical
        return built

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fixture_ops, '_driver_schema', schema)
        patch.setattr(RelationalMaterializer, '_build_graph', graph)
        yield


def _runs_what_the_engine_runs(case: dict[str, Any]) -> None:
    from sdm_relational_engine.rfm.backend.duckdb import DuckDBSampler
    from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler
    from sdm_relational_engine.rfm.rfm import KumoRelational

    def model(method: str) -> tuple[Boundary, Any]:
        return (
            Boundary(KumoRelational, method, lambda m: m._sampler),
            _model_describer(method),
        )

    def sampler(owner: type, method: str) -> tuple[Boundary, Any]:
        return Boundary(owner, method, lambda s: s), _missing_indices(method)

    watched = [
        (
            'task_table',
            *model('_get_task_table'),
            task_table_want,
            task_table_got,
        ),
        ('context', *model('_get_context'), context_want, context_got),
        (
            'missing_indices',
            *sampler(DuckDBSampler, '_get_missing_index_dict'),
            duckdb_missing_want,
            duckdb_missing_got,
        ),
        (
            'missing_indices',
            *sampler(SQLiteSampler, '_warn_missing_indices'),
            sqlite_missing_want,
            sqlite_missing_got,
        ),
    ]
    with contextlib.ExitStack() as stack:
        for _, boundary, describe, _, _ in watched:
            stack.enter_context(boundary.installed(describe))
        got = _outcome(case)
    assert got[0] == _recorded(case)[0], got
    if _fixed(case, FIXED_IN_THE_RUNS):
        assert same(got[1], _recorded(case)[1]), (got, _recorded(case))
        backend = case['input']['backend']
        assert [len(boundary.calls) for _, boundary, _, _, _ in watched] == [
            1,
            1,
            int(backend == 'duckdb'),
            int(backend == 'sqlite'),
        ]
    total = 0
    problems: list[str] = []
    for name, boundary, _, want, given in watched:
        count, found = _compare(name, boundary, want, given)
        if _fixed(case, FIXED_IN_THE_RUNS):
            assert count == len(boundary.calls), (
                f'{name}: native comparison skipped'
            )
        total += count
        problems += [f'{name}: {problem}' for problem in found]
    if total == 0 and _refused_as_reserved(case):
        total, problems = _reserved_by_the_core(case)
    assert total >= 1, 'no engine method was called'
    assert not problems, '\n'.join(problems[:20])


def _refused_as_reserved(case: dict[str, Any]) -> bool:
    error = case.get('expected_error', {})
    return 'reserved by the SQL backends' in error.get('message', '')


def _reserved_by_the_core(case: dict[str, Any]) -> tuple[int, list[str]]:
    r"""A SQL sampler refuses a column named like its own while it is made,
    before any method is called that the core's runs stand for. So the case
    runs again with that check skipped, the task table's call is described,
    and the core's run of it must raise the error today's sampler raised."""
    from sdm_relational_engine.rfm.base import sql_sampler
    from sdm_relational_engine.rfm.rfm import KumoRelational

    boundary = Boundary(KumoRelational, '_get_task_table', lambda m: m._sampler)
    check = vars(sql_sampler)['_reject_reserved_columns']
    sql_sampler._reject_reserved_columns = lambda graph: None
    try:
        with boundary.installed(_model_describer('_get_task_table')):
            _outcome(case)
    finally:
        sql_sampler._reject_reserved_columns = check
    for call in boundary.calls:
        call['result'] = None
        call['error'] = ValueError(case['expected_error']['message'])
    return _compare('task_table', boundary, task_table_want, task_table_got)


def test_an_unseeded_context_on_a_fresh_model() -> None:
    r"""With ``random_seed=None`` the neighbor sampler is not seeded again,
    and draws on from where it was built (seed 123 on both sides), so the
    first context a fresh in-memory model samples is the same in the core.
    Items and their reviews draw neighbors (32 of each item's reviews), and
    the entity draws before them, from the system's entropy, are taken from
    today's call. Later calls on the same model are not compared here: both
    generators carry on from the calls before, and relationallib's state
    cannot be read to hand the core's run (the sampling crate's
    ``a_sampler_built_from_its_state_draws_on_where_it_left_off`` holds the
    core's side)."""
    from sdm_client import PredictiveQuery
    from sdm_relational_engine.rfm.rfm import KumoRelational

    boundary = Boundary(
        KumoRelational, '_get_context', lambda model: model._sampler
    )
    timeless = {
        'backend': 'memory',
        'data': {'generator': 'marketplace', 'seed': 0},
        'tables': ['items', 'reviews'],
    }
    with (
        boundary.installed(_model_describer('_get_context')),
        _driver_schema(timeless) as schema,
    ):
        PredictiveQuery(
            'PREDICT items.category FOR EACH items.item_id', schema
        ).materialize(indices=[1, 2], num_context_samples=30, random_seed=None)
    assert len(boundary.calls) == 1, len(boundary.calls)
    inputs = json.loads(boundary.calls[0]['described'][0])
    assert inputs['arguments']['random_seed'] is None
    count, problems = _compare('context', boundary, context_want, context_got)
    assert count == 1
    assert not problems, '\n'.join(problems)
