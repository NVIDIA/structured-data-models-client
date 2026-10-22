# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recording mode's own checks (T3.2; Phase 3 design choice 2): the
warehouse drivers' captures, the placeholders a capture is written with,
the draws read from its statements, ``capture_check.py captures`` holding
a capture complete, and ``captures-only`` holding a captures commit to the
final candidate. The acceptance tests are the test author's; these are the
implementer's own.

    python -m pytest contract/tests/test_t32_recording.py -q
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract' / 'runners'
REFERENCE = ROOT / 'contract' / 'reference'


def _load(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run = _load('run_python')
check = _load('capture_check')
pa = pytest.importorskip('pyarrow')
pytest.importorskip('duckdb')


# --- fake warehouse drivers --------------------------------------------------------


class FakeCursor:
    r"""A Databricks SQL cursor answering from a script of tables."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self._answers = answers
        self._table: Any = None
        self.description: Any = None

    def execute(self, operation: str, parameters: Any = None) -> None:
        self._table = self._answers[operation.split('\nLIMIT')[0]]
        self.description = [
            (field.name, str(field.type), None, None, None, None, None)
            for field in self._table.schema
        ]

    def fetchall_arrow(self) -> Any:
        return self._table

    def fetchall(self) -> list[tuple[Any, ...]]:
        return [tuple(row.values()) for row in self._table.to_pylist()]

    def fetchone(self) -> tuple[Any, ...] | None:
        rows = self.fetchall()
        return rows[0] if rows else None

    def close(self) -> None:
        pass

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class FakeConnection:
    def __init__(self, answers: dict[str, Any]) -> None:
        self._answers = answers

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._answers)


FakeCursor.__module__ = 'databricks.sql.client'
FakeConnection.__module__ = 'databricks.sql.client'

SAMPLE = (
    'SELECT `user_id`\nFROM `{c}`.`{s}`.`users`\n'
    'TABLESAMPLE (12.5 PERCENT) REPEATABLE (7)'
)
VIEW_READ = (
    'SELECT `price` FROM (SELECT * FROM `{c}`.`{s}`.`orders` AS src '
    'WHERE (src.price > 5))'
)
NAMES = {'catalog': 'privcat_x', 'schema': 'privschema_y'}


def _sql(text: str) -> str:
    return text.replace('{c}', NAMES['catalog']).replace('{s}', NAMES['schema'])


def _answers() -> dict[str, Any]:
    return {
        _sql(SAMPLE): pa.table({'user_id': pa.array([1, 3], pa.int64())}),
        _sql(VIEW_READ): pa.table({'price': [9.5]}),
        'SELECT current_catalog()': pa.table({'c': [NAMES['catalog']]}),
    }


def _drive(log: Any) -> None:
    r"""What an old client does on the captured connection: a catalog read
    as rows, a random draw as Arrow, a view-derived read as Arrow."""
    connection = run._CapturedDatabricks(FakeConnection(_answers()), log)
    log.stage = 'from_source'
    with connection.cursor() as cursor:
        cursor.execute('SELECT current_catalog()')
        cursor.fetchall()
    log.stage = 'materialize'
    with connection.cursor() as cursor:
        cursor.execute(_sql(SAMPLE))
        cursor.fetchall_arrow()
    with connection.cursor() as cursor:
        cursor.execute(_sql(VIEW_READ) + '\nLIMIT 1000')
        cursor.fetchall_arrow()


class FakeLive:
    leg = 'databricks-sql'
    zone = 'Etc/UTC'
    names = NAMES
    private = ['privcat_x', 'privschema_y', 'warehouse-host.invalid']

    def __init__(self, tables: list[str]) -> None:
        self.tables = [(name, pa.table({'x': [1]})) for name in tables]


def _reference_line(key: str) -> dict[str, Any]:
    path = next(REFERENCE.glob('*.jsonl'))
    with path.open(encoding='utf-8') as handle:
        for text in handle:
            if '"case": "base.binary@postgres"' in text:
                return json.loads(text)
    raise AssertionError('no base.binary@postgres reference line')


@functools.cache
def _base_binary() -> dict[str, Any]:
    ((case, _),) = run.load_cases(
        [ROOT / 'contract' / 'cases' / 's1' / 'base.binary.json']
    )
    return case


def _case() -> dict[str, Any]:
    return copy.deepcopy(_base_binary())


def _capture() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    case, line, tables = _built_capture()
    return copy.deepcopy(case), copy.deepcopy(line), list(tables)


@functools.cache
def _built_capture() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    r"""A complete capture of base.binary on databricks-sql from the fake
    driver, as the recording mode builds it, written and read back."""
    case = _case()
    bases = run.load_bases()
    tables = check._duck_tables(case, bases)
    log = run.DriverLog()
    log.recorder = run.Recorder()
    _drive(log)
    observation = _reference_line('base.binary@postgres')
    observation['case'] = 'base.binary@databricks-sql'
    observation['driver']['capture'] = log.record()
    live = FakeLive(tables)
    relations = [
        {
            'statement': 2,
            'sql': _sql(VIEW_READ),
            'table': pa.table({'price': [9.5, 12.0]}),
        }
    ]
    line = run.capture_line(live, case, bases, log, observation, relations)
    line['versions']['databricks-sql-connector'] = '4.0.0'
    text = run.written(line, run.Placeheld(live.names, live.private))
    return case, json.loads(text), tables


# --- the warehouse captures --------------------------------------------------------


def test_the_databricks_capture_records_every_call_and_keeps_each_result() -> (
    None
):
    log = run.DriverLog()
    log.recorder = run.Recorder()
    _drive(log)
    assert [s['stage'] for s in log.statements] == [
        'from_source',
        'materialize',
        'materialize',
    ]
    assert [r['method'] for s in log.statements for r in s['results']] == [
        'fetchall',
        'fetchall_arrow',
        'fetchall_arrow',
    ]
    assert sorted(log.recorder.kept) == [(0, 0), (1, 0), (2, 0)]
    assert log.incomplete == []


def test_a_read_the_driver_lacks_is_not_offered() -> None:
    r"""The reference's reader picks the first Arrow read the cursor has
    (reader.py's ``hasattr`` order): the capture offers only the driver's."""

    class Bare:
        description = None

        def execute(self, operation: str) -> None:
            pass

    Bare.__module__ = 'databricks.sql.client'
    captured = run._CapturedDatabricks(Bare(), run.DriverLog())
    assert not hasattr(captured, 'fetchall_arrow')
    assert not hasattr(captured, 'fetch_arrow_all')


def test_the_spark_session_passes_the_reference_dispatch() -> None:
    class Frame:
        columns = ['a']

        def toArrow(self) -> Any:  # noqa: N802 - Spark's name
            return pa.table({'a': [1]})

    class Session:
        version = '4.0.0'

        def sql(self, query: str, args: Any = None) -> Frame:
            return Frame()

    log = run.DriverLog()
    log.recorder = run.Recorder()
    captured = run._CapturedSpark(Session(), log)
    # sdm_connectors.source._is_spark_session reads the type's MRO.
    assert any(
        cls.__name__ == 'SparkSession' and cls.__module__.startswith('pyspark.')
        for cls in type(captured).__mro__
    )
    assert captured.version == '4.0.0'
    assert captured.sql('SELECT ?', args=[1]).toArrow().num_rows == 1
    assert log.statements[0]['parameters'][0]['type'] == 'builtins.int'
    assert log.recorder.kept[(0, 0)].num_rows == 1


def test_psycopg_copy_stream_and_executemany_are_recorded() -> None:
    r"""P2-T2.1a-2: a psycopg statement run by copy, stream or executemany
    is recorded, never passed through unseen."""

    class Copy:
        def __init__(self) -> None:
            self._blocks = [b'a', b'b']

        def __enter__(self) -> Copy:
            return self

        def __exit__(self, *exc: Any) -> None:
            pass

        def read(self) -> bytes:
            return self._blocks.pop(0) if self._blocks else b''

    class Connection:
        description = [('a', 23, None, None, None, None, None)]

        def copy(self, statement: str, params: Any = None) -> Copy:
            return Copy()

        def stream(self, query: str, params: Any = None) -> Any:
            yield (1,)
            yield (2,)

        def executemany(self, query: str, rows: Any) -> None:
            assert list(rows) == [(1,), (2,)]

    Connection.__module__ = 'psycopg.connection'
    log = run.DriverLog()
    captured = run._CapturedPsycopg(Connection(), log)
    with captured.copy('COPY t TO STDOUT') as copy_:
        while copy_.read():
            pass
    assert list(captured.stream('SELECT a FROM t')) == [(1,), (2,)]
    captured.executemany('INSERT INTO t VALUES (%s)', iter([(1,), (2,)]))
    assert [s['sql'] for s in log.statements] == [
        'COPY t TO STDOUT',
        'SELECT a FROM t',
        'INSERT INTO t VALUES (%s)',
    ]
    assert [r['rows'] for r in log.statements[0]['results']] == [2]
    assert [r['rows'] for r in log.statements[1]['results']] == [2]
    assert run._CAPTURES['postgres'] is run._CapturedPsycopg


# --- draws and placeholders ------------------------------------------------------------


@pytest.mark.parametrize(
    ('sql', 'draw'),
    [
        (
            'SELECT a\nFROM `c`.`s`.`t`\nTABLESAMPLE (12.5 PERCENT) REPEATABLE (42)',
            ('tablesample', '`c`.`s`.`t`', '12.5', '42'),
        ),
        (
            'SELECT a\nFROM t\nTABLESAMPLE (1.5e-05 PERCENT) REPEATABLE (4294967296)',
            ('tablesample', 't', '1.5e-05', '4294967296'),
        ),
        (
            'SELECT a\nFROM t\nTABLESAMPLE (100.0 PERCENT)',
            ('tablesample', 't', '100.0', None),
        ),
        (
            'SELECT a\nFROM "D"."S"."T"\nSAMPLE ROW (5 ROWS)',
            ('sample_rows', '"D"."S"."T"', '5', None),
        ),
    ],
)
def test_a_draw_is_read_as_written(sql: str, draw: tuple[Any, ...]) -> None:
    (found,) = run.draws_of(sql)
    assert (
        found['kind'],
        found['table'],
        found['fraction'],
        found['seed'],
    ) == draw


def test_names_are_placeheld_in_text_and_tables_and_others_refuse_the_write() -> (
    None
):
    held = run.Placeheld(NAMES, [*NAMES.values(), 'warehouse-host.invalid'])
    assert (
        held.text('`PRIVCAT_X`.`privschema_y`.t') == '`{catalog}`.`{schema}`.t'
    )
    table = pa.Table.from_arrays(
        [pa.array(['privcat_x', 'a']), pa.array([1, 2])],
        names=['privcat_x', 'privcat_x'],
    )
    placed = held.table(table)
    assert placed.schema.names == ['{catalog}', '{catalog}']
    assert placed.column(0).to_pylist() == ['{catalog}', 'a']
    assert held.leaked('connected to warehouse-host.invalid')
    with pytest.raises(run.RunnerFailure, match='holds a private name'):
        run.written(
            {'case': 'x@databricks-sql', 'host': 'warehouse-host.invalid'}, held
        )
    with pytest.raises(run.RunnerFailure, match='a table of its capture'):
        run.written(
            {
                'case': 'x@databricks-sql',
                't': pa.table({'h': ['warehouse-host.invalid']}),
            },
            held,
        )


def test_an_unknown_placeholder_fails_the_run_by_name() -> None:
    case = {'id': 'x', 'input': {'from_source': {'tables': {'t': '{other}.t'}}}}
    with pytest.raises(run.RunnerFailure, match=r'\{other\}'):
        run.resolved(case, NAMES)
    case['input']['from_source']['tables']['t'] = '{catalog}.{schema}.t'
    assert run.resolved(case, NAMES)['input']['from_source']['tables']['t'] == (
        'privcat_x.privschema_y.t'
    )


def test_the_live_configuration_reads_the_environment_and_marks_it_private(
    tmp_path: Path,
) -> None:
    where = tmp_path / 'live.json'
    where.write_text(
        json.dumps(
            {
                'format': 'sdm-live/1',
                'private': ['a-workspace-host'],
                'legs': {
                    'snowflake': {
                        'names': {'catalog': 'privdb_z', 'schema': 'sch_test'},
                        'driver': {'account': {'env': 'SDM_TEST_ACCOUNT'}},
                    }
                },
            }
        )
    )
    environ = {'SDM_LIVE_CONFIG': str(where), 'SDM_TEST_ACCOUNT': 'acct-1234'}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('SDM_TEST_ACCOUNT', 'acct-1234')
        spec, names, private = run.live_configuration('snowflake', environ)
    assert spec['driver'] == {'account': 'acct-1234'}
    assert names == {'catalog': 'privdb_z', 'schema': 'sch_test'}
    assert {'acct-1234', 'a-workspace-host', 'privdb_z', 'sch_test'} <= set(
        private
    )


# --- capture_check.py captures -----------------------------------------------------------


def test_a_complete_capture_passes_and_holds_no_private_name() -> None:
    case, line, tables = _capture()
    assert check.capture_problems(line, case, tables) == []
    text = json.dumps(line)
    assert 'privcat_x' not in text and '{catalog}' in text
    assert line['draws'] == [
        {
            'statement': 1,
            'kind': 'tablesample',
            'table': '`{catalog}`.`{schema}`.`users`',
            'fraction': '12.5',
            'seed': '7',
        }
    ]


@pytest.mark.parametrize(
    ('fault', 'named'),
    [
        (lambda line: line['draws'].clear(), 'its draws are not'),
        (
            lambda line: line['draws'][0].update(fraction='12.50'),
            'its draws are not',
        ),
        (lambda line: line['draws'][0].update(seed=None), 'its draws are not'),
        (lambda line: line['tables'].pop(), 'its tables are'),
        (lambda line: line['statements'].pop(), 'its statements are not'),
        (
            lambda line: line['statements'][1]['results'].clear(),
            'no result read',
        ),
        (
            lambda line: line['statements'][1]['results'][0].update(arrow=None),
            'no Arrow',
        ),
        (
            lambda line: line['statements'][1]['results'][0].update(arrow='x'),
            'not an Arrow',
        ),
        (
            lambda line: line['statements'][1]['results'][0].update(rows=9),
            'rows held',
        ),
        (lambda line: line['relations'].clear(), 'did not evaluate'),
        (lambda line: line.update(zone=''), 'no session zone'),
        (
            lambda line: line['versions'].pop('databricks-sql-connector'),
            "'s driver recorded",
        ),
        (lambda line: line['recording'].update(commit=None), 'no commit'),
        (
            lambda line: line.update(incomplete=['fetchmany_arrow']),
            'could not record',
        ),
        (lambda line: line.update(format='sdm-capture/9'), 'not sdm-capture/1'),
        (lambda line: line.pop('observation'), 'no reference line'),
    ],
)
def test_an_incomplete_capture_fails_by_name(fault: Any, named: str) -> None:
    case, line, tables = _capture()
    fault(line)
    problems = check.capture_problems(line, case, tables)
    assert any(named in problem for problem in problems), problems


def test_a_case_changed_after_its_capture_is_pending_live() -> None:
    case, line, _ = _capture()
    problems, pending = check.captures([case], [line], ['databricks-sql'])
    assert problems == [] and pending == []
    changed = copy.deepcopy(case)
    changed['input']['materialize']['indices'] = [0]
    problems, pending = check.captures([changed], [line], ['databricks-sql'])
    assert pending == [
        'base.binary@databricks-sql: its input changed after its capture'
    ]
    problems, pending = check.captures([case], [], ['databricks-sql'])
    assert pending == ['base.binary@databricks-sql: no capture']
    problems, _ = check.captures([case], [line, line], ['databricks-sql'])
    assert problems == ['base.binary@databricks-sql: captured twice']


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env={
            'GIT_AUTHOR_NAME': 't',
            'GIT_AUTHOR_EMAIL': 't@example.invalid',
            'GIT_COMMITTER_NAME': 't',
            'GIT_COMMITTER_EMAIL': 't@example.invalid',
            'GIT_CONFIG_GLOBAL': '/dev/null',
            'GIT_CONFIG_NOSYSTEM': '1',
            'PATH': '/usr/bin:/bin',
        },
    ).stdout.strip()


def test_a_captures_only_commit_must_name_the_final_candidate(
    tmp_path: Path,
) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    (repo / 'README').write_text('x\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'final')
    final = _git(repo, 'rev-parse', 'HEAD')
    captures = repo / 'contract' / 'captures'
    captures.mkdir(parents=True)
    (captures / 'e1.jsonl').write_text(
        json.dumps({'recording': {'commit': final}}) + '\n'
    )
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'captures')
    assert check.captures_only(repo, final) == []
    (captures / 'e1.jsonl').write_text(
        json.dumps({'recording': {'commit': '0' * 40}}) + '\n'
    )
    (repo / 'README').write_text('y\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'more')
    problems = check.captures_only(repo, final)
    assert any('README: changed outside' in p for p in problems)
    assert any('recorded from 0000' in p for p in problems)
    assert check.captures_only(repo, 'HEAD') == [
        'no commit after the final candidate'
    ]


# --- the case format ------------------------------------------------------------------------


def _s2(**changes: Any) -> dict[str, Any]:
    case = _case()
    case.update(id='t32.s2', slice='S2', warehouses=['databricks-sql'])
    for key, value in changes.items():
        if value is None:
            case.pop(key, None)
        else:
            case[key] = value
    return case


def _problem(case: dict[str, Any]) -> str | None:
    return run._invalid(run.validator('case'), case) or run.warehouse_problem(
        case
    )


@pytest.mark.parametrize(
    ('change', 'refused'),
    [
        ({}, False),
        ({'slice': 'S1'}, True),
        ({'warehouses': None}, True),
        ({'warehouses': []}, True),
        ({'warehouses': ['databricks-sql', 'databricks-sql']}, True),
        ({'warehouses': ['bigquery']}, True),
        ({'databases': ['duckdb'], 'exclusion': 'KO3-45'}, True),
    ],
)
def test_warehouses_is_read_as_the_format_says(
    change: dict[str, Any], refused: bool
) -> None:
    assert (_problem(_s2(**change)) is not None) is refused


@pytest.mark.parametrize(
    ('given', 'legs', 'refused'),
    [
        (
            {
                'view': {
                    'kind': 'metric_view',
                    'name': 'v',
                    'yaml': 'version: 0.1',
                }
            },
            ['databricks-sql', 'databricks-spark'],
            False,
        ),
        (
            {'view': {'kind': 'metric_view', 'name': '{shop_metric_view}'}},
            ['databricks-sql'],
            False,
        ),
        ({'view': {'kind': 'metric_view', 'name': 'v'}}, ['snowflake'], True),
        (
            {
                'view': {
                    'kind': 'semantic_view',
                    'name': 'v',
                    'trust_expressions': True,
                }
            },
            ['snowflake'],
            False,
        ),
        (
            {
                'view': {
                    'kind': 'semantic_view',
                    'name': 'v',
                    'trust_expressions': 1,
                }
            },
            ['snowflake'],
            True,
        ),
        ({'view': {'kind': 'cube', 'name': 'v'}}, ['snowflake'], True),
        ({'view': {'kind': 'metric_view'}}, ['databricks-sql'], True),
        ({'connect': 'warehouse'}, ['databricks-sql'], False),
        ({'connect': 'spark'}, ['databricks-spark'], False),
        ({'connect': 'snowpark'}, ['snowflake'], False),
        ({'connect': 'lakebase'}, ['lakebase'], False),
        (
            {'connect': 'warehouse'},
            ['databricks-sql', 'databricks-spark'],
            True,
        ),
        ({'connect': 'jdbc'}, ['databricks-sql'], True),
    ],
)
def test_view_and_connect_are_read_as_the_format_says(
    given: dict[str, Any], legs: list[str], refused: bool
) -> None:
    case = _s2(warehouses=legs)
    case['input'].update(given)
    assert (_problem(case) is not None) is refused


def test_view_and_connect_need_warehouses() -> None:
    case = _case()
    case['input']['connect'] = 'warehouse'
    assert _problem(case) is not None


def test_s2_cases_run_only_on_their_warehouses_and_s1_on_every_leg() -> None:
    assert run.case_legs(_s2(warehouses=['snowflake', 'databricks-sql'])) == (
        'databricks-sql',
        'snowflake',
    )
    assert run.case_legs(_case()) == run.LEGS
    batching = _case()
    batching['id'] = 'multicall.entities.100000'
    assert run.case_legs(batching) == run.DATABASES


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity'])
def test_a_non_json_constant_is_refused(tmp_path: Path, constant: str) -> None:
    r"""P2-T2.1i-1: NaN and Infinity are not JSON; both loaders refuse them."""
    case = _case()
    case['input']['materialize']['rows_per_call'] = 7
    text = json.dumps(case).replace(
        '"rows_per_call": 7', f'"rows_per_call": {constant}'
    )
    (tmp_path / 'case.json').write_text(text)
    with pytest.raises(run.RunnerFailure, match='not JSON'):
        run.load_cases([tmp_path / 'case.json'])


def test_a_negative_zero_call_is_call_zero(tmp_path: Path) -> None:
    case = _case()
    case['input']['fail_call'] = 7
    (tmp_path / 'case.json').write_text(
        json.dumps(case).replace('"fail_call": 7', '"fail_call": -0')
    )
    ((loaded, _),) = run.load_cases([tmp_path / 'case.json'])
    assert loaded['input']['fail_call'] == 0


def test_deep_and_wide_capture_lines_are_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    r"""Outside JSON (implementer.md's probes): thousands of nested levels
    are refused by name, a very wide line is read, neither crashes."""
    deep = tmp_path / 'deep.jsonl'
    cases = ROOT / 'contract' / 'cases' / 's1' / 'base.binary.json'
    for depth, named in (
        (5_000, 'a capture line is a JSON list'),
        (200_000, 'unreadable'),
    ):
        deep.write_text('[' * depth + ']' * depth + '\n')
        assert (
            check.main(
                ['captures', '--cases', str(cases), '--capture', str(deep)]
            )
            == 1
        )
        assert named in capsys.readouterr().out
    wide = tmp_path / 'wide.jsonl'
    wide.write_text(json.dumps({f'k{i}': i for i in range(200_000)}) + '\n')
    assert (
        check.main(['captures', '--cases', str(cases), '--capture', str(wide)])
        == 1
    )
    assert (
        'FAILS None: no case of these runs on that leg'
        in capsys.readouterr().out
    )


def test_a_deep_reference_line_is_refused_by_name() -> None:
    case, line, _ = _capture()
    nested: Any = []
    for _ in range(50_000):
        nested = [nested]
    line['observation']['public']['schema'] = nested
    problems, _ = check.captures([case], [line], ['databricks-sql'])
    assert problems and all('base.binary@databricks-sql' in p for p in problems)


def test_a_stale_installed_client_is_refused(tmp_path: Path) -> None:
    r"""P2-MID2-7: python-core refuses a client whose contract version is
    not the checkout's."""
    source = tmp_path / 'lib.rs'
    source.write_text('pub const CONTRACT_VERSION: u32 = 28;\n')
    assert run.stale_client(28, source) is None
    assert 'a stale wheel' in (run.stale_client(27, source) or '')
    assert 'a stale wheel' in (run.stale_client(None, source) or '')
    assert run.stale_client(27, tmp_path / 'none.rs') is None
    # The checkout's own declaration is read: no contract is version 0.
    assert run.stale_client(0) is not None
