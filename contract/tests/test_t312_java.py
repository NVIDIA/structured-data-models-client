# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.12's acceptance tests of the Java runner's S1 legs
(``contract/runners/run_java.py``; the brief's exits 1, 3 and 4).

- Which cases run on a Java leg: an S1 case on DuckDB, SQLite and Postgres
  whose paths are all Java's (no ranking, no forecast: J2), but one whose
  inputs Java's types cannot say, which is registered; S0 on DuckDB's leg
  alone.
- What the runner records: the core section replayed into the Python
  runner's own capture, the driver's statements with DuckDB's query log
  counting a key set kept as a temporary table once, and a model call the
  endpoint received as other bytes than the core wrote failing the runner
  (seeded).
- Every S1 case on a few of Java's paths, on the Java host, equal to the
  reference on each database (JAR-gated: ``SDM_JAVA_JAR``; Postgres where
  ``SDM_POSTGRES_TEST_DSN`` names a server).
- No empty S0 or S1 Java path (``coverage.py``'s Java rule, exit 3).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'contract/runners'))
import run_java  # noqa: E402
import run_python  # noqa: E402

CASES = ROOT / 'contract/cases'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _case(name: str) -> dict[str, Any]:
    for path in (CASES / 's1').glob('*.json'):
        case = json.loads(path.read_text(encoding='utf-8'))
        if case['id'] == name:
            return case
    raise AssertionError(name)


def test_t312_an_s1_case_on_java_s_paths_has_a_leg_on_each_of_its_databases() -> (
    None
):
    case = _case('base.binary')
    for database in ('duckdb', 'sqlite', 'postgres'):
        assert run_java.on_java(case, database), database
    assert not run_java.on_java(case, 'databricks-sql')


def test_t312_a_ranking_or_a_forecast_has_no_java_leg() -> None:
    for name in ('rank.top-k', 'forecast.lag-1'):
        case = _case(name)
        assert not any(
            run_java.on_java(case, database)
            for database in ('duckdb', 'sqlite', 'postgres')
        ), name


def test_t312_s0_runs_on_duckdb_s_leg_alone() -> None:
    case = json.loads(
        (CASES / 's0/s0.accept.sum-window.json').read_text(encoding='utf-8')
    )
    assert run_java.on_java(case, 'duckdb')
    assert not run_java.on_java(case, 'sqlite')


def test_t312_a_case_java_s_types_cannot_say_has_no_leg_and_is_registered() -> (
    None
):
    import yaml

    register = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    entry = next(e for e in register['entries'] if e['id'] == 'JV-1')
    assert sorted(entry['cases']) == sorted(run_java.UNSAID)
    for name in run_java.UNSAID:
        assert not run_java.on_java(_case(name), 'duckdb'), name


def test_t312_a_java_row_on_a_duckdb_line_moves_from_no_outcome() -> None:
    # Claude T3.12 r1 P2-2: no baseline holds a Java line, so a row on a
    # DuckDB line (``<case>``, which only a baseline would hold) moves from
    # ``none``, the ratchet's old outcome for it; a SQLite or Postgres
    # line's moves from python-ref's line, as every core runner's does.
    import yaml

    assert not list((ROOT / 'contract/baselines').glob('*/java.jsonl'))
    register = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    wrong = [
        f'{entry["id"]}: {item["case"]} old {item["old"]}'
        for entry in register['entries']
        for item in entry.get('case_outcomes') or []
        if item.get('runner') == 'java'
        and '@' not in item['case']
        and item.get('old') != 'none'
    ]
    assert not wrong, wrong


def test_t312_duckdb_s_log_counts_a_key_set_kept_as_a_table_once() -> None:
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    table = pa.table({'k': pa.array([1, 2], pa.int64())})
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    ipc = base64.b64encode(sink.getvalue().to_pybytes()).decode()
    keep = 'CREATE TEMPORARY TABLE "keys" AS SELECT * FROM "__sdm_keys_1"'
    events = [
        {'event': 'cursor', 'stage': 'materialize'},
        {
            'event': 'bound',
            'stage': 'materialize',
            'handle': 1,
            'name': 'keys',
            'sql': keep,
            'arrow': ipc,
        },
        {
            'event': 'statement',
            'stage': 'materialize',
            'handle': 1,
            'sql': 'SELECT 1',
        },
        {
            'event': 'answered',
            'stage': 'materialize',
            'arrow': ipc,
            'columns': [{'type': 'BIGINT', 'precision': 0, 'scale': 0}],
        },
    ]
    record = run_java.driver_record(events, ['', keep, 'SELECT 1'], 'duckdb')
    assert record['query_log'] == {
        'statements': 1,
        'bindings': 1,
        'compared': 1,
        'differing': [],
    }
    assert len(record['capture']['bindings']) == 1
    assert (
        record['capture']['statements'][0]['results'][0]['method']
        == 'query_arrow'
    )
    assert record['capture']['statements'][0]['results'][0]['values'][
        'types'
    ] == ['BIGINT']


def test_t312_postgres_columns_are_described_as_psycopg_displays_them() -> None:
    import pyarrow as pa

    table = pa.table({'a': [1], 'b': ['x'], 'c': [1], 'd': [[1]]})
    columns = [
        {'type': 'int4', 'precision': 10, 'scale': 0},
        {'type': 'varchar', 'precision': 3, 'scale': 0},
        {'type': 'numeric', 'precision': 10, 'scale': 2},
        {'type': '_int4', 'precision': 10, 'scale': 0},
    ]
    assert run_java.described(table, columns, 'postgres') == [
        'int4',
        'varchar(3)',
        'numeric(10,2)',
        'int4[]',
    ]
    assert run_java.described(table, columns, 'sqlite') == [
        'int64',
        'string',
        'int64',
        'list<item: int64>',
    ]


def test_t312_a_call_the_endpoint_received_otherwise_fails_the_runner() -> None:
    case = _case('base.binary')
    body = b'{"model": "kumo-relational"}'
    events = [
        {
            'event': 'stepped',
            'stage': 'predict',
            'step': {
                'message': 'step',
                'step': 'call_model',
                'id': 1,
                'body': 4,
            },
        },
        {
            'event': 'taken',
            'stage': 'predict',
            'id': 4,
            'bytes': base64.b64encode(body).decode(),
        },
    ]
    assert run_java.calls_sent(events) == [body]
    run_java.check_sent(
        case,
        events,
        SimpleNamespace(requests=[('POST', '/v1/predictions', body)]),
    )
    # Seeded: the endpoint received a body the core did not write.
    other = SimpleNamespace(
        requests=[('POST', '/v1/predictions', body.replace(b' ', b''))]
    )
    with pytest.raises(run_java.RunnerFailure, match='other bytes'):
        run_java.check_sent(case, events, other)


# --- exit 1: the Java host's S1 lines against the reference ---------------------------------

#: A few S1 cases, a binary, a multiclass and a regression query, an edit and
#: a request asked in several calls, each with no registered difference.
SAMPLE = (
    'base.binary',
    'base.multiclass',
    'base.regression',
    'edit.accepted-meaning-key-links',
    'multicall.rows-per-call-3',
)


def _jar() -> Path | None:
    given = os.environ.get('SDM_JAVA_JAR')
    if not given:
        assert not os.environ.get('SDM_REQUIRE_JAVA'), (
            'SDM_REQUIRE_JAVA is set but SDM_JAVA_JAR is not'
        )
        return None
    return Path(given)


@pytest.mark.parametrize('database', ['duckdb', 'sqlite', 'postgres'])
def test_t312_s1_cases_on_the_java_host_equal_the_reference(
    tmp_path: Path, database: str
) -> None:
    jar = _jar()
    if jar is None:
        return
    if database == 'postgres' and not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        assert not os.environ.get('SDM_REQUIRE_POSTGRES')
        return
    cases = tmp_path / 'cases'
    cases.mkdir()
    for name in SAMPLE:
        for path in (CASES / 's1').glob('*.json'):
            if json.loads(path.read_text(encoding='utf-8'))['id'] == name:
                (cases / path.name).write_bytes(path.read_bytes())
    out = tmp_path / 'java.jsonl'
    command = [
        sys.executable,
        '-I',
        str(ROOT / 'contract/runners/run_java.py'),
        '--database',
        database,
        '--jar',
        str(jar),
        '--out',
        str(out),
    ]
    native = os.environ.get('SDM_JAVA_NATIVE')
    if native:
        command += ['--native', native]
    done = subprocess.run(
        [*command, str(cases)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-4000:]
    ran = {
        json.loads(line)['case']
        for line in out.read_text(encoding='utf-8').splitlines()
    }
    assert ran == {run_python.case_key(name, database) for name in SAMPLE}
    reference = tmp_path / 'reference.jsonl'
    reference.write_text(
        ''.join(
            line + '\n'
            for line in REFERENCE.read_text(encoding='utf-8').splitlines()
            if json.loads(line)['case'] in ran
        ),
        encoding='utf-8',
    )
    judged = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/rust_core/compare.py'),
            '--cases',
            str(cases),
            '--database',
            database,
            '--reference',
            str(reference),
            '--register',
            str(REGISTER),
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert judged.returncode == 0, judged.stdout[-4000:] + judged.stderr[-2000:]


# --- exit 3: no empty S0 or S1 Java path ---------------------------------------------------


def test_t312_every_s0_and_s1_java_path_is_covered() -> None:
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/rust_core/coverage.py'),
            '--gate',
            'java:S1',
            '--checkpoint',
            'HEAD',
            '--repo',
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    empty = [
        line
        for line in done.stdout.splitlines()
        if 'java: S0 path' in line or 'java: S1 path' in line
    ]
    assert empty == [], '\n'.join(empty)


def _tool(name: str) -> Any:
    import importlib.util

    path = ROOT / f'scripts/rust_core/{name}.py'
    spec = importlib.util.spec_from_file_location(f't312_{name}', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f't312_{name}'] = module
    spec.loader.exec_module(module)
    return module


class _Tree:
    r"""A candidate tree of the given files (``coverage``'s ``root``)."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    def read(self, path: str) -> str:
        return self.files[path]


def test_t312_a_java_path_s_test_is_a_junit_method_the_tree_defines() -> None:
    coverage = _tool('coverage')
    given = 'java/src/test/java/com/nvidia/sdm/T312SchemaTest.java'
    assert (
        coverage._test_problem(
            f'{given}#theShopIsDescribedAsTheReferenceDescribesIt', ROOT, 'java'
        )
        is None
    )
    assert coverage._test_problem(f'{given}#noSuchMethod', ROOT, 'java') == (
        'no test the tree defines'
    )
    assert coverage._test_problem('java/Other.java#x', ROOT, 'java') is not None
    # A parameterized test, its source annotation between, is a test.
    parameterized = 'java/src/test/java/com/nvidia/sdm/T312PredictTest.java'
    assert (
        coverage._test_problem(
            f'{parameterized}#eachKindOfQueryIsMaterializedAndPredicted',
            ROOT,
            'java',
        )
        is None
    )


def test_t312_a_java_method_that_is_no_junit_test_claims_nothing() -> None:
    # Codex T3.12 r1 P1-5: a method with no JUnit annotation is no test.
    coverage = _tool('coverage')
    path = 'java/src/test/java/com/nvidia/sdm/Helpers.java'
    tree = _Tree(
        {
            path: 'class Helpers {\n'
            '  @Test\n'
            '  void aTest() {}\n'
            '\n'
            '  void helper() {}\n'
            '\n'
            '  @Test\n'
            '  @Timeout(5)\n'
            '  public void anotherTest() {}\n'
            '}\n'
        }
    )
    assert coverage._test_problem(f'{path}#aTest', tree, 'java') is None
    assert coverage._test_problem(f'{path}#anotherTest', tree, 'java') is None
    assert coverage._test_problem(f'{path}#helper', tree, 'java') == (
        'no test the tree defines'
    )


def test_t312_a_python_path_takes_no_java_test_and_a_java_path_no_pytest() -> (
    None
):
    # Codex T3.12 r1 P1-5: a Python path claimed by a Java test alone is
    # neither a valid claim (rule 4) nor covered (the S1 rule).
    import copy

    import yaml

    coverage = _tool('coverage')
    java_test = (
        'java/src/test/java/com/nvidia/sdm/T312SchemaTest.java'
        '#theShopIsDescribedAsTheReferenceDescribesIt'
    )
    assert coverage._test_problem(java_test, ROOT, 'python') is not None
    pytest_test = 'contract/tests/test_t312_java.py::test_t312_a_ranking_or_a_forecast_has_no_java_leg'
    assert coverage._test_problem(pytest_test, ROOT, 'java') is not None
    manifests = {
        language: yaml.safe_load(
            (
                ROOT / f'docs/design/rust-core/phase-0/scope-{language}.yaml'
            ).read_text(encoding='utf-8')
        )
        for language in ('python', 'java')
    }
    broken = copy.deepcopy(manifests)
    [path] = [
        p
        for p in broken['python']['paths']
        if p['slice'] == 'S1' and not p.get('tests')
    ][:1]
    path['cases'] = []
    path['tests'] = [java_test]
    assert path['id'] in coverage.uncovered(broken, {}, 'python', 'S1')
    report = coverage.Report()
    coverage.rule_4(broken, {}, report)
    assert any(java_test in p for p in report.problems), report.problems
    # The manifest's own schema refuses it too, by its language.
    import jsonschema

    schema = json.loads(
        (ROOT / 'docs/design/rust-core/phase-0/scope.schema.json').read_text(
            encoding='utf-8'
        )
    )
    errors = list(
        jsonschema.Draft202012Validator(schema).iter_errors(broken['python'])
    )
    assert errors, 'a Python path claiming a Java test passed the schema'
    java = copy.deepcopy(manifests['java'])
    [claimed] = [p for p in java['paths'] if p.get('tests')][:1]
    claimed['tests'] = [pytest_test]
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(java))


def test_t312_what_only_the_core_keeps_is_taken_out_of_java_s_lines_too() -> (
    None
):
    # The Java host runs the same core as rust-host: a column only the core
    # keeps (core_only_columns, approved for the core's runners) is left
    # out of its side against the reference, as theirs is.
    compare = _tool('compare')
    assert 'java' in compare.MORE_EXACT_RUNNERS
