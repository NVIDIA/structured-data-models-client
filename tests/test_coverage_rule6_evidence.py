# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage rule 6's evidence, failing closed (T1.32g part 1; the
implementer's own tests, beside the test author's seeds in
``tests/test_t132g_cell_claims.py``).

Each seed is this repository's HEAD with ``contract/reference``'s line of
``key.uint8`` changed, written as a commit whose objects live in a temporary
object directory (nothing of the repository is written), and judged by
``scripts/rust_core/coverage.py --check``: the evidence missing, emptied or
retyped fails ``rule 6`` naming the case; a line nested past the parser or
not an observation is refused by name (``read:``), never a traceback.
:func:`cell_type` is held to pyarrow's own text of each type it reads.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / 'scripts/rust_core/coverage.py'
CASE = 'key.uint8'


def _coverage_module() -> Any:
    spec = importlib.util.spec_from_file_location('coverage_rule6', COVERAGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(env: dict[str, str], *args: str, text_in: str | None = None) -> str:
    return subprocess.run(
        ['git', '-C', str(ROOT), *args],
        input=text_in,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


def _environment(tmp: Path) -> dict[str, str]:
    common = Path(_git(dict(os.environ), 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = ROOT / common
    (tmp / 'objects').mkdir()
    return {
        **os.environ,
        'GIT_OBJECT_DIRECTORY': str(tmp / 'objects'),
        'GIT_ALTERNATE_OBJECT_DIRECTORIES': str(common / 'objects'),
        'GIT_INDEX_FILE': str(tmp / 'index'),
        'GIT_AUTHOR_NAME': 'seed',
        'GIT_AUTHOR_EMAIL': 'seed@example.invalid',
        'GIT_COMMITTER_NAME': 'seed',
        'GIT_COMMITTER_EMAIL': 'seed@example.invalid',
    }


def _reference_path() -> str:
    [path] = [
        name
        for name in _git(
            dict(os.environ),
            'ls-tree',
            '--name-only',
            'HEAD:contract/reference/',
        ).splitlines()
        if name.endswith('.jsonl')
    ]
    return f'contract/reference/{path}'


def seeded(
    tmp: Path, change: Callable[[list[str]], list[str]]
) -> subprocess.CompletedProcess[str]:
    r"""``coverage.py --check`` of HEAD with the reference's lines changed."""
    env = _environment(tmp)
    path = _reference_path()
    lines = _git(dict(os.environ), 'show', f'HEAD:{path}').splitlines()
    blob = _git(
        env,
        'hash-object',
        '-w',
        '--stdin',
        text_in='\n'.join(change(lines)) + '\n',
    )
    _git(env, 'read-tree', 'HEAD')
    _git(env, 'update-index', '--cacheinfo', f'100644,{blob},{path}')
    commit = _git(
        env, 'commit-tree', _git(env, 'write-tree'), '-p', 'HEAD', '-m', 'seed'
    )
    return subprocess.run(
        [
            sys.executable,
            str(COVERAGE),
            '--check',
            '--repo',
            str(ROOT),
            '--rev',
            commit,
            '--checkpoint',
            'HEAD',
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )


def _line(change: Callable[[dict[str, Any]], None]) -> Any:
    r"""A change of ``key.uint8``'s line, the others kept."""

    def apply(lines: list[str]) -> list[str]:
        found = []
        for text in lines:
            line = json.loads(text)
            if line['case'] == CASE:
                change(line)
                text = json.dumps(line, sort_keys=True)
            found.append(text)
        return found

    return apply


def _samples(line: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        field
        for statement in line['driver']['capture']['statements']
        if statement['stage'] == 'from_source'
        for result in statement['results']
        for field in result.get('schema') or []
    ]


def _retype(value: Any) -> Callable[[dict[str, Any]], None]:
    def change(line: dict[str, Any]) -> None:
        for field in _samples(line):
            if field['name'] == 'user_id':
                field['type'] = value

    return change


def _set(*path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(line: dict[str, Any]) -> None:
        held = line
        for key in path[:-1]:
            held = held[key]
        held[path[-1]] = value

    return change


def fails_naming(done: subprocess.CompletedProcess[str], rule: str) -> None:
    output = done.stdout + done.stderr
    assert 'Traceback' not in output, output
    assert 'RecursionError' not in output, output
    assert done.returncode == 1, output
    assert any(
        line.startswith(f'{rule}:') and (rule == 'read' or CASE in line)
        for line in output.splitlines()
    ), output


def test_the_unchanged_reference_passes(tmp_path: Path) -> None:
    done = seeded(tmp_path, lambda lines: lines)
    assert done.returncode == 0, done.stdout + done.stderr


SEEDS: dict[str, Callable[[list[str]], list[str]]] = {
    'missing': lambda lines: [
        x for x in lines if json.loads(x)['case'] != CASE
    ],
    'no driver section': _line(_set('driver', value=None)),
    'no capture': _line(_set('driver', 'capture', value=None)),
    'no statements': _line(_set('driver', 'capture', 'statements', value=[])),
    'no schema past from_source': _line(_set('public', 'schema', value=None)),
    'no outcome': _line(_set('outcome', value=None)),
    'an empty schema': _line(
        _set('public', 'schema', value={'tables': [], 'links': []})
    ),
    'a type retyped as a number': _line(_retype(8)),
    'a type retyped as another type': _line(_retype('uint16')),
    'a type it does not know': _line(_retype('uint8 but stranger')),
    'a key retyped': _line(
        lambda line: line['public']['schema']['tables'][1].update(primary_key=7)
    ),
}


@pytest.mark.parametrize('seed', list(SEEDS))
def test_evidence_missing_empty_or_retyped_fails_rule_6(
    seed: str, tmp_path: Path
) -> None:
    fails_naming(seeded(tmp_path, SEEDS[seed]), 'rule 6')


@pytest.mark.parametrize(
    'line',
    [
        '[' * 100_000 + ']' * 100_000,
        json.dumps({f'k{i}': i for i in range(100_000)}),
        '{"case": ',
        '"a string"',
    ],
    ids=['nested 100,000 deep', '100,000 wide', 'not JSON', 'not an object'],
)
def test_a_reference_line_it_cannot_read_is_refused_by_name(
    line: str, tmp_path: Path
) -> None:
    fails_naming(seeded(tmp_path, lambda lines: [*lines, line]), 'read')


def test_the_cell_types_are_pyarrow_s_text_through_the_admission_table() -> (
    None
):
    pa = pytest.importorskip('pyarrow')
    cell_type = _coverage_module().cell_type
    table = [
        (pa.int8(), 'int8'),
        (pa.uint64(), 'uint64'),
        (pa.float16(), 'float32'),
        (pa.float32(), 'float32'),
        (pa.float64(), 'float64'),
        (pa.decimal128(18, 2), 'decimal128'),
        (pa.decimal256(40, 2), 'decimal256'),
        (pa.string(), 'utf8'),
        (pa.large_string(), 'utf8'),
        (pa.binary(), 'binary'),
        (pa.large_binary(), 'binary'),
        (pa.binary(16), 'binary'),
        (pa.date32(), 'date32'),
        (pa.date64(), 'timestamp-us'),
        (pa.timestamp('s'), 'timestamp-us'),
        (pa.timestamp('ms'), 'timestamp-us'),
        (pa.timestamp('us'), 'timestamp-us'),
        (pa.timestamp('ns'), 'timestamp-ns'),
        (pa.timestamp('s', tz='America/New_York'), 'timestamp-us-tz'),
        (pa.timestamp('ns', tz='UTC'), 'timestamp-ns-tz'),
        (pa.duration('s'), 'duration-us'),
        (pa.duration('ns'), 'duration-ns'),
        (pa.time32('s'), 'time64'),
        (pa.time64('ns'), 'time64'),
        (pa.list_(pa.int32()), 'list'),
        (pa.large_list(pa.int32()), 'list'),
        (pa.list_(pa.int64(), 2), 'list'),
        (pa.struct([('a', pa.int32()), ('b', pa.string())]), 'struct'),
        (pa.map_(pa.string(), pa.int32()), 'map'),
        (pa.dictionary(pa.int32(), pa.string()), 'dictionary-utf8'),
        (pa.dictionary(pa.int8(), pa.large_string()), 'dictionary-utf8'),
        (pa.dictionary(pa.int32(), pa.int64()), 'int64'),
        (pa.dictionary(pa.int32(), pa.timestamp('s')), 'timestamp-us'),
        (pa.run_end_encoded(pa.int32(), pa.string()), 'utf8'),
        (pa.bool_(), 'bool'),
        (pa.null(), 'null'),
        (pa.month_day_nano_interval(), None),
        (pa.dense_union([pa.field('n', pa.int32())]), None),
    ]
    if hasattr(pa, 'string_view'):
        table += [(pa.string_view(), 'utf8'), (pa.binary_view(), 'binary')]
    if hasattr(pa, 'decimal32'):
        table += [(pa.decimal32(5, 2), 'decimal128')]
    if hasattr(pa, 'list_view'):
        table += [(pa.list_view(pa.int32()), 'list')]
    found = {str(arrow): cell_type(str(arrow)) for arrow, _ in table}
    assert found == {str(arrow): cell for arrow, cell in table}


def test_a_warehouse_only_case_is_held_to_its_captures_reference_line() -> None:
    # Codex and Claude T3.20 r1 P1-1: a case with no DuckDB line is held to
    # the old client's reference line in each of its warehouse captures,
    # the sample named as the warehouse names the relation; a claim no leg
    # shows, a case with no capture, and a sample read as rows all fail.
    coverage = _coverage_module()
    case = {
        'id': 'w.case',
        'cells': ['int64/primary-key', 'timestamp-us/time-column'],
        'input': {},
    }
    sample = 'SELECT "id", "ts" FROM "{catalog}"."{schema}"."ORDERS" LIMIT 1000'

    def line(types: Any) -> dict[str, Any]:
        result: dict[str, Any] = {'rows': 2}
        if types is not None:
            result['schema'] = [
                {'name': 'id', 'type': types[0]},
                {'name': 'ts', 'type': types[1]},
            ]
        return {
            'case': 'w.case@snowflake',
            'outcome': {'result': 'success', 'stage': 'from_source'},
            'driver': {
                'capture': {
                    'statements': [
                        {
                            'stage': 'from_source',
                            'sql': sample,
                            'results': [result],
                        }
                    ]
                }
            },
            'public': {
                'schema': {
                    'tables': [
                        {
                            'name': 'orders',
                            'columns': [{'name': 'id'}, {'name': 'ts'}],
                            'primary_key': 'id',
                            'time_column': 'ts',
                        }
                    ],
                    'links': [],
                }
            },
        }

    def problems(captured: dict[str, Any]) -> list[str]:
        report = coverage.Report()
        coverage.rule_6({'w.case': case}, {}, report, captured)
        return list(report.problems)

    held = line(('int64', 'timestamp[us]'))
    assert problems({'w.case': [held]}) == []
    narrower = line(('int8', 'timestamp[us]'))
    assert problems({'w.case': [narrower, held]}) == []
    [one] = problems({'w.case': [narrower]})
    assert one.startswith('rule 6: case w.case: claims int64/primary-key')
    [none] = problems({})
    assert 'no reference observation' in none
    [rows] = problems({'w.case': [line(None)]})
    assert 'a sample read as rows' in rows


def _databricks_omitted_features(
    change: Callable[[dict[str, Any]], None],
    source: str = 'users',
) -> list[str]:
    """The real ordinary capture through the capture reader and rule 6."""
    coverage = _coverage_module()
    case_id = 'java.databricks-sql.types'
    path = f'contract/captures/databricks-sql/{case_id}/capture.json'
    capture = json.loads((ROOT / path).read_text())
    change(capture)
    case = json.loads((ROOT / f'contract/cases/s2/{case_id}.json').read_text())
    case['input']['from_source']['tables']['users'] = source
    case['cells'] = ['binary/feature', 'list/feature']
    real_tree = coverage._module('ratchet').Tree(
        ROOT, _git(dict(os.environ), 'rev-parse', 'HEAD')
    )

    class CaptureTree:
        repo = real_tree.repo
        sha = real_tree.sha

        def files(self, prefix: str) -> list[str]:
            assert prefix == coverage.CAPTURES
            return [path]

        def read(self, name: str) -> str | None:
            return json.dumps(capture) if name == path else real_tree.read(name)

    report = coverage.Report()
    captured = coverage.read_captures(CaptureTree(), report, {case_id})
    coverage.rule_6({case_id: case}, {}, report, captured)
    return list(report.problems)


def test_real_databricks_omitted_columns_keep_their_evidenced_input_roles() -> (
    None
):
    # E96(b): the reference reads token:binary and tags:list<int32>, then
    # leaves them out. Both real catalog constraint reads return no rows.
    assert _databricks_omitted_features(lambda capture: None) == []


@pytest.mark.parametrize(
    'source',
    [
        '{catalog}.{schema}.users',
        '`{catalog}`.`{schema}`.`users`',
        '`{catalog}`.{schema}.USERS',
    ],
)
def test_qualified_source_must_match_the_proven_sample_identity(
    source: str,
) -> None:
    assert (
        _databricks_omitted_features(lambda capture: None, source=source) == []
    )


@pytest.mark.parametrize(
    'source',
    [
        '{catalog}.another.users',
        'another.{schema}.users',
        '`{CATALOG}`.`{schema}`.`users`',
        '{schema}.users',
        'users trailing',
        'catalog..users',
    ],
)
def test_a_single_sample_does_not_prove_a_different_or_unknown_source(
    source: str,
) -> None:
    # The genuine sample and complete zero-row proof remain unchanged.
    # A matching final name cannot replace the case's catalog/schema.
    problems = _databricks_omitted_features(lambda capture: None, source=source)
    assert len(problems) == 2, problems
    assert all(
        'rule 6: case java.databricks-sql.types: claims ' in p for p in problems
    )


@pytest.mark.parametrize(
    'raw', ['missing', 'null', 'object', 'string', 'empty']
)
def test_serialized_derived_bindings_never_replace_missing_raw_evidence(
    raw: str,
) -> None:
    def change(capture: dict[str, Any]) -> None:
        statements = capture['reference']['driver']['capture']['statements']
        for statement, kept in zip(
            statements, capture['statements'], strict=True
        ):
            parameters = kept.get('parameters')
            if isinstance(parameters, list) and len(parameters) == 2:
                statement['_coverage_parameters'] = list(parameters)
        if raw == 'missing':
            capture.pop('statements')
        else:
            capture['statements'] = {
                'null': None,
                'object': {},
                'string': 'not a statement list',
                'empty': [],
            }[raw]

    problems = _databricks_omitted_features(change)
    assert len(problems) == 2, problems
    assert all(
        'rule 6: case java.databricks-sql.types: claims ' in p for p in problems
    )


def test_constraint_absence_does_not_cross_sampled_namespaces() -> None:
    def change(capture: dict[str, Any]) -> None:
        statements = capture['reference']['driver']['capture']['statements']
        recorded = capture['statements']
        samples = []
        for index, (statement, kept) in enumerate(
            zip(statements, recorded, strict=True)
        ):
            if kept.get('parameters') == ['{schema}', 'users']:
                # Both catalog reads now prove absence only for a.users.
                kept['parameters'][0] = 'a'
                statement['parameters'][0]['value'] = repr('a')
            relation = '`{catalog}`.`{schema}`.`users`'
            if relation in kept['text'] and kept['text'].endswith('LIMIT 1000'):
                a = '`{catalog}`.`a`.`users`'
                b = '`{catalog}`.`b`.`users`'
                kept['text'] = kept['text'].replace(relation, a)
                statement['sql'] = kept['text']
                other_kept, other_statement = copy.deepcopy((kept, statement))
                other_kept['text'] = other_kept['text'].replace(a, b)
                other_statement['sql'] = other_kept['text']
                samples.append((index, other_kept, other_statement))
        assert len(samples) == 1
        index, kept, statement = samples[0]
        recorded.insert(index + 1, kept)
        statements.insert(index + 1, statement)

    # Both a.users and b.users are sampled, but the case names only b.users
    # and no catalog read establishes constraint absence for b.users.
    problems = _databricks_omitted_features(change, source='{catalog}.b.users')
    assert len(problems) == 2, problems
    assert all(
        'rule 6: case java.databricks-sql.types: claims ' in p for p in problems
    )


@pytest.mark.parametrize('kind', ['keys', 'foreign'])
@pytest.mark.parametrize(
    'fault',
    [
        'missing',
        'nonempty',
        'unknown count',
        'failed',
        'missing parameters',
        'wrong namespace',
        'wrong table',
        'unrecognized SQL',
    ],
)
def test_omitted_warehouse_roles_need_complete_empty_constraint_evidence(
    kind: str, fault: str
) -> None:
    def change(capture: dict[str, Any]) -> None:
        recorded = capture['statements']
        statements = capture['reference']['driver']['capture']['statements']
        selected = [
            i
            for i, kept in enumerate(recorded)
            if kept.get('parameters') == ['{schema}', 'users']
            and (
                "constraint_type IN ('PRIMARY KEY', 'UNIQUE')"
                if kind == 'keys'
                else "constraint_type = 'FOREIGN KEY'"
            )
            in kept['text']
        ]
        assert len(selected) == 1
        [index] = selected
        statement, kept = statements[index], recorded[index]
        if fault == 'missing':
            statements.pop(index)
            recorded.pop(index)
        elif fault == 'nonempty':
            statement['results'][0]['rows'] = 1
        elif fault == 'unknown count':
            statement['results'][0]['rows'] = None
        elif fault == 'failed':
            statement['error'] = {'type': 'RuntimeError'}
        elif fault == 'missing parameters':
            kept.pop('parameters')
        elif fault in ('wrong namespace', 'wrong table'):
            parameter = 0 if fault == 'wrong namespace' else 1
            kept['parameters'][parameter] = 'another'
            statement['parameters'][parameter]['value'] = repr('another')
        elif fault == 'unrecognized SQL':
            kept['text'] += ' AND 1 = 0'
            statement['sql'] = kept['text']
        else:
            raise AssertionError(fault)

    problems = _databricks_omitted_features(change)
    assert len(problems) == 2
    assert all(
        problem.startswith('rule 6: case java.databricks-sql.types: claims ')
        for problem in problems
    ), problems


def test_a_row_read_sample_is_typed_by_the_arrow_its_capture_holds() -> None:
    # T3.20 r2: a Postgres leg's driver reads rows; its capture holds each
    # read as the Arrow the core's Postgres provider builds from the
    # driver's type codes, and rule 6 reads that Arrow's types for a sample
    # its reference line holds as rows.
    pa = pytest.importorskip('pyarrow')
    coverage = _coverage_module()
    sample = 'SELECT "id", "ts" FROM "{schema}"."orders" LIMIT 1000'
    line = {
        'case': 'w.case@lakebase',
        'outcome': {'result': 'success', 'stage': 'from_source'},
        'driver': {
            'capture': {
                'statements': [
                    {'stage': 'from_source', 'sql': sample, 'results': [{}]}
                ]
            }
        },
        'public': {
            'schema': {
                'tables': [
                    {
                        'name': 'orders',
                        'columns': [{'name': 'id'}, {'name': 'ts'}],
                        'primary_key': 'id',
                        'time_column': 'ts',
                    }
                ],
                'links': [],
            }
        },
    }
    capture = {'statements': [{'text': sample, 'result': 'tables/a.arrows'}]}
    table = pa.table(
        {
            'id': pa.array([1], pa.int32()),
            'ts': pa.array([0], pa.timestamp('us')),
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    blobs = {'tables/a.arrows': sink.getvalue().to_pybytes()}
    coverage.row_schemas(capture, line, blobs.get)
    case = {
        'id': 'w.case',
        'cells': ['int32/primary-key', 'timestamp-us/time-column'],
        'input': {},
    }
    report = coverage.Report()
    coverage.rule_6({'w.case': case}, {}, report, {'w.case': [line]})
    assert report.problems == []


def test_the_captures_tables_are_read_through_one_git_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Claude T3.20 r3 P3: each Lakebase sample's Arrow was read with its own
    # git cat-file, which made a coverage run take minutes; every table of
    # one commit is read through one git cat-file --batch process.
    coverage = _coverage_module()
    started: list[Any] = []
    real = coverage.subprocess.Popen

    def counting(*args: Any, **kwargs: Any) -> Any:
        started.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(coverage.subprocess, 'Popen', counting)
    paths = [
        name
        for name in _git(
            dict(os.environ),
            'ls-tree',
            '-r',
            '--name-only',
            'HEAD',
            'contract/captures/lakebase/base.binary/tables/',
        ).splitlines()
    ][:3]
    assert len(paths) == 3
    tree = coverage._module('ratchet').Tree(
        ROOT, _git(dict(os.environ), 'rev-parse', 'HEAD')
    )
    blobs = coverage.Blobs(tree)
    try:
        read = [blobs.read(path) for path in paths]
        assert blobs.read('contract/captures/no-such-file') is None
    finally:
        blobs.close()
    assert [args for args in started if 'cat-file' in args] == [
        ['git', '-C', str(ROOT), 'cat-file', '--batch']
    ]
    for path, data in zip(paths, read, strict=True):
        assert data == (ROOT / path).read_bytes()
