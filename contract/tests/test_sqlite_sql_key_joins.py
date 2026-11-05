# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite provider, executed SQL goldens and exact-key joins."""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

RUNNER = ROOT / 'contract/runners/run_python.py'

VECTORS_FILE = ROOT / 'contract/sqlite-key-pair-vectors.json'

GOLDENS = ROOT / 'rust/crates/sdm-data/tests'

DATABASE = 'sqlite'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


M = _module(
    'sqlite_key_pairs_record',
    ROOT / 'contract/support/sqlite_key_scenarios.py',
)

R = M.R


def test_the_core_reports_a_sqlite_provider() -> None:
    r"""Exit (the staging rule's marker, T2.1a): the core under test reports
    a SQLite provider (``sdm_client._native.PROVIDERS``), so a SQLite job
    and the gate run its core legs there."""

    def check() -> None:
        from sdm_client import _native

        assert DATABASE in tuple(_native.PROVIDERS)

    check()


@functools.cache
def _cases_on_sqlite() -> dict[str, dict[str, Any]]:
    r"""python-core over every case on SQLite, as the SQLite job runs it:
    each observation by its key."""
    with tempfile.TemporaryDirectory(prefix='t22-cases-') as tmp:
        out = Path(tmp) / 'python-core.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(RUNNER),
                '--mode',
                'python-core',
                '--database',
                DATABASE,
                '--out',
                str(out),
                str(ROOT / 'contract/cases'),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


_RAW = re.compile(r'r(#*)"(.*?)"\1', re.S)

_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '0': '\0'}


def literals(source: str) -> list[str]:
    r"""Every string literal of a Rust source, its escapes read: ``"..."``
    (``\n``, ``\t``, ``\r``, ``\0``, ``\\``, ``\"``, ``\'`` and a line
    continuation) and raw ``r"..."``, ``r#"..."#``; comments and character
    literals skipped (``sdm-testkit``'s ``t132a_keys_sql.rs`` reads them
    so)."""
    out: list[str] = []
    i = 0
    while i < len(source):
        c = source[i]
        if source.startswith('//', i):
            end = source.find('\n', i)
            i = len(source) if end < 0 else end
        elif c == "'":
            if source[i + 2 : i + 3] == "'":
                i += 3
            elif source[i + 1 : i + 2] == '\\':
                close = source.find("'", i + 2, i + 12)
                i = i + 1 if close < 0 else close + 1
            else:
                i += 1
        elif c == 'r' and (
            i == 0 or not (source[i - 1].isalnum() or source[i - 1] == '_')
        ):
            raw = _RAW.match(source, i)
            if raw is None:
                i += 1
                continue
            out.append(raw.group(2))
            i = raw.end()
        elif c == '"':
            text = []
            i += 1
            while i < len(source) and source[i] != '"':
                if source[i] == '\\':
                    i += 1
                    if source[i] == '\n':
                        while i + 1 < len(source) and source[i + 1].isspace():
                            i += 1
                    else:
                        text.append(_ESCAPES.get(source[i], source[i]))
                else:
                    text.append(source[i])
                i += 1
            out.append(''.join(text))
            i += 1
        else:
            i += 1
    return out


def goldens(directory: Path = GOLDENS) -> set[str]:
    r"""The SQLite goldens: every string literal of ``sdm-data``'s test
    files whose name holds ``sqlite``."""
    return {
        text
        for path in sorted(directory.glob('*sqlite*.rs'))
        for text in literals(path.read_text(encoding='utf-8'))
    }


def _scratch_shape(sql: str) -> str:
    """Ignore only opaque scratch-name digests, preserving distinct relations."""
    names: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        name = match.group()
        if match.lastgroup != 'scratch':
            return name
        return f'"__sdm_seeds_digest_{names.setdefault(name, len(names))}__"'

    tokens = re.compile(
        r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/|"
        r'(?P<scratch>"__sdm_seeds_[0-9a-f]{16}__")(?!")|"(?:[^"]|"")*"',
        re.DOTALL,
    )
    return tokens.sub(replace, sql)


def unpinned(
    observations: dict[str, dict[str, Any]], pinned: set[str]
) -> list[str]:
    r"""Each statement a run made that no golden pins, by key and public
    call; every one when none was made."""
    ran = [
        (key, statement['stage'], statement['sql'])
        for key, line in sorted(observations.items())
        for statement in ((line.get('driver') or {}).get('capture') or {}).get(
            'statements', []
        )
    ]
    if not ran:
        return ['no statement ran: no evidence']
    shapes = {_scratch_shape(sql) for sql in pinned}
    return [
        f'{k} {stage}: {sql}'
        for k, stage, sql in ran
        if _scratch_shape(sql) not in shapes
    ]


def test_the_golden_check_reads_rust_and_fails_closed() -> None:
    r"""Control: the literal reader reads Rust's strings, and the check
    fails on a statement no golden pins and on no statements at all."""
    source = (
        'let a = "x\\n\\"y\\""; // "not a pin"\n'
        'let b = r#"raw "q""#; let c = \'"\'; let d = "long \\\n    line";'
    )
    assert literals(source) == ['x\n"y"', 'raw "q"', 'long line']
    seen = {
        'k@sqlite': {
            'driver': {
                'capture': {
                    'statements': [
                        {'stage': 'from_source', 'sql': 'SELECT 1'},
                        {'stage': 'materialize', 'sql': 'SELECT 2'},
                    ]
                }
            }
        }
    }
    assert unpinned(seen, {'SELECT 1', 'SELECT 2'}) == []
    assert unpinned(seen, {'SELECT 1'}) == ['k@sqlite materialize: SELECT 2']
    assert unpinned({}, {'SELECT 1'}) == ['no statement ran: no evidence']


def test_scratch_digest_matching_preserves_all_other_sql() -> None:
    first = 'SELECT * FROM temp."__sdm_seeds_0123456789abcdef__" WHERE x = 1'
    second = first.replace('0123456789abcdef', 'fedcba9876543210')
    assert _scratch_shape(first) == _scratch_shape(second)
    assert _scratch_shape(first) != _scratch_shape(
        second.replace('x = 1', 'x = 2')
    )
    assert _scratch_shape(first) != _scratch_shape(
        second.replace('temp.', 'main.')
    )
    assert _scratch_shape(first) != _scratch_shape(
        second.replace('fedcba9876543210', 'bad')
    )
    for wrapper in ("SELECT '%s'", '-- %s\nSELECT 1', '/* %s */ SELECT 1'):
        left = wrapper % '"__sdm_seeds_0123456789abcdef__"'
        right = wrapper % '"__sdm_seeds_fedcba9876543210__"'
        assert _scratch_shape(left) != _scratch_shape(right)
    # Two distinct key sets cannot collapse into one relation.
    assert _scratch_shape(first + second) != _scratch_shape(first + first)


def test_every_statement_python_core_ran_on_sqlite_is_a_golden() -> None:
    r"""Exit "L4 goldens executed" (SQLite): every statement python-core ran
    on SQLite, over every case, is pinned apart from scratch-name digests in
    ``rust/crates/sdm-data/tests/sqlite*.rs``: each golden that matches was
    executed on SQLite, and any other SQL change without a golden update fails.
    The digest includes current resolved request metadata absent from old pins."""

    def check() -> None:
        problems = unpinned(_cases_on_sqlite(), goldens())
        assert not problems, '\n'.join(problems[:40])

    check()


def recorded() -> dict[str, Any]:
    from scripts.rust_core.fixture_tables import unpack

    found: dict[str, Any] = unpack(json.loads(VECTORS_FILE.read_text('utf-8')))
    return found


def _by_id() -> dict[str, dict[str, Any]]:
    return {vector['id']: vector for vector in recorded()['vectors']}


def test_the_join_check_fails_on_a_wrong_join() -> None:
    r"""Control: :func:`joins` passes the reference's own request on SQLite
    (``base.regression@sqlite``: the shop, its question and entities, as
    ``pair.bigint.bigint`` asks) and fails it with one order moved to
    another entity, and with one entity's age taken from another."""
    vector = _by_id()['pair.bigint.bigint']
    observation = _core_runs()['pair.bigint.bigint']
    [body] = [p['parsed'] for p in observation['public']['payloads']]
    assert M.joins(body, M.offset(vector)) == []
    orders = body['predict']['related_tables']['orders']
    at = orders['columns'].index('user_id')
    moved = json.loads(json.dumps(body))
    moved['predict']['related_tables']['orders']['rows'][0][at] = {
        'number': '5'
    }
    assert M.joins(moved, M.offset(vector)) != []
    users = body['context']['related_tables']['users']
    at = users['columns'].index('age')
    aged = json.loads(json.dumps(body))
    aged['context']['related_tables']['users']['rows'][0][at] = {'number': '99'}
    assert M.joins(aged, M.offset(vector)) != []


@functools.cache
def _core_runs() -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix='t22-pairs-') as tmp:
        result: dict[str, dict[str, Any]] = M.run(
            'python-core', M.VECTORS, Path(tmp)
        )
        return result


def test_each_key_pair_preserves_exact_values_and_joins():
    observed = _core_runs()
    assert set(observed) == {vector['id'] for vector in recorded()['vectors']}
    for vector in recorded()['vectors']:
        key = vector['id']
        observation = observed[key]
        got = R.project(observation, vector)
        assert observation['identity']['native_loaded'], key
        if key in {'wide.ubigint.hugeint', 'wide.ubigint.ubigint'}:
            # SQLite stores these unsigned values as text; native joins preserve it.
            assert got['outcome'] == {
                'result': 'success',
                'stage': 'predict',
                'error': None,
                'code': None,
            }, key
            assert got['schema'] == vector['reference']['schema'], key
            assert got['warnings'] == vector['reference']['warnings'], key
            assert got['result']['rows'] == 2, key
        else:
            assert got == vector['reference'], key
        if got['outcome']['result'] == 'success':
            assert M.joined(observation, vector) is True, key
