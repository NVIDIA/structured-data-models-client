# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P6.capture: the dialect-call capture holds what the brief asks for.

- The committed ``tests/parity/dialects/*.json`` are what the capture script
  writes today.
- Every call is well formed, and a plain function's arguments are its whole
  signature.
- Every ``dialect.py`` function, every SQL fragment of ``base/utils.py`` a
  backend imports, quoting and every moved helper is captured, per backend.
- Every execute step of every SQL transcript, live ones too, is a captured
  statement of that scenario.
- The synthetic branches of task 2 are there.
"""

from __future__ import annotations

import inspect
import json
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _dialects import (
    BACKENDS,
    CAPTURE_DIR,
    MOVED_HELPERS,
    TRANSCRIPT_DIALECT,
    TRANSCRIPTS,
    UTILS_FRAGMENTS,
    backend_module,
    decode,
    dialect_functions,
    dialect_module,
    imports,
    is_statement,
    load_capture,
    plain_function,
    run_capture,
)

#: The SQL transcripts the corpus held when Phase 6 started; it may only grow.
SQL_TRANSCRIPTS_AT_BASE = 56


@pytest.fixture(scope='module')
def capture() -> dict[str, dict[str, Any]]:
    return load_capture()


def _calls(
    capture: dict[str, dict[str, Any]], backend: str, function: str
) -> list[dict[str, Any]]:
    return [c for c in capture[backend]['calls'] if c['function'] == function]


# --- The files ------------------------------------------------------------------


def test_the_capture_is_what_the_script_writes() -> None:
    with tempfile.TemporaryDirectory(prefix='sdm-p6-capture-') as tmp:
        out = Path(tmp)
        run_capture(out, hashseed='7')
        written = sorted(p.name for p in out.iterdir())
        assert written == sorted(f'{b}.json' for b in BACKENDS), (
            f'the capture wrote {written}'
        )
        stale = [
            name
            for name in written
            if (out / name).read_bytes() != (CAPTURE_DIR / name).read_bytes()
        ]
    assert not stale, (
        f'tests/parity/dialects differs from what the capture writes today: '
        f'{stale}'
    )


def test_every_call_is_well_formed(capture: dict[str, dict[str, Any]]) -> None:
    problems = []
    for backend, data in capture.items():
        for i, call in enumerate(data['calls']):
            where = f'{backend}#{i} {call.get("function")!r}'
            if not isinstance(call, dict):
                problems.append(f'{where}: not an object')
                continue
            name = call.get('function')
            if not isinstance(name, str) or not name:
                problems.append(f'{where}: no function')
                continue
            if not isinstance(call.get('args'), dict):
                problems.append(f'{where}: args is not an object')
                continue
            sources = call.get('sources')
            if (
                not isinstance(sources, list)
                or not sources
                or not all(
                    isinstance(s, str)
                    and re.fullmatch(r'(scenario|synthetic):.+', s)
                    for s in sources
                )
            ):
                problems.append(f'{where}: sources {sources!r}')
            function = plain_function(backend, name)
            helper = name in MOVED_HELPERS[backend] or (
                function is None and re.fullmatch(r'_?[A-Z]\w*\.\w+', name)
            )
            if function is None and not helper:
                problems.append(f'{where}: names no function this format has')
                continue
            if function is not None:
                params = set(inspect.signature(function).parameters)
                if set(call['args']) != params:
                    problems.append(
                        f'{where}: args {sorted(call["args"])} are not its '
                        f'parameters {sorted(params)}'
                    )
            statement = function is not None and is_statement(function)
            keys = set(call) - {'function', 'args', 'sources'}
            want = {'sql', 'params'} if statement else {'result'}
            if keys != want:
                problems.append(
                    f'{where}: holds {sorted(keys)}, not {sorted(want)}'
                )
            elif statement and not isinstance(call['sql'], str):
                problems.append(f'{where}: sql is not text')
            elif statement and not (
                call['params'] is None or isinstance(call['params'], list)
            ):
                problems.append(f'{where}: params is neither null nor a list')
    assert not problems, f'{len(problems)} malformed calls:\n' + '\n'.join(
        problems[:60]
    )


# --- Every function --------------------------------------------------------------


def test_every_function_is_captured(capture: dict[str, dict[str, Any]]) -> None:
    from sdm_connectors.sql import quote_ident
    from sdm_relational_engine.rfm.base import utils

    missing = []
    for backend in BACKENDS:
        names = {call['function'] for call in capture[backend]['calls']}
        wanted = set(dialect_functions(backend))
        wanted |= {
            f'utils.{name}'
            for name in UTILS_FRAGMENTS
            if imports(backend, getattr(utils, name))
        }
        if vars(dialect_module(backend)).get('quote_ident') is quote_ident:
            wanted.add('sql.quote_ident')
        wanted |= set(MOVED_HELPERS[backend])
        missing += [f'{backend}: {name}' for name in sorted(wanted - names)]
    assert not missing, f'{len(missing)} functions never captured:\n' + (
        '\n'.join(missing)
    )


@pytest.mark.parametrize('backend', BACKENDS)
def test_the_moved_helpers_exist(backend: str) -> None:
    r"""The helpers named in ``CONTRACT.md`` are attributes of the engine's
    classes, so the list keeps up with the code."""
    for name in MOVED_HELPERS[backend]:
        owner, attribute = name.split('.')
        found = [
            vars(backend_module(backend, module))[owner]
            for module in ('table', 'sampler')
            if owner in vars(backend_module(backend, module))
        ]
        assert found, f'{backend}: no class {owner}'
        assert attribute in vars(found[0]), f'{backend}: {name} is not defined'


# --- Every scenario --------------------------------------------------------------


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def test_every_scenario_is_captured(capture: dict[str, dict[str, Any]]) -> None:
    statements: dict[str, dict[tuple[str, str], set[str]]] = {
        backend: defaultdict(set) for backend in BACKENDS
    }
    for backend, data in capture.items():
        for call in data['calls']:
            if 'sql' in call:
                key = (call['sql'], _canonical(call['params']))
                statements[backend][key].update(call['sources'])
    transcripts = sorted(TRANSCRIPTS.rglob('transcript.json'))
    checked, problems = 0, []
    for path in transcripts:
        transcript = json.loads(path.read_text())
        dialect = TRANSCRIPT_DIALECT.get(transcript['backend'])
        if dialect is None:
            continue
        checked += 1
        name = transcript['name']
        source = f'scenario:{name}'
        steps = [s for s in transcript['steps'] if s['op'] == 'execute']
        assert steps, f'{name} has no execute step'
        for k, step in enumerate(steps):
            key = (step['sql'], _canonical(step['params']))
            if source not in statements[dialect].get(key, set()):
                problems.append(
                    f'{name}: execute step {k} is no captured {dialect} '
                    f'statement of this scenario: {step["sql"][:120]!r}'
                )
                break
    assert checked >= SQL_TRANSCRIPTS_AT_BASE, (
        f'{checked} SQL transcripts, fewer than {SQL_TRANSCRIPTS_AT_BASE}'
    )
    assert not problems, (
        f'{len(problems)} of {checked} scenarios are not captured:\n'
        + '\n'.join(problems)
    )


# --- The synthetic branches ------------------------------------------------------


def _arg(call: dict[str, Any], name: str) -> Any:
    return decode(call['args'][name])


def _length(value: Any) -> int:
    return len(value)


def _result(call: dict[str, Any]) -> Any:
    return decode(call['result'])


def _is_expression(projection: str) -> bool:
    return re.search(r'\sAS\s', projection) is not None


Check = tuple[str, str, str, Callable[[dict[str, Any]], bool]]


def _checks() -> list[Check]:
    r"""``(backend, function, what, holds)``: some call of ``function`` in
    ``backend``'s capture must make ``holds`` true."""
    out: list[Check] = []
    for b in BACKENDS:
        out += [
            (
                b,
                'holds_any',
                'two values or more',
                lambda c: len(_arg(c, 'values')) >= 2,
            ),
            (
                b,
                'holds_any',
                'no values',
                lambda c: len(_arg(c, 'values')) == 0,
            ),
            (
                b,
                'key_counts',
                'a composite key',
                lambda c: any(len(p.columns) >= 2 for p in _arg(c, 'parts')),
            ),
            (
                b,
                'repeated_key',
                'a composite key',
                lambda c: len(_arg(c, 'columns')) >= 2,
            ),
            (
                b,
                'first_rows' if b in ('duckdb', 'sqlite') else 'sample',
                'an expression column',
                lambda c: any(
                    _is_expression(p) for p in _arg(c, 'projections')
                ),
            ),
            (
                b,
                'rows_by_key',
                'no ids',
                lambda c: _length(_arg(c, 'ids')) == 0,
            ),
        ]
    for b in ('postgres', 'databricks'):
        out += [
            (b, 'chunk_rows', 'no rows', lambda c: len(_arg(c, 'rows')) == 0),
            (
                b,
                'chunk_rows',
                'two 900 KB chunks or more',
                lambda c: (
                    _arg(c, 'max_bytes') == 900_000 and len(_result(c)) >= 2
                ),
            ),
        ]
    out += [
        # The scan caps.
        (
            'postgres',
            'PostgresTable._prefiltered_sample_source',
            'a block sample',
            lambda c: isinstance(_result(c), str),
        ),
        (
            'postgres',
            'PostgresTable._prefiltered_sample_source',
            'no prefilter',
            lambda c: _result(c) is None,
        ),
        (
            'postgres',
            'sample',
            'a TABLESAMPLE SYSTEM source',
            lambda c: 'TABLESAMPLE SYSTEM' in _arg(c, 'table'),
        ),
        (
            'snow',
            'SnowTable._prefilter',
            'a clause',
            lambda c: isinstance(_result(c), str),
        ),
        (
            'snow',
            'SnowTable._prefilter',
            'no prefilter',
            lambda c: _result(c) is None,
        ),
        (
            'snow',
            'SnowTable._listing',
            'a listing',
            lambda c: isinstance(_result(c), dict),
        ),
        (
            'snow',
            'SnowTable._listing',
            'no listing',
            lambda c: _result(c) is None,
        ),
        (
            'snow',
            'sample',
            'a SAMPLE SYSTEM source',
            lambda c: 'SAMPLE SYSTEM (' in _arg(c, 'table'),
        ),
        (
            'snow',
            'entity_rows',
            'a seeded draw through a prefilter',
            lambda c: (
                _arg(c, 'seeded') is not None
                and _arg(c, 'prefilter') is not None
            ),
        ),
        (
            'databricks',
            'DatabricksTable._is_large',
            'a large table',
            lambda c: _result(c) is True,
        ),
        (
            'databricks',
            'DatabricksTable._is_large',
            'a small table',
            lambda c: _result(c) is False,
        ),
        (
            'databricks',
            'sample',
            'a bounded source',
            lambda c: (
                re.fullmatch(
                    r'\(SELECT \* FROM .+ LIMIT \d+\)',
                    _arg(c, 'table'),
                    re.DOTALL,
                )
                is not None
            ),
        ),
        # Zoned bounds and string time columns.
        (
            'postgres',
            'serialize_time',
            'a UTC bound',
            lambda c: (
                _arg(c, 'utc') is True and str(_result(c)).endswith('+00:00')
            ),
        ),
        (
            'postgres',
            'serialize_time',
            'a bare bound',
            lambda c: _arg(c, 'utc') is False and _result(c) is not None,
        ),
        (
            'postgres',
            'window',
            'a timestamptz column',
            lambda c: _arg(c, 'time_type') == 'timestamp with time zone',
        ),
        (
            'postgres',
            'PostgresSampler._utc_bounds',
            'UTC bounds',
            lambda c: _result(c) is True,
        ),
        (
            'postgres',
            'PostgresSampler._utc_bounds',
            'a text column',
            lambda c: _result(c) is False,
        ),
        (
            'databricks',
            'DatabricksSampler._bound_format',
            'a zoned format',
            lambda c: _result(c) == '%Y-%m-%d %H:%M:%S+00:00',
        ),
        (
            'databricks',
            'DatabricksSampler._bound_format',
            'a bare format',
            lambda c: _result(c) == '%Y-%m-%d %H:%M:%S',
        ),
        (
            'databricks',
            'window',
            'a bound naming UTC',
            lambda c: any(
                str(bound).endswith('+00:00') for bound in _arg(c, 'bounds')
            ),
        ),
        ('snow', 'zone_of', 'LTZ', lambda c: _result(c) == 'LTZ'),
        ('snow', 'zone_of', 'TZ', lambda c: _result(c) == 'TZ'),
        ('snow', 'zone_of', 'a naive type', lambda c: _result(c) is None),
        ('snow', 'window', 'an LTZ column', lambda c: _arg(c, 'zone') == 'LTZ'),
        ('snow', 'window', 'a TZ column', lambda c: _arg(c, 'zone') == 'TZ'),
        (
            'snow',
            'neighbors',
            'an LTZ column',
            lambda c: _arg(c, 'zone') == 'LTZ',
        ),
        ('snow', 'neighbors', 'a TZ column', lambda c: _arg(c, 'zone') == 'TZ'),
        ('snow', 'SnowSampler._zone', 'LTZ', lambda c: _result(c) == 'LTZ'),
        ('snow', 'SnowSampler._zone', 'TZ', lambda c: _result(c) == 'TZ'),
        ('snow', 'SnowSampler._zone', 'no zone', lambda c: _result(c) is None),
        (
            'snow',
            'SnowSampler._bound_format',
            'a zoned format',
            lambda c: _result(c) == '%Y-%m-%dT%H:%M:%S+00:00',
        ),
        (
            'snow',
            'SnowSampler._bound_format',
            'a naive format',
            lambda c: _result(c) == '%Y-%m-%d %H:%M:%S',
        ),
        (
            'duckdb',
            'time_column',
            'a zoned column',
            lambda c: _arg(c, 'zoned') is True,
        ),
        (
            'duckdb',
            '_duckdb_type',
            'TIMESTAMPTZ',
            lambda c: _result(c) == 'TIMESTAMPTZ',
        ),
        (
            'duckdb',
            'window',
            'zoned end times',
            lambda c: 'tz=' in c['args']['end_times']['$arrow']['type'],
        ),
        (
            'duckdb',
            'DuckDBTable._is_zoned_time',
            'a zoned column',
            lambda c: _result(c) is True,
        ),
        (
            'duckdb',
            'DuckDBTable._is_zoned_time',
            'a naive column',
            lambda c: _result(c) is False,
        ),
    ]
    out += [
        (
            b,
            'utils.prefilter_fraction',
            'a fraction',
            lambda c: _result(c) is not None,
        )
        for b in ('postgres', 'snow')
    ]
    out += [
        (
            b,
            'utils.prefilter_fraction',
            'no prefilter',
            lambda c: _result(c) is None,
        )
        for b in ('postgres', 'snow')
    ]
    return out


def test_the_synthetic_branches_are_captured(
    capture: dict[str, dict[str, Any]],
) -> None:
    missing = []
    for backend, function, what, holds in _checks():
        calls = _calls(capture, backend, function)
        ok = False
        for call in calls:
            try:
                ok = bool(holds(call))
            except Exception:  # noqa: BLE001 - a call this check can't read
                ok = False
            if ok:
                break
        if not ok:
            missing.append(f'{backend}: {function} with {what}')
    spellings = {
        json.dumps(call['result'], sort_keys=True)
        for call in _calls(
            capture, 'sqlite', 'SQLiteSampler._probe_time_spelling'
        )
    }
    if len(spellings) < 2:
        missing.append(
            'sqlite: SQLiteSampler._probe_time_spelling giving two spellings'
        )
    assert not missing, (
        f'{len(missing)} branches the capture does not reach:\n'
        + '\n'.join(missing)
    )
