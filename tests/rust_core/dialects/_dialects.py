# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 6 acceptance checks share: the backends and their
functions, the capture, today's Python answer for a call, the Rust answer
through ``sdm_core._native.dialect_render``, and the comparison.

The interface these rely on is ``CONTRACT.md``, beside this file.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAPTURE_DIR = REPO / 'tests' / 'parity' / 'dialects'
CAPTURE_SCRIPT = REPO / 'scripts' / 'parity' / 'capture_dialects.py'
TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'
ENGINE = 'sdm_relational_engine.rfm'

BACKENDS = ('databricks', 'duckdb', 'postgres', 'snow', 'sqlite')

#: The dialect of each transcript backend that runs SQL.
TRANSCRIPT_DIALECT = {
    'databricks': 'databricks',
    'duckdb': 'duckdb',
    'lakebase': 'postgres',
    'postgres': 'postgres',
    'snowflake': 'snow',
    'sqlite': 'sqlite',
}

#: The SQL fragments of ``base/utils.py`` the dialects take over.
UTILS_FRAGMENTS = (
    'ordered_sample_sql',
    'key_count_select',
    'count_distinct_values',
    'repeated_key_select',
    'sample_order_terms',
    'percent_literal',
    'prefilter_fraction',
)

#: The moved helpers each backend's capture must hold (``CONTRACT.md``).
MOVED_HELPERS = {
    'databricks': (
        'DatabricksSampler._bound_format',
        'DatabricksTable._is_large',
    ),
    'duckdb': ('DuckDBTable._is_zoned_time',),
    'postgres': (
        'PostgresSampler._utc_bounds',
        'PostgresTable._prefiltered_sample_source',
    ),
    'snow': (
        'SnowSampler._zone',
        'SnowSampler._bound_format',
        'SnowTable._prefilter',
        'SnowTable._listing',
    ),
    'sqlite': ('SQLiteSampler._probe_time_spelling', '_TimeSpelling.spell'),
}

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'scripts' / 'parity'))

from _codec import decode, encode, first_difference  # noqa: E402

# --- The Python side -----------------------------------------------------------


def dialect_module(backend: str) -> Any:
    return importlib.import_module(f'{ENGINE}.backend.{backend}.dialect')


def backend_module(backend: str, name: str) -> Any:
    r"""The backend's ``table`` or ``sampler`` module."""
    return importlib.import_module(f'{ENGINE}.backend.{backend}.{name}')


def dialect_functions(backend: str) -> dict[str, Any]:
    r"""Every function defined at the top level of the backend's
    ``dialect.py``, by name."""
    module = dialect_module(backend)
    return {
        name: value
        for name, value in vars(module).items()
        if inspect.isfunction(value) and value.__module__ == module.__name__
    }


def imports(backend: str, function: Any) -> bool:
    r"""Whether the backend's ``dialect.py`` or ``table.py`` imports
    ``function`` by name."""
    for name in ('dialect', 'table'):
        module = backend_module(backend, name)
        if any(value is function for value in vars(module).values()):
            return True
    return False


def plain_function(backend: str, name: str) -> Any | None:
    r"""The Python function a captured name stands for, or ``None`` for a
    moved helper."""
    if name.startswith('utils.'):
        module = importlib.import_module(f'{ENGINE}.base.utils')
        return getattr(module, name.split('.', 1)[1], None)
    if name == 'sql.quote_ident':
        from sdm_connectors.sql import quote_ident

        return quote_ident
    if name.startswith('view_expr.'):
        module = importlib.import_module(f'{ENGINE}.view_expr')
        return getattr(module, name.split('.', 1)[1], None)
    if '.' in name:
        return None
    return dialect_functions(backend).get(name)


def is_statement(function: Any) -> bool:
    annotation = inspect.signature(function).return_annotation
    return annotation in ('Statement', 'Statement | None') or (
        getattr(annotation, '__name__', None) == 'Statement'
    )


def output(function: Any, value: Any) -> dict[str, Any]:
    r"""A call's output as the capture writes it."""
    if is_statement(function):
        from _parity.values import encode as encode_params

        sql, params = value
        return {
            'sql': sql,
            'params': None if params is None else encode_params(list(params)),
        }
    return {'result': encode(value)}


def python_call(backend: str, name: str, args: dict[str, Any]) -> Any:
    r"""What today's Python gives for a captured call that is not a moved
    helper, as the capture writes it."""
    function = plain_function(backend, name)
    assert function is not None, f'{backend}: no function {name}'
    return output(function, function(**decode(args)))


def python_randint(args: dict[str, Any]) -> dict[str, Any]:
    from sdm_relational_engine.rfm.base.utils import random_state

    given = decode(args)
    rng = random_state(given['random_seed'])
    drawn = rng.randint(given['low'], given['high'], size=given['size'])
    return {'result': encode(drawn.tolist())}


def expected(call: dict[str, Any]) -> dict[str, Any]:
    r"""A captured call's output."""
    if 'result' in call:
        return {'result': call['result']}
    return {'sql': call['sql'], 'params': call['params']}


# --- The capture ---------------------------------------------------------------


def load_capture(root: Path = CAPTURE_DIR) -> dict[str, dict[str, Any]]:
    r"""Every backend's capture file, read, by backend."""
    found = sorted(p.stem for p in root.glob('*.json'))
    assert found == sorted(BACKENDS), (
        f'{root} holds {found}, not one file per backend {sorted(BACKENDS)}'
    )
    out = {}
    for backend in BACKENDS:
        data = json.loads((root / f'{backend}.json').read_text())
        assert isinstance(data, dict), f'{backend}.json is not an object'
        assert data.get('format') == 1, f'{backend}.json: format is not 1'
        assert data.get('backend') == backend, (
            f'{backend}.json: backend is {data.get("backend")!r}'
        )
        assert isinstance(data.get('calls'), list) and data['calls'], (
            f'{backend}.json has no calls'
        )
        out[backend] = data
    return out


def run_capture(out: Path, hashseed: str) -> None:
    r"""Run the capture script into ``out`` under ``PYTHONHASHSEED``."""
    assert CAPTURE_SCRIPT.is_file(), f'no capture script at {CAPTURE_SCRIPT}'
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = hashseed
    proc = subprocess.run(
        [sys.executable, str(CAPTURE_SCRIPT), '--out', str(out)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, (
        f'the capture exited {proc.returncode}\n--- stdout\n'
        f'{proc.stdout[-4000:]}\n--- stderr\n{proc.stderr[-8000:]}'
    )


def capture_module() -> Any:
    r"""``scripts/parity/capture_dialects.py``, imported."""
    spec = importlib.util.spec_from_file_location(
        '_p6_capture_dialects', CAPTURE_SCRIPT
    )
    assert spec and spec.loader, f'cannot import {CAPTURE_SCRIPT}'
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- The Rust core, through the binding --------------------------------------


_NATIVE: dict[str, Any] = {}


def native() -> Any:
    r"""``sdm_core._native``, freshly built from this checkout and installed
    into this environment, once per process.

    A build that fails, or a module without ``dialect_render``, fails every
    later call the same way, without building again.
    """
    if 'error' in _NATIVE:
        raise AssertionError(_NATIVE['error'])
    if 'module' in _NATIVE:
        return _NATIVE['module']
    try:
        from _rc import install_local_wheel

        install_local_wheel()
        importlib.invalidate_caches()
        module = importlib.import_module('sdm_core._native')
        assert hasattr(module, 'dialect_render'), (
            'sdm_core._native has no dialect_render'
        )
    except Exception as error:  # noqa: BLE001 - kept for every later call
        _NATIVE['error'] = f'the binding is not usable: {error}'
        raise AssertionError(_NATIVE['error']) from error
    _NATIVE['module'] = module
    return module


def rust_call(backend: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    r"""What the Rust core gives for a call, through the binding, in the
    capture's form."""
    got = native().dialect_render(backend, name, json.dumps(args))
    assert isinstance(got, tuple) and len(got) == 2, (
        f'dialect_render returned {got!r}, not a 2-tuple'
    )
    sql, text = got
    assert isinstance(text, str), f'dialect_render gave {type(text)} as JSON'
    if sql is None:
        return {'result': json.loads(text)}
    assert isinstance(sql, str), f'dialect_render gave {type(sql)} as SQL'
    return {'sql': sql, 'params': json.loads(text)}


def mismatch(want: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""How ``got`` differs from ``want``: the SQL byte for byte, then the
    parameters or result by value and JSON type."""
    if set(got) != set(want):
        return f'gave {sorted(got)} where {sorted(want)} was captured'
    if 'sql' in want and got['sql'] != want['sql']:
        a, b = want['sql'], got['sql']
        at = next(
            (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
            min(len(a), len(b)),
        )
        return (
            f'SQL differs at character {at}: captured '
            f'{a[max(0, at - 40) : at + 40]!r}, got {b[max(0, at - 40) : at + 40]!r}'
        )
    key = 'params' if 'sql' in want else 'result'
    found = first_difference(want[key], got[key], f'${key}')
    if found:
        return found
    return None


def describe(backend: str, index: int, call: dict[str, Any]) -> str:
    sources = call.get('sources') or ['?']
    more = f' (+{len(sources) - 1})' if len(sources) > 1 else ''
    return f'{backend}#{index} {call.get("function")} [{sources[0]}{more}]'
