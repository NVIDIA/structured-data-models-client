# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P6.dialect-calls, through Python and the binding: every captured call is
run through both implementations.

- Today's Python, called with the decoded arguments, gives exactly the
  captured SQL and parameters or result. A moved helper is replayed through
  ``capture_dialects.replay_helper``, which must reach the engine's own
  attribute and return what it returned.
- ``sdm_core._native.dialect_render``, freshly built, gives the same: the SQL
  byte for byte, and the parameters or result by value and JSON type.
- The binding refuses what it cannot read with ``ValueError``, never a panic.
"""

from __future__ import annotations

import contextlib
import functools
import json
from collections.abc import Iterator
from typing import Any

import pytest
from _codec import same
from _dialects import (
    BACKENDS,
    backend_module,
    capture_module,
    describe,
    encode,
    expected,
    load_capture,
    mismatch,
    native,
    plain_function,
    python_call,
    rust_call,
)


@pytest.fixture(scope='module')
def capture() -> dict[str, dict[str, Any]]:
    return load_capture()


def _owner(backend: str, name: str) -> tuple[type, str]:
    owner, attribute = name.split('.', 1)
    for module in ('table', 'sampler'):
        found = vars(backend_module(backend, module)).get(owner)
        if isinstance(found, type):
            assert attribute in vars(found), f'{name} is not defined on {owner}'
            return found, attribute
    raise AssertionError(
        f'{backend}: no class {owner} in table.py or sampler.py'
    )


@contextlib.contextmanager
def _spy(cls: type, attribute: str) -> Iterator[list[Any]]:
    r"""Record what ``cls.attribute`` returns while the block runs."""
    original = vars(cls)[attribute]
    returned: list[Any] = []

    def wrap(function: Any) -> Any:
        @functools.wraps(function)
        def spied(*args: Any, **kwargs: Any) -> Any:
            value = function(*args, **kwargs)
            returned.append(value)
            return value

        return spied

    replacement: Any
    if isinstance(original, functools.cached_property):
        replacement = functools.cached_property(wrap(original.func))
        replacement.__set_name__(cls, attribute)
    elif isinstance(original, property):
        replacement = property(wrap(original.fget), original.fset)
    elif isinstance(original, staticmethod):
        replacement = staticmethod(wrap(original.__func__))
    elif isinstance(original, classmethod):
        replacement = classmethod(wrap(original.__func__))
    else:
        replacement = wrap(original)
    setattr(cls, attribute, replacement)
    try:
        yield returned
    finally:
        setattr(cls, attribute, original)


def test_today_python_gives_every_captured_call(
    capture: dict[str, dict[str, Any]],
) -> None:
    failures, checked = [], 0
    for backend, data in capture.items():
        for i, call in enumerate(data['calls']):
            if plain_function(backend, call['function']) is None:
                continue
            checked += 1
            try:
                got = python_call(backend, call['function'], call['args'])
            except Exception as error:  # noqa: BLE001 - reported per call
                failures.append(
                    f'{describe(backend, i, call)}: raised {error!r}'
                )
                continue
            problem = mismatch(expected(call), got)
            if problem:
                failures.append(f'{describe(backend, i, call)}: {problem}')
    assert checked, 'no captured call to check'
    assert not failures, (
        f"today's Python differs on {len(failures)} of {checked} captured "
        f'calls:\n' + '\n'.join(failures[:40])
    )


def test_moved_helpers_replay_through_the_engine(
    capture: dict[str, dict[str, Any]],
) -> None:
    replay = capture_module().replay_helper
    failures, checked = [], 0
    for backend, data in capture.items():
        for i, call in enumerate(data['calls']):
            if plain_function(backend, call['function']) is not None:
                continue
            checked += 1
            where = describe(backend, i, call)
            try:
                cls, attribute = _owner(backend, call['function'])
                with _spy(cls, attribute) as returned:
                    got = replay(backend, call['function'], call['args'])
            except Exception as error:  # noqa: BLE001 - reported per call
                failures.append(f'{where}: raised {error!r}')
                continue
            if not returned:
                failures.append(f'{where}: the engine attribute was not called')
                continue
            if not same(encode(got), call['result']):
                failures.append(
                    f'{where}: replay_helper gave {encode(got)!r}, captured '
                    f'{call["result"]!r}'
                )
            elif not any(same(encode(v), call['result']) for v in returned):
                failures.append(
                    f'{where}: the engine attribute never returned the '
                    f'captured result'
                )
    assert checked, 'no moved helper captured'
    assert not failures, (
        f'{len(failures)} of {checked} moved-helper calls do not replay:\n'
        + '\n'.join(failures[:40])
    )


def test_the_binding_gives_every_captured_call(
    capture: dict[str, dict[str, Any]],
) -> None:
    native()
    failures, checked = [], 0
    for backend, data in capture.items():
        for i, call in enumerate(data['calls']):
            checked += 1
            try:
                got = rust_call(backend, call['function'], call['args'])
            except Exception as error:  # noqa: BLE001 - reported per call
                failures.append(
                    f'{describe(backend, i, call)}: raised {error!r}'
                )
                continue
            problem = mismatch(expected(call), got)
            if problem:
                failures.append(f'{describe(backend, i, call)}: {problem}')
    assert not failures, (
        f'the binding differs on {len(failures)} of {checked} captured '
        f'calls:\n' + '\n'.join(failures[:40])
    )


@pytest.mark.parametrize(
    ('backend', 'function', 'args_json'),
    [
        ('mysql', 'list_tables', '{}'),
        ('snowflake', 'current_namespace', '{}'),
        ('sqlite', 'no_such_function', '{}'),
        ('sqlite', 'SnowTable._prefilter', '{}'),
        ('sqlite', 'list_tables', 'not json'),
        ('sqlite', 'list_tables', '[]'),
        ('sqlite', 'table_info', '{}'),
        ('sqlite', 'table_info', '{"table": 5}'),
        ('postgres', 'set_seed', '{"random_seed": "7"}'),
        ('duckdb', 'rows_by_key', '{"table": "t"}'),
        (
            'sqlite',
            'rows_by_key',
            json.dumps(
                {
                    'table': 't',
                    'key': 'k',
                    'ids': {'$arrow': {'type': 'no_such_type', 'values': []}},
                    'projections': ['x'],
                    'unique': True,
                    'tie_break': [],
                    'count_matches': False,
                }
            ),
        ),
    ],
)
def test_the_binding_refuses_what_it_cannot_read(
    backend: str, function: str, args_json: str
) -> None:
    with pytest.raises(ValueError):
        native().dialect_render(backend, function, args_json)


@pytest.mark.parametrize('backend', BACKENDS)
def test_shared_functions_answer_under_every_backend(backend: str) -> None:
    args = {'ident': 'a"b', 'char': '"'}
    assert rust_call(backend, 'sql.quote_ident', args) == {'result': '"a""b"'}
    args = {
        'source': 't',
        'projections': ['x'],
        'order_terms': [],
        'num_rows': 3,
    }
    assert rust_call(backend, 'utils.ordered_sample_sql', args) == {
        'result': 'SELECT x FROM t LIMIT 3'
    }
