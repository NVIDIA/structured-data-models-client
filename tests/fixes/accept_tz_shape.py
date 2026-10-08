# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.tz-shape: Databricks and Snowflake statements name UTC instants.

The same two predictions run through the public API over stub drivers
(``_warehouse_stub.py``) whose ``orders.ordered_at`` is declared with each
time type, at this checkout and, in a child process, at ``BASE``. The
statements that bound a time window (those defining ``__KUMO_END_TIME__``)
are compared, in order:

* Databricks, through the SQL connector and a notebook's Spark session:
  ``timestamp`` (LTZ), ``timestamp_ntz`` and ``date`` columns keep today's SQL,
  and every time bound, in the key list and outside it, names the same
  instant as today's naive string, written with ``+00:00``. A ``string``
  column keeps today's statements exactly.
* Snowflake: ``TIMESTAMP_NTZ`` and ``DATE`` columns keep today's statements
  exactly. ``TIMESTAMP_LTZ`` and ``TIMESTAMP_TZ`` columns compare against
  ``TO_TIMESTAMP_LTZ`` / ``TO_TIMESTAMP_TZ`` with an explicit format, both
  for the key list's times and for every outer ``?`` bound, and every time
  string carries ``+00:00``, names today's instant, and matches that format.

The end-to-end run on a local Spark is ``test_fix_spark_local.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from _fixes import base_env, run_child

_base_cache: dict[tuple[str, str], Any] = {}
_head_cache: dict[tuple[str, str], Any] = {}


@pytest.fixture(scope='module')
def base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp('base')


def _at_base(base_root: Path, kind: str, time_type: str) -> Any:
    key = (kind, time_type)
    if key not in _base_cache:
        _base_cache[key] = run_child(
            'warehouse', kind, time_type, env=base_env(base_root)
        )
    return _base_cache[key]


def _at_head(kind: str, time_type: str) -> Any:
    key = (kind, time_type)
    if key not in _head_cache:
        _head_cache[key] = run_child('warehouse', kind, time_type)
    return _head_cache[key]


def _windowed(result: Any) -> list[dict[str, Any]]:
    return [s for s in result['statements'] if '__KUMO_END_TIME__' in s['sql']]


def _times(statement: dict[str, Any]) -> tuple[list[Any], list[str]]:
    r"""A windowed statement's key list with its times taken out, and the
    times, in order: those in the key list, then the outer bounds."""
    params = statement['params'] or []
    assert params, statement['sql']
    items = json.loads(params[0])
    times: list[str] = []
    shape: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            rest = {}
            for name, value in item.items():
                if isinstance(value, str) and name != 'id':
                    times.append(value)
                    rest[name] = '<time>'
                else:
                    rest[name] = value
            shape.append(rest)
        elif isinstance(item, list):
            shape.append([item[0], *['<time>' for _ in item[1:]]])
            times += [str(v) for v in item[1:] if v is not None]
        else:
            shape.append(item)
    for bound in params[1:]:
        times.append(str(bound))
    return shape, times


def _same_instant(head: str, base: str) -> bool:
    got = pd.Timestamp(head)
    want = pd.Timestamp(base)
    if want.tzinfo is None:
        want = want.tz_localize('UTC')
    return got.tzinfo is not None and got == want


def _assert_utc_bounds(
    head: list[dict[str, Any]], base: list[dict[str, Any]], what: str
) -> None:
    assert len(head) == len(base) > 0, (
        f'{what}: {len(head)} windowed statements, {len(base)} at BASE'
    )
    for index, (got, want) in enumerate(zip(head, base, strict=True)):
        got_shape, got_times = _times(got)
        want_shape, want_times = _times(want)
        assert got_shape == want_shape, f'{what}, statement {index}'
        assert len(got_times) == len(want_times), f'{what}, statement {index}'
        for g, w in zip(got_times, want_times, strict=True):
            assert g.endswith('+00:00'), (
                f'{what}, statement {index}: bound {g!r} names no UTC '
                f'instant (BASE sent {w!r})'
            )
            assert _same_instant(g, w), (
                f'{what}, statement {index}: bound {g!r} is not the instant '
                f'BASE meant by {w!r}'
            )


# Databricks #################################################################


@pytest.mark.parametrize('kind', ('databricks', 'spark'))
@pytest.mark.parametrize('time_type', ('timestamp', 'timestamp_ntz', 'date'))
def test_databricks_bounds_name_utc_instants(
    base_root: Path, kind: str, time_type: str
) -> None:
    head = _windowed(_at_head(kind, time_type))
    base = _windowed(_at_base(base_root, kind, time_type))
    assert [s['sql'] for s in head] == [s['sql'] for s in base], (
        'the windowed SQL changed; only its bounds should'
    )
    _assert_utc_bounds(head, base, f'{kind} {time_type}')


@pytest.mark.parametrize('kind', ('databricks', 'spark'))
def test_databricks_string_time_column_is_unchanged(
    base_root: Path, kind: str
) -> None:
    head = _windowed(_at_head(kind, 'string'))
    base = _windowed(_at_base(base_root, kind, 'string'))
    assert head and head == base


def test_databricks_payloads_do_not_change(base_root: Path) -> None:
    r"""Over the stub, which compares in UTC, the fix changes no payload."""
    for kind in ('databricks', 'spark'):
        for time_type in ('timestamp', 'timestamp_ntz', 'date'):
            assert (
                _at_head(kind, time_type)['payloads']
                == _at_base(base_root, kind, time_type)['payloads']
            ), f'{kind} {time_type}'


# Snowflake ##################################################################

#: Snowflake format elements, as the regular expressions they match.
_FORMAT_ELEMENTS = (
    ('YYYY', r'\d{4}'),
    ('HH24', r'\d{2}'),
    ('TZH', r'[+-]\d{2}'),
    ('TZM', r'\d{2}'),
    ('MM', r'\d{2}'),
    ('DD', r'\d{2}'),
    ('MI', r'\d{2}'),
    ('SS', r'\d{2}'),
)


def _format_regex(text: str) -> re.Pattern[str]:
    r"""A Snowflake timestamp format as a regular expression over the text
    it reads; refuses an element this check doesn't know."""
    out = ''
    rest = text
    while rest:
        if rest.startswith('"'):
            end = rest.index('"', 1)
            out += re.escape(rest[1:end])
            rest = rest[end + 1 :]
            continue
        found = re.match(r'FF[0-9]?', rest)
        if found:
            out += r'(?:\d+)?'
            rest = rest[found.end() :]
            continue
        for element, pattern in _FORMAT_ELEMENTS:
            if rest.startswith(element):
                out += pattern
                rest = rest[len(element) :]
                break
        else:
            if rest[0].isalpha():
                raise AssertionError(f'unknown format element in {text!r}')
            out += re.escape(rest[0])
            rest = rest[1:]
    return re.compile(out)


#: ``TO_TIMESTAMP_LTZ(x, 'format')`` or ``TO_TIMESTAMP_TZ(...)``: the
#: function, its first argument, and the format.
_CONVERSION = re.compile(
    r'(TO_TIMESTAMP_(?:LTZ|TZ))\(\s*'
    r'([^,()]+(?:\([^()]*\))?)\s*,\s*'
    r"'([^']+)'\s*\)",
    re.IGNORECASE,
)


@pytest.mark.parametrize('time_type', ('TIMESTAMP_NTZ(9)', 'DATE'))
def test_snowflake_naive_and_date_columns_are_unchanged(
    base_root: Path, time_type: str
) -> None:
    head = _windowed(_at_head('snowflake', time_type))
    base = _windowed(_at_base(base_root, 'snowflake', time_type))
    assert head and head == base
    for statement in head:
        assert 'TO_TIMESTAMP_' not in statement['sql'].upper()


@pytest.mark.parametrize(
    ('time_type', 'function'),
    [
        ('TIMESTAMP_LTZ(9)', 'TO_TIMESTAMP_LTZ'),
        ('TIMESTAMP_TZ(9)', 'TO_TIMESTAMP_TZ'),
    ],
)
def test_snowflake_zoned_columns_compare_against_typed_instants(
    base_root: Path, time_type: str, function: str
) -> None:
    head = _windowed(_at_head('snowflake', time_type))
    base = _windowed(_at_base(base_root, 'snowflake', time_type))
    _assert_utc_bounds(head, base, f'snowflake {time_type}')
    for index, statement in enumerate(head):
        sql = statement['sql']
        found = _CONVERSION.findall(sql)
        assert found, f'statement {index} converts no bound:\n{sql}'
        assert {f.upper() for f, _, _ in found} == {function}, (
            f'statement {index} should convert with {function} only:\n{sql}'
        )
        assert 'TIMESTAMP_NTZ' not in sql.upper(), (
            f'statement {index} still reads a bound as naive:\n{sql}'
        )
        formats = {fmt for _, _, fmt in found}
        assert len(formats) == 1, formats
        fmt = formats.pop()
        assert 'TZH' in fmt.upper(), f'the format {fmt!r} reads no offset'
        pattern = _format_regex(fmt)
        _, times = _times(statement)
        for value in times:
            assert pattern.fullmatch(value), (
                f'statement {index}: {value!r} does not match {fmt!r}'
            )
        # The key list's two times and every outer `?` bound are converted;
        # the only bare placeholder left is the key list itself.
        converted = [arg for _, arg, _ in found]
        assert sum(arg.strip().startswith('?') for arg in converted) == (
            len(statement['params']) - 1
        ), f'statement {index}: not every outer bound is converted:\n{sql}'
        assert len(converted) - (len(statement['params']) - 1) >= 1, (
            f"statement {index}: the key list's times are not converted"
        )
        stripped = _CONVERSION.sub('', sql)
        assert stripped.count('?') == 1, (
            f'statement {index}: a bound is compared unconverted:\n{sql}'
        )
