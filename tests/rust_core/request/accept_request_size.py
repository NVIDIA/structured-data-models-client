# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.size: the core counts a payload's bytes as Python's
``payload_size_bytes`` does, and refuses at the same byte.

- 2,000 generated payloads (seed 20261023), holding text with control,
  non-ASCII, astral and escaped characters and floats whose ``repr`` is
  awkward, written as JSON in spellings other than Python's, give through
  ``sdm_core._native.payload_size_bytes`` what Python gives.
- Frames whose relational payload is 30 MiB less one byte, exactly 30 MiB,
  and 30 MiB and one byte, by Python's count, give through the host what
  they give today: the first two are gathered and sent alike, the third is
  refused alike.
"""

from __future__ import annotations

import json
import math
import random
import struct
from typing import Any

import pytest
from _request import recorder, run_flow, short

SEED = 20261023
COUNT = 2000
LIMIT = 30 * 1024 * 1024

ASCII = [chr(c) for c in range(32, 127)]
SPECIAL = ['\x00', '\x01', '\x08', '\t', '\n', '\r', '\x1f', '\x7f', '"', '\\']
NON_ASCII = ['é', 'ü', 'ß', 'ÿ', '\x80', '\xa0', '☃', '中', ' ', ' ']
NON_BMP = ['﻿', '￿', '😀', '𝄞', '\U0010ffff']
AWKWARD = [
    0.1,
    0.1 + 0.2,
    1e16,
    1e-7,
    1e-5,
    1.5e-5,
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    -0.0,
    0.0,
    1.0,
    -1.0,
    123456789012345680.0,
    1e21,
    1e22,
    9007199254740993.0,
    3.141592653589793,
    100.0,
    -2.5e-310,
]


def _text(rng: random.Random) -> str:
    pools = [ASCII, ASCII, SPECIAL, NON_ASCII, NON_BMP]
    return ''.join(
        rng.choice(rng.choice(pools)) for _ in range(rng.randint(0, 24))
    )


def _float(rng: random.Random) -> float:
    if rng.random() < 0.5:
        return rng.choice(AWKWARD)
    while True:
        (value,) = struct.unpack(
            '<d', rng.getrandbits(64).to_bytes(8, 'little')
        )
        if math.isfinite(value):
            return value


def _value(rng: random.Random, depth: int) -> Any:
    kind = rng.choice(
        ['text', 'float', 'float', 'int', 'bool', 'null', 'list', 'object']
    )
    if depth >= 3 and kind in ('list', 'object'):
        kind = 'text'
    if kind == 'text':
        return _text(rng)
    if kind == 'float':
        return _float(rng)
    if kind == 'int':
        return rng.choice(
            [
                0,
                -1,
                2**53 + 1,
                2**63 - 1,
                -(2**63),
                rng.randint(-(10**9), 10**9),
            ]
        )
    if kind == 'bool':
        return rng.random() < 0.5
    if kind == 'null':
        return None
    if kind == 'list':
        return [_value(rng, depth + 1) for _ in range(rng.randint(0, 5))]
    return _object(rng, depth + 1)


def _object(rng: random.Random, depth: int) -> dict[str, Any]:
    return {_text(rng): _value(rng, depth) for _ in range(rng.randint(0, 5))}


def _spell_float(rng: random.Random, value: float) -> str:
    r"""``value`` as a JSON number of another spelling than Python's
    ``repr``, still a float (an exponent or a point), parsing to the same
    double."""
    for _ in range(4):
        text = rng.choice(
            [
                repr(value),
                f'{value:.17e}',
                f'{value:.17E}',
                f'{value:.{rng.randint(17, 24)}e}',
            ]
        )
        if float(text) == value and math.copysign(1, float(text)) == (
            math.copysign(1, value)
        ):
            return text
    return repr(value)


def spell(rng: random.Random, value: Any, indent: int = 0) -> str:
    r"""JSON text of ``value`` in a spelling picked by ``rng``: other float
    forms, escaped or raw non-ASCII, and other spacing."""
    gap = rng.choice(['', ' ', '\n  '])
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, float):
        return _spell_float(rng, value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=rng.random() < 0.3)
    if isinstance(value, list):
        return '[' + gap + (',' + gap).join(spell(rng, v) for v in value) + ']'
    items = (
        f'{json.dumps(k, ensure_ascii=rng.random() < 0.3)}{gap}:{gap}'
        f'{spell(rng, v)}'
        for k, v in value.items()
    )
    return '{' + gap + (',' + gap).join(items) + gap + '}'


def test_generated_payloads_count_alike() -> None:
    from sdm_relational_engine.rfm.inference.payload import payload_size_bytes

    module = recorder().module
    rng = random.Random(SEED)
    problems = []
    for i in range(COUNT):
        payload = _object(rng, 0)
        text = spell(rng, payload)
        parsed = json.loads(text)
        want = payload_size_bytes(parsed)
        assert want == payload_size_bytes(payload), 'the spelling is unfaithful'
        try:
            got = module.payload_size_bytes(text)
        except Exception as error:  # noqa: BLE001 - reported
            problems.append(f'payload {i}: raised {short(str(error))}')
            continue
        if got != want or type(got) is not int:
            problems.append(
                f'payload {i}: {got!r} instead of {want}: {short(text)}'
            )
    assert not problems, '\n'.join(problems[:30])


@pytest.mark.parametrize(
    'text',
    [
        '{"a": NaN}',
        '{"a": [1, Infinity]}',
        '{"a": -Infinity}',
        '[1, 2]',
        '"text"',
        '3',
        'null',
        '{"a": ',
        '',
        '{"a": 1} trailing',
    ],
)
def test_text_that_is_not_a_payload_is_refused(text: str) -> None:
    module = recorder().module
    with pytest.raises(ValueError):
        module.payload_size_bytes(text)


# --- At 30 MiB ----------------------------------------------------------------------

QUERY = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
USERS = 1000
ORDERS_PER_USER = 6
#: The most characters added to one user's note, below the text cap.
NOTE_ROOM = 200


def _calls(gather_only: bool) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = [
        {'call': 'Schema.from_data', 'args': {'tables': ['users', 'orders']}},
        {'call': 'PredictiveQuery', 'args': {'query': QUERY}},
        {
            'call': 'PredictiveQuery.materialize',
            'args': {
                'indices': [1],
                'num_context_samples': 1000,
                'anchor_time': {'$timestamp': '2024-09-01T00:00:00'},
            },
        },
    ]
    if not gather_only:
        calls.append(
            {'call': 'SDMClient.predict', 'args': {'model': 'kumo-relational'}}
        )
    return calls


#: Float columns of awkward magnitudes every frame carries.
FLOATS = 4


def _frames(digits: list[int], notes: list[str]) -> dict[str, Any]:
    r"""The users and their orders: notes as given, ``FLOATS`` float
    columns, and one integer column per entry of ``digits``, each value of
    exactly that many digits."""
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(SEED % 2**32)
    n = USERS
    rows = ORDERS_PER_USER * n
    users = pd.DataFrame(
        {
            'user_id': np.arange(n),
            'age': rng.randint(18, 80, n),
            'note': notes,
        }
    )
    orders = {
        'order_id': np.arange(rows),
        'user_id': rng.randint(0, n, rows),
        'at': pd.Timestamp('2024-01-01')
        + pd.to_timedelta(rng.randint(0, 300 * 24, rows), unit='h'),
        'amount': rng.rand(rows) * 100,
    }
    for j in range(FLOATS):
        orders[f'f{j}'] = rng.rand(rows) * 10.0 ** rng.randint(-8, 17)
    for j, d in enumerate(digits):
        orders[f'g{j:03d}'] = rng.randint(
            10 ** (d - 1), 10**d, rows, dtype=np.int64
        )
    return {'users': users, 'orders': pd.DataFrame(orders)}


def _measure(frames: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    r"""The size today's Python measures of the one payload, and the
    payload, from gathering alone."""
    import sdm_relational_engine.rfm.rfm as rfm

    seen: list[tuple[int, dict[str, Any]]] = []
    original = rfm.payload_size_bytes

    def spy(payload: dict[str, Any]) -> int:
        size = original(payload)
        seen.append((size, payload))
        return size

    rfm.payload_size_bytes = spy
    try:
        flow = run_flow(_calls(True), frames, core=False)
    finally:
        rfm.payload_size_bytes = original
    assert len(seen) == 1, (
        f'{len(seen)} payloads were measured; gathering gave {flow.error}'
    )
    return seen[0]


def _orders_rows(payload: dict[str, Any]) -> int:
    return sum(
        len(payload[section]['related_tables']['orders']['rows'])
        for section in ('context', 'predict')
    )


def _users_once(payload: dict[str, Any]) -> list[int]:
    seen: dict[int, int] = {}
    for section in ('context', 'predict'):
        table = payload[section]['related_tables']['users']
        at = table['columns'].index('user_id')
        for row in table['rows']:
            seen[row[at]] = seen.get(row[at], 0) + 1
    return sorted(user for user, count in seen.items() if count == 1)


def _notes() -> list[str]:
    return [f'wé{i} lorem ipsum ☃ dolor' for i in range(USERS)]


def _grow(notes: list[str], users: list[int], extra: int) -> list[str]:
    r"""``notes`` with ``extra`` more ASCII characters, spread over the
    first word of the notes of ``users``, each of which the payload holds
    once."""
    out = list(notes)
    for user in users:
        if extra <= 0:
            break
        room = NOTE_ROOM - (len(out[user]) - len(notes[user]))
        take = min(room, extra)
        if take > 0:
            head, _, tail = out[user].partition(' ')
            out[user] = f'{head}{"x" * take} {tail}'
            extra -= take
    assert extra == 0, f'{extra} bytes are left to add'
    return out


def _columns_for(gap: int, rows: int, overhead: int) -> list[int]:
    r"""Digit counts of integer columns adding at most ``gap`` bytes and
    leaving less than two rows' worth: an integer column of ``d`` digits
    adds ``rows * (d + 1) + overhead``. Fifteen digits at most, since the
    payload writes an integer past 2**53 as text."""
    digits: list[int] = []
    full = rows * 16 + overhead
    while gap >= full:
        digits.append(15)
        gap -= full
    last = (gap - overhead) // rows - 1
    if last >= 1:
        digits.append(min(last, 15))
    return digits


@pytest.fixture(scope='module')
def boundary() -> dict[int, dict[str, Any]]:
    r"""Frames whose payload is ``LIMIT + d`` bytes, by Python's count, for
    ``d`` of -1, 0 and 1."""
    notes = _notes()
    base, payload = _measure(_frames([], notes))
    rows = _orders_rows(payload)
    one, _ = _measure(_frames([1], notes))
    overhead = one - base - 2 * rows
    room = NOTE_ROOM * len(_users_once(payload))
    digits = _columns_for(LIMIT - 1 - base - rows, rows, overhead)
    size, payload = _measure(_frames(digits, notes))
    assert size <= LIMIT - 1, f'{size} bytes overshoot {LIMIT - 1}'
    assert LIMIT - 1 - size <= room, f'{LIMIT - 1 - size} bytes exceed {room}'
    out: dict[int, dict[str, Any]] = {}
    grown = notes
    current = size
    for d in (-1, 0, 1):
        target = LIMIT + d
        for _ in range(6):
            if current == target:
                break
            assert current < target, f'{current} passed {target}'
            grown = _grow(grown, _users_once(payload), target - current)
            current, payload = _measure(_frames(digits, grown))
        assert current == target, f'could not reach {target}: {current}'
        out[d] = _frames(digits, grown)
    return out


def _compare(today: Any, core: Any) -> list[str]:
    problems = [f'[{a}] {m}' for a, m in core.problems]
    if (today.error_at, today.error) != (core.error_at, core.error):
        problems.append(
            f'today ends at call {today.error_at} with '
            f'{short(today.error, 600)}, the core at call {core.error_at} '
            f'with {short(core.error, 600)}'
        )
    if json.dumps(today.payloads, sort_keys=True) != json.dumps(
        core.payloads, sort_keys=True
    ):
        problems.append('the payloads sent differ')
    if today.warnings != core.warnings:
        problems.append(
            f'the warnings differ: {short(today.warnings)} became '
            f'{short(core.warnings)}'
        )
    return problems


@pytest.mark.parametrize('d', [-1, 0, 1])
def test_the_limit_is_where_it_is_today(
    boundary: dict[int, dict[str, Any]], d: int
) -> None:
    frames = boundary[d]
    today = run_flow(_calls(False), frames, core=False)
    if d <= 0:
        assert today.error is None and len(today.payloads) == 1, today.error
    else:
        assert today.error_at == 2 and 'exceeds the 30MB limit' in str(
            today.error
        ), today.error
    core = run_flow(_calls(False), frames, core=True)
    problems = _compare(today, core)
    assert not problems, '\n'.join(problems)
