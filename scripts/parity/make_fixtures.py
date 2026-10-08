#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Write the parity fixtures from the test files they are converted from.

    make_fixtures.py [--check]

Each case below is a case of a test in one of the files ``SOURCES`` names: the
same inputs, run through the operation that test exercises
(``tests/parity/_fixture_ops.py``). Its expected output is what today's code
gives. Where the test states the answer, the case states it too (``want``),
and writing stops if today's code disagrees, so a fixture cannot quietly
record a regression. The original tests are not changed.

``--check`` writes nothing in the repository and fails if the stored fixtures
differ from what would be written. For each file that differs it says which
cases changed, and where in each, and keeps what it would have written in a
temporary directory it names, so a difference that comes and goes can be read
after the fact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tempfile
import uuid
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PARITY = REPO / 'tests' / 'parity'
sys.path.insert(0, str(PARITY))

from _fixture_ops import (  # noqa: E402
    FixtureError,
    _client_run,
    build_frame,
    build_series,
    encode,
    run,
    same,
    warned,
)

FIXTURES = PARITY / 'fixtures'
_MISSING = object()
#: Cases whose output disagrees with what their test says, under
#: ``--keep-going``, which reports them all rather than stopping at the first.
DISAGREEING: list[str] = []
KEEP_GOING = False


def _disagree(message: str) -> None:
    if not KEEP_GOING:
        raise SystemExit(message)
    DISAGREEING.append(message)


CLIENT = 'packages/structured-data-models-client/tests'
ENGINE = 'packages/structured-data-models-relational-engine/tests'
SOURCES = {
    'dates': f'{CLIENT}/test_dates_the_column_decides.py',
    'wire': f'{CLIENT}/test_wire_integer_exactness.py',
    'time-and-keys': f'{CLIENT}/test_inferred_time_and_keys.py',
    'relationships': f'{CLIENT}/test_inferred_relationships.py',
    'decomposition': f'{CLIENT}/test_decomposition_is_faithful.py',
    'binary-target': f'{CLIENT}/test_a_binary_target_is_a_classification.py',
    'json-payload': f'{ENGINE}/test_json_payload.py',
    'dtype-fidelity': f'{ENGINE}/test_payload_dtype_fidelity.py',
    'composite-key': f'{ENGINE}/test_composite_primary_key.py',
    'pandas-executor': f'{ENGINE}/query/test_pandas_executor.py',
    'local-sampler': f'{ENGINE}/backend/local/test_local_sampler.py',
    'local-graph-store': f'{ENGINE}/backend/local/test_local_graph_store.py',
    'sampler-utils': f'{ENGINE}/base/test_utils.py',
    'driver': 'tests/rust_core/rfm/test_driver_fixtures.py',
    'sampler-sequences': 'tests/rust_core/rfm/test_sequences.py',
}
#: The message fixtures come from the catalog of warnings and refusals: each
#: case reaches one or more of its entries (``_fixture_messages.py``).
CATALOG_SOURCE = 'packages/structured-data-models-client/diagnostics.json'


def series(values: Any, dtype: str | None = None) -> dict[str, Any]:
    r"""A series as fixture JSON, checked to rebuild as itself."""
    made = (
        values
        if isinstance(values, pd.Series)
        else pd.Series(values, dtype=dtype)
    )
    spec = {
        'dtype': str(made.dtype),
        'values': [encode(v) for v in made.tolist()],
    }
    rebuilt = build_series(spec)
    pd.testing.assert_series_equal(
        rebuilt.reset_index(drop=True),
        made.reset_index(drop=True),
        check_names=False,
    )
    return spec


def frame(data: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    made = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    spec = {'columns': {str(c): series(made[c]) for c in made.columns}}
    pd.testing.assert_frame_equal(build_frame(spec), made)
    return spec


class Group:
    r"""The cases converted from one test file."""

    def __init__(self, name: str, source: str | None = None) -> None:
        self.name = name
        self.source = source or SOURCES.get(name, CATALOG_SOURCE)
        self.cases: list[dict[str, Any]] = []
        self._ids: set[str] = set()

    def add(
        self, label: str, operation: str, input: Any, want: Any = _MISSING
    ) -> dict[str, Any]:
        label = re.sub(r'[^A-Za-z0-9_.-]+', '-', label).strip('-') or 'case'
        case_id = f'{self.name}/{operation}/{label}'
        suffix = 2
        while case_id in self._ids:
            case_id = f'{self.name}/{operation}/{label}-{suffix}'
            suffix += 1
        self._ids.add(case_id)
        input = json.loads(json.dumps(input))
        case: dict[str, Any] = {
            'id': case_id,
            'source': self.source,
            'operation': operation,
            'input': input,
        }
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                got = json.loads(json.dumps(run(operation, input)))
        except FixtureError as error:
            case['expected_error'] = error.described()
            got = _MISSING
            if want is not _MISSING and not (
                isinstance(want, Raises) and want.matches(error)
            ):
                _disagree(f'{case_id}: raised {error}, test says {want}')
        else:
            case['expected'] = got
            if isinstance(want, Raises):
                _disagree(f'{case_id}: returned {got!r:.300}, test says {want}')
            elif want is not _MISSING and not same(
                got, json.loads(json.dumps(want))
            ):
                _disagree(f'{case_id}: gave {got!r}, test says {want!r}')
        case['expected_warnings'] = warned(caught)
        self.cases.append(case)
        return case

    def flow(
        self,
        label: str,
        input: dict[str, Any],
        stand_in: dict[str, Any] | None = None,
        want: Any = _MISSING,
    ) -> dict[str, Any]:
        r"""A ``client.run`` case, its model answered as the parity stand-in
        answers (``scripts/parity/_parity/nim.py``, misbehaving as
        ``stand_in`` says), with those responses written into its input."""
        from _parity.nim import SyntheticSession

        session = SyntheticSession(stand_in)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                _client_run(json.loads(json.dumps(input)), session=session)
            except Exception:  # noqa: BLE001, S110 - the case records it
                pass
        input = {
            **input,
            'responses': [
                {k: v for k, v in r.items() if k != 'request'}
                for r in session.responses
            ],
        }
        return self.add(label, 'client.run', input, want)


class Raises:
    r"""The test says this raises, with ``text`` in its message."""

    def __init__(self, text: str = '', code: str | None = None) -> None:
        self.text, self.code = text, code

    def matches(self, error: FixtureError) -> bool:
        return self.text in error.message and (
            self.code is None or error.code == self.code
        )

    def __repr__(self) -> str:
        return f'Raises({self.text!r}, {self.code!r})'


# --- test_dates_the_column_decides.py ---------------------------------------


def dates() -> Group:
    g = Group('dates')
    for i, (values, decided) in enumerate(
        [
            (['02/01/2024', '15/03/2024', '06/04/2024'], True),
            (['01/02/2024', '03/15/2024', '04/06/2024'], False),
            (['31/12/2023'], True),
            (['12/31/2023'], False),
            (['02/01/2024', '03/05/2024', '06/04/2024'], None),
            (['15/03/2024', '03/15/2024'], None),
            (['2024-01-02', '2024-03-15'], None),
            ([], None),
            (['03/15/2024 00:00:00', '04/16/2024 00:00:00'], False),
            (['15/03/2024 08:30', '16/04/2024 09:00'], True),
            (['15/03/2024T08:30:00+01:00'], True),
            (['01/02/2024 00:00:00', '03/05/2024 00:00:00'], None),
            (['15/03/2024 batch7'], None),
            (['15/03/2024-REV2'], None),
            (['01/02/2024'] * 1000 + ['03/15/2024'], False),
            (['01/02/2024'] * 2000, None),
        ]
    ):
        g.add(f'case-{i}', 'dates.day_comes_first', {'values': values}, decided)

    for i, (values, unambiguous) in enumerate(
        [
            (['2024-01-02', '2024-03-15'], True),
            (['02/01/2024', '15/03/2024'], True),
            (['01/02/2024', '03/15/2024'], True),
            (['02/01/2024', '03/05/2024'], False),
            (['15/03/2024', '03/15/2024'], False),
            (['02/01/2024', '03/05/2024', '06/04/2024'], False),
            (['01/02/2024'] * 1000 + ['03/15/2024'], True),
        ]
    ):
        g.add(
            f'case-{i}',
            'dates.looks_unambiguous',
            {'values': values},
            unambiguous,
        )
    for zone in [
        '-0800',
        '-08:00',
        '-08',
        '+0530',
        'Z',
        '+00:00',
        'UTC',
        'America/Denver',
    ]:
        values = [
            f'2021-02-{(i % 28) + 1:02d} 13:24:26.000 {zone}'
            for i in range(120)
        ]
        g.add(
            f'zone{zone}', 'dates.looks_unambiguous', {'values': values}, True
        )

    for i, (values, read_as) in enumerate(
        [
            (['02/01/2024', '15/03/2024'], ['2024-01-02', '2024-03-15']),
            (['01/02/2024', '03/15/2024'], ['2024-01-02', '2024-03-15']),
            (['2024-01-02', '2024-03-15'], ['2024-01-02', '2024-03-15']),
        ]
    ):
        g.add(
            f'case-{i}',
            'dates.read',
            {'values': values},
            [f'{day}T00:00:00' for day in read_as],
        )
    g.add(
        'timestamp-day-first',
        'dates.read',
        {
            'values': ['15/03/2024 08:30', '02/01/2024 09:00'],
            'conversion': {'dayfirst': True},
        },
        ['2024-03-15T08:30:00', '2024-01-02T09:00:00'],
    )

    old = ['1899-01-01', '1899-02-14', '1875-06-30']
    g.add('old-plausible', 'dates.conversion_that_works', {'values': old}, None)
    g.add(
        'old-declared',
        'dates.conversion_that_works',
        {'values': old, 'only_plausible_years': False},
        {},
    )
    codes = [f'{4711 + i}-01-02' for i in range(5)]
    g.add('part-codes', 'dates.conversion_that_works', {'values': codes}, None)
    g.add(
        'year-2201',
        'dates.conversion_that_works',
        {'values': ['2201-01-02'] * 20},
        None,
    )
    g.add(
        'year-2201-declared',
        'dates.conversion_that_works',
        {'values': ['2201-01-02'] * 20, 'only_plausible_years': False},
    )

    for conversion, written in [
        ({}, ''),
        ({'utc': True}, ', utc=True'),
        ({'dayfirst': True}, ', dayfirst=True'),
        ({'dayfirst': False}, ', dayfirst=False'),
    ]:
        g.add(
            written.strip(', ') or 'default',
            'dates.conversion_text',
            conversion,
            {
                'text': written,
                'arguments': [
                    k.split('=')[0].strip() for k in written.split(', ') if k
                ],
            },
        )

    for value in ['02/01/2020', '2020-01-02-REV-7', 'not a date at all']:
        g.add(
            f'year-first-{value}',
            'dates.pattern',
            {'pattern': 'YEAR_FIRST', 'value': value},
            False,
        )
    for zone in ['+0200', ' +0200', ' UTC', 'Z', ' -08:00', '']:
        year_first = f'2020-01-01 00:00{zone}'
        slash = f'13/02/2021 00:00{zone}'
        g.add(
            f'year-first{zone}',
            'dates.pattern',
            {'pattern': 'YEAR_FIRST', 'value': year_first},
            True,
        )
        g.add(
            f'slash{zone}',
            'dates.pattern',
            {'pattern': 'DAY_OR_MONTH_FIRST', 'value': slash, 'whole': False},
            True,
        )
        g.add(
            f'two-shapes{zone}',
            'dates.shapes_written',
            {'values': [year_first, slash]},
            ['day-or-month-first', 'year-first'],
        )
    g.add(
        'one-late-slash',
        'dates.shapes_written',
        {
            'values': [f'2020-01-{1 + i % 28:02d}' for i in range(5000)]
            + ['02/01/2020']
        },
        ['day-or-month-first', 'year-first'],
    )
    g.add(
        'zone-on-every-row',
        'dates.shapes_written',
        {'values': [f'2020-01-{d:02d} 00:00 +0100' for d in range(1, 29)]},
        ['year-first'],
    )
    mixed = ['13/02/2020 00:00 +0100', '14/02/2020 00:00 -0800', '15/02/2020']
    g.add(
        'mixed-offset-slash',
        'dates.reads_as_a_date',
        {'values': mixed},
        [True] * 3,
    )

    iso = [f'2021-02-{d:02d}' for d in range(1, 29)]
    zoned = [f'2021-02-{d:02d} 00:00 +0100' for d in range(1, 29)]
    day_first = [f'{d:02d}/02/2021' for d in range(1, 29)]
    month_first = [f'02/{d:02d}/2021' for d in range(1, 29)]
    undecided = ['02/01/2020', '03/01/2020'] * 14
    junk, slash_junk, bad_iso = ['not a date'], ['99/99/2020'], ['2021-13-45']
    arms = {
        'iso': iso,
        'zoned': zoned,
        'iso-and-zoned': iso[:14] + zoned[:14],
        'day-first': day_first,
        'month-first': month_first,
        'undecided': undecided,
        'iso-and-day-first': iso[:14] + day_first[:14],
        'iso-and-undecided': iso[:14] + undecided[:14],
        'iso-junk': iso + junk,
        'iso-slash-junk': iso + slash_junk,
        'iso-bad-iso': iso + bad_iso,
        'day-first-junk': day_first + junk,
        'day-first-slash-junk': day_first + slash_junk,
        'month-first-junk': month_first + junk,
        'bad-iso-and-day-first': bad_iso * 14 + day_first[:14],
        'all-junk': junk * 28,
        'all-slash-junk': slash_junk * 28,
        'all-bad-iso': bad_iso * 28,
        'mixed-offset-slash': mixed * 10,
        'trailing-space': ['2024-09-01 00:00:00 ', '2024-09-02 00:00:00 '] * 14,
        'leading-space': ['  2024-09-01', '  2024-09-02'] * 14,
        'a-tab': ['2024-09-01\t'] * 14,
        'counted-mixed-offset': [
            '13/02/2020 00:00 +0100',
            '14/02/2020 00:00 -0800',
            'not a date',
        ]
        * 10,
        'counted-partial-zone': [
            '2020-01-01',
            '2020-01-02 00:00 +0100',
            'not a date',
        ]
        * 10,
        'counted-day-first': day_first + ['not a date'],
        'both-faults': ['2021-02-01', '13/02/2021', 'not a date'] * 10,
        'two-ways-only': ['2021-02-01', '13/02/2021'] * 10,
        'cst-zone': [
            f'2021-02-{(i % 28) + 1:02d} 13:24:26 CST' for i in range(120)
        ],
        'utc-zone': [
            f'2021-02-{(i % 28) + 1:02d} 13:24:26 UTC' for i in range(120)
        ],
        'one-unread': [f'2021-02-{(i % 28) + 1:02d}' for i in range(119)]
        + ['not a date'],
        'some-rows-zoned': ['2020-01-01', '2020-01-02 00:00 +0100'] * 45,
        'two-offsets': ['2020-01-01 00:00 +0100', '2020-01-02 00:00 -0800']
        * 45,
        'bare-date-and-z': ['2020-01-01', '2020-01-02T00:00:00Z'] * 45,
    }
    for label, values in arms.items():
        g.add(label, 'time_column.read', {'values': values})

    def written(shape: str, day: int, zone: str) -> str:
        stem = {
            'year-first': f'2021-02-{day:02d}',
            'day-first': f'{day:02d}/02/2021',
            'month-first': f'02/{day:02d}/2021',
        }[shape]
        return f'{stem} 00:00:00{zone}' if zone else stem

    zones = ['', 'Z', '+0100', ' +0100', ' -08:00', ' UTC', ' ']
    junks = {
        'clean': [],
        'junk': ['not a date'],
        'slash-junk': ['99/99/2020'],
        'bad-iso': ['2021-13-45'],
    }
    for shape in ['year-first', 'day-first', 'month-first']:
        for z, zone in enumerate(zones):
            for name, extra in junks.items():
                values = [
                    written(shape, day, zone) for day in range(1, 29)
                ] + extra
                g.add(
                    f'{shape}-zone{z}-{name}',
                    'time_column.read',
                    {'values': values},
                )
    return g


# --- test_wire_integer_exactness.py -----------------------------------------


def wire() -> Group:
    g = Group('wire')
    big = 9007199254740991 + 2
    int64_max = 2**63 - 1
    obj = 'object'

    def column(
        label: str,
        values: list[Any],
        want: Any = _MISSING,
        dtype: str = obj,
        declared: str | None = None,
    ) -> None:
        given: dict[str, Any] = {'series': series(values, dtype)}
        if declared is not None:
            given['dtype'] = declared
        g.add(label, 'wire.serialize_column', given, want)

    for label, values, dtype, want in [
        ('whole-decimal', [Decimal(str(big)), Decimal('1')], obj, 'int64'),
        ('fractional-decimal', [Decimal('12.3456')], obj, 'float64'),
        (
            'wide-decimal',
            [Decimal('123456789012345678901234567890')],
            obj,
            'string',
        ),
        ('int64-max-decimal', [Decimal(str(int64_max))], obj, 'int64'),
        ('uint64-in-range', [5, 6], 'uint64', 'int64'),
        ('uint64-past', [2**64 - 1, 1], 'uint64', 'string'),
        ('UInt64-past', [2**64 - 1, 1], 'UInt64', 'string'),
        ('plain-int', [big, 1], None, 'int64'),
        ('decimal-object', [Decimal('123')], obj, 'int64'),
    ]:
        g.add(
            label, 'wire.infer_dtype', {'series': series(values, dtype)}, want
        )
    import pyarrow as pa

    for label, values, arrow, want in [
        ('arrow-whole', [Decimal('123')], pa.decimal128(38, 0), 'int64'),
        (
            'arrow-fractional',
            [Decimal('1.50')],
            pa.decimal128(38, 2),
            'float64',
        ),
        ('arrow-narrow', [Decimal('1')], pa.decimal128(18, 0), 'int64'),
    ]:
        g.add(
            label,
            'wire.infer_dtype',
            {'series': series(pd.Series(values, dtype=pd.ArrowDtype(arrow)))},
            want,
        )

    column(
        'past-safe',
        [Decimal(str(big)), Decimal('1')],
        {'dtype': 'int64', 'cells': [str(big), 1]},
    )
    column(
        'null',
        [Decimal(str(big)), None],
        {'dtype': 'int64', 'cells': [str(big), None]},
    )
    column(
        'decimal-nan',
        [Decimal('NaN'), Decimal('1')],
        {'dtype': 'int64', 'cells': [None, 1]},
    )
    column(
        'fractional',
        [Decimal('12.3456'), Decimal('1.00')],
        {'dtype': 'float64', 'cells': [12.3456, 1.0]},
    )
    column(
        'too-wide',
        [Decimal('123456789012345678901234567890')],
        {'dtype': 'string', 'cells': ['123456789012345678901234567890']},
    )
    for value in [2**63, 2**64, 10**20]:
        column(f'past-int64-{value}', [Decimal(str(value))])
    for label, value in [
        ('1e400', Decimal('1e400')),
        ('1E30', Decimal('1E+30')),
        ('2-70', Decimal(str(2**70))),
    ]:
        column(f'digits-{label}', [value])
    column('mixed-width', [Decimal(str(2**63)), Decimal('1')])
    column(
        'fractional-past-safe',
        [Decimal('12345678901234567.8901')],
        Raises('too large'),
    )
    column(
        'infinity',
        [Decimal('Infinity')],
        Raises('JSON-representable', 'INVALID_REQUEST'),
    )
    column(
        'uint64-declared-int64',
        [5, 6],
        {'dtype': 'int64', 'cells': [5, 6]},
        dtype='uint64',
        declared='int64',
    )
    for dtype in ['uint64', 'UInt64']:
        column(
            f'{dtype}-past-declared-int64',
            [2**64 - 1, 1],
            Raises('outside the range'),
            dtype=dtype,
            declared='int64',
        )
        g.add(
            f'{dtype}-past-inferred',
            'wire.serialize_column',
            {'series': series([2**64 - 1, 1], dtype)},
            {'dtype': 'string', 'cells': [str(2**64 - 1), '1']},
        )
    g.add(
        'plain-int-column',
        'wire.serialize_column',
        {'series': series([big, 1]), 'dtype': 'int64'},
        {'dtype': 'int64', 'cells': [str(big), 1]},
    )
    g.add(
        'float-column',
        'wire.serialize_column',
        {'series': series([1.5, 2.5])},
        {'dtype': 'float64', 'cells': [1.5, 2.5]},
    )
    column('large-float', [1.2345678901234568e16], dtype='float64')

    def cell(label: str, value: Any, dtype: str, want: Any = _MISSING) -> None:
        g.add(
            label,
            'wire.serialize_cell',
            {'value': encode(value), 'dtype': dtype},
            want,
        )

    for label, value in [
        ('int', big),
        ('decimal', Decimal(str(big))),
        ('np-int64', np.int64(big)),
    ]:
        cell(f'{label}-past-safe-in-float64', value, 'float64', Raises())
    for label, value in [
        ('int', 3),
        ('decimal', Decimal('3')),
        ('np-int64', np.int64(3)),
    ]:
        cell(f'{label}-3-in-float64', value, 'float64', 3.0)
    cell('int64-max', int64_max, 'int64', str(int64_max))
    cell('int64-max-plus-one', int64_max + 1, 'int64', Raises())
    cell('bool-in-float64', True, 'float64', 1.0)
    cell(
        '10-400-in-float64',
        10**400,
        'float64',
        Raises('outside the range', 'INVALID_REQUEST'),
    )
    cell('float-1.5', 1.5, 'float64', 1.5)
    cell('2-53', 2**53, 'float64', float(2**53))
    cell(
        'decimal-past-cutoff',
        Decimal('9007199254740991.1'),
        'float64',
        Raises('too large'),
    )
    cell('decimal-1.5', Decimal('1.5'), 'float64', 1.5)
    cell('2-53-plus-one', 2**53 + 1, 'float64', Raises())
    cell(
        'numeric-string-too-wide',
        '9223372036854775809',
        'float64',
        Raises('losing'),
    )
    cell('numeric-string-1.5', '1.5', 'float64', 1.5)
    cell('numeric-string-42', '42', 'float64', 42.0)

    def ready(label: str, value: Any, want: Any = _MISSING) -> None:
        g.add(label, 'wire.json_ready', {'value': encode(value)}, want)

    ready('nested', {'a': [1, 2], 'b': {'c': 3}}, {'a': [1, 2], 'b': {'c': 3}})
    ready(
        'seed-past-float', {'seed': 2**53 + 1}, Raises(code='INVALID_REQUEST')
    )
    ready('np-seed-past-float', {'seed': np.int64(2**53 + 1)}, Raises())
    ready('np-seed-4', {'seed': np.int64(4)}, {'seed': 4})
    ready('uint64-seed', {'seed': np.uint64(2**64 - 1)}, Raises())
    ready('2-53', 2**53, 2**53)
    ready('2-53-plus-two', 2**53 + 2, 2**53 + 2)
    ready('minus-2-53', -(2**53), -(2**53))
    ready('2-53-plus-one', 2**53 + 1, Raises())
    ready('minus-2-53-minus-one', -(2**53) - 1, Raises())
    ready('bool', {'explain': True}, {'explain': True})
    ready('nested-past-float', {'opts': {'seeds': [1, 2**53 + 1]}}, Raises())
    return g


# --- test_inferred_time_and_keys.py and test_inferred_relationships.py ------


def _describe(g: Group, label: str, tables: dict[str, Any], **how: Any) -> None:
    g.add(
        label,
        'schema.from_data',
        {'tables': {name: frame(f) for name, f in tables.items()}, **how},
    )


def time_and_keys() -> Group:
    g = Group('time-and-keys')
    _describe(
        g,
        'stype-timestamp-sets-time-column',
        {
            'review': {
                'rid': [1, 2, 3],
                'date': ['2021-01-01', '2021-02-01', '2021-03-01'],
            }
        },
        edits=[
            {
                'table': 'review',
                'column': 'date',
                'set': 'stype',
                'value': 'timestamp',
            }
        ],
    )
    two = {
        'a': pd.to_datetime(['2021-01-01', '2021-02-01']),
        'b': pd.to_datetime(['2022-01-01', '2022-02-01']),
    }
    _describe(g, 'two-timestamps-decline', {'t': two})
    _describe(
        g,
        'naming-one-of-two-timestamps',
        {'t': two},
        edits=[
            {'table': 't', 'column': 'b', 'set': 'stype', 'value': 'timestamp'}
        ],
    )
    _describe(
        g,
        'second-timestamp-does-not-displace',
        {
            't': {
                'seq': [1, 2],
                'created': pd.to_datetime(['2021-01-01', '2021-02-01']),
            }
        },
        edits=[
            {
                'table': 't',
                'column': 'seq',
                'set': 'dtype',
                'value': 'timestamp[us]',
            },
            {
                'table': 't',
                'column': 'seq',
                'set': 'stype',
                'value': 'timestamp',
            },
        ],
    )
    _describe(
        g,
        'primary-key-is-not-time-column',
        {'t': {'k': [1, 2], 'v': [1.0, 2.0]}},
        primary_keys={'t': 'k'},
        edits=[
            {
                'table': 't',
                'column': 'k',
                'set': 'dtype',
                'value': 'timestamp[us]',
            },
            {'table': 't', 'column': 'k', 'set': 'stype', 'value': 'timestamp'},
        ],
    )
    for i, values in enumerate(
        [
            ['2013-01-01', '2013-06-15', '2014-02-20'],
            [
                '2013-01-01T04:05:06',
                '2013-06-15T00:00:00',
                '2014-02-20T23:59:59',
            ],
            [
                '2013-01-01 04:05:06',
                '2013-06-15 00:00:00',
                '2014-02-20 23:59:59',
            ],
            ['2013/01/01', '2013/06/15', '2014/02/20'],
            [
                '2013-01-01T04:05:06Z',
                '2013-06-15T00:00:00Z',
                '2014-02-20T00:00:00Z',
            ],
            [
                '2013-01-01T04:05:06+05:30',
                '2013-06-15T00:00:00-08:00',
                '2014-02-20T00:00:00+0000',
            ],
        ]
    ):
        _describe(
            g, f'text-dates-{i}', {'t': {'date': values * 4, 'v': [1.0] * 12}}
        )
    for label, column in [
        ('codes', ['4711-01-02', 'A-12-9', 'ZX-0001'] * 4),
        ('starts-with-date', ['2013-01-01-REV-7', '2013-01-01-REV-8'] * 6),
        ('partly-dates', ['2013-01-01'] * 11 + ['not a date']),
        ('impossible-dates', ['2024-02-31', '2024-99-99', '2024-00-00'] * 4),
        ('part-code-dates', ['4711-01-02', '4711-01-03'] * 6),
        ('two-formats', ['2024-01-01'] * 11 + ['March 5, 2024']),
        ('dates-and-z', ['2024-01-01T10:00:00Z', '2024-06-15T11:00:00Z'] * 6),
        (
            'dates-and-offsets',
            ['2024-01-01T10:00:00+05:30', '2024-06-15T11:00:00-08:00'] * 6,
        ),
    ]:
        _describe(g, label, {'t': {'d': column, 'v': [1.0] * len(column)}})
    _describe(
        g,
        'found-time-column',
        {
            't': {
                'when': pd.to_datetime(['2021-01-01'] * 4),
                'other': ['2021-01-01'] * 4,
                'v': [1.0] * 4,
            }
        },
    )
    airports = {
        'IATA_CODE': ['ATL', 'ORD', 'LAX'],
        'CITY': ['Atlanta', 'Chicago', 'LA'],
    }
    sales = {
        'SALE_ID': [1, 2, 3, 4],
        'IATA_CODE': ['ATL', 'ORD', 'ATL', 'LAX'],
        'AMOUNT': [10.0, 20.0, 30.0, 40.0],
    }
    _describe(g, 'keyless-parent', {'airports': airports, 'sales': sales})
    _describe(
        g,
        'keyless-parent-declared',
        {'airports': airports, 'sales': sales},
        primary_keys={'airports': 'IATA_CODE'},
    )
    line_items = {
        'ORDER_NO': ['O1', 'O2'],
        'LINE_NO': [1, 1],
        'sku': ['A', 'B'],
    }
    shipments = {'SHIPMENT_ID': [1, 2, 3], 'ORDER_NO': ['O1', 'O1', 'O2']}
    _describe(
        g,
        'unique-foreign-key',
        {'line_items': line_items, 'shipments': shipments},
    )
    _describe(
        g,
        'declared-parent-key',
        {
            'orders': {'ORDER_NO': ['O1', 'O2'], 'total': [1.0, 2.0]},
            'line_items': line_items,
            'shipments': shipments,
        },
        primary_keys={'orders': 'ORDER_NO'},
    )
    _describe(
        g,
        'lone-table',
        {'airports': {'IATA_CODE': ['ATL', 'ORD'], 'CITY': ['Atlanta', 'Chi']}},
    )
    child = {'cid': [1, 2], 'code': ['a', 'b']}
    _describe(
        g,
        'repeated-column',
        {
            'parent': {'code': ['a', 'a', 'b'], 'n': [1.0, 2.0, 3.0]},
            'child': child,
        },
    )
    _describe(
        g,
        'float-measure',
        {
            'parent': {'amount': [1.5, 2.5, 3.5]},
            'child': {'cid': [1, 2], 'amount': [1.5, 2.5]},
        },
    )
    _describe(
        g,
        'null-bearing',
        {'parent': {'code': ['a', None, 'b']}, 'child': child},
    )
    _describe(
        g,
        'table-with-key',
        {'parent': {'code': ['a', 'b'], 'n': [1, 2]}, 'child': child},
        primary_keys={'parent': 'n'},
    )
    _describe(
        g,
        'time-column-not-a-key',
        {
            'parent': {'day': ['2021-01-01', '2021-01-02'], 'n': [1.0, 2.0]},
            'child': {'cid': [1, 2], 'day': ['2021-01-01', '2021-01-02']},
        },
        time_columns={'parent': 'day'},
    )
    return g


def relationships() -> Group:
    g = Group('relationships')
    frames = {
        'alpha': {'ID': [1, 2, 3], 'a': [1.0, 2.0, 3.0]},
        'beta': {'ID': [10, 20, 30], 'b': [1.0, 2.0, 3.0]},
        'child': {'cid': [1, 2, 3], 'ID': [1, 2, 3]},
    }
    keys = {'alpha': 'ID', 'beta': 'ID', 'child': 'cid'}
    _describe(g, 'shared-key-name', frames, primary_keys=keys)
    _describe(
        g,
        'shared-key-name-reversed',
        {name: frames[name] for name in reversed(list(frames))},
        primary_keys=keys,
    )
    _describe(
        g,
        'two-parents-sharing-a-name',
        {
            'airlines': {
                'IATA_CODE': ['AA', 'DL'],
                'name': ['American', 'Delta'],
            },
            'airports': {
                'IATA_CODE': ['ATL', 'ORD'],
                'city': ['Atlanta', 'Chicago'],
            },
        },
        primary_keys={'airlines': 'IATA_CODE', 'airports': 'IATA_CODE'},
    )
    _describe(
        g,
        'unambiguous-key',
        {
            'users': {'user_id': [1, 2, 3], 'seg': list('abc')},
            'orders': {'order_id': [1, 2, 3], 'user_id': [1, 2, 3]},
        },
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
    )
    _describe(
        g,
        'one-ambiguity-does-not-suppress-another',
        {
            'alpha': {'ID': [1, 2], 'x': [1.0, 2.0]},
            'beta': {'ID': [3, 4], 'y': [1.0, 2.0]},
            'users': {'user_id': [1, 2], 'seg': ['a', 'b']},
            'events': {'ev_id': [1, 2], 'user_id': [1, 2], 'ID': [1, 2]},
        },
        primary_keys={
            'alpha': 'ID',
            'beta': 'ID',
            'users': 'user_id',
            'events': 'ev_id',
        },
    )
    _describe(
        g,
        'not-linked-to-itself',
        {'users': {'user_id': [1, 2, 3], 'seg': list('abc')}},
        primary_keys={'users': 'user_id'},
    )
    pairs = {'a': [1, 1, 2], 'b': [1, 2, 1], 'v': [1.0, 2.0, 3.0]}
    _describe(
        g,
        'composite-is-not-a-target',
        {'pairs': pairs, 'child': {'cid': [1, 2, 3], 'a': [1, 1, 2]}},
        primary_keys={'pairs': ['a', 'b'], 'child': 'cid'},
    )
    _describe(
        g,
        'ambiguity-scoped-to-column',
        {
            'users': {'ID': [1, 2], 'seg': ['a', 'b']},
            'orders': {'ID': [1, 2], 'account_id': [7, 8], 'v': [1.0, 2.0]},
            'accounts': {'account_id': [7, 8], 'tier': ['x', 'y']},
        },
        primary_keys={'users': 'ID', 'orders': 'ID', 'accounts': 'account_id'},
    )
    _describe(
        g,
        'composite-component-does-not-contest',
        {
            'pairs': pairs,
            'owner': {'a': [1, 2], 'name': ['x', 'y']},
            'child': {'cid': [1, 2], 'a': [1, 2]},
        },
        primary_keys={'pairs': ['a', 'b'], 'owner': 'a', 'child': 'cid'},
    )
    repeated = pd.DataFrame(
        {'user_id': list(range(200)), 'line_no': [1] * 200, 'v': [1.0] * 200}
    )
    repeated.loc[[7, 8], 'user_id'] = 6
    repeated.loc[[7, 8], 'line_no'] = [2, 3]
    _describe(g, 'repeated-key-names-composite', {'t': repeated})
    return g


# --- test_a_binary_target_is_a_classification.py ----------------------------


def binary_target() -> Group:
    g = Group('binary-target')
    query = 'PREDICT people.churn FOR EACH people.person_id'

    def task(
        label: str, target: list[Any], want: Any = _MISSING, **options: Any
    ) -> None:
        data = pd.DataFrame(
            {
                'person_id': range(len(target)),
                'age': [20.0 + i for i in range(len(target))],
                'churn': target,
            }
        )
        g.add(
            label,
            'tabular.task',
            {
                'query': query,
                'table': 'people',
                'target': 'churn',
                'frame': frame(data),
                'options': encode(options),
            },
            want,
        )

    zero_one = [i % 2 for i in range(40)]
    task('zero-one', zero_one)
    for pair in [[0, 1], [1, 2], [-1, 1], [7, 9]]:
        task(f'pair-{pair[0]}-{pair[1]}', [pair[i % 2] for i in range(40)])
    task('fractional-pair', [0.5 if i % 2 else 1.5 for i in range(40)])
    task('whole-floats', [2.0 if i % 2 else 3.0 for i in range(40)])
    task(
        'decimal-not-quite-whole',
        [
            Decimal('2.0000000000000000001') if i % 2 else Decimal('3')
            for i in range(40)
        ],
    )
    task('single-valued', [1] * 40)
    task('continuous', [float(i) for i in range(40)])
    task('three-whole-numbers', [i % 3 for i in range(40)])
    task('categorical', ['yes' if i % 2 else 'no' for i in range(40)])
    with_null: list[Any] = list(zero_one)
    with_null[3] = None
    task('null-is-not-a-value', with_null)
    task('override-to-regression', zero_one, task='regression')
    task(
        'override-to-classification',
        [float(i) for i in range(40)],
        task='classification',
    )
    task(
        'forecasting-refused',
        zero_one,
        Raises('forecasting'),
        task='forecasting',
    )
    for label, bad in [
        ('list', ['regression']),
        ('dict', {'kind': 'r'}),
        ('int', 42),
    ]:
        task(f'task-not-a-string-{label}', zero_one, Raises(), task=bad)
    g.add(
        'huge-integers',
        'tabular.two_whole_numbers',
        {'series': series(pd.Series([10**400, 10**401], dtype=object))},
        True,
    )
    for label, values, dtype in [
        ('zero-one', [0, 1, 0], 'int64'),
        ('floats', [2.0, 3.0], 'float64'),
        ('fractional', [0.5, 1.5], 'float64'),
        ('three', [0, 1, 2], 'int64'),
        ('one', [1, 1], 'int64'),
        ('with-null', [0.0, None, 1.0], 'float64'),
    ]:
        g.add(
            label,
            'tabular.two_whole_numbers',
            {'series': series(values, dtype)},
        )
    return g


# --- test_decomposition_is_faithful.py --------------------------------------


def decomposition() -> Group:
    g = Group('decomposition')
    for label, value in [
        ('double-quote', 'a"b'),
        ('single-quote', "a'b"),
        ('comma', 'a,b'),
        ('open-paren', 'a(b'),
        ('close-paren', 'a)b'),
        ('backslash', 'a\\b'),
        ('empty', ''),
        ('blank', '   '),
        ('none', None),
        ('true', True),
        ('float', 7.5),
        ('nan', float('nan')),
        ('na', pd.NA),
        ('nat', pd.NaT),
    ]:
        g.add(label, 'decompose.as_literal', {'value': encode(value)}, Raises())
    for label, value, want in [
        ('np-int64-past-safe', np.int64(9007199254740993), '9007199254740993'),
        ('decimal-10', Decimal(10), '10'),
        ('decimal-10.00', Decimal('10.00'), '10'),
        (
            'decimal-10-30',
            Decimal(10) ** 30,
            '"1000000000000000000000000000000"',
        ),
        ('int', 7, _MISSING),
        ('string', 'item-1', _MISSING),
        ('negative', -3, _MISSING),
    ]:
        g.add(label, 'decompose.as_literal', {'value': encode(value)}, want)

    rows = 300
    shop = {
        'users': frame({'uid': [1, 2, 3], 'seg': list('abc')}),
        'items': frame({'iid': [10, 11, 12], 'kind': list('xyz')}),
        'buys': frame(
            {
                'bid': list(range(rows)),
                'uid': [1, 2, 3] * (rows // 3),
                'iid': [10, 11, 12] * (rows // 3),
                'channel': ['web', 'app'] * (rows // 2),
                'when': pd.to_datetime('2025-06-01')
                + pd.to_timedelta(range(rows), unit='D'),
            }
        ),
    }
    how = {
        'tables': shop,
        'primary_keys': {'users': 'uid', 'items': 'iid', 'buys': 'bid'},
        'time_columns': {'buys': 'when'},
    }
    ranking = 'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 FOR EACH users.uid'
    for label, query, options, want in [
        (
            'ranking',
            ranking,
            {'indices': [1, 2, 3], 'candidates': [10, 11, 12]},
            _MISSING,
        ),
        (
            'decimal-candidates',
            ranking,
            {
                'indices': [1, 2, 3],
                'candidates': [Decimal(10), Decimal(11), Decimal(12)],
            },
            _MISSING,
        ),
        (
            'repeated-candidate',
            ranking,
            {'indices': [1], 'candidates': [10, 10, 11]},
            _MISSING,
        ),
        (
            'top-k-option',
            ranking,
            {'indices': [1], 'candidates': [10, 11, 12], 'top_k': 1},
            _MISSING,
        ),
        ('no-candidates', ranking, {'indices': [1]}, Raises()),
        (
            'empty-candidates',
            ranking,
            {'indices': [1], 'candidates': []},
            Raises(),
        ),
        (
            'forecast',
            'PREDICT COUNT(buys.*, 0, 30, days) FORECAST 3 TIMEFRAMES FOR EACH users.uid',
            {'indices': [1]},
            _MISSING,
        ),
        (
            'forecast-keeps-filter',
            "PREDICT COUNT(buys.* WHERE buys.channel = 'web', 0, 30, days) "
            'FORECAST 2 TIMEFRAMES FOR EACH users.uid',
            {'indices': [1]},
            _MISSING,
        ),
        (
            'forecast-refuses-candidates',
            'PREDICT COUNT(buys.*, 0, 30, days) FORECAST 3 TIMEFRAMES FOR EACH users.uid',
            {'indices': [1], 'candidates': [10]},
            Raises(),
        ),
        (
            'candidates-on-plain-question',
            'PREDICT COUNT(buys.*, 0, 30, days) FOR EACH users.uid',
            {'indices': [1], 'candidates': [10]},
            Raises(),
        ),
        (
            'plain-question',
            'PREDICT COUNT(buys.*, 0, 30, days) FOR EACH users.uid',
            {'indices': [1]},
            None,
        ),
    ]:
        g.add(
            label,
            'decompose.parts',
            {**how, 'query': query, 'options': encode(options)},
            want,
        )
    for column in ['buys.channel', 'buys.bid', 'buys.uid']:
        g.add(
            f'ranking-{column}',
            'decompose.parts',
            {
                **how,
                'query': f'PREDICT LIST_DISTINCT({column}, 0, 30, days) RANK TOP 2 FOR EACH users.uid',
                'options': {'indices': [1], 'candidates': [1, 2]},
            },
        )
    return g


# --- test_json_payload.py -----------------------------------------------------


def json_payload() -> Group:
    g = Group('json-payload')

    def cell(label: str, value: Any, dtype: str, want: Any = _MISSING) -> None:
        g.add(
            label,
            'payload.cell_json_value',
            {'value': encode(value), 'dtype': dtype},
            want,
        )

    cell(
        'uint64-max', np.uint64(2**64 - 1), 'int64', Raises('outside the range')
    )
    cell('2-40', 2**40, 'int64', 2**40)
    cell('2-62', 2**62, 'int64', str(2**62))
    cell('minus-2-63', -(2**63), 'int64', str(-(2**63)))
    cell('decimal-1e400', Decimal('1e400'), 'float64', Raises('strings'))
    cell('10-400', 10**400, 'float64', Raises('strings'))
    cell('2-53-plus-one', 2**53 + 1, 'float64', Raises())
    cell(
        'wide-whole-decimal',
        Decimal('9223372036854775809.00'),
        'float64',
        Raises('strings'),
    )
    cell('decimal-1.5', Decimal('1.5'), 'float64', 1.5)
    cell('int-42', 42, 'float64', 42.0)
    for label, value, survives in [
        ('2-53', 2**53, True),
        ('2-53-minus-one', 2**53 - 1, True),
        ('minus-2-53', -(2**53), True),
        ('minus-2-53-minus-one', -(2**53) - 1, False),
    ]:
        cell(label, value, 'float64', float(value) if survives else Raises())
    cell('numeric-string-too-wide', '9223372036854775809', 'float64', Raises())
    cell('numeric-string-1.5', '1.5', 'float64', 1.5)
    for label, value, want in [
        (
            'huge-fractional',
            Decimal('1' + '0' * 310 + '.1'),
            Raises('too large'),
        ),
        ('decimal-1.5', Decimal('1.5'), 1.5),
        ('decimal-1e400', Decimal('1e400'), '1' + '0' * 400),
    ]:
        g.add(label, 'payload.json_value', {'value': encode(value)}, want)
    for i, (values, want) in enumerate(
        [
            ([Decimal('1.5'), Decimal('2.5')], 'float64'),
            ([Decimal('1'), Decimal('2')], 'int64'),
            ([Decimal('9223372036854775809'), Decimal('1')], 'string'),
            ([Decimal('1.5'), Decimal('9223372036854775809.00')], 'string'),
        ]
    ):
        for order, ordered in (
            ('forward', values),
            ('reverse', list(reversed(values))),
        ):
            g.add(
                f'case-{i}-{order}',
                'payload.decimal_dtype_name',
                {'series': series(ordered, 'object')},
                want,
            )
    for label, values, want in [
        ('lists-and-null', [['a'], None, np.array(['b'])], 'stringlist'),
        ('list-then-string', [['a'], 'b'], Raises('only arrays or nulls')),
        ('string-then-list', ['b', ['a']], Raises('only arrays or nulls')),
        ('null-item', [['a', None]], Raises('must not contain null items')),
    ]:
        g.add(
            label,
            'payload.dtype_name',
            {'series': series(pd.Series(values, dtype=object))},
            want,
        )
    link = 'temporal_link_prediction'
    for label, value, want in [
        ('array', np.array([1, '2']), ['1', '2']),
        ('scalar', 'item-1', Raises('stringlist arrays')),
        ('null-item', ['item-1', None], Raises('must not contain null')),
    ]:
        g.add(
            label,
            'payload.target_json_value',
            {'task_type': link, 'value': encode(value)},
            want,
        )
    for value in [float('inf'), float('-inf')]:
        g.add(
            f'non-finite-{value}',
            'payload.feature_column',
            {'feature': series([1.0, 1.0, 1.0, value] + [1.0] * 4)},
            Raises("Column 'USERS.FEATURE' row 3"),
        )
    for label, values in [
        ('fractional-decimals', [Decimal(f'-{i}.50') for i in range(8)]),
        ('whole-decimals', [Decimal(i) for i in range(8)]),
        (
            'wide-decimals',
            [Decimal(f'1234567890123456789012345678{i:02d}') for i in range(8)],
        ),
    ]:
        g.add(
            label,
            'payload.feature_column',
            {'feature': series(pd.Series(values, dtype=object))},
        )
    return g


# --- test_payload_dtype_fidelity.py -----------------------------------------


def dtype_fidelity() -> Group:
    g = Group('dtype-fidelity')
    n = 24
    zone = dt.timezone(dt.timedelta(hours=-8))
    cases = {
        'int64': np.arange(n, dtype='int64'),
        'int32': np.arange(n, dtype='int32'),
        'uint8': np.arange(n, dtype='uint8'),
        'float64': np.arange(n, dtype='float64') / 2,
        'float32': (np.arange(n) / 2).astype('float32'),
        'bool': np.array([True, False] * (n // 2)),
        'str': [f's{i}' for i in range(n)],
        'str_low_cardinality': ['a', 'b'] * (n // 2),
        'datetime': pd.date_range('2024-01-01', periods=n),
        'datetime_tz': pd.date_range('2024-01-01', periods=n, tz=zone),
        'timedelta': pd.to_timedelta(np.arange(n), unit='D'),
        'timedelta_beyond_js_safe': pd.to_timedelta(
            np.arange(n) + 4000, unit='D'
        ),
        'bytes': pd.Series([bytes([i % 3, 7]) for i in range(n)], dtype=object),
        'decimal': pd.Series(
            [Decimal(f'{i}.1234') for i in range(n)], dtype=object
        ),
        'decimal_low_cardinality': pd.Series(
            [Decimal('1.25'), Decimal('2.50')] * (n // 2), dtype=object
        ),
        'object_int': pd.Series([1, 2, 3] * (n // 3), dtype=object),
        'object_int_with_null': pd.Series(
            [1, 2, None] * (n // 3), dtype=object
        ),
        'object_float': pd.Series([1.5, 2.5, 3.5] * (n // 3), dtype=object),
        'object_bool': pd.Series([True, False] * (n // 2), dtype=object),
        'object_mixed': pd.Series([1, 'a', 2.5, None] * (n // 4), dtype=object),
        'object_all_null': pd.Series([None] * n, dtype=object),
        # uuid4 in the test; fixed here, since the fixture stores its values.
        'object_uuid': pd.Series(
            [uuid.uuid5(uuid.NAMESPACE_OID, f'parity-{i}') for i in range(n)],
            dtype=object,
        ),
        'object_date': pd.Series(
            [dt.date(2024, 1, 1 + i % 28) for i in range(n)], dtype=object
        ),
        'object_datetime': pd.Series(
            [dt.datetime(2024, 1, 1 + i % 28, tzinfo=dt.UTC) for i in range(n)],
            dtype=object,
        ),
        'object_timedelta': pd.Series(
            [pd.Timedelta(days=i) for i in range(n)], dtype=object
        ),
        'object_int_beyond_js_safe': pd.Series(
            [9007199254740993 + 2 * i for i in range(n)], dtype=object
        ),
        'int64_beyond_js_safe': np.array(
            [9007199254740993 + 2 * i for i in range(n)], dtype='int64'
        ),
        'category_int': pd.Categorical([1, 2, 3] * (n // 3)),
        'category_str': pd.Categorical(['a', 'b'] * (n // 2)),
        'nullable_int64': pd.array([1, 2, None] * (n // 3), dtype='Int64'),
        'nullable_float64': pd.array(
            [1.5, 2.5, None] * (n // 3), dtype='Float64'
        ),
        'nullable_boolean': pd.array(
            [True, False, None] * (n // 3), dtype='boolean'
        ),
        'string_extension': pd.array(
            [f's{i}' for i in range(n)], dtype='string'
        ),
        'stringlist': pd.Series([['a', 'b']] * n, dtype=object),
        'intlist': pd.Series([[1, 2, 3]] * n, dtype=object),
        'floatlist': pd.Series([[1.5, 2.5]] * n, dtype=object),
        'ndarray_float': pd.Series([np.array([1.0, 2.0])] * n, dtype=object),
        'all_nan_float': pd.Series([np.nan] * n),
        'float_with_nan': pd.Series([1.5, np.nan] * (n // 2)),
    }
    for label, feature in cases.items():
        g.add(
            label,
            'payload.feature_column',
            {'feature': series(pd.Series(feature))},
        )
    return g


# --- test_composite_primary_key.py ------------------------------------------


def composite_key() -> Group:
    from sdm_relational_engine.rfm.base.composite_key import SEPARATOR

    g = Group('composite-key')
    for label, parts in [
        ('separator-in-second', ['a', f'b{SEPARATOR}c']),
        ('separator-in-first', [f'a{SEPARATOR}b', 'c']),
        ('customer-region', ['AB-10015', 'Central']),
        ('three-parts', ['a', 'b', 'c']),
        ('holds-separator', [f'holds{SEPARATOR}separator', 'y']),
        ('holds-backslash', ['holds\\backslash', 'z']),
        ('empty-first', ['', 'empty-first-part']),
        ('numbers-and-text', [7, 'x']),
        ('numpy-bool', [np.bool_(True), 'x']),
        ('python-bool', [True, 'x']),
    ]:
        g.add(label, 'composite.encode_values', {'parts': encode(parts)})
    for label, parts in [
        ('customer-region', ('AB-10015', 'Central')),
        ('three-parts', ('a', 'b', 'c')),
        ('holds-separator', (f'holds{SEPARATOR}separator', 'y')),
        ('holds-backslash', ('holds\\backslash', 'z')),
        ('empty-first', ('', 'empty-first-part')),
    ]:
        from sdm_relational_engine.rfm.base.composite_key import encode_values

        g.add(
            label,
            'composite.decode_value',
            {'value': encode_values(list(parts)), 'arity': len(parts)},
            list(parts),
        )
    for label, value in [
        ('nan', float('nan')),
        ('none', None),
        ('nat', pd.NaT),
        ('np-nan', np.nan),
    ]:
        g.add(
            f'missing-{label}',
            'composite.encode_values',
            {'parts': encode(['a', value])},
            Raises(),
        )
    g.add(
        'approximate', 'composite.encode_values', {'parts': encode(['a', 1.5])}
    )
    for columns in [
        ['Customer ID', 'Region'],
        ['Cust.ID', 'Region'],
        ['back`tick', 'Region'],
        ['has space', 'Region'],
        ['a', 'b', 'c'],
    ]:
        g.add(
            '-'.join(columns), 'composite.encode_identity', {'columns': columns}
        )
    from sdm_relational_engine.rfm.base.composite_key import (
        DERIVED_PREFIX,
        encode_identity,
    )

    for columns in [
        ['Customer ID', 'Region'],
        ['a_b', 'c'],
        ['a', 'b_c'],
        ['a', 'b', 'c'],
        ['holds:colon', 'y'],
        ['Cust.ID', 'R'],
        ['back`tick', 'R'],
        ['ünïcode', 'x'],
    ]:
        g.add(
            '-'.join(columns),
            'composite.decode_identity',
            {'derived': DERIVED_PREFIX + encode_identity(columns)},
            columns,
        )
    for label, derived in [
        ('prefix-only', DERIVED_PREFIX),
        ('not-hex', DERIVED_PREFIX + 'nothex'),
        ('customer-id', 'Customer ID'),
    ]:
        g.add(label, 'composite.decode_identity', {'derived': derived}, None)
    g.add(
        'too-long',
        'composite.encode_identity',
        {'columns': ['x' * 120]},
        Raises(),
    )
    g.add(
        'plain-name', 'composite.decode_identity', {'derived': 'user_id'}, None
    )
    people = {
        'Customer ID': ['AB-10015', 'AB-10015', 'ZZ-00001', 'ZZ-00001'],
        'Region': ['Central', 'East', 'Central', 'West'],
        'Segment': ['Consumer', 'Consumer', 'Corporate', 'Corporate'],
    }
    g.add(
        'superstore',
        'composite.encode_frame',
        {'frame': frame(people), 'columns': ['Customer ID', 'Region']},
    )
    g.add(
        'missing-column',
        'composite.encode_frame',
        {'frame': frame(people), 'columns': ['Customer ID', 'Nope']},
        Raises(),
    )
    return g


# --- query/test_pandas_executor.py ------------------------------------------


def pandas_executor() -> Group:
    g = Group('pandas-executor')
    users = frame(
        pd.DataFrame(
            {
                'AGE': [10, None, 30],
                'GENDER': ['male', None, 'female'],
                'CAT': pd.Series([1, pd.NA, 3], dtype='Int64'),
            }
        )
    )
    for column in ['AGE', 'GENDER', 'CAT']:
        for filter_na in (False, True):
            g.add(
                f'{column}-filter-na-{filter_na}',
                'labels.column',
                {
                    'fqn': f'USERS.{column}',
                    'tables': {'USERS': users},
                    'filter_na': filter_na,
                },
            )
    g.add(
        'star',
        'labels.column',
        {'fqn': 'USERS.*', 'tables': {'USERS': users}, 'filter_na': True},
    )
    lefts = {
        'int': (
            series([2, 4, pd.NA, 4, 6], 'Int64'),
            {'value': '4', 'dtype': 'int'},
        ),
        'float': (
            series([2.5, 4.5, pd.NA, 4.5, 6.5], 'Float64'),
            {'value': '4.5', 'dtype': 'float'},
        ),
        'timestamp': (
            series(
                pd.Series(
                    [
                        '2019-06-07',
                        '2023-01-01',
                        pd.NaT,
                        '2023-01-01',
                        '2025-12-31',
                    ],
                    dtype='datetime64[ns]',
                )
            ),
            {'value': '2023-01-01', 'dtype': 'date'},
        ),
        'int-against-float': (
            series([2, 4, pd.NA, 4, 6], 'Int64'),
            {'value': '4.5', 'dtype': 'float'},
        ),
    }
    expected = {
        '=': [False, True, False, True, False],
        '!=': [True, False, False, False, True],
        '<=': [True, True, False, True, False],
        '>=': [False, True, False, True, True],
        '<': [True, False, False, False, False],
        '>': [False, False, False, False, True],
    }
    from predictive_query.schema.typing import RelOp

    for name, (left, right) in lefts.items():
        for op in RelOp:
            want = (
                {'dtype': 'bool', 'values': expected[op.value]}
                if name != 'int-against-float' and op.value in expected
                else _MISSING
            )
            g.add(
                f'{name}-{op.name}',
                'labels.rel_op',
                {'left': left, 'op': op.value, 'right': right},
                want,
            )
    for op, want in [
        ('=', [False, False, True, False, False]),
        ('!=', [True, True, False, True, True]),
    ]:
        g.add(
            f'null-{RelOp(op).name}',
            'labels.rel_op',
            {
                'left': lefts['int'][0],
                'op': op,
                'right': {'value': 'NULL', 'dtype': None},
            },
            {'dtype': 'bool', 'values': want},
        )
    g.add(
        'in',
        'labels.member_op',
        {
            'left': lefts['int'][0],
            'op': 'IN',
            'right': {
                'value': [
                    {'value': '2', 'dtype': 'int'},
                    {'value': '6', 'dtype': 'int'},
                ],
                'dtype': 'intlist',
            },
        },
        {'dtype': 'bool', 'values': [True, False, False, False, True]},
    )
    left, right = (
        series([True, False, True, False]),
        series([True, True, False, False]),
    )
    for op, want in [
        ('AND', [True, False, False, False]),
        ('OR', [True, True, True, False]),
        ('NOT', [False, True, False, True]),
    ]:
        g.add(
            op,
            'labels.bool_op',
            {'left': left, 'op': op, 'right': None if op == 'NOT' else right},
            {'dtype': 'bool', 'values': want},
        )
    feat = series([2, 4, pd.NA, 4, 6, 6, 2], 'Int64')
    for op in ['SUM', 'AVG', 'MIN', 'MAX', 'COUNT', 'LIST_DISTINCT']:
        for filter_na in (False, True):
            g.add(
                f'{op}-filter-na-{filter_na}',
                'labels.aggregation_type',
                {
                    'op': op,
                    'feat': feat,
                    'batch': [0, 0, 1, 3, 3, 3, 4],
                    'batch_size': 6,
                    'filter_na': filter_na,
                },
            )
    times = series(
        pd.to_datetime(pd.Series(['2025-01-01', '2025-01-10', '2025-02-01']))
    )
    for op in ['MIN', 'MAX']:
        for filter_na in (False, True):
            g.add(
                f'timestamp-{op}-filter-na-{filter_na}',
                'labels.aggregation_type',
                {
                    'op': op,
                    'feat': times,
                    'batch': [0, 0, 2],
                    'batch_size': 4,
                    'filter_na': filter_na,
                },
            )
    fact = frame({'test': range(1, 9)})
    for start in (0, None):
        for filter_na in (False, True):
            g.add(
                f'sum-start-{start}-filter-na-{filter_na}',
                'labels.aggregation',
                {
                    'aggregation': {
                        'target': 'fact.test',
                        'aggr': 'SUM',
                        'start': start,
                        'end': 2,
                        'unit': 'days',
                    },
                    'tables': {'fact': fact},
                    'times': {
                        'fact': series(
                            pd.Series(pd.date_range('2023-01-01', periods=8))
                        )
                    },
                    'batches': {'fact': [0, 0, 0, 0, 1, 1, 1, 1]},
                    'anchor_time': series(
                        pd.Series(pd.to_datetime(['2023-01-01', '2023-01-06']))
                    ),
                    'filter_na': filter_na,
                },
            )
    return g


# --- test_local_sampler.py -----------------------------------------------------


def _ints(values: Any) -> dict[str, Any]:
    r"""An ``int64`` array, as ``relationallib`` takes its arrays."""
    return encode(np.array(values, dtype=np.int64))


def _sampler(
    node_types: list[str],
    edge_types: list[tuple[str, str, str]],
    colptr: dict[str, Any],
    row: dict[str, Any],
    node_time: Any,
    **seed: Any,
) -> dict[str, Any]:
    return {
        'node_types': node_types,
        'edge_types': [list(e) for e in edge_types],
        'colptr': colptr,
        'row': row,
        'node_time': node_time,
        **{k: encode(v) for k, v in seed.items()},
    }


def _call(
    num_neighbors: Any,
    offsets: Any,
    entity: str,
    seed_node: Any,
    seed_time: Any,
    **reseed: Any,
) -> dict[str, Any]:
    return {
        'num_neighbors': num_neighbors,
        'sampling_time_offsets': offsets,
        'entity_table': entity,
        'seed_node': seed_node,
        'seed_time': seed_time,
        **{k: encode(v) for k, v in reseed.items()},
    }


def local_sampler() -> Group:
    g = Group('local-sampler')
    op = 'sampling.neighbor_sample'

    # test_relationallib_sampler_validation
    two = ['a', 'b']
    edges = [('a', 'to', 'b'), ('b', 'rev_to', 'a')]
    colptr = {
        'a__to__b': _ints([0, 1, 2, 0, 1, 2]),
        'b__rev_to__a': _ints([0, 1, 2, 0, 1, 2]),
    }
    row = {
        'a__to__b': _ints([0, 1, 1, 0]),
        'b__rev_to__a': _ints([0, 1, 1, 0]),
    }
    node_time = {'a': _ints([1, 2]), 'b': _ints([1, 2])}
    hops = {'a__to__b': _ints([3, 3]), 'b__rev_to__a': _ints([3, 3])}
    seed_node, seed_time = _ints([0, 1]), _ints([0, 1])
    g.add(
        'validation-not-1d',
        op,
        {
            'sampler': _sampler(
                two,
                edges,
                {'a__to__b': 'invalid', 'b__to__a': None},
                row,
                node_time,
                seed=123,
            ),
            'calls': [],
        },
        Raises('is not 1D'),
    )
    valid = _sampler(two, edges, colptr, row, node_time, seed=123)
    for label, call, want in [
        (
            'validation-invalid-edge',
            _call({'invalid': 1}, {}, 'a', seed_node, seed_time),
            Raises('invalid'),
        ),
        (
            'validation-entity-table',
            _call(hops, {}, 'k', seed_node, seed_time),
            Raises('Invalid entity table'),
        ),
        (
            'validation-size-mismatch',
            _call(hops, {}, 'a', seed_node, _ints([0])),
            Raises('size mismatch'),
        ),
        (
            'validation-same-edge-type',
            _call(
                hops,
                {'a__to__b': [[0, 5], [None, 0]]},
                'a',
                seed_node,
                seed_time,
            ),
            Raises('same edge type'),
        ),
    ]:
        g.add(label, op, {'sampler': valid, 'calls': [call]}, want)

    # test_relationallib_sampler_temporal, _static and _temporal_range
    ab = ['A', 'B']
    both = [('A', 'to', 'B'), ('B', 'to', 'A')]
    colptr = {'A__to__B': _ints([0, 2, 4]), 'B__to__A': _ints([0, 1, 2, 3])}
    row = {'A__to__B': _ints([0, 2, 0, 2]), 'B__to__A': _ints([0, 1, 1])}
    g.add(
        'temporal',
        op,
        {
            'sampler': _sampler(
                ab,
                both,
                colptr,
                row,
                {'A': _ints([1, 0, 3]), 'B': _ints([2, 1])},
            ),
            'calls': [
                _call(
                    {'A__to__B': _ints([1, -1]), 'B__to__A': _ints([1, 1])},
                    {},
                    'A',
                    _ints([0, 1, 2]),
                    _ints([3, 1, 3]),
                )
            ],
        },
    )
    g.add(
        'static',
        op,
        {
            'sampler': _sampler(ab, both, colptr, row, []),
            'calls': [
                _call(
                    {'A__to__B': _ints([1, -1]), 'B__to__A': _ints([1, 1])},
                    {},
                    'A',
                    _ints([0, 1, 2]),
                    encode(np.array([])),
                )
            ],
        },
    )
    g.add(
        'temporal-range',
        op,
        {
            'sampler': _sampler(
                ab,
                both,
                colptr,
                row,
                {'A': _ints([1, 0, 3]), 'B': _ints([2, 1])},
            ),
            'calls': [
                _call(
                    {},
                    {
                        'A__to__B': _ints([[0, 1], [-1, 2]]),
                        'B__to__A': [[None, 0], [0, 0]],
                    },
                    'A',
                    _ints([0, 1, 2]),
                    _ints([3, 1, 3]),
                )
            ],
        },
    )

    # test_relationallib_seeds_as_random_state
    ten = {
        'colptr': {'B__to__A': _ints([0, 10])},
        'row': {'B__to__A': _ints(range(10))},
    }
    pick = _call(
        {'B__to__A': _ints([3])}, {}, 'A', _ints([0]), encode(np.array([]))
    )
    for seed in (0, 123, 2**31, 2**32 - 1, 2**32, 2**63, 2**64 + 3, 10**30):
        g.add(
            f'seed-{seed}-constructor',
            op,
            {
                'sampler': _sampler(
                    ab,
                    [('B', 'to', 'A')],
                    ten['colptr'],
                    ten['row'],
                    {},
                    seed=seed,
                ),
                'calls': [pick],
            },
        )
        g.add(
            f'seed-{seed}-reseeded',
            op,
            {
                'sampler': _sampler(
                    ab,
                    [('B', 'to', 'A')],
                    ten['colptr'],
                    ten['row'],
                    {},
                    seed=1,
                ),
                'calls': [{**pick, 'reseed': encode(seed)}],
            },
        )
    # test_relationallib_refuses_seeds_below_a_c_int
    for label, seed in (('below-c-int', -(2**31) - 1), ('float', 1.5)):
        g.add(
            f'seed-{label}',
            op,
            {
                'sampler': _sampler(['A'], [], {}, {}, {}, seed=seed),
                'calls': [],
            },
            Raises(code='TypeError'),
        )

    # test_a_seconds_bound_brackets_the_calendar_offset
    for kwds in (
        {'months': 3},
        {'months': -3},
        {'years': 1},
        {'years': -1},
        {'days': 5},
    ):
        for bound in ('earliest', 'latest'):
            g.add(
                f'{"-".join(f"{k}-{v}" for k, v in kwds.items())}-{bound}',
                'sampling.date_offset_to_seconds',
                {'offset': {'n': 1, 'kwds': kwds}, 'bound': bound},
            )
    # The month and fixed-length windows: an anchor plus the window's start.
    for label, anchor, kwds in [
        ('month-1', '2022-11-01', {'months': 1}),
        ('month-2', '2022-10-01', {'months': 2}),
        ('month-3', '2022-09-01', {'months': 3}),
        ('month-3-new-year', '2023-01-01', {'months': 3}),
        ('days-30', '2023-01-01', {'days': 30}),
        ('weeks-4', '2023-01-01', {'weeks': 4}),
    ]:
        for sign in (1, -1):
            g.add(
                f'{label}-{"plus" if sign > 0 else "minus"}',
                'sampling.date_offset_apply',
                {
                    'timestamps': [pd.Timestamp(anchor).value],
                    'offset': {'n': 1, 'kwds': kwds},
                    'sign': sign,
                },
            )
    return g


# --- test_local_graph_store.py ---------------------------------------------------


def _graph_spec(graph: Any) -> dict[str, Any]:
    r"""A graph as ``sampling.graph_store`` is told about it: each table's
    keys, time columns and columns' dtypes and stypes, and its links."""
    return {
        'tables': [
            {
                'name': t.name,
                'primary_key': t._primary_key,
                'time_column': t._time_column,
                'end_time_column': t._end_time_column,
                'columns': [
                    {
                        'name': c.name,
                        'dtype': str(c.dtype),
                        'stype': str(c.stype),
                    }
                    for c in t.columns
                ],
            }
            for t in graph.tables.values()
        ],
        'edges': [
            {'src_table': e.src_table, 'fkey': e.fkey, 'dst_table': e.dst_table}
            for e in graph.edges
        ],
    }


def local_graph_store() -> Group:
    from predictive_query.schema.typing import Stype
    from sdm_relational_engine.rfm import Graph

    g = Group('local-graph-store')
    op = 'sampling.graph_store'

    # test_local_graph_store
    df_dict = {
        'USERS': pd.DataFrame(
            {
                'USER_ID': pd.Series([0, 1, pd.NA, 2, 1, 3, 4], dtype='Int64'),
                'TIME': ['1990-01-01'] * 6 + [None],
                'TEXT': [
                    'this is!',
                    'text',
                    None,
                    '',
                    ' abc',
                    'hello!world',
                    '?. :',
                ],
            }
        ),
        'ORDERS': pd.DataFrame(
            {
                'ORDER_ID': [str(i) for i in range(9)],
                'USER_ID': pd.Series(
                    [0, 1, 4, 2, 5, pd.NA, 4, 3, 3], dtype='Int64'
                ),
                'TIME': pd.date_range('2023-01-01', periods=8)[::-1].tolist()
                + [pd.NaT],
            }
        ),
        'VIEWS': pd.DataFrame(
            {
                'USER_ID': pd.Series(
                    [0, 1, 4, 2, 5, pd.NA, 4, 3, 3], dtype='Int64'
                ),
                'TIME': pd.date_range('2023-01-01', periods=9)[::-1],
            }
        ),
        'RETURNS': pd.DataFrame({'ORDER_ID': ['2', '4', '7']}),
    }
    graph = Graph.from_data(df_dict, verbose=False)
    graph['USERS'].time_column = 'TIME'
    graph['USERS']['TEXT'].stype = Stype.text
    graph['RETURNS']['ORDER_ID'].stype = Stype.text
    tables = {name: frame(df) for name, df in df_dict.items()}
    g.add('shop', op, {'tables': tables, 'graph': _graph_spec(graph)})
    for name in df_dict:
        alone = Graph.from_data({name: df_dict[name]}, verbose=False)
        if name == 'USERS':
            alone['USERS'].time_column = 'TIME'
        g.add(
            f'only-{name}',
            op,
            {'tables': {name: tables[name]}, 'graph': _graph_spec(alone)},
        )

    # test_sanitization_reason_precedence_and_reference_count
    users = pd.DataFrame(
        {
            'USER_ID': pd.Series([pd.NA, 1, 1, 2], dtype='Int64'),
            'TIME': [None, None, None, '2025-01-01'],
        }
    )
    graph = Graph.from_data({'USERS': users}, verbose=False)
    graph['USERS'].time_column = 'TIME'
    g.add(
        'sanitization-precedence',
        op,
        {'tables': {'USERS': frame(users)}, 'graph': _graph_spec(graph)},
    )
    # The same table with no valid row left: the store refuses it.
    empty = pd.DataFrame(
        {
            'USER_ID': pd.Series([pd.NA, 1, 1], dtype='Int64'),
            'TIME': [None, None, '2025-01-01'],
        }
    )
    spec = _graph_spec(graph)
    g.add(
        'no-valid-rows',
        op,
        {
            'tables': {'USERS': frame(empty.iloc[:2])},
            'graph': spec,
        },
        Raises('Found no valid rows'),
    )
    return g


# --- base/test_utils.py -----------------------------------------------------------


def sampler_utils() -> Group:
    g = Group('sampler-utils')
    for seed in (
        0,
        1,
        42,
        2**31,
        2**32 - 1,
        2**32 + 7 + (3 << 64),
        2**32,
        2**33,
    ):
        g.add(
            f'random-state-{seed}',
            'sampling.random_state',
            {'seed': encode(seed), 'low': 0, 'high': 2**31, 'size': 8},
        )
    return g


# --- sequences read back as text --------------------------------------------


#: Rows of a sequence column stored as text, each read alone and beside the
#: others: what numpy reads, what it refuses, and what it reads oddly.
SEQUENCE_ROWS = (
    '[1.5, 2, 3]',
    '[ 1 , 2 ]',
    '[]',
    '',
    '4',
    '[1,2,]',
    '[1, 2, ]',
    '  ',
    '[nan, inf, -Infinity]',
    '[1e40, -1e-50, 0.1]',
    '[+3, -4, .5, 6.]',
    '[[1, 2]]',
    '[1,,2]',
    '[1; 2]',
    '[1 2]',
    '[0x10]',
    '[1e]',
    'not a list',
    '[9223372036854775808, -9223372036854775809, - 5]',
)


def sampler_sequences() -> Group:
    g = Group('sampler-sequences')
    op = 'sampling.to_list'
    for i, row in enumerate(SEQUENCE_ROWS):
        for dtype, stype in (('string', 'sequence'), ('intlist', None)):
            g.add(
                f'row-{i}-{dtype}',
                op,
                {
                    'series': 'arrow',
                    'values': [row, '[1]'],
                    'dtype': dtype,
                    'stype': stype,
                },
            )
            # Where numpy refuses a row and every row is JSON, the column is
            # whatever json.loads (or orjson.loads) gives, which the core
            # refuses to guess at (tests/rust_core/rfm/test_sequences.py).
            given = g.cases[-1].get('expected', {}).get('values', [])
            if any(isinstance(value, list) for value in given):
                g.cases.pop()
    # Each holder of text, with every row read, with a missing row, and with
    # a row that is neither numbers nor JSON; and no rows at all.
    for series in ('arrow', 'str', 'object'):
        for label, values in (
            ('read', ['[1, 2.5]', '[]', '[3]']),
            ('missing', ['[1, 2.5]', None]),
            ('unread', ['[1, 2.5]', '[1 2]']),
            ('none', []),
            ('all-missing', [None, None]),
        ):
            g.add(
                f'{series}-{label}',
                op,
                {
                    'series': series,
                    'values': values,
                    'dtype': 'string',
                    'stype': 'sequence',
                },
            )
    g.add(
        'floatlist',
        op,
        {
            'series': 'arrow',
            'values': ['[1, 2]'],
            'dtype': 'floatlist',
            'stype': None,
        },
    )
    g.add(
        'stringlist-unread',
        op,
        {
            'series': 'arrow',
            'values': ['[a]'],
            'dtype': 'stringlist',
            'stype': None,
        },
    )
    g.add(
        'not-a-sequence',
        op,
        {
            'series': 'arrow',
            'values': ['[1, 2]'],
            'dtype': 'string',
            'stype': 'text',
        },
    )
    return g


# --- the sampling driver, end to end ----------------------------------------

#: The marketplace's tables (scripts/parity/_parity/data.py), and the large
#: one's, which has no reviews.
MARKETPLACE = ['customers', 'items', 'reviews', 'stores', 'purchases']
LARGE = ['customers', 'items', 'stores', 'purchases']
#: The marketplace's one key inference cannot tell from a foreign key.
STORE_KEY = {'stores': 'store_code'}
COUNT = 'PREDICT COUNT(purchases.*, 0, 30, days) FOR EACH customers.customer_id'
VIP = COUNT + " WHERE customers.segment = 'vip'"
#: PostgreSQL loads a case's tables into this schema, made afresh.
PG_SCHEMA = 'parity_driver'


def _driver_input(
    backend: str,
    query: str,
    options: dict[str, Any],
    *,
    generator: str = 'marketplace',
    tables: list[str] | None = None,
    setup: list[str] | None = None,
    primary_keys: dict[str, Any] | None = None,
    time_columns: dict[str, str] | None = None,
    relationships: list[list[str]] | None = None,
    stypes: dict[str, str] | None = None,
    column_types: dict[str, str] | None = None,
    check_keys: bool | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {'generator': generator, 'seed': 0}
    if backend == 'postgres':
        data['namespace'] = {'schema': PG_SCHEMA}
    if column_types:
        data['column_types'] = column_types
    out: dict[str, Any] = {
        'backend': backend,
        'data': data,
        'tables': list(
            tables
            or (LARGE if generator == 'marketplace-large' else MARKETPLACE)
        ),
        'primary_keys': dict(
            STORE_KEY if primary_keys is None else primary_keys
        ),
        'query': query,
        'options': encode({'random_seed': 7, **options}),
    }
    for key, value in (
        ('setup', setup),
        ('time_columns', time_columns),
        ('relationships', relationships),
        ('stypes', stypes),
        ('check_keys', check_keys),
    ):
        if value is not None:
            out[key] = value
    return out


class _Driver:
    r"""The ``driver`` group, with checks that each case does what its label
    says it does."""

    def __init__(self) -> None:
        self.group = Group('driver')

    def add(
        self,
        label: str,
        input: dict[str, Any],
        raises: str | None = None,
        warns: tuple[str, ...] = (),
        quiet: tuple[str, ...] = (),
        warns_first: tuple[str, ...] = (),
    ) -> None:
        r"""``warns`` must each be in a warning given by a question answered,
        ``quiet`` in none, and ``warns_first`` in a warning given before an
        error."""
        want: Any = Raises(raises) if raises is not None else _MISSING
        self.group.add(label, 'driver.materialize', input, want)
        case = self.group.cases[-1]
        given = [m for _, m in case.get('expected', {}).get('warnings', [])]
        for text in warns:
            if not any(text in m for m in given):
                raise SystemExit(f'{case["id"]}: no warning says {text!r}')
        before = [
            m for _, m in case.get('expected_error', {}).get('warnings', [])
        ]
        for text in warns_first:
            if not any(text in m for m in before):
                raise SystemExit(
                    f'{case["id"]}: no warning before the error says {text!r}'
                )
        for text in quiet:
            if any(text in m for m in given):
                raise SystemExit(f'{case["id"]}: a warning says {text!r}')


def _duckdb_keys() -> list[str]:
    r"""DuckDB: every key declared but the reviews', every time column NOT
    NULL, a one-to-one table whose foreign key is UNIQUE, and indexes of each
    shape DuckDB lists: one column, a quoted column, two columns, an
    expression, and one on a table the question leaves out."""
    return [
        'ALTER TABLE customers ADD PRIMARY KEY (customer_id)',
        'ALTER TABLE customers ALTER COLUMN signup_at SET NOT NULL',
        'ALTER TABLE items ADD PRIMARY KEY (item_id)',
        'ALTER TABLE stores ADD PRIMARY KEY (store_code)',
        'ALTER TABLE purchases ADD PRIMARY KEY (purchase_id)',
        'ALTER TABLE purchases ALTER COLUMN purchased_at SET NOT NULL',
        'ALTER TABLE purchases ALTER COLUMN customer_id SET NOT NULL',
        'CREATE INDEX purchases_customer ON purchases (customer_id)',
        'CREATE INDEX purchases_item_time ON purchases (item_id, purchased_at)',
        'CREATE INDEX purchases_cents ON purchases ((amount * 100))',
        'CREATE TABLE cards ("Card No" BIGINT PRIMARY KEY, '
        'customer_id BIGINT UNIQUE, issued_at TIMESTAMP NOT NULL)',
        'INSERT INTO cards SELECT 1000 + customer_id, customer_id, signup_at '
        'FROM customers WHERE customer_id % 3 = 0',
        'CREATE INDEX cards_number ON cards ("Card No")',
        'CREATE INDEX reviews_item ON reviews (item_id)',
        'CREATE TABLE archive AS SELECT * FROM purchases LIMIT 5',
        'CREATE INDEX archive_customer ON archive (customer_id)',
    ]


def _duckdb_every_key() -> list[str]:
    r"""DuckDB: every primary key declared and every foreign key indexed."""
    return [
        'ALTER TABLE customers ADD PRIMARY KEY (customer_id)',
        'ALTER TABLE items ADD PRIMARY KEY (item_id)',
        'ALTER TABLE reviews ADD PRIMARY KEY (review_id)',
        'ALTER TABLE stores ADD PRIMARY KEY (store_code)',
        'ALTER TABLE purchases ADD PRIMARY KEY (purchase_id)',
        'CREATE INDEX reviews_item ON reviews (item_id)',
        'CREATE INDEX purchases_customer ON purchases (customer_id)',
        'CREATE INDEX purchases_item ON purchases (item_id)',
        'CREATE INDEX purchases_store ON purchases (store_code)',
    ]


#: SQLite's tables rebuilt with keys and constraints, as SQLite can only
#: declare them when a table is made. Times stay text, as SQLite keeps them.
_SQLITE_TABLES = {
    'customers': 'customer_id INTEGER PRIMARY KEY, segment TEXT, '
    'signup_at TEXT NOT NULL',
    'items': 'item_id INTEGER PRIMARY KEY, category TEXT, price REAL',
    'stores': 'store_code TEXT PRIMARY KEY, region TEXT',
    'purchases': 'purchase_id INTEGER PRIMARY KEY, '
    'customer_id INTEGER NOT NULL, item_id INTEGER, amount REAL, '
    'purchased_at TEXT NOT NULL, store_code TEXT',
}


def _sqlite_rebuilt(*tables: str) -> list[str]:
    out = []
    for table in tables:
        out += [
            f'ALTER TABLE {table} RENAME TO {table}_loaded',
            f'CREATE TABLE {table} ({_SQLITE_TABLES[table]})',
            f'INSERT INTO {table} SELECT * FROM {table}_loaded',
            f'DROP TABLE {table}_loaded',
        ]
    return out


def _sqlite_keys() -> list[str]:
    r"""SQLite: keys and NOT NULL time columns, a one-to-one table whose
    foreign key is UNIQUE, an index on one column, and one on two columns
    whose first is a foreign key."""
    return [
        *_sqlite_rebuilt('customers', 'items', 'stores', 'purchases'),
        'CREATE INDEX purchases_customer ON purchases (customer_id)',
        'CREATE INDEX purchases_item_time ON purchases (item_id, purchased_at)',
        'CREATE TABLE cards (card_no INTEGER PRIMARY KEY, '
        'customer_id INTEGER UNIQUE, issued_at TEXT NOT NULL)',
        'INSERT INTO cards SELECT 1000 + customer_id, customer_id, signup_at '
        'FROM customers WHERE customer_id % 3 = 0',
    ]


def _sqlite_every_key() -> list[str]:
    return [
        *_sqlite_rebuilt('customers', 'items', 'stores', 'purchases'),
        'CREATE INDEX reviews_item ON reviews (item_id)',
        'CREATE UNIQUE INDEX reviews_id ON reviews (review_id)',
        'CREATE INDEX purchases_customer ON purchases (customer_id)',
        'CREATE INDEX purchases_item ON purchases (item_id)',
        'CREATE INDEX purchases_store ON purchases (store_code)',
    ]


def _postgres_keys() -> list[str]:
    r"""PostgreSQL: every key declared, NOT NULL time columns and foreign
    keys, and an index on a foreign key."""
    s = PG_SCHEMA
    return [
        f'ALTER TABLE {s}.customers ADD PRIMARY KEY (customer_id)',
        f'ALTER TABLE {s}.customers ALTER COLUMN signup_at SET NOT NULL',
        f'ALTER TABLE {s}.items ADD PRIMARY KEY (item_id)',
        f'ALTER TABLE {s}.reviews ADD PRIMARY KEY (review_id)',
        f'ALTER TABLE {s}.stores ADD PRIMARY KEY (store_code)',
        f'ALTER TABLE {s}.purchases ADD PRIMARY KEY (purchase_id)',
        f'ALTER TABLE {s}.purchases ALTER COLUMN purchased_at SET NOT NULL',
        f'ALTER TABLE {s}.purchases ALTER COLUMN customer_id SET NOT NULL',
        f'CREATE INDEX purchases_customer ON {s}.purchases (customer_id)',
    ]


#: Questions the engine refuses inside ``KumoRelational._get_task_table`` or
#: ``_get_context``: label -> (query, options).
_REFUSED: dict[str, tuple[str, dict[str, Any]]] = {
    # An entity filter no customer passes: the empty sample's diagnosis
    # names the filter.
    'filter-keeps-none': (
        COUNT + " WHERE customers.segment = 'nobody'",
        {'indices': [1]},
    ),
    # The same, with the context's window running past the latest time.
    'filter-keeps-none-late': (
        COUNT + " WHERE customers.segment = 'nobody'",
        {
            'indices': [1],
            'anchor_time': pd.Timestamp('2024-12-29'),
            'context_anchor_time': pd.Timestamp('2024-12-20'),
        },
    ),
    # A static question that labels nobody.
    'static-keeps-none': (
        'PREDICT customers.segment FOR EACH customers.customer_id '
        "WHERE customers.segment = 'nobody'",
        {'indices': [1]},
    ),
    'anchor-before-data': (
        COUNT,
        {'indices': [1], 'anchor_time': pd.Timestamp('2023-01-01')},
    ),
    'context-before-data': (
        COUNT,
        {
            'indices': [1],
            'anchor_time': pd.Timestamp('2024-10-01'),
            'context_anchor_time': pd.Timestamp('2023-01-01'),
        },
    ),
    # Each customer's own time, asked of a temporal question.
    'entity-anchor-temporal': (
        COUNT,
        {'indices': [1], 'anchor_time': 'entity'},
    ),
    'no-pq-iterations': (COUNT, {'indices': [1], 'max_pq_iterations': 0}),
    'entity-context-only': (
        'PREDICT customers.segment FOR EACH customers.customer_id',
        {
            'indices': [1],
            'anchor_time': pd.Timestamp('2024-10-01'),
            'context_anchor_time': 'entity',
        },
    ),
    'entity-prediction-only': (
        'PREDICT customers.segment FOR EACH customers.customer_id',
        {
            'indices': [1],
            'anchor_time': 'entity',
            'context_anchor_time': pd.Timestamp('2024-10-01'),
        },
    ),
    # Each entity's own time, of a table without one.
    'entity-anchor-timeless': (
        'PREDICT items.category FOR EACH items.item_id',
        {'indices': [1], 'anchor_time': 'entity'},
    ),
    'negative-lag': (COUNT, {'indices': [1], 'lag_timesteps': -1}),
    'exclude-unknown-table': (
        COUNT,
        {'indices': [1], 'exclude_cols_dict': {'nothing': ['segment']}},
    ),
    'exclude-unknown-column': (
        COUNT,
        {'indices': [1], 'exclude_cols_dict': {'customers': ['nothing']}},
    ),
    # A key that does not cast to the entity key's type.
    'uncastable-index': (COUNT, {'indices': ['one']}),
}


def driver() -> Group:
    d = _Driver()
    small = {'indices': [1, 2], 'num_context_samples': 30}
    index_warning = 'for optimal database querying'

    # Keys, constraints and indexes the corpus's tables never have.
    keyed = MARKETPLACE[:-1] + ['purchases', 'cards']
    d.add(
        'duckdb-keys-and-indexes',
        _driver_input(
            'duckdb', COUNT, small, tables=keyed, setup=_duckdb_keys()
        ),
        warns=(index_warning,),
    )
    d.add(
        'duckdb-every-key-indexed',
        _driver_input('duckdb', COUNT, small, setup=_duckdb_every_key()),
        quiet=(index_warning,),
    )
    d.add(
        'duckdb-one-keyed-table',
        _driver_input(
            'duckdb',
            'PREDICT customers.segment FOR EACH customers.customer_id',
            {**small, 'model': 'kumo-relational'},
            tables=['customers'],
            primary_keys={},
            setup=['ALTER TABLE customers ADD PRIMARY KEY (customer_id)'],
        ),
        quiet=(index_warning,),
    )
    d.add(
        'sqlite-keys-and-indexes',
        _driver_input(
            'sqlite', COUNT, small, tables=keyed, setup=_sqlite_keys()
        ),
        warns=(index_warning,),
    )
    d.add(
        'sqlite-every-key-indexed',
        _driver_input('sqlite', COUNT, small, setup=_sqlite_every_key()),
        quiet=(index_warning,),
    )
    d.add(
        'postgres-keys',
        _driver_input('postgres', COUNT, small, setup=_postgres_keys()),
    )
    for backend in ('duckdb', 'sqlite', 'postgres'):
        setup = {
            'duckdb': _duckdb_every_key,
            'sqlite': _sqlite_every_key,
            'postgres': _postgres_keys,
        }[backend]()
        d.add(
            f'{backend}-keys-stores',
            _driver_input(
                backend,
                'PREDICT COUNT(purchases.*, 0, 30, days) '
                'FOR EACH stores.store_code',
                {
                    'indices': ['S01', 'S02'],
                    'num_context_samples': 30,
                    'num_neighbors': [8, 8, 8],
                },
                setup=setup,
            ),
        )

    # Four hops: a customer's purchases, their items, those items' other
    # purchases and their stores, so the stores keyed by text reached at the
    # second hop are reached again at the fourth.
    for backend in ('duckdb', 'sqlite'):
        d.add(
            f'{backend}-stores-again',
            _driver_input(
                backend,
                COUNT,
                {**small, 'num_neighbors': [4, 4, 4, 4]},
                tables=['customers', 'items', 'stores', 'purchases'],
            ),
        )

    # Purchases at a store the stores table does not hold: every key a hop
    # looks up dangles.
    for backend in ('duckdb', 'sqlite', 'postgres'):
        table = (
            f'{PG_SCHEMA}.purchases' if backend == 'postgres' else 'purchases'
        )
        d.add(
            f'{backend}-dangling-stores',
            _driver_input(
                backend,
                COUNT,
                {**small, 'num_neighbors': [8, 8]},
                tables=['customers', 'items', 'stores', 'purchases'],
                setup=[f"UPDATE {table} SET store_code = 'S99'"],
            ),
        )

    # Tables too large for the sampler's first draw or its first batch.
    for backend in ('memory', 'duckdb', 'sqlite', 'postgres'):
        large = {'generator': 'marketplace-large'}
        d.add(
            f'{backend}-few-vips',
            _driver_input(
                backend,
                VIP,
                {
                    'indices': [1],
                    'num_context_samples': 30,
                    'max_pq_iterations': 1,
                },
                **large,
            ),
            warns=('To improve coverage',),
        )
        d.add(
            f'{backend}-vips-over-two-batches',
            _driver_input(
                backend,
                VIP,
                {
                    'indices': [1],
                    'num_context_samples': 1000,
                    'max_pq_iterations': 11,
                },
                **large,
            ),
        )
        d.add(
            f'{backend}-enough-in-one-batch',
            _driver_input(
                backend,
                COUNT,
                {
                    'indices': [1],
                    'num_context_samples': 30,
                    'max_pq_iterations': 2,
                },
                **large,
            ),
            quiet=('To improve coverage',),
        )

    # Sequences held as text, a table of nothing but keys, on DuckDB a column
    # of lists, and in memory a table keyed by a pair of columns. A SQL
    # sampler refuses such a table: its derived key column is named
    # `__kumo_key_...`, a name the SQL backends reserve.
    extras = ['customers', 'items', 'stores', 'purchases', 'cards']
    for backend in ('memory', 'duckdb', 'sqlite', 'postgres'):
        keys = {'stores': 'store_code', 'cards': 'card_no'}
        tables = extras
        if backend == 'memory':
            keys['stock'] = ['store_code', 'item_id']
            tables = [*extras, 'stock']
        d.add(
            f'{backend}-sequences',
            _driver_input(
                backend,
                COUNT,
                {**small, 'num_neighbors': [8, 8, 8]},
                generator='marketplace-extras',
                tables=tables,
                primary_keys=keys,
                stypes={'items.dims': 'sequence'},
                setup=[
                    'ALTER TABLE items ADD COLUMN sizes DOUBLE[]',
                    'UPDATE items SET sizes = [item_id * 1.5, 2.0]',
                ]
                if backend == 'duckdb'
                else None,
            ),
        )
    d.add(
        'duckdb-a-pair-of-key-columns',
        _driver_input(
            'duckdb',
            COUNT,
            small,
            generator='marketplace-extras',
            tables=[*extras, 'stock'],
            primary_keys={
                'stores': 'store_code',
                'cards': 'card_no',
                'stock': ['store_code', 'item_id'],
            },
        ),
        # Phase 9's fix: the key's derived column is the client's own, no
        # longer refused as a reserved name.
    )

    # Columns of the types a database has besides the corpus's, read as
    # features of the items: each kept, cast or refused as today's Python
    # keeps, casts or refuses it.
    items_sum = (
        'PREDICT SUM(purchases.amount, 0, 30, days) FOR EACH items.item_id'
    )
    four = ['customers', 'items', 'stores', 'purchases']
    duckdb_types = {
        'released': ('DATE', "DATE '2024-01-01' + CAST(item_id AS INTEGER)"),
        'code': ('UUID', 'md5(CAST(item_id AS VARCHAR))::UUID'),
        'cost': ('DECIMAL(10,2)', 'price'),
        'spec': ('JSON', """'{"w": 1, "tags": ["a"]}'"""),
        'listed_at': ('TIMESTAMPTZ', "TIMESTAMPTZ '2024-01-01 00:00:00+02'"),
        'huge': ('HUGEINT', 'item_id'),
        'tiny': ('UTINYINT', 'item_id'),
        'weight': ('FLOAT', 'price'),
        'heavy': ('BOOLEAN', 'item_id % 2 = 0'),
    }
    postgres_types = {
        'released': ('DATE', "DATE '2024-01-01' + item_id::int"),
        'code': ('UUID', 'md5(item_id::text)::uuid'),
        'cost': ('NUMERIC(10,2)', 'price'),
        'units': (
            'NUMERIC(10,0)',
            "CASE WHEN item_id = 2 THEN 'NaN'::numeric ELSE item_id END",
        ),
        'ratio': (
            'NUMERIC',
            "CASE WHEN item_id = 3 THEN 'NaN'::numeric ELSE price / 3 END",
        ),
        'spec': (
            'JSONB',
            "jsonb_build_object('w', item_id, 'é', jsonb_build_array(1.5, 'x', true, null))",
        ),
        'note': ('JSON', """'"just text"'"""),
        'host': ('INET', "'192.168.0.1'"),
        'page': ('XML', "'<a/>'"),
        'fee': ('MONEY', '12.5'),
        'heavy': ('BOOLEAN', 'item_id % 2 = 0'),
        'weight': ('REAL', 'price'),
        'rank': ('SMALLINT', 'item_id::smallint'),
    }

    def typed(backend: str, columns: dict[str, tuple[str, str]]) -> list[str]:
        table = f'{PG_SCHEMA}.items' if backend == 'postgres' else 'items'
        out = []
        for name, (sql_type, value) in columns.items():
            out += [
                f'ALTER TABLE {table} ADD COLUMN {name} {sql_type}',
                f'UPDATE {table} SET {name} = {value}',
            ]
        return out

    for backend, columns in (
        ('duckdb', duckdb_types),
        ('postgres', postgres_types),
    ):
        d.add(
            f'{backend}-column-types',
            _driver_input(
                backend,
                items_sum,
                small,
                tables=four,
                setup=typed(backend, columns),
            ),
        )
        # A date past what `datetime64[ns]` holds, and a time of day, which
        # pandas reads as no time at all.
        d.add(
            f'{backend}-a-date-past-2262',
            _driver_input(
                backend,
                items_sum,
                small,
                tables=four,
                setup=typed(
                    backend,
                    {
                        'released': (
                            'DATE',
                            "CASE WHEN item_id = 3 THEN DATE '9999-12-31' "
                            "ELSE DATE '2024-01-01' END",
                        )
                    },
                ),
            ),
            raises='Out of bounds nanosecond timestamp',
        )
        d.add(
            f'{backend}-a-time-of-day',
            _driver_input(
                backend,
                items_sum,
                small,
                tables=four,
                setup=typed(backend, {'opens': ('TIME', "TIME '10:00:00'")}),
            ),
            # Phase 9's fix: a time of day is text, not the time column
            # whose values all read as NaT.
        )

    # Foreign keys with nulls, and, with the key check skipped, primary keys
    # with nulls: numpy reads integers with a null as floats.
    for backend in ('duckdb', 'sqlite', 'postgres'):
        table = (
            f'{PG_SCHEMA}.purchases' if backend == 'postgres' else 'purchases'
        )
        d.add(
            f'{backend}-null-foreign-keys',
            _driver_input(
                backend,
                COUNT,
                {**small, 'num_neighbors': [8, 8]},
                tables=four,
                setup=[
                    f'UPDATE {table} SET item_id = NULL WHERE purchase_id % 7 = 0',
                    f'UPDATE {table} SET store_code = NULL '
                    'WHERE purchase_id % 11 = 0',
                ],
            ),
        )
        d.add(
            f'{backend}-null-primary-keys',
            _driver_input(
                backend,
                COUNT,
                small,
                tables=four,
                check_keys=False,
                setup=[
                    f'UPDATE {table} SET purchase_id = NULL '
                    'WHERE customer_id IN (1, 2)'
                ],
            ),
        )

    # Keys that are floats, and an anchor with a time zone.
    d.add(
        'duckdb-float-keys',
        _driver_input(
            'duckdb',
            COUNT,
            {'indices': [1.0, 2.0], 'num_context_samples': 30},
            column_types={
                'customers.customer_id': 'DOUBLE',
                'purchases.customer_id': 'DOUBLE',
            },
        ),
    )
    for backend in ('memory', 'duckdb'):
        d.add(
            f'{backend}-zoned-anchor',
            _driver_input(
                backend,
                COUNT,
                {
                    **small,
                    'anchor_time': pd.Timestamp('2024-10-01T02:00:00+02:00'),
                },
            ),
        )

    # Questions the engine refuses.
    for backend in ('memory', 'duckdb'):
        for label, (query, options) in _REFUSED.items():
            d.add(
                f'{backend}-{label}',
                _driver_input(
                    backend, query, {'num_context_samples': 30, **options}
                ),
                raises='',
            )
    # Refusals after the sampler has read times it could not parse: today's
    # Python warns of them as it reads them, before it raises, in sampling
    # the targets, after it, and in the context's step.
    garbled_refusals = (
        'filter-keeps-none',
        'negative-lag',
        'exclude-unknown-column',
    )
    for backend in ('duckdb', 'sqlite', 'postgres'):
        for label in garbled_refusals:
            query, options = _REFUSED[label]
            d.add(
                f'{backend}-garbled-{label}',
                _driver_input(
                    backend,
                    query,
                    {'num_context_samples': 30, **options},
                    generator='marketplace-garbled',
                    time_columns={'purchases': 'purchased_at'},
                ),
                raises='',
                warns_first=('Could not parse',),
            )
    for backend in ('memory', 'duckdb'):
        # A static question of a table without a time column, and one at
        # each entity's own time, that label nobody.
        d.add(
            f'{backend}-timeless-keeps-none',
            _driver_input(
                backend,
                'PREDICT items.category FOR EACH items.item_id '
                "WHERE items.category = 'nobody'",
                small,
            ),
            raises='',
        )
        d.add(
            f'{backend}-own-time-keeps-none',
            _driver_input(
                backend,
                'PREDICT customers.segment FOR EACH customers.customer_id '
                "WHERE customers.segment = 'nobody'",
                {**small, 'anchor_time': 'entity'},
            ),
            raises='',
        )
        # A static question whose anchor no purchase is old enough for.
        d.add(
            f'{backend}-no-purchase-yet',
            _driver_input(
                backend,
                'PREDICT purchases.amount FOR EACH purchases.purchase_id',
                {
                    **small,
                    'anchor_time': pd.Timestamp('2024-01-01T12:00:00'),
                    'context_anchor_time': pd.Timestamp('2024-01-01T12:00:00'),
                },
            ),
            raises='',
        )
        # A what-if clause.
        d.add(
            f'{backend}-assuming',
            _driver_input(
                backend,
                COUNT + ' ASSUMING COUNT(purchases.*, 0, 7, days) > 0',
                small,
            ),
        )
        # More tables two hops away than a request may carry.
        d.add(
            f'{backend}-too-many-tables',
            _driver_input(
                backend,
                COUNT,
                {**small, 'num_neighbors': [4, 4]},
                generator='marketplace-wide',
                tables=[
                    *MARKETPLACE,
                    *(f'notes_{n:02d}' for n in range(1, 13)),
                ],
            ),
            raises='',
        )
    # A target the schema declares multicategorical, which the engine has
    # no task type for (rfm.py, _get_task_type).
    for backend in ('memory', 'duckdb'):
        d.add(
            f'{backend}-multicategorical-target',
            _driver_input(
                backend,
                'PREDICT customers.segment FOR EACH customers.customer_id',
                small,
                stypes={'customers.segment': 'multicategorical'},
            ),
            raises='',
        )
    # Customers nobody recorded the signup time of: none existed at any
    # anchor, and the diagnosis finds no earliest time to name.
    d.add(
        'duckdb-no-signup-times',
        _driver_input(
            'duckdb',
            'PREDICT customers.segment FOR EACH customers.customer_id',
            small,
            setup=['UPDATE customers SET signup_at = NULL'],
            time_columns={'customers': 'signup_at'},
        ),
        raises='',
    )

    # A table without a key whose time column holds no time. (SQLite and
    # PostgreSQL are loaded from frames whose missing times they read as
    # numbers, so the question is asked of the other two.)
    for backend in ('memory', 'duckdb'):
        d.add(
            f'{backend}-a-table-without-times',
            _driver_input(
                backend,
                'PREDICT customers.segment FOR EACH customers.customer_id',
                small,
                generator='marketplace-untimed',
                time_columns={'purchases': 'purchased_at'},
            ),
        )
    d.add(
        'memory-an-unknown-key',
        _driver_input(
            'memory', COUNT, {'indices': [999_999], 'num_context_samples': 30}
        ),
        raises='',
    )
    d.add(
        'sqlite-earliest-time-unreadable',
        _driver_input(
            'sqlite',
            COUNT,
            small,
            setup=[
                "UPDATE purchases SET purchased_at = '' WHERE purchase_id = 3"
            ],
        ),
        # Phase 9's fix: the unreadable earliest time is dropped, as the
        # others are, where today's failed the request on it.
    )
    d.add(
        'duckdb-a-scored-key-repeats',
        _driver_input(
            'duckdb',
            COUNT,
            small,
            check_keys=False,
            setup=[
                'INSERT INTO customers SELECT * FROM customers WHERE customer_id = 1'
            ],
        ),
        raises='',
    )
    d.add(
        'duckdb-an-unknown-key',
        _driver_input(
            'duckdb', COUNT, {'indices': [999_999], 'num_context_samples': 30}
        ),
        raises='',
    )
    return d.group


def arrow_switch() -> Group:
    import importlib.util

    source = REPO / 'tests/rust_core/test_public_dtypes_on_both_paths.py'
    spec = importlib.util.spec_from_file_location('public_dtypes', source)
    assert spec is not None and spec.loader is not None
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    group = Group('arrow-switch', str(source.relative_to(REPO)))
    selected = [
        'time:arrow_timestamp_us',
        'time:arrow_timestamp_ns',
        'time:arrow_date32',
        'key:arrow_string',
        'feature:arrow_bool',
        'feature:arrow_date32',
        'feature:arrow_timestamp_us',
        'feature:arrow_large_string',
        'key:uint64_past_int64',
        'feature:arrow_duration',
        'feature:arrow_dictionary',
        'feature:arrow_time32',
        'feature:arrow_time64',
        'key:arrow_large_string',
    ]
    for name in selected:
        tables = cases.frames(name)
        recorded = group.flow(
            name,
            {
                'tables': {
                    table: frame(data) for table, data in tables.items()
                },
                'calls': [
                    {'call': 'Schema.from_data'},
                    {
                        'call': 'PredictiveQuery',
                        'args': {'query': cases.QUESTION},
                    },
                    {
                        'call': 'PredictiveQuery.materialize',
                        'args': {
                            'indices': encode(
                                list(tables['users']['user_id'][:2])
                            ),
                            'anchor_time': encode(pd.Timestamp(cases.ANCHOR)),
                            'random_seed': 7,
                        },
                    },
                    {
                        'call': 'SDMClient.predict',
                        'args': {'model': 'kumo-relational'},
                    },
                    {'call': 'PredictionResult.to_pandas'},
                ],
                'show_bodies': True,
            },
        )
        if name in {'feature:arrow_date32', 'feature:arrow_timestamp_us'}:
            assert recorded['expected_error']['code'] == 'INVALID_REQUEST'
        elif name in cases.FIXED and name != 'key:uint64_past_int64':
            assert recorded['expected_error']['code'] == 'UNSUPPORTED_REQUEST'
        else:
            assert 'expected' in recorded, recorded
    return group


GROUPS = (
    dates,
    wire,
    time_and_keys,
    relationships,
    binary_target,
    decomposition,
    json_payload,
    dtype_fidelity,
    composite_key,
    pandas_executor,
    local_sampler,
    local_graph_store,
    sampler_utils,
    sampler_sequences,
    driver,
    arrow_switch,
)


def _first_difference(old: Any, new: Any, path: str) -> str | None:
    r"""Where ``old`` and ``new`` first differ, and how, or ``None``."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in [*old, *(k for k in new if k not in old)]:
            if key not in old or key not in new:
                return f'{path}.{key} only in the {"new" if key in new else "stored"} case'
            found = _first_difference(old[key], new[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(old, list) and isinstance(new, list):
        for i, (a, b) in enumerate(zip(old, new, strict=False)):
            found = _first_difference(a, b, f'{path}[{i}]')
            if found:
                return found
        if len(old) != len(new):
            return f'{path}: {len(old)} items stored, {len(new)} written'
        return None
    if old == new and type(old) is type(new):
        return None
    shown = [json.dumps(v, ensure_ascii=False)[:200] for v in (old, new)]
    return f'{path}: stored {shown[0]}, written {shown[1]}'


#: How many changed cases ``--check`` describes per file.
SHOWN_CASES = 10


def _case_differences(stored: str | None, written: str) -> list[str]:
    r"""The cases of a fixture file that differ, each with where it first
    differs, at most ``SHOWN_CASES`` of them."""
    if stored is None:
        return ['not stored']
    try:
        old = {c['id']: c for c in json.loads(stored)['cases']}
    except (ValueError, KeyError, TypeError):
        return ['the stored file is not a fixture file']
    new = {c['id']: c for c in json.loads(written)['cases']}
    lines = [f'case {i} only stored' for i in old if i not in new]
    lines += [f'case {i} only written' for i in new if i not in old]
    for i in old:
        if i in new:
            found = _first_difference(old[i], new[i], '')
            if found:
                lines.append(f'case {i}: {found}')
    if not lines:
        return ['the same cases, written differently']
    more = len(lines) - SHOWN_CASES
    return lines[:SHOWN_CASES] + ([f'and {more} more'] if more > 0 else [])


def _groups() -> tuple[Any, ...]:
    from _fixture_messages import message_groups

    return (*GROUPS, *message_groups(Group, frame, Raises, encode))


HOST_EVENTS = REPO / 'tests/rust_core/shadow/host_events.json'


def capture_host_events() -> dict:
    sys.path.insert(0, str(REPO / 'scripts/parity'))

    from unittest.mock import patch

    import pandas as pd
    from _container_events import capture_container_events
    from _scalar_events import capture_scalar_events
    from sdm_client import Schema, _impl

    sys.path.insert(
        0, str(REPO / 'packages/structured-data-models-client/tests')
    )
    import test_a_metric_view_filter_holds as metric
    import test_semantic_model_snowflake as semantic

    definition = metric._VIEW.replace(
        'expr: source.quantity', 'expr: mystery(source.quantity)'
    )
    metric_source = metric.DatabricksSource(
        metric._Warehouse(definition), catalog='cat', schema='sch'
    ).semantic_model('sales_mv')
    semantic_source = semantic._model()
    events = {
        'containers': capture_container_events(),
        'scalars': capture_scalar_events(),
    }
    with patch.object(
        semantic,
        '_SEMANTIC_VIEW_YAML',
        semantic._SEMANTIC_VIEW_YAML.replace(
            'expr: racks.HALL', 'expr: mystery(racks.HALL)'
        ),
    ):
        for name, source in [
            ('metric', metric_source),
            ('semantic', semantic_source),
        ]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                Schema.from_source(source)
            events[name] = [
                {
                    'category': f'{warning.category.__module__}.{warning.category.__qualname__}',
                    'message': str(warning.message),
                }
                for warning in caught
            ]
    if not _impl.OLD:
        from sdm_core import _native

        with patch.object(
            _native,
            'data_schema',
            side_effect=_native.CorePanic(
                'the core panicked; this is a bug in it'
            ),
        ):
            try:
                Schema.from_data({'users': pd.DataFrame({'user_id': [1, 2]})})
            except Exception as error:
                events['panic'] = {
                    'class': f'{type(error).__module__}.{type(error).__qualname__}',
                    'code': getattr(error, 'code', None),
                    'message': str(error),
                    'visible_cause': error.__cause__ is not None
                    or not error.__suppress_context__,
                }
            else:
                raise AssertionError('the injected binding panic was swallowed')
        assert not any(
            name.split('.')[0] in {'sdm_relational_engine', 'predictive_query'}
            for name in sys.modules
        )
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--host-events', action='store_true')
    parser.add_argument(
        '--only', help='comma-separated groups to write, for working on them'
    )
    parser.add_argument(
        '--keep-going',
        action='store_true',
        help='report every case its test disagrees with, write the rest, and '
        'exit 1',
    )
    args = parser.parse_args(argv)
    if args.host_events:
        events = capture_host_events()
        if args.check:
            expected = json.loads(HOST_EVENTS.read_text())
            assert events == {name: expected[name] for name in events}, events
        else:
            assert 'panic' in events
            HOST_EVENTS.write_text(json.dumps(events, indent=2) + '\n')
        print(f'{len(events)} host event recordings match')
        return 0
    global KEEP_GOING
    KEEP_GOING = args.keep_going
    only = set(args.only.split(',')) if args.only else None
    written = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for make in _groups():
            if only is not None and make.__name__.replace('_', '-') not in only:
                continue
            group = make()
            text = (
                json.dumps(
                    {'source': group.source, 'cases': group.cases},
                    indent=1,
                    ensure_ascii=False,
                )
                + '\n'
            )
            written[FIXTURES / f'{group.name}.json'] = text
            print(f'{group.name}: {len(group.cases)} cases')
    if args.check:
        stale = [
            p
            for p, text in written.items()
            if not p.exists() or p.read_text() != text
        ]
        if stale:
            kept = Path(tempfile.mkdtemp(prefix='make-fixtures-check-'))
            for path in stale:
                (kept / path.name).write_text(written[path], encoding='utf-8')
                stored = path.read_text() if path.exists() else None
                for line in _case_differences(stored, written[path]):
                    print(f'  {path.name}: {line}', file=sys.stderr)
            print(f'what would be written is in {kept}', file=sys.stderr)
        for path in stale:
            print(f'differs: {path.relative_to(REPO)}', file=sys.stderr)
        return 1 if stale else 0
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for path, text in written.items():
        path.write_text(text, encoding='utf-8')
    print(
        f'{sum(len(json.loads(t)["cases"]) for t in written.values())} cases written'
    )
    for message in DISAGREEING:
        print(f'disagrees: {message}', file=sys.stderr)
    return 1 if DISAGREEING else 0


if __name__ == '__main__':
    raise SystemExit(main())
