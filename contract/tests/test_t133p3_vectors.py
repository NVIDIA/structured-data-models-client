# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.33 part 3's rows held to their own shape and its pending list.

Written by the package's independent test author, before the change.

    python -m pytest contract/tests/test_t133p3_vectors.py -q

``rust/crates/sdm-testkit/tests/t133p3_reads.rs`` runs each row of
``contract/t133p3-vectors.json`` through real DuckDB; this file holds that
the rows say what the owner decided (2026-10-14, items 24 and 26): every
type a lossless connection hands over is read (none left out), a number
falls back to text only past decimal128(38, 0), and a time-named column of
values that are no times is text with no time column, beside the kept time
column.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ROWS: list[dict[str, Any]] = json.loads(
    (ROOT / 'contract/t133p3-vectors.json').read_text(encoding='utf-8')
)['rows']


def _gate() -> Any:
    path = ROOT / 'contract/tests/t133p3_gate.py'
    spec = importlib.util.spec_from_file_location('t133p3_gate', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _gate()
LOSSLESS = 'SET arrow_lossless_conversion = true'
TYPE_SET: dict[str, Any] = json.loads(
    (ROOT / 'contract/type-set-vectors.json').read_text(encoding='utf-8')
)
TYPES = {r['id']: r for r in TYPE_SET['vectors'] + TYPE_SET['reached']}
I64 = {'type': 'int64'}
D38 = {'type': 'decimal128', 'precision': 38, 'scale': 0}
UTF8 = {'type': 'utf8'}


def test_the_pending_list_only_shrinks() -> None:
    assert G.only_shrinks() == []
    ids = {row['id'] for row in ROWS} | {
        i for i, r in TYPES.items() if r.get('cells') is not None
    }
    assert set(G.gating()['pending']['rust-host']) <= ids


def test_each_row_is_a_witness_and_a_reading() -> None:
    ids = [row['id'] for row in ROWS]
    assert len(ids) == len(set(ids))
    for row in ROWS:
        last = row['witness'][-1]
        assert last.startswith('CREATE OR REPLACE TABLE users AS ')
        assert last.endswith(f' AS {row["column"]} FROM users')
        assert row['want']['dtype'] in ('string', 'int64', 'timestamp[us]')
        assert row['want']['cells'] in ('cast', 'null', None)
        assert row['why']


def test_every_lossless_type_is_read_none_left_out() -> None:
    lossless = {
        i: r
        for i, r in TYPES.items()
        if (r.get('witness') or [''])[0] == LOSSLESS
    }
    assert {
        'lossless.uuid',
        'lossless.bit',
        'lossless.timetz',
        'lossless.hugeint',
        'lossless.uhugeint',
        'lossless.varint',
        'lossless.json',
    } <= set(lossless)
    for row in lossless.values():
        assert 'left_out' not in row['inference'], row['id']
    for i in ('lossless.uuid', 'lossless.bit', 'lossless.json'):
        assert lossless[i]['inference'] == {'dtype': 'string'}
        assert lossless[i]['cells'] == 'cast'
    timetz = lossless['lossless.timetz']
    control = {r['id']: r for r in ROWS}['time.control']['want']
    assert timetz['inference'] == {'dtype': 'time'}
    assert timetz['cells'] == control['cells'] == 'null'


def test_the_big_integer_rule_has_each_tier() -> None:
    tiers = {
        'lossless.hugeint': I64,
        'lossless.hugeint-wide': D38,
        'lossless.hugeint-past-38-digits': UTF8,
        'lossless.uhugeint': I64,
        'lossless.uhugeint-wide': D38,
        'lossless.uhugeint-past-38-digits': UTF8,
        'varint': I64,
        'varint.large': I64,
        'varint.view': I64,
        'lossless.varint': I64,
        'varint-wide': D38,
        'varint-past-38-digits': UTF8,
    }
    for row_id, received in tiers.items():
        row = TYPES[row_id]
        assert row['received'] == received, row_id
        assert row['cells'] == 'cast', row_id
        dtype = 'string' if received == UTF8 else 'int'
        assert row['inference'] == {'dtype': dtype}, row_id
    # VARINT with no session setting, in each storage DuckDB gives it.
    for row_id in ('varint', 'varint-wide', 'varint-past-38-digits'):
        assert not TYPES[row_id]['witness'][0].startswith('SET ')


def test_a_time_named_column_of_no_times_is_text() -> None:
    by_id = {r['id']: r for r in ROWS}
    for row_id in ('text-time.time-frame', 'text-time.time-perspective'):
        want = by_id[row_id]['want']
        assert want['dtype'] == 'string'
        assert want['time_column'] is None
    kept = by_id['text-time.time-frame-dates']['want']
    assert kept['stype'] == 'timestamp'
    assert kept['time_column'] == 'time_frame'


def test_a_varint_key_has_each_tier() -> None:
    # The subagent's P2-1 (fix round 1): a VARINT key, users.user_id and
    # orders.user_id, in each tier; materialize's seed joins run on it.
    by_id = {r['id']: r for r in ROWS}
    tiers = {
        'varint-key.int64': (I64, 'int64'),
        'varint-key.decimal': (D38, 'int64'),
        'varint-key.text': (UTF8, 'string'),
    }
    for row_id, (received, dtype) in tiers.items():
        row = by_id[row_id]
        assert row['column'] == 'user_id', row_id
        assert row['want'] == {
            'dtype': dtype,
            'cells': 'cast',
            'received': received,
        }, row_id
        orders, users = row['witness']
        assert orders.startswith('CREATE OR REPLACE TABLE orders ')
        assert (
            '::VARINT AS user_id' in orders and '::VARINT AS user_id' in users
        )
        assert not any(s.startswith('SET ') for s in row['witness'])
