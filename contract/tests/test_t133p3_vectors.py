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


def test_the_pending_list_only_shrinks() -> None:
    assert G.only_shrinks() == []
    ids = {row['id'] for row in ROWS}
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
    lossless = {r['id']: r for r in ROWS if r['witness'][0] == LOSSLESS}
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
        assert row['want']['cells'] is not None, row['id']
    text = {'lossless.uuid', 'lossless.bit', 'lossless.json'}
    assert all(lossless[i]['want']['dtype'] == 'string' for i in text)
    by_id = {r['id']: r for r in ROWS}
    assert by_id['lossless.timetz']['want'] == by_id['time.control']['want']
    for row in ROWS:
        if 'past-38-digits' in row['id']:
            assert row['want'] == {'dtype': 'string', 'cells': 'cast'}


def test_a_time_named_column_of_no_times_is_text() -> None:
    by_id = {r['id']: r for r in ROWS}
    for row_id in ('text-time.time-frame', 'text-time.time-perspective'):
        want = by_id[row_id]['want']
        assert want['dtype'] == 'string'
        assert want['time_column'] is None
    kept = by_id['text-time.time-frame-dates']['want']
    assert kept['stype'] == 'timestamp'
    assert kept['time_column'] == 'time_frame'
