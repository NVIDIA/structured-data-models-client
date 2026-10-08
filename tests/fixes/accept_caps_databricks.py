# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.caps, Databricks: a bounded read ahead of the inference sample.

Adi's decision (``BRIEF.md``, task 2): a large table's inference sample reads
``SELECT ... FROM (SELECT <cols> FROM t LIMIT M) ...``, then today's hash
order and ``LIMIT 1000`` over it; its size comes from metadata,
``DESCRIBE DETAIL`` for a Delta table. Through the public API over the stub
driver (``_warehouse_stub.py``), with the SQL connector and a notebook's
Spark session:

* a Delta table ``DESCRIBE DETAIL`` says is 10 TiB is read through the
  bounded subset, today's select list, order and limit outside it, ``M``
  between 10,000 and 1,000,000;
* a 1 MiB table, and a table whose size can't be read (``DESCRIBE DETAIL``
  fails), send today's statements byte for byte, apart from the metadata
  reads;
* the size is read with ``DESCRIBE DETAIL`` of the table;
* a table's expression sample reads the same bounded subset as its source
  sample, so the two line up row by row.

Today's statements come from running ``BASE``'s code in a child process.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from _fixes import assert_imports_base, base_env, run_child

TIB = 1 << 40
MIB = 1 << 20
ORDERS = '`main`.`shop`.`orders`'
USERS = '`main`.`shop`.`users`'


@pytest.fixture(scope='module')
def base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('base')
    assert_imports_base(base_env(root), root)
    return root


_cache: dict[tuple[str, str, str], Any] = {}


def _run(
    kind: str, config: dict[str, Any], base_root: Path | None = None
) -> Any:
    text = json.dumps(config, sort_keys=True)
    key = ('base' if base_root else 'head', kind, text)
    if key not in _cache:
        env = base_env(base_root) if base_root else None
        _cache[key] = run_child('warehouse', kind, '-', text, env=env)
    return _cache[key]


def _detail(size: int, files: int = 1) -> dict[str, Any]:
    import _warehouse_stub as stub

    return stub.detail(size, files)


def _samples(result: Any, table: str) -> list[str]:
    return [
        s['sql']
        for s in result['statements']
        if '__kumo_order_' in s['sql'] and table in s['sql']
    ]


def _today(base_root: Path, kind: str, table: str) -> str:
    found = _samples(_run(kind, {'questions': []}, base_root), table)
    assert len(found) == 1, found
    return found[0]


def _metadata(sql: str) -> bool:
    flat = ' '.join(sql.split()).upper()
    return flat.startswith(('DESCRIBE', 'SHOW', 'ANALYZE')) or (
        'INFORMATION_SCHEMA.TABLES' in flat and 'TABLE_TYPE' in flat
    )


_OFFSET = re.compile(r'(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?)\+00:00')


def _without_offsets(params: Any) -> Any:
    r"""Parameters with ``+00:00`` taken off time bounds: the time-zone fix
    (``accept_tz_shape.py``) adds it, and this file checks the scans."""
    if params is None:
        return None
    return [_OFFSET.sub(r'\1', p) if isinstance(p, str) else p for p in params]


def _without_metadata(result: Any) -> list[Any]:
    return [
        {'sql': s['sql'], 'params': _without_offsets(s['params'])}
        for s in result['statements']
        if not _metadata(s['sql'])
    ]


def _bounded(today: str, table: str) -> re.Pattern[str]:
    r"""Today's statement over a bounded subset of ``table``."""
    head, sep, tail = today.partition(f' FROM {table} ')
    assert sep, today
    return re.compile(
        re.escape(head)
        + r' FROM \(SELECT (?P<cols>.+) FROM '
        + re.escape(table)
        + r' LIMIT (?P<m>\d+)\)(?: (?:AS )?`?\w+`?)? '
        + re.escape(tail)
    )


@pytest.mark.parametrize('kind', ('databricks', 'spark'))
def test_a_large_delta_table_is_read_through_a_bounded_subset(
    base_root: Path, kind: str
) -> None:
    config = {
        'questions': [],
        'details': {'orders': _detail(10 * TIB, 100_000)},
        'prefiltered_rows': 1000,
    }
    result = _run(kind, config)
    sent = _samples(result, ORDERS)
    assert sent, result['statements']
    today = _today(base_root, kind, ORDERS)
    found = _bounded(today, ORDERS).fullmatch(' '.join(sent[0].split()))
    assert found, f"not a bounded read of today's statement:\n{sent[0]}"
    assert 10_000 <= int(found.group('m')) <= 1_000_000, found.group('m')
    cols = found.group('cols').strip()
    if cols != '*':
        for name in re.findall(r'`(\w+)`', today.split(' FROM ')[0]):
            assert f'`{name}`' in cols, (name, cols)
    # USERS says nothing of its size, so it is read as today.
    assert _samples(result, USERS) == [_today(base_root, kind, USERS)]


@pytest.mark.parametrize('kind', ('databricks', 'spark'))
def test_the_size_is_read_with_describe_detail(kind: str) -> None:
    config = {
        'questions': [],
        'details': {'orders': _detail(10 * TIB, 100_000)},
    }
    described = [
        ' '.join(s['sql'].split())
        for s in _run(kind, config)['statements']
        if ' '.join(s['sql'].split()).upper().startswith('DESCRIBE DETAIL')
    ]
    assert any(ORDERS in sql for sql in described), described


@pytest.mark.parametrize('kind', ('databricks', 'spark'))
@pytest.mark.parametrize(
    'details',
    [
        pytest.param(
            {
                'orders': {
                    'format': 'delta',
                    'sizeInBytes': MIB,
                    'numFiles': 1,
                },
                'users': {'format': 'delta', 'sizeInBytes': MIB, 'numFiles': 1},
            },
            id='small',
        ),
        pytest.param({}, id='unknown'),
    ],
)
def test_small_and_unsized_tables_send_todays_statements(
    base_root: Path, kind: str, details: dict[str, Any]
) -> None:
    config = {'details': details}
    head = _run(kind, config)
    base = _run(kind, config, base_root)
    assert _without_metadata(head) == _without_metadata(base)
    assert head['payloads'] == base['payloads']


@pytest.mark.parametrize('kind', ('databricks', 'spark'))
def test_a_large_table_predicts_as_before_over_the_stub(
    base_root: Path, kind: str
) -> None:
    r"""The stub returns the same rows either way, so only the inference
    sample's statements may differ."""
    config = {
        'details': {'orders': _detail(10 * TIB, 100_000)},
        'prefiltered_rows': 1000,
    }
    head = _run(kind, config)
    base = _run(kind, config, base_root)
    keep = [
        s for s in _without_metadata(head) if '__kumo_order_' not in s['sql']
    ]
    want = [
        s for s in _without_metadata(base) if '__kumo_order_' not in s['sql']
    ]
    assert keep == want


def test_the_expression_sample_reads_the_same_subset() -> None:
    import _warehouse_stub as stub
    from sdm_relational_engine.rfm.backend.databricks import DatabricksTable

    warehouse = stub.Warehouse(
        kind='databricks',
        details={'orders': stub.detail(10 * TIB, 100_000)},
        prefiltered_rows=1000,
    )
    table = DatabricksTable(
        stub.source(warehouse),
        'orders',
        catalog='main',
        schema='shop',
        columns=[
            'order_id',
            'amount',
            dict(name='twice', expr='2 * `amount`', dtype='float'),
        ],
        primary_key='order_id',
    )
    assert len(table._source_sample_df) > 0
    table._get_expr_sample_df([table['twice']])
    subsets = [
        re.search(r'FROM \((SELECT .+ LIMIT \d+)\)', ' '.join(s.sql.split()))
        for s in warehouse.statements
        if '__kumo_order_' in s.sql
    ]
    assert len(subsets) >= 2 and all(subsets), [
        s.sql for s in warehouse.statements
    ]
    assert len({found.group(1) for found in subsets if found}) == 1
