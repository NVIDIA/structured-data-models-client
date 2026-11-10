# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-table oracles for provider execution, independent of SDM planning."""

import pyarrow as pa


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _table(connection, relation):
    assert list(relation) == ['source'] and len(relation['source']) == 1
    return connection.sql(
        f'SELECT * FROM {_quote(relation["source"][0])}'
    ).to_arrow_table()


def _filtered(table, filters):
    positions = list(range(table.num_rows))
    for condition in filters:
        assert condition['filter'] == 'not_null'
        values = table[condition['column']].to_pylist()
        positions = [i for i in positions if values[i] is not None]
    return table.take(pa.array(positions, pa.int64()))


def _rows(schema, rows):
    return pa.Table.from_arrays(
        [
            pa.array([r[i] for r in rows], type=t)
            for i, (_, t) in enumerate(schema)
        ],
        names=[name for name, _ in schema],
    )


def source_answer(connection, fetch, *, postgres=False):
    """Expected values from raw input rows, never the product's SQL or admission."""
    request = fetch['request']
    kind = request['request']
    if kind == 'describe':
        part = request['part']
        table = request['tables'][0]['source'][0] if request['tables'] else None
        if part == 'tables':
            sql, params = (
                'SELECT table_name FROM information_schema.tables '
                "WHERE table_schema='main'",
                [],
            )
        elif part == 'columns':
            sql, params = 'SELECT * FROM pragma_table_info(?)', [table]
        elif part in {'keys', 'foreign_keys'}:
            columns = (
                'constraint_type, constraint_column_names'
                if part == 'keys'
                else 'constraint_index, referenced_table, constraint_column_names, referenced_column_names'
            )
            sql = (
                f'SELECT {columns} FROM duckdb_constraints() WHERE table_name=?'
            )
            if part == 'foreign_keys':
                sql += " AND constraint_type='FOREIGN KEY'"
            params = [table]
        else:
            assert part == 'indexes'
            sql, params = (
                'SELECT table_name, expressions FROM duckdb_indexes()',
                [],
            )
        return connection.sql(sql, params=params).to_arrow_table()
    if kind == 'key_counts':
        rows = []
        for index, key in enumerate(request['keys']):
            values = list(
                zip(
                    *[
                        _table(connection, key['table'])[c].to_pylist()
                        for c in key['columns']
                    ],
                    strict=True,
                )
            )
            nonnull = [v for v in values if all(x is not None for x in v)]
            rows.append(
                [
                    index,
                    len(values),
                    len(set(nonnull)),
                    len(values) - len(nonnull),
                ]
            )
        return _rows(
            [
                ('k', pa.int32()),
                ('n', pa.int64()),
                ('d', pa.int64()),
                ('z', pa.int64()),
            ],
            rows,
        )
    if kind == 'time_bounds':
        rows = []
        for index, column in enumerate(request['columns']):
            values = [
                v
                for v in _table(connection, column['table'])[
                    column['column']
                ].to_pylist()
                if v is not None
            ]
            rows.append([index, min(values), max(values)])
        return _rows(
            [
                ('table_index', pa.int32()),
                ('min_date', pa.timestamp('us')),
                ('max_date', pa.timestamp('us')),
            ],
            rows,
        )
    source = _filtered(
        _table(connection, request['table']), request.get('filters', [])
    )
    if kind == 'count_rows':
        return _rows([('rows', pa.int64())], [[source.num_rows]])
    columns = request['columns']
    projected = source.select(columns)
    if kind in {'scan', 'preview'}:
        # No SQL ORDER BY was requested: preserve complete membership, not
        # an accidental physical order, when the caller asks for a prefix.
        return projected
    if kind == 'sample_entities':
        if postgres:
            return projected
        # DuckDB owns this reservoir algorithm. Exercise it directly on the
        # filtered source, bypassing SDM's SQL builder and result admission.
        relation = connection.from_arrow(source)
        seed = int(request['seed']['v'])
        return (
            relation.query(
                'source_rows',
                f'SELECT * FROM source_rows USING SAMPLE reservoir({request["n"]} ROWS) REPEATABLE ({seed})',
            )
            .to_arrow_table()
            .select(columns)
        )
    assert kind in {'neighbors', 'lookup'}
    assert len(request['key']) == 1
    keys = source[request['key'][0]].to_pylist()
    values = [column.to_pylist() for column in projected.columns]
    rows = []
    for batch, seed in enumerate(fetch['keys']):
        matches = [i for i, key in enumerate(keys) if key == seed]
        for i in matches[:1] if kind == 'lookup' else matches:
            own = [column[i] for column in values]
            rows.append(
                [batch, *own]
                if kind == 'neighbors'
                else [batch, *own, 1, len(matches)]
                if postgres
                else [batch, len(matches), *own]
            )
    schema = [('__kumo_batch__', pa.int64())]
    own_schema = [(f.name, f.type) for f in projected.schema]
    if kind == 'neighbors':
        schema += own_schema
    elif postgres:
        schema += own_schema + [
            ('__kumo_row__', pa.int64()),
            ('__kumo_matches__', pa.int64()),
        ]
    else:
        schema += [('__kumo_matches__', pa.int64())] + own_schema
    return _rows(schema, rows)


def assert_source_answer(
    actual, expected, request, canonical, *, postgres=False
):
    assert [(f.name, f.type) for f in actual.schema] == [
        (f.name, f.type) for f in expected.schema
    ]
    rows, source = canonical(actual), canonical(expected)
    kind = request['request']
    limit = request.get('limit') if kind in {'scan', 'preview'} else None
    if kind == 'sample_entities' and postgres:
        limit = request['n']
        assert len({tuple(row) for row in rows}) == len(rows)
    if limit is not None and limit < len(source):
        assert len(rows) == limit
        remaining = list(source)
        for row in rows:
            assert row in remaining
            remaining.remove(row)
    else:
        assert rows == source
