# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every connector, against real data, round-tripped into a prediction.

A connector is only correct if what comes back out of it still predicts. So
each probe loads through the connector and then scores with a live NIM.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3

import data
import pandas as pd
from harness import TABULAR, fail, note, probe, save
from sdm_client import KumoTabular, Request, SDMClient
from sdm_connectors import ConnectorError, connect, read, read_table

AREA = 'connectors'
SCRATCH = pathlib.Path.home() / 'sdm-qa' / 'scratch'
SCRATCH.mkdir(parents=True, exist_ok=True)


def scores(
    client, frame: pd.DataFrame, target: str, task: str = 'classification'
):
    context = frame.head(len(frame) - 3)
    predict = frame.tail(3).drop(columns=[target]).reset_index(drop=True)
    return client.predict(
        Request(context=context, predict=predict, target=target, task=task),
        model=KumoTabular(),
    ).to_pandas()


def run(client):
    adult = data.adult().head(400)

    # --- local files -------------------------------------------------------
    with probe(
        AREA, 'local csv', 'reads back identical, and still predicts'
    ) as e:
        path = SCRATCH / 'adult.csv'
        adult.to_csv(path, index=False)
        frame = read('local', path=str(path))
        if len(frame) != len(adult) or list(frame.columns) != list(
            adult.columns
        ):
            fail(e, f'{frame.shape} vs {adult.shape}')
        elif len(scores(client, frame, 'income')) != 3:
            fail(e, 'the round-tripped frame did not score')

    with probe(
        AREA, 'local parquet', 'dtypes survive, and it still predicts'
    ) as e:
        path = SCRATCH / 'adult.parquet'
        adult.to_parquet(path, index=False)
        frame = read('local', path=str(path))
        if not frame.dtypes.equals(adult.dtypes):
            fail(
                e,
                f'dtypes changed: {frame.dtypes[frame.dtypes != adult.dtypes]}',
            )
        elif len(scores(client, frame, 'income')) != 3:
            fail(e, 'the round-tripped frame did not score')

    with probe(
        AREA, 'a gzipped csv', 'compression is detected from the suffix'
    ) as e:
        path = SCRATCH / 'adult.csv.gz'
        adult.to_csv(path, index=False, compression='gzip')
        frame = read('local', path=str(path))
        if len(frame) != len(adult):
            fail(e, f'{len(frame)} rows')

    with probe(
        AREA, 'a parquet directory', 'partitioned parquet reads as one frame'
    ) as e:
        directory = SCRATCH / 'adult_parts'
        directory.mkdir(exist_ok=True)
        for i, part in enumerate([adult.head(200), adult.tail(200)]):
            part.to_parquet(directory / f'part-{i}.parquet', index=False)
        frame = read('local', path=str(directory))
        if len(frame) != 400:
            fail(e, f'{len(frame)} rows from a 2-part directory')

    with probe(
        AREA, 'a file that is not there', 'a ConnectorError naming the path'
    ) as e:
        try:
            read('local', path=str(SCRATCH / 'nope.csv'))
            fail(e, 'a missing file was accepted')
        except (ConnectorError, FileNotFoundError, OSError) as exc:
            note(e, f'{type(exc).__name__}: {str(exc)[:120]}')

    with probe(
        AREA, 'a suffix with no reader', 'refused, listing what it does read'
    ) as e:
        path = SCRATCH / 'adult.xlsx'
        path.write_bytes(b'not really a spreadsheet')
        try:
            read('local', path=str(path))
            fail(e, 'an unknown suffix was accepted')
        except ConnectorError as exc:
            note(e, str(exc)[:140])

    with probe(
        AREA,
        'an s3:// path given to the local reader',
        'refused, pointing at the s3 connector',
    ) as e:
        try:
            read('local', path='s3://bucket/key.csv')
            fail(e, 'the local reader accepted an s3 URI')
        except ConnectorError as exc:
            note(e, str(exc)[:140])

    # --- sqlite ------------------------------------------------------------
    with probe(
        AREA, 'sqlite, a real database', 'a query returns a frame that predicts'
    ) as e:
        connection = connect(
            'sqlite', database=str(data.DATA / 'northwind.sqlite')
        )
        frame = read_table(connection, query='select * from Orders limit 400')
        if len(frame) != 400:
            fail(e, f'{len(frame)} rows')
        else:
            note(
                e,
                f'{frame.shape}, dtypes={sorted(set(map(str, frame.dtypes)))}',
            )

    with probe(
        AREA, 'sqlite, a table name with a space', 'quoted, not split into two'
    ) as e:
        connection = connect(
            'sqlite', database=str(data.DATA / 'northwind.sqlite')
        )
        frame = read_table(
            connection, query='select * from "Order Details" limit 50'
        )
        if len(frame) != 50:
            fail(e, f'{len(frame)} rows')

    with probe(
        AREA,
        'sqlite, a quote in a table name',
        'refused, since only plain identifiers are accepted',
    ) as e:
        path = SCRATCH / 'quoted.sqlite'
        path.unlink(missing_ok=True)
        with sqlite3.connect(path) as raw:
            raw.execute('create table "odd""name" (a integer)')
            raw.execute('insert into "odd""name" values (1), (2)')
        connection = connect('sqlite', database=str(path))
        try:
            read_table(connection, table='odd"name')
            fail(e, 'a quoted identifier was interpolated into SQL')
        except ConnectorError as exc:
            note(e, str(exc)[:140])
        frame = read_table(connection, query='select * from "odd""name"')
        if len(frame) != 2:
            fail(e, f'{len(frame)} rows through an explicit query')

    with probe(
        AREA,
        'sqlite, an injection attempt through a table name',
        'no second statement runs',
    ) as e:
        path = SCRATCH / 'inject.sqlite'
        path.unlink(missing_ok=True)
        with sqlite3.connect(path) as raw:
            raw.execute('create table keep (a integer)')
            raw.execute('insert into keep values (1)')
        connection = connect('sqlite', database=str(path))
        try:
            read_table(connection, table='keep; drop table keep')
            fail(e, 'a second statement was allowed through a table name')
        except Exception as exc:  # noqa: BLE001
            note(e, f'{type(exc).__name__}: {str(exc)[:100]}')
        with sqlite3.connect(path) as raw:
            still = raw.execute(
                "select count(*) from sqlite_master where name='keep'"
            ).fetchone()[0]
        if not still:
            fail(e, 'the table was dropped -- the identifier was not escaped')

    with probe(
        AREA, 'sqlite, round trip into a prediction', 'warehouse data scores'
    ) as e:
        connection = connect(
            'sqlite', database=str(data.DATA / 'northwind.sqlite')
        )
        frame = read_table(
            connection,
            query='select ProductID, UnitPrice, '
            'UnitsInStock, Discontinued from Products',
        )
        frame = frame.dropna()
        if len(frame) < 20:
            fail(e, f'only {len(frame)} usable rows')
        elif len(scores(client, frame, 'Discontinued')) != 3:
            fail(e, 'sqlite data did not score')

    # --- duckdb ------------------------------------------------------------
    with probe(
        AREA, 'duckdb, a query over a parquet file', 'the frame predicts'
    ) as e:
        path = SCRATCH / 'adult.parquet'
        connection = connect('duckdb')
        frame = read_table(
            connection,
            query=(
                f'select age, education_num, hours_per_week, income '
                f"from read_parquet('{path}')"
            ),
        )
        if len(frame) != 400:
            fail(e, f'{len(frame)} rows')
        elif len(scores(client, frame, 'income')) != 3:
            fail(e, 'duckdb data did not score')

    with probe(
        AREA, 'duckdb, an in-memory table', 'a registered relation is readable'
    ) as e:
        connection = connect('duckdb')
        connection.execute('create table t as select 1 a union all select 2')
        frame = read_table(connection, table='t')
        if len(frame) != 2:
            fail(e, f'{len(frame)} rows')

    with probe(
        AREA,
        'duckdb, a syntax error',
        'a ConnectorError, not a raw driver error',
    ) as e:
        connection = connect('duckdb')
        try:
            read_table(connection, query='select * from')
            fail(e, 'invalid SQL was accepted')
        except Exception as exc:  # noqa: BLE001
            kind = type(exc).__name__
            note(e, f'{kind}: {str(exc)[:100]}')
            if not isinstance(exc, ConnectorError):
                fail(e, f'{kind} escaped the connector error surface')

    with probe(
        AREA,
        'a backend that is not installed',
        'a MissingBackendError naming the extra',
    ) as e:
        try:
            connect('not-a-backend')
            fail(e, 'an unknown backend was accepted')
        except ConnectorError as exc:
            note(e, str(exc)[:140])

    # --- limit and pushdown -------------------------------------------------
    with probe(
        AREA,
        'limit is applied by the warehouse, not after the fact',
        'exactly the asked rows',
    ) as e:
        connection = connect(
            'sqlite', database=str(data.DATA / 'northwind.sqlite')
        )
        frame = read_table(connection, query='select * from Orders limit 7')
        if len(frame) != 7:
            fail(e, f'{len(frame)} rows for limit=7')

    with probe(AREA, 'a negative limit', 'refused, not silently reversed') as e:
        connection = connect(
            'sqlite', database=str(data.DATA / 'northwind.sqlite')
        )
        try:
            frame = read_table(
                connection, query='select * from Orders limit -5'
            )
            note(e, f'sqlite treats limit=-5 as no limit: {len(frame)} rows')
        except (ConnectorError, ValueError) as exc:
            note(e, f'{type(exc).__name__}: {str(exc)[:120]}')

    with probe(AREA, 'both a query and a table', 'refused as ambiguous') as e:
        connection = connect(
            'sqlite', database=str(data.DATA / 'northwind.sqlite')
        )
        try:
            read_table(connection, query='select 1', table='Orders')
            fail(e, 'both were accepted')
        except ConnectorError as exc:
            note(e, f'{type(exc).__name__}: {str(exc)[:120]}')

    with probe(
        AREA, 'neither a query nor a table', 'refused, saying which to pass'
    ) as e:
        connection = connect(
            'sqlite', database=str(data.DATA / 'northwind.sqlite')
        )
        try:
            read_table(connection)
            fail(e, 'neither was required')
        except (ConnectorError, TypeError) as exc:
            note(e, f'{type(exc).__name__}: {str(exc)[:120]}')


def run_exactness(client):  # noqa: ARG001
    """Every route a large integer can take into the SDK.

    The wire layer's rule -- an integer float64 cannot hold exactly must not
    travel as a number -- is only as strong as the weakest reader.
    """
    import sqlite3 as sqlite3_module

    from sdm_client.wire.dtypes import infer_tfm_dtype, serialize_column

    big = 9_007_199_254_740_993

    def exact(frame, column='id'):
        series = frame[column]
        cells = serialize_column(series, infer_tfm_dtype(series))
        return (
            str(big) in [str(cell) for cell in cells],
            series.dtype,
            cells[:2],
        )

    with probe(AREA, 'parquet, a nullable integer with a null', 'exact') as e:
        path = SCRATCH / 'nullable.parquet'
        pd.DataFrame(
            {'id': pd.array([big, None], dtype='Int64'), 'v': [1.0, 2.0]}
        ).to_parquet(path, index=False)
        ok, dtype, cells = exact(read('local', path=str(path)))
        if not ok:
            fail(e, f'{dtype} -> {cells}')

    with probe(AREA, 'parquet, a plain int64', 'exact') as e:
        path = SCRATCH / 'plain.parquet'
        pd.DataFrame({'id': [big, big + 2], 'v': [1.0, 2.0]}).to_parquet(
            path, index=False
        )
        ok, dtype, cells = exact(read('local', path=str(path)))
        if not ok:
            fail(e, f'{dtype} -> {cells}')

    with probe(AREA, 'sqlite, an integer column with a null', 'exact') as e:
        path = SCRATCH / 'exact.sqlite'
        path.unlink(missing_ok=True)
        with sqlite3_module.connect(path) as raw:
            raw.execute('create table t (id integer, v real)')
            raw.execute(f'insert into t values ({big}, 1.0), (NULL, 2.0)')
        ok, dtype, cells = exact(
            read_table(connect('sqlite', database=str(path)), table='t')
        )
        if not ok:
            fail(e, f'{dtype} -> {cells}')

    with probe(AREA, 'duckdb, a BIGINT column with a null', 'exact') as e:
        connection = connect('duckdb')
        connection.execute(
            f'create table t as select * from '
            f'(values ({big}::BIGINT, 1.0), (NULL, 2.0)) t(id, v)'
        )
        ok, dtype, cells = exact(read_table(connection, table='t'))
        if not ok:
            fail(e, f'{dtype} -> {cells}')

    # KNOWN BUG bugs/014
    with probe(AREA, 'duckdb, a SUM over a BIGINT column', 'exact') as e:
        connection = connect('duckdb')
        connection.execute(
            f'create table t as select {big}::BIGINT id from range(3)'
        )
        frame = read_table(connection, query='select sum(id) id from t')
        cells = serialize_column(frame['id'], infer_tfm_dtype(frame['id']))
        if str(cells[0]) != str(big * 3):
            fail(e, f'sum is {big * 3}, got {cells[0]!r} ({frame["id"].dtype})')

    # KNOWN BUG bugs/009
    with probe(
        AREA, 'csv, a large integer with a blank in the column', 'exact'
    ) as e:
        path = SCRATCH / 'blank.csv'
        path.write_text(f'id,v\n{big},1.0\n,2.0\n{big + 2},3.0\n')
        ok, dtype, cells = exact(read('local', path=str(path)))
        if not ok:
            fail(e, f'{dtype} -> {cells}')

    with probe(AREA, 'csv, a large integer with no blank', 'exact') as e:
        path = SCRATCH / 'noblank.csv'
        path.write_text(f'id,v\n{big},1.0\n{big + 2},3.0\n')
        ok, dtype, cells = exact(read('local', path=str(path)))
        if not ok:
            fail(e, f'{dtype} -> {cells}')

    with probe(AREA, 'a python int list through data=', 'exact') as e:
        ok, dtype, cells = exact(
            read('local', data={'id': [big, big + 2], 'v': [1.0, 2.0]})
        )
        if not ok:
            fail(e, f'{dtype} -> {cells}')

    # KNOWN BUG bugs/009
    with probe(
        AREA, 'a python int list through data= with a None', 'exact'
    ) as e:
        ok, dtype, cells = exact(
            read(
                'local', data={'id': [big, None, big + 2], 'v': [1.0, 2.0, 3.0]}
            )
        )
        if not ok:
            fail(e, f'{dtype} -> {cells}')


if __name__ == '__main__':
    os.environ.setdefault('SDM_TABULAR_URL', TABULAR)
    with SDMClient(url=TABULAR, timeout=300.0) as client:
        run(client)
        run_exactness(client)
    save('connectors.json')
