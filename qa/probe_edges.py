# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The data shapes real tables actually contain, and the error contract."""

from __future__ import annotations

import data
import numpy as np
import pandas as pd
from harness import RELATIONAL, TABULAR, fail, note, probe, save
from sdm_client import (
    KumoRelational,
    KumoTabular,
    NimRequestError,
    PredictiveQuery,
    Request,
    SDMClient,
    StructuredDataError,
    UnknownModelError,
)
from sdm_client.wire.dtypes import infer_tfm_dtype, serialize_column

AREA = 'edges'
SAFE = 9_007_199_254_740_991


def frame(n=60, **columns):
    base = {
        'row_id': list(range(n)),
        'v': [float(i) for i in range(n)],
        'y': [i % 2 for i in range(n)],
    }
    base.update(columns)
    return pd.DataFrame(base)


def score(
    client, context, predict, target='y', task='classification', **kwargs
):
    return client.predict(
        Request(context=context, predict=predict, target=target, task=task),
        model=KumoTabular(),
        **kwargs,
    ).to_pandas()


def run(client, relational):
    # --- wire encoding of awkward values ----------------------------------
    # KNOWN BUG bugs/001
    with probe(
        AREA, 'a Decimal from a warehouse', 'exact, not rounded through float'
    ) as e:
        from decimal import Decimal

        series = pd.Series([Decimal('9007199254740993'), Decimal('1')])
        cells = serialize_column(series, infer_tfm_dtype(series))
        if '9007199254740993' not in [str(c) for c in cells]:
            fail(e, f'{Decimal("9007199254740993")} became {cells[0]!r}')
        else:
            note(e, f'dtype={infer_tfm_dtype(series)} cells={cells}')

    # KNOWN BUG bugs/004
    with probe(
        AREA, 'a uint64 past int64', 'never declared int64 while out of range'
    ) as e:
        series = pd.Series([2**64 - 1, 1], dtype='uint64')
        dtype = infer_tfm_dtype(series)
        cells = serialize_column(series, dtype)
        if dtype == 'int64' and int(cells[0]) > 2**63 - 1:
            fail(e, f'declared int64 but sent {cells[0]}')
        else:
            note(e, f'dtype={dtype} cells={cells}')

    with probe(
        AREA, 'a nullable integer with a null', 'no float widening, no rounding'
    ) as e:
        series = pd.Series([SAFE + 2, None], dtype='Int64')
        cells = serialize_column(series, infer_tfm_dtype(series))
        if str(cells[0]) != str(SAFE + 2):
            fail(e, f'{SAFE + 2} became {cells[0]!r}')

    with probe(
        AREA, 'a bool column', 'true/false, not the strings True/False'
    ) as e:
        series = pd.Series([True, False, None], dtype='boolean')
        cells = serialize_column(series, infer_tfm_dtype(series))
        note(e, f'dtype={infer_tfm_dtype(series)} cells={cells}')

    with probe(
        AREA, 'a tz-aware timestamp', 'the same instant as its naive UTC twin'
    ) as e:
        aware = pd.Series(pd.to_datetime(['2024-03-01T12:00:00-05:00']))
        naive = pd.Series(pd.to_datetime(['2024-03-01T17:00:00']))
        a = serialize_column(aware, infer_tfm_dtype(aware))
        n = serialize_column(naive, infer_tfm_dtype(naive))
        if a != n:
            fail(e, f'{a} != {n}')

    with probe(
        AREA,
        'a year outside the pandas range',
        'a StructuredDataError, not a pandas one',
    ) as e:
        series = pd.Series(['1500-01-01', '3000-01-01'])
        try:
            cells = serialize_column(series, 'timestamp')
            note(e, f'accepted: {cells}')
        except StructuredDataError as exc:
            note(e, str(exc)[:120])
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    with probe(
        AREA, 'a NaN in a float column', 'null on the wire, not the string nan'
    ) as e:
        series = pd.Series([1.0, np.nan])
        cells = serialize_column(series, infer_tfm_dtype(series))
        if str(cells[1]).lower() == 'nan':
            fail(e, f'NaN went out as {cells[1]!r}')
        else:
            note(e, f'cells={cells}')

    with probe(
        AREA, 'an infinity', 'refused, not sent as a JSON non-value'
    ) as e:
        series = pd.Series([1.0, np.inf])
        try:
            cells = serialize_column(series, infer_tfm_dtype(series))
            fail(e, f'inf went out as {cells[1]!r}')
        except StructuredDataError as exc:
            note(e, str(exc)[:120])
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    # --- awkward frames, live ---------------------------------------------
    with probe(
        AREA,
        'a big int through a live prediction',
        'ids survive the round trip',
    ) as e:
        context = frame(60, big=[SAFE + i for i in range(60)])
        predict = pd.DataFrame(
            {'row_id': [900, 901], 'v': [1.0, 2.0], 'big': [SAFE + 1, SAFE + 2]}
        )
        result = score(client, context, predict)
        if len(result) != 2:
            fail(e, f'{len(result)} rows')

    with probe(
        AREA,
        'a column that is all null in context but filled to score',
        'no silent stringification',
    ) as e:
        context = frame(60, note=[None] * 60)
        predict = pd.DataFrame(
            {'row_id': [900, 901], 'v': [1.0, 2.0], 'note': ['alpha', 'beta']}
        )
        try:
            result = score(client, context, predict)
            note(e, f'served {len(result)} rows')
        except StructuredDataError as exc:
            note(e, f'refused: {str(exc)[:140]}')

    with probe(
        AREA,
        'a numeric column that is all null in context',
        'numbers are not stringified',
    ) as e:
        context = frame(60, measure=[None] * 60)
        predict = pd.DataFrame(
            {'row_id': [900, 901], 'v': [1.0, 2.0], 'measure': [3.5, 4.5]}
        )
        try:
            result = score(client, context, predict)
            note(e, f'served {len(result)} rows')
        except StructuredDataError as exc:
            note(e, f'refused: {str(exc)[:140]}')

    with probe(AREA, 'a single feature column', 'one feature is enough') as e:
        context = pd.DataFrame(
            {
                'v': [float(i) for i in range(60)],
                'y': [i % 2 for i in range(60)],
            }
        )
        predict = pd.DataFrame({'v': [1.0, 2.0]})
        result = score(client, context, predict)
        if len(result) != 2:
            fail(e, f'{len(result)} rows')

    with probe(AREA, 'an empty context', 'refused locally, with a reason') as e:
        try:
            score(client, frame(0), pd.DataFrame({'row_id': [1], 'v': [1.0]}))
            fail(e, 'an empty context was accepted')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])

    # KNOWN BUG bugs/005
    with probe(
        AREA, 'an empty predict frame', 'refused locally, with a reason'
    ) as e:
        try:
            result = score(client, frame(60), frame(0).drop(columns=['y']))
            fail(e, f'accepted; result has columns {list(result.columns)}')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])

    with probe(
        AREA, 'a target column missing from the context', 'named locally'
    ) as e:
        try:
            score(
                client,
                frame(60).drop(columns=['y']),
                pd.DataFrame({'row_id': [1], 'v': [1.0]}),
            )
            fail(e, 'a missing target was accepted')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])

    with probe(
        AREA,
        'a target that is all null',
        'refused: there is nothing to learn from',
    ) as e:
        context = frame(60)
        context['y'] = None
        try:
            result = score(
                client, context, pd.DataFrame({'row_id': [1], 'v': [1.0]})
            )
            fail(
                e,
                f'an all-null target was accepted, predicting {list(result["prediction"])}',
            )
        except StructuredDataError as exc:
            note(e, str(exc)[:140])

    with probe(
        AREA, 'a target with one class', 'refused, or that one class returned'
    ) as e:
        context = frame(60)
        context['y'] = 1
        try:
            result = score(
                client, context, pd.DataFrame({'row_id': [1], 'v': [1.0]})
            )
            if set(result['prediction']) != {1}:
                fail(
                    e,
                    f'predicted {set(result["prediction"])} from a single-class context',
                )
            else:
                note(e, 'the one class was returned')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])

    with probe(
        AREA,
        'columns the context does not have',
        'named locally, not silently ignored',
    ) as e:
        predict = pd.DataFrame({'row_id': [1], 'v': [1.0], 'surprise': ['x']})
        try:
            result = score(client, frame(60), predict)
            note(e, f'extra column ignored, {len(result)} rows scored')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])

    with probe(
        AREA, 'a duplicate column name', 'refused, not silently deduplicated'
    ) as e:
        context = frame(60)
        context.columns = ['row_id', 'v', 'y']
        duplicated = pd.concat([context, context[['v']]], axis=1)
        try:
            score(client, duplicated, pd.DataFrame({'row_id': [1], 'v': [1.0]}))
            fail(e, 'a duplicated column name was accepted')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    # --- sessions ---------------------------------------------------------
    with probe(
        AREA,
        'a session used with a different schema',
        'refused, not silently mis-scored',
    ) as e:
        context = frame(60)
        base = Request(
            context=context,
            predict=pd.DataFrame({'row_id': [1], 'v': [1.0]}),
            target='y',
            task='classification',
        )
        other = Request(
            context=data.wine().head(60),
            predict=data.wine().tail(2).drop(columns=['quality']),
            target='quality',
            task='classification',
        )
        with client.fit(base, model=KumoTabular()) as fitted:
            try:
                result = fitted.predict(other).to_pandas()
                fail(
                    e, f'a mismatched schema scored anyway: {len(result)} rows'
                )
            except StructuredDataError as exc:
                note(e, str(exc)[:140])

    # --- numpy scalars in inference --------------------------------------
    with probe(
        AREA,
        'a numpy scalar in inference',
        'converted or refused, never a raw TypeError',
    ) as e:
        built = data.northwind_ready()
        request = PredictiveQuery(
            'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
            built,
        ).materialize(
            indices=list(built['Customers'].data['CustomerID'].head(2)),
            num_context_samples=60,
            inference_config={'num_estimators': np.int64(2)},
        )
        try:
            relational.predict(request, model=KumoRelational()).to_pandas()
        except StructuredDataError as exc:
            note(e, f'refused: {str(exc)[:140]}')
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    # --- duplicate entities -----------------------------------------------
    with probe(
        AREA, 'the same entity asked for twice', 'answered twice, in order'
    ) as e:
        built = data.northwind_ready()
        ids = list(built['Orders'].data['OrderID'].head(3))
        asked = [ids[0], ids[0], ids[1], ids[2]]
        try:
            request = PredictiveQuery(
                'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
            ).materialize(indices=asked, num_context_samples=200)
            result = relational.predict(
                request, model=KumoRelational()
            ).to_pandas()
            if len(result) != len(asked):
                fail(
                    e,
                    f'asked for {len(asked)}, got {len(result)}: {list(result["id"])}',
                )
            else:
                note(e, f'ids={list(result["id"])}')
        except StructuredDataError as exc:
            note(e, f'refused: {str(exc)[:140]}')

    # --- the error contract ------------------------------------------------
    with probe(
        AREA,
        'an unreachable host',
        'a StructuredDataError, not a requests exception',
    ) as e:
        try:
            with SDMClient(url='http://127.0.0.1:9', timeout=5.0) as dead:
                dead.models()
            fail(e, 'an unreachable host succeeded')
        except StructuredDataError as exc:
            note(e, f'{exc.code}: {str(exc)[:100]}')
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    with probe(
        AREA,
        'a model the endpoint does not serve',
        'an UnknownModelError naming it',
    ) as e:
        try:
            client.predict(
                Request(
                    context=frame(60),
                    predict=pd.DataFrame({'row_id': [1], 'v': [1.0]}),
                    target='y',
                    task='classification',
                ),
                model=KumoRelational(),
            )
            fail(e, 'the tabular endpoint served kumo-relational')
        except (UnknownModelError, NimRequestError, StructuredDataError) as exc:
            note(e, f'{type(exc).__name__}: {str(exc)[:120]}')

    with probe(
        AREA, 'a url with a credential in the query string', 'refused'
    ) as e:
        try:
            SDMClient(url='http://localhost:8000?api_key=secret')
            fail(e, 'a query string on the base url was accepted')
        except StructuredDataError as exc:
            note(e, str(exc)[:120])

    with probe(
        AREA, 'a url with userinfo', 'refused over plaintext to a remote host'
    ) as e:
        # localhost is exempt on purpose -- the traffic never leaves the box.
        try:
            SDMClient(url='http://user:secret@example.com')
            fail(e, 'userinfo over plaintext was accepted')
        except StructuredDataError as exc:
            note(e, str(exc)[:120])

    with probe(
        AREA,
        'a timeout too small to complete',
        'a StructuredDataError, not a socket error',
    ) as e:
        try:
            with SDMClient(url=TABULAR, timeout=0.001) as impatient:
                impatient.predict(
                    Request(
                        context=data.adult().head(400),
                        predict=data.adult().tail(3).drop(columns=['income']),
                        target='income',
                        task='classification',
                    ),
                    model=KumoTabular(),
                )
            note(e, 'it completed within 1ms')
        except StructuredDataError as exc:
            note(e, f'{exc.code}: {str(exc)[:100]}')
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )


if __name__ == '__main__':
    with (
        SDMClient(url=TABULAR, timeout=300.0) as client,
        SDMClient(url=RELATIONAL, timeout=900.0) as relational,
    ):
        run(client, relational)
    save('edges.json')
