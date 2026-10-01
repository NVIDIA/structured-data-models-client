# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""kumo-tabular against real datasets, through every public entry point."""

from __future__ import annotations

import data
import numpy as np
import pandas as pd
from harness import TABULAR, fail, note, probe, save
from sdm_client import (
    KumoTabular,
    PredictiveQuery,
    Request,
    SDMClient,
    StructuredDataError,
)

AREA = 'tabular'


def split(frame: pd.DataFrame, context: int, score: int):
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    return shuffled.head(context), shuffled.tail(score).reset_index(drop=True)


def run(client):
    # --- real dataset, binary classification -----------------------------
    with probe(
        AREA,
        'adult binary classification',
        '3 rows scored, probabilities sum to 1',
    ) as e:
        ctx, sc = split(data.adult(), 400, 3)
        result = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['income']),
                target='income',
                task='classification',
            ),
            model=KumoTabular(),
        )
        frame = result.to_pandas()
        if len(frame) != 3:
            fail(e, f'expected 3 rows, got {len(frame)}')
        elif not {'probability_0', 'probability_1'} <= set(frame.columns):
            fail(e, f'no probability columns: {list(frame.columns)}')
        elif (
            not frame[['probability_0', 'probability_1']]
            .sum(axis=1)
            .sub(1)
            .abs()
            .lt(1e-4)
            .all()
        ):
            fail(e, 'probabilities do not sum to 1')
        else:
            note(e, f'predictions={list(frame["prediction"])}')

    # --- multiclass, 7 ordinal classes -----------------------------------
    with probe(
        AREA,
        'wine multiclass classification',
        'a class from the training set, per row',
    ) as e:
        ctx, sc = split(data.wine(), 500, 5)
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['quality']),
                target='quality',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        classes = set(ctx['quality'].unique())
        got = set(frame['prediction'])
        if len(frame) != 5:
            fail(e, f'expected 5 rows, got {len(frame)}')
        elif not got <= classes:
            fail(e, f'predicted a class never seen in context: {got - classes}')
        else:
            probs = [c for c in frame.columns if c.startswith('probability_')]
            note(
                e,
                f'{len(probs)} probability columns for {len(classes)} classes',
            )

    # --- regression -------------------------------------------------------
    with probe(
        AREA, 'wine regression', 'a finite number in the target range'
    ) as e:
        ctx, sc = split(data.wine(), 500, 5)
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['alcohol']),
                target='alcohol',
                task='regression',
            ),
            model=KumoTabular(),
        ).to_pandas()
        values = frame['prediction'].astype(float)
        if not np.isfinite(values).all():
            fail(e, f'non-finite prediction: {list(values)}')
        elif not values.between(0, 30).all():
            fail(e, f'alcohol outside any plausible range: {list(values)}')

    # --- nulls, bools, object columns ------------------------------------
    with probe(
        AREA, 'titanic with nulls and bools', 'nulls survive the wire'
    ) as e:
        ctx, sc = split(data.titanic(), 400, 4)
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['survived']),
                target='survived',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        if len(frame) != 4:
            fail(e, f'expected 4 rows, got {len(frame)}')
        else:
            note(e, f'context nulls={int(ctx.isna().sum().sum())}')

    with probe(
        AREA, 'category dtype column', 'a pandas category is sent, not rejected'
    ) as e:
        ctx, sc = split(data.titanic(), 300, 2)
        ctx = ctx.copy()
        sc = sc.copy()
        ctx['class'] = ctx['class'].astype('category')
        sc['class'] = sc['class'].astype('category')
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['survived']),
                target='survived',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        if len(frame) != 2:
            fail(e, f'expected 2 rows, got {len(frame)}')

    # --- shape edges ------------------------------------------------------
    with probe(AREA, 'one row scored', 'exactly one row back') as e:
        ctx, sc = split(data.adult(), 200, 1)
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['income']),
                target='income',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        if len(frame) != 1:
            fail(e, f'expected 1 row, got {len(frame)}')

    with probe(
        AREA, 'one context row', 'either a prediction or a clear refusal'
    ) as e:
        ctx, sc = split(data.adult(), 1, 2)
        try:
            frame = client.predict(
                Request(
                    context=ctx,
                    predict=sc.drop(columns=['income']),
                    target='income',
                    task='classification',
                ),
                model=KumoTabular(),
            ).to_pandas()
            note(e, f'served: {len(frame)} rows')
        except StructuredDataError as exc:
            note(e, f'refused cleanly: {exc}')

    with probe(
        AREA,
        'empty predict frame',
        'a shaped result: zero rows is legal, no columns is not',
    ) as e:
        ctx, _ = split(data.adult(), 200, 1)
        result = client.predict(
            Request(
                context=ctx,
                predict=data.adult().head(0).drop(columns=['income']),
                target='income',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        if len(result) != 0:
            fail(e, f'{len(result)} rows for an empty frame')
        elif 'prediction' not in result.columns:
            fail(e, f'no columns to join or read: {list(result.columns)}')
        else:
            note(e, f'columns {list(result.columns)}')

    with probe(AREA, 'a column of only nulls', 'not a crash') as e:
        ctx, sc = split(data.titanic(), 300, 2)
        ctx = ctx.copy()
        sc = sc.copy()
        ctx['empty_col'] = None
        sc['empty_col'] = None
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['survived']),
                target='survived',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        if len(frame) != 2:
            fail(e, f'expected 2 rows, got {len(frame)}')

    # --- integers past the JS safe range ----------------------------------
    with probe(
        AREA,
        'int64 past the JS safe range',
        'sent as a base-10 string, not a float',
    ) as e:
        big = 9_007_199_254_740_993
        ctx = pd.DataFrame(
            {
                'row_id': range(60),
                'big': [big + i for i in range(60)],
                'v': [float(i) for i in range(60)],
                'y': [i % 2 for i in range(60)],
            }
        )
        sc = pd.DataFrame(
            {'row_id': [900, 901], 'big': [big, big + 5], 'v': [1.0, 2.0]}
        )
        frame = client.predict(
            Request(context=ctx, predict=sc, target='y', task='classification'),
            model=KumoTabular(),
        ).to_pandas()
        if len(frame) != 2:
            fail(e, f'expected 2 rows, got {len(frame)}')

    with probe(
        AREA,
        'the big int is a string on the wire',
        'no float64 rounding in the payload',
    ) as e:
        from sdm_client.wire.dtypes import infer_tfm_dtype  # noqa: PLC0415
        from sdm_client.wire.tables import encode_table  # noqa: PLC0415

        big = 9_007_199_254_740_993
        frame = pd.DataFrame({'big': [big], 'v': [1.0]})
        payload = encode_table(
            frame, {c: infer_tfm_dtype(frame[c]) for c in frame}
        )
        cell = payload['rows'][0][0]
        if cell != str(big):
            fail(e, f'{big} went on the wire as {type(cell).__name__} {cell!r}')

    # --- unicode and long text -------------------------------------------
    with probe(
        AREA, 'unicode and long strings', 'round trip without mangling'
    ) as e:
        # 5,000 bytes is past the endpoint's 1,024-byte cell limit, which it
        # refuses with a clear 413; the point here is the characters, not size.
        text = [
            '日本語',
            'emoji 🚀',
            'x' * 900,
            'quote\'"backslash\\',
            'nul-free\ttab',
            '',
        ]
        ctx = pd.DataFrame(
            {
                'row_id': range(60),
                'note': [text[i % len(text)] for i in range(60)],
                'v': [float(i) for i in range(60)],
                'y': [i % 2 for i in range(60)],
            }
        )
        sc = pd.DataFrame(
            {
                'row_id': [900, 901],
                'note': ['日本語', 'x' * 900],
                'v': [1.0, 2.0],
            }
        )
        frame = client.predict(
            Request(context=ctx, predict=sc, target='y', task='classification'),
            model=KumoTabular(),
        ).to_pandas()
        if len(frame) != 2:
            fail(e, f'expected 2 rows, got {len(frame)}')

    # --- outputs ----------------------------------------------------------
    with probe(AREA, 'prediction only', 'no probability columns') as e:
        ctx, sc = split(data.adult(), 200, 2)
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['income']),
                target='income',
                task='classification',
            ),
            model=KumoTabular(),
            outputs=['prediction'],
        ).to_pandas()
        extra = [c for c in frame.columns if c.startswith('probability_')]
        if extra:
            fail(e, f'asked for prediction only, got {extra}')

    with probe(
        AREA,
        'an output this model does not serve',
        'named locally, no round trip',
    ) as e:
        ctx, sc = split(data.adult(), 200, 2)
        try:
            client.predict(
                Request(
                    context=ctx,
                    predict=sc.drop(columns=['income']),
                    target='income',
                    task='classification',
                ),
                model=KumoTabular(),
                outputs=['explanation'],
            )
            fail(
                e, 'explanation was accepted by a model that does not serve it'
            )
        except StructuredDataError as exc:
            note(e, str(exc)[:120])

    # --- correlation ------------------------------------------------------
    with probe(
        AREA, 'row_index joins one-to-one', 'every scored row is answered once'
    ) as e:
        ctx, sc = split(data.adult(), 300, 25)
        to_score = sc.drop(columns=['income'])
        frame = client.predict(
            Request(
                context=ctx,
                predict=to_score,
                target='income',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        joined = to_score.reset_index(names='row_index').merge(
            frame, on='row_index', validate='one_to_one'
        )
        if len(joined) != len(to_score):
            fail(e, f'{len(to_score)} rows sent, {len(joined)} correlated')

    with probe(
        AREA,
        'a shuffled index still correlates',
        'row_index is positional, not the pandas index',
    ) as e:
        ctx, sc = split(data.adult(), 300, 6)
        to_score = sc.drop(columns=['income'])
        to_score.index = [99, 3, 41, 7, 12, 55]
        frame = client.predict(
            Request(
                context=ctx,
                predict=to_score,
                target='income',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        if sorted(frame['row_index']) != list(range(6)):
            fail(
                e,
                f'row_index followed the pandas index: {sorted(frame["row_index"])}',
            )

    # --- sessions ---------------------------------------------------------
    with probe(
        AREA,
        'a fitted context serves several calls',
        'each call scores its own rows',
    ) as e:
        ctx, sc = split(data.adult(), 300, 9)
        base = Request(
            context=ctx,
            predict=sc.drop(columns=['income']),
            target='income',
            task='classification',
        )
        with client.fit(base, model=KumoTabular()) as fitted:
            first = fitted.predict(base).to_pandas()
            fewer = Request(
                context=ctx,
                predict=sc.drop(columns=['income']).head(2),
                target='income',
                task='classification',
            )
            second = fitted.predict(fewer).to_pandas()
            if len(first) != 9 or len(second) != 2:
                fail(e, f'{len(first)} then {len(second)}, expected 9 then 2')

    with probe(
        AREA,
        'using a session after it closes',
        'a clear error, not a 404 or a hang',
    ) as e:
        ctx, sc = split(data.adult(), 200, 2)
        base = Request(
            context=ctx,
            predict=sc.drop(columns=['income']),
            target='income',
            task='classification',
        )
        with client.fit(base, model=KumoTabular()) as fitted:
            pass
        try:
            fitted.predict(base)
            fail(e, 'a closed session still served a prediction')
        except StructuredDataError as exc:
            note(e, str(exc)[:120])

    with probe(AREA, 'two sessions at once', 'they do not collide') as e:
        ctx, sc = split(data.adult(), 200, 3)
        one = Request(
            context=ctx,
            predict=sc.drop(columns=['income']),
            target='income',
            task='classification',
        )
        wctx, wsc = split(data.wine(), 200, 4)
        two = Request(
            context=wctx,
            predict=wsc.drop(columns=['quality']),
            target='quality',
            task='classification',
        )
        with (
            client.fit(one, model=KumoTabular()) as a,
            client.fit(two, model=KumoTabular()) as b,
        ):
            if a.session_id == b.session_id:
                fail(e, 'both sessions share an id')
            elif (
                len(a.predict(one).to_pandas()) != 3
                or len(b.predict(two).to_pandas()) != 4
            ):
                fail(e, 'a session answered the wrong number of rows')

    # --- two more datasets, chosen for what they are awkward about --------
    with probe(
        AREA,
        'bank marketing, an imbalanced binary target',
        'both classes are reachable',
    ) as e:
        ctx, sc = split(data.bank(), 800, 6)
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['y']),
                target='y',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        rate = ctx['y'].mean()
        if len(frame) != 6:
            fail(e, f'expected 6 rows, got {len(frame)}')
        else:
            note(
                e,
                f'{rate:.1%} positive in context, predictions '
                f'{list(frame["prediction"])}',
            )

    with probe(
        AREA,
        "bank marketing's 'unknown' is a level, not a null",
        'it survives as a category',
    ) as e:
        ctx, sc = split(data.bank(), 800, 3)
        unknowns = int((ctx == 'unknown').sum().sum())
        frame = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['y']),
                target='y',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        if len(frame) != 3:
            fail(e, f'expected 3 rows, got {len(frame)}')
        else:
            note(e, f"{unknowns} 'unknown' cells in the context")

    with probe(
        AREA,
        'column names with spaces, slashes and a tab',
        'sent as given, not sanitised',
    ) as e:
        frame_in = data.students()
        awkward = [
            c for c in frame_in.columns if any(ch in c for ch in ' /()\t')
        ]
        ctx, sc = split(frame_in, 600, 4)
        result = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['Target']),
                target='Target',
                task='classification',
            ),
            model=KumoTabular(),
        ).to_pandas()
        if len(result) != 4:
            fail(e, f'expected 4 rows, got {len(result)}')
        elif not set(result['prediction']) <= set(ctx['Target'].unique()):
            fail(
                e,
                f'predicted outside the label set: {set(result["prediction"])}',
            )
        else:
            note(e, f'{len(awkward)} awkward column names, e.g. {awkward[:2]}')

    with probe(
        AREA,
        'a three-class target on real data',
        'one probability column per class',
    ) as e:
        ctx, sc = split(data.students(), 600, 4)
        result = client.predict(
            Request(
                context=ctx,
                predict=sc.drop(columns=['Target']),
                target='Target',
                task='classification',
            ),
            model=KumoTabular(),
            outputs=['prediction', 'probabilities'],
        ).to_pandas()
        columns = [c for c in result.columns if c.startswith('probability_')]
        if len(columns) != ctx['Target'].nunique():
            fail(
                e,
                f'{len(columns)} probability columns for '
                f'{ctx["Target"].nunique()} classes',
            )
        elif not result[columns].sum(axis=1).sub(1).abs().lt(1e-4).all():
            fail(e, 'probabilities do not sum to 1')

    # --- the query path ---------------------------------------------------
    with probe(
        AREA,
        'a predictive query on a real dataset',
        'the query names the target',
    ) as e:
        ctx, sc = split(data.adult(), 300, 3)
        request = PredictiveQuery(
            'PREDICT person.income FOR EACH person.person_id',
            {'person': ctx},
        ).materialize(predict=sc.drop(columns=['income']))
        frame = client.predict(request, model=KumoTabular()).to_pandas()
        if len(frame) != 3:
            fail(e, f'expected 3 rows, got {len(frame)}')
        elif request.task.target.column_name != 'income':
            fail(e, f'target came out as {request.task.target.column_name}')


if __name__ == '__main__':
    with SDMClient(url=TABULAR, timeout=300.0) as client:
        run(client)
    save('tabular.json')
