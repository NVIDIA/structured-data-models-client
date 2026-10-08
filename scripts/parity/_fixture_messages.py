# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The message fixtures: a case for each warning and refusal a user meets.

``packages/structured-data-models-client/diagnostics.json`` lists every warning and refusal of
the client and every engine raise that reaches users. Phase 8.5 ports the
modules that give most of them, so each is pinned here by a case that gives
it, the way a user meets it: mostly as public calls over small frames
(``client.run``), with the model answered as the parity stand-in answers.
What no input reaches is listed in ``tests/parity/UNREACHABLE_DIAGNOSTICS.md``
with the reason.

Each case records the error it ends in, class, code, message and the keys of
its ``details``, and every warning its caller sees, in order
(``tests/parity/FORMAT.md``, Fixtures). ``make_fixtures.py`` writes them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd

REL = 'kumo-relational'
TAB = 'kumo-tabular'

COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
INACTIVE = 'PREDICT COUNT(orders.*, 0, 60, days)=0 FOR EACH users.user_id'
SUM = 'PREDICT SUM(orders.quantity, 0, 30, days) FOR EACH products.product_id'
FORECAST = (
    'PREDICT SUM(orders.quantity, 0, 30, days) '
    'FORECAST 3 TIMEFRAMES FOR EACH products.product_id'
)
RANKING = (
    'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) '
    'RANK TOP 2 FOR EACH users.user_id'
)
LINK = (
    'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) '
    'FOR EACH users.user_id'
)
CATEGORY = 'PREDICT products.category FOR EACH products.product_id'
ANCHOR = {'$timestamp': '2024-09-01T00:00:00'}


def shop() -> dict[str, pd.DataFrame]:
    r"""A shop small enough to inline in every case: six users, four
    products, and forty order lines over 2024, every user buying."""
    users = pd.DataFrame(
        {
            'user_id': pd.Series([1, 2, 3, 4, 5, 6], dtype='int64'),
            'age': pd.Series([25, 32, 41, 19, 57, 38], dtype='int64'),
            'country': ['US', 'DE', 'IN', 'US', 'DE', 'IN'],
        }
    )
    products = pd.DataFrame(
        {
            'product_id': pd.Series([1, 2, 3, 4], dtype='int64'),
            'category': ['books', 'games', 'tools', 'books'],
            'price': [10.5, 20.0, 7.25, 12.0],
        }
    )
    lines = 40
    orders = pd.DataFrame(
        {
            'order_id': pd.Series(range(1, lines + 1), dtype='int64'),
            'user_id': pd.Series(
                [i % 6 + 1 for i in range(lines)], dtype='int64'
            ),
            'product_id': pd.Series(
                [(i * 7) % 4 + 1 for i in range(lines)], dtype='int64'
            ),
            'quantity': pd.Series(
                [i % 4 + 1 for i in range(lines)], dtype='int64'
            ),
            'ordered_at': pd.Series(
                [
                    pd.Timestamp('2024-01-02')
                    + pd.Timedelta(days=9 * i, hours=8 + i % 9)
                    for i in range(lines)
                ],
                dtype='datetime64[us]',
            ),
        }
    )
    return {'users': users, 'products': products, 'orders': orders}


def people(rows: int = 12) -> pd.DataFrame:
    r"""One table for the tabular model: a key, two features, two targets."""
    return pd.DataFrame(
        {
            'user_id': pd.Series(range(rows), dtype='int64'),
            'age': [20.0 + i for i in range(rows)],
            'segment': ['north' if i % 3 else 'south' for i in range(rows)],
            'churn': pd.Series([i % 2 for i in range(rows)], dtype='int64'),
        }
    )


def call(name: str, **args: Any) -> dict[str, Any]:
    return {'call': name, 'args': args}


def rows(table: str, start: int, stop: int, drop: list[str] | None = None):
    spec: dict[str, Any] = {'table': table, 'rows': [start, stop]}
    if drop:
        spec['drop'] = drop
    return {'$frame': spec}


def whole(table: str, drop: list[str] | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {'table': table}
    if drop:
        spec['drop'] = drop
    return {'$frame': spec}


def message_groups(
    Group: type,
    frame: Callable[[Any], dict[str, Any]],
    Raises: type,
    encode_value: Callable[[Any], Any],
) -> tuple[Callable[[], Any], ...]:
    r"""The groups of message cases, given ``make_fixtures``'s ``Group``,
    ``frame``, ``Raises`` and the fixture value encoder."""

    def tables(source: dict[str, pd.DataFrame]) -> dict[str, Any]:
        return {name: frame(table) for name, table in source.items()}

    SHOP = tables(shop())
    PEOPLE = tables({'people': people()})

    def schema(**args: Any) -> dict[str, Any]:
        return call('Schema.from_data', **args)

    def gathered(query: str, **options: Any) -> list[dict[str, Any]]:
        r"""The shop's schema, ``query``, and a gathering of it."""
        options.setdefault('random_seed', 7)
        return [
            schema(),
            call('PredictiveQuery', query=query),
            call('PredictiveQuery.materialize', **options),
        ]

    def flow(
        g: Any,
        label: str,
        calls: list[dict[str, Any]],
        *,
        data: dict[str, Any] | None = None,
        stand_in: dict[str, Any] | None = None,
        want: Any = None,
    ) -> None:
        g.flow(
            label,
            {'tables': SHOP if data is None else data, 'calls': calls},
            stand_in=stand_in,
            **({} if want is None else {'want': want}),
        )

    predict = call('SDMClient.predict', model=REL)
    to_pandas = call('PredictionResult.to_pandas')

    def messages_client() -> Any:
        r"""``SDMClient``, ``FittedModel`` and what they refuse or warn."""
        g = Group('messages-client')
        count = gathered(COUNT, indices=[1, 2], num_context_samples=12)
        fit = call('SDMClient.fit', model=REL)
        # FittedModel.predict on a closed handle.
        flow(
            g,
            'scoring-a-closed-session',
            [
                *count,
                fit,
                call('FittedModel.close'),
                call('FittedModel.predict'),
            ],
            want=Raises('this fitted model is closed'),
        )
        # A ranking scored against a session.
        flow(
            g,
            'scoring-a-ranking-against-a-session',
            [
                *count,
                fit,
                call('PredictiveQuery', query=RANKING),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    candidates=[1, 2, 3],
                    anchor_time=ANCHOR,
                    num_context_samples=12,
                    random_seed=7,
                ),
                call('FittedModel.predict'),
            ],
            want=Raises('questions and a session pins one context'),
        )
        # A session asked a different question.
        flow(
            g,
            'scoring-another-question-against-a-session',
            [
                *count,
                fit,
                call('PredictiveQuery', query=INACTIVE),
                call(
                    'PredictiveQuery.materialize',
                    indices=[3],
                    num_context_samples=12,
                    random_seed=7,
                ),
                call('FittedModel.predict'),
            ],
            want=Raises('this session pinned a different question'),
        )
        # Scoring rows the pinned context answers.
        flow(
            g,
            'scoring-rows-the-session-pinned',
            [
                *count,
                fit,
                call(
                    'PredictiveQuery.materialize',
                    indices=[1, 2, 3, 4, 5, 6],
                    num_context_samples=12,
                    random_seed=7,
                ),
                call('FittedModel.predict'),
                to_pandas,
            ],
        )
        flow(
            g,
            'scoring-rows-a-static-session-pinned',
            [
                *gathered(CATEGORY, indices=[1]),
                fit,
                call('PredictiveQuery.materialize', indices=[2, 3, 4]),
                call('FittedModel.predict'),
                to_pandas,
            ],
        )
        # Holdout.
        for label, holdout in (
            ('holdout-as-one-string', 'u1'),
            ('holdout-matching-nothing', [101, 102]),
            ('holdout-in-the-context', [1, 2, 3, 4, 5, 6]),
            ('holdout-empty', []),
        ):
            flow(
                g,
                label,
                [
                    *count,
                    call('SDMClient.fit', model=REL, holdout=holdout),
                    call(
                        'PredictiveQuery.materialize',
                        indices=[3, 4],
                        num_context_samples=12,
                        random_seed=7,
                    ),
                    call('FittedModel.predict'),
                    to_pandas,
                ],
            )
        # A ranking pinned in a session.
        ranking = gathered(
            RANKING,
            indices=[1],
            candidates=[1, 2, 3],
            anchor_time=ANCHOR,
            num_context_samples=12,
        )
        flow(
            g,
            'pinning-a-ranking',
            [*ranking, fit],
            want=Raises('a session pins one context'),
        )
        # Endpoints that cannot pin a context.
        flow(
            g,
            'fitting-on-a-serving-endpoint',
            [
                *count,
                call('SDMClient.for_databricks_serving', endpoint='rfm'),
                fit,
            ],
            want=Raises('cannot pin a context'),
        )
        flow(
            g,
            'fitting-where-sessions-are-not-served',
            [*count, fit],
            stand_in={
                'fail': [
                    {
                        'at': 0,
                        'status': 404,
                        'body': {'title': 'Not Found', 'status': 404},
                    }
                ]
            },
            want=Raises('this endpoint does not serve sessions'),
        )
        flow(
            g,
            'a-session-without-an-id',
            [*count, fit],
            stand_in={'fail': [{'at': 0, 'status': 200, 'body': {}}]},
            want=Raises('named no session id'),
        )
        # outputs.
        for label, outputs, text in (
            ('outputs-empty', [], 'at least one field'),
            ('outputs-a-string', 'prediction', 'not a single string'),
            ('outputs-not-names', [1, 2], 'sequence of field names, got'),
            ('outputs-not-a-sequence', 5, 'sequence of field names, got'),
            ('outputs-the-model-does-not-produce', ['embedding'], 'produce'),
        ):
            flow(
                g,
                label,
                [*count, call('SDMClient.predict', model=REL, outputs=outputs)],
                want=Raises(text),
            )
        flow(
            g,
            'request-not-a-request',
            [*count, call('SDMClient.predict', model=REL, request='rows')],
            want=Raises('request must be a Request'),
        )
        # Rankings: outputs, cold candidates, missing probabilities.
        for label, outputs, text in (
            ('ranking-outputs-empty', [], 'at least one field'),
            (
                'ranking-outputs-not-its-own',
                ['probabilities', 'scores'],
                'a ranking gives back',
            ),
            ('ranking-outputs-one-not-its-own', ['scores'], 'is not among'),
        ):
            flow(
                g,
                label,
                [
                    *ranking,
                    call('SDMClient.predict', model=REL, outputs=outputs),
                ],
                want=Raises(text),
            )
        for label, candidates in (
            ('ranking-one-candidate-cold', [1, 2, 99]),
            ('ranking-two-candidates-cold', [1, 98, 99]),
        ):
            flow(
                g,
                label,
                [
                    *gathered(
                        RANKING,
                        indices=[1, 2],
                        candidates=candidates,
                        anchor_time=ANCHOR,
                        num_context_samples=12,
                    ),
                    predict,
                    to_pandas,
                ],
            )
        for label, candidates in (
            ('ranking-every-candidate-cold', [99]),
            ('ranking-both-candidates-cold', [98, 99]),
        ):
            flow(
                g,
                label,
                [
                    *gathered(
                        RANKING,
                        indices=[1],
                        candidates=candidates,
                        anchor_time=ANCHOR,
                        num_context_samples=12,
                    ),
                    predict,
                ],
                want=Raises('nothing to rank and nothing was sent'),
            )
        flow(
            g,
            'ranking-answered-without-probabilities',
            [*ranking, predict],
            stand_in={'omit': ['probabilities']},
            want=Raises('without saying how sure it was'),
        )
        flow(
            g,
            'ranking-in-order',
            [*ranking, predict, to_pandas],
        )
        # Fields the answer left out.
        flow(
            g,
            'a-field-asked-for-and-not-answered',
            [
                *gathered(CATEGORY, indices=[1, 2]),
                call(
                    'SDMClient.predict',
                    model=REL,
                    outputs=['prediction', 'probabilities'],
                ),
                to_pandas,
            ],
            stand_in={'omit': ['probabilities']},
        )
        forecast = gathered(
            FORECAST, indices=[1], anchor_time=ANCHOR, num_context_samples=12
        )
        explained = call(
            'SDMClient.predict',
            model=REL,
            outputs=['prediction', 'explanation'],
        )
        flow(
            g,
            'forecast-explained-by-step',
            [*forecast, explained, to_pandas],
        )
        flow(
            g,
            'forecast-a-step-unexplained',
            [*forecast, explained, to_pandas],
            stand_in={'omit_at': [{'at': 1, 'fields': ['explanation']}]},
        )
        flow(
            g,
            'forecast-no-step-explained',
            [*forecast, explained, to_pandas],
            stand_in={'omit': ['explanation']},
        )
        # How a ranking orders its candidates (task 6 of phase 8): ties keep
        # the order the candidates were named in, and a NaN score is ordered
        # as Python's stable sort leaves it. The port decides later whether
        # to keep that order or refuse NaN.
        nan = {'$float': 'nan'}
        for label, scores, top in (
            ('ranking-in-order', [[0.2, 0.9], [0.7, 0.1], [0.5, 0.5]], 2),
            ('ranking-ties-keep-the-named-order', [[0.5], [0.7], [0.5]], 3),
            ('ranking-a-nan-score-first', [[nan], [0.7], [0.2]], 3),
            ('ranking-a-nan-score-between', [[0.2], [nan], [0.7]], 3),
            ('ranking-a-nan-score-last', [[0.7], [0.2], [nan]], None),
        ):
            g.add(
                label,
                'client.ranked',
                {'candidates': [3, 1, 2], 'scores': scores, 'top': top},
            )
        # Models.
        for label, model, text in (
            ('model-unknown-id', 'kumo-nothing', 'kumo-nothing'),
            ('model-a-bare-handle', {'$model': 'Model'}, 'Model'),
            ('model-not-a-model', 3, 'model must be a model handle'),
        ):
            flow(
                g,
                label,
                [*count, call('SDMClient.predict', model=model)],
                want=Raises(text),
            )
        return g

    def tabular(
        g: Any,
        label: str,
        calls: list[dict[str, Any]],
        stand_in: dict[str, Any] | None = None,
        want: Any = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        flow(
            g,
            label,
            calls,
            data=PEOPLE if data is None else data,
            stand_in=stand_in,
            want=want,
        )

    context = whole('people')
    scored = rows('people', 0, 3, drop=['churn'])

    def direct(**args: Any) -> dict[str, Any]:
        r"""``Request(...)`` over ``people``, asking churn as a
        classification, with ``args`` over that."""
        given = {
            'context': context,
            'predict': scored,
            'target': 'churn',
            'task': 'classification',
            **args,
        }
        return call('Request', **{k: v for k, v in given.items() if v != '-'})

    tab_predict = call('SDMClient.predict', model=TAB)

    def messages_request() -> Any:
        r"""Building a request: ``Request``, ``PredictiveQuery`` and its
        ``materialize``, the decomposition of rankings and forecasts, and
        ``PredictionResult.merge``."""
        g = Group('messages-request')
        # Request(...) from frames.
        for label, args, text in (
            ('task-unknown', {'task': 'clustering'}, 'task must be one of'),
            ('context-not-a-frame', {'context': {'age': 1}}, 'DataFrame'),
            ('predict-not-a-frame', {'predict': [1, 2]}, 'DataFrame'),
            (
                'related-tables',
                {'related_tables': {'orders': whole('people')}},
                'related_tables',
            ),
            (
                'context-without-rows',
                {'context': rows('people', 0, 0)},
                'context has no rows',
            ),
            ('context-without-the-target', {'target': 'tier'}, 'no target'),
            (
                'predict-carrying-the-target',
                {'predict': rows('people', 0, 3)},
                'predict carries the target column',
            ),
            (
                'predict-with-a-column-context-lacks',
                {'context': whole('people', drop=['age'])},
                'that context does not',
            ),
            (
                'predict-missing-a-column-context-has',
                {'predict': rows('people', 0, 3, drop=['churn', 'age'])},
                'predict is missing column(s)',
            ),
        ):
            tabular(g, label, [direct(**args)], want=Raises(text))
        unlabelled = tables(
            {
                'people': people().assign(
                    churn=pd.Series([None] * 12, dtype='float64')
                )
            }
        )
        tabular(
            g,
            'context-without-a-known-target',
            [direct()],
            data=unlabelled,
            want=Raises('at least one row with a known target'),
        )
        listed = people().astype({'churn': object})
        listed['churn'] = [[i % 2] for i in range(12)]
        tabular(
            g,
            'target-holding-lists',
            [direct()],
            data=tables({'people': listed}),
            want=Raises('multi-label'),
        )
        constant = people().assign(churn=1)
        for label, task in (
            ('one-class-context', 'classification'),
            ('constant-regression-context', 'regression'),
        ):
            tabular(
                g,
                label,
                [direct(task=task), tab_predict, to_pandas],
                data=tables({'people': constant}),
            )
        tabular(
            g,
            'scoring-rows-the-context-answers',
            [
                schema(),
                direct(
                    predict=rows('people', 0, 3, drop=['churn']),
                    schema={'$schema': None},
                    entity_table='people',
                ),
                tab_predict,
                to_pandas,
            ],
        )
        worded = people().astype({'age': object})
        worded['age'] = ['old' if i % 2 else 'young' for i in range(12)]
        tabular(
            g,
            'scored-values-the-schema-does-not-mean',
            [
                schema(),
                direct(
                    predict={'$frame': {'table': 'worded', 'rows': [0, 3]}},
                    schema={'$schema': None},
                    entity_table='people',
                ),
            ],
            data=tables(
                {'people': people(), 'worded': worded.drop(columns=['churn'])}
            ),
            want=Raises('the rows being scored hold'),
        )
        flow(
            g,
            'replacing-a-gathered-half',
            [
                *gathered(COUNT, indices=[1]),
                call('Request.assign', half='predict', value=whole('users')),
            ],
            want=Raises('cannot be replaced on a request gathered'),
        )
        # PredictiveQuery(...) and materialize(...).
        for label, query in (('query-empty', '  '), ('query-not-text', 5)):
            flow(
                g,
                label,
                [schema(), call('PredictiveQuery', query=query)],
                want=Raises('query must be a non-empty string'),
            )
        flow(
            g,
            'query-that-does-not-resolve',
            [
                schema(),
                call(
                    'PredictiveQuery',
                    query='PREDICT COUNT(orders.*, 0, 30, days) FOR EACH '
                    'users.nobody',
                ),
                call('PredictiveQuery.materialize', indices=[1]),
            ],
            want=Raises('is not a query'),
        )
        materialize = [schema(), call('PredictiveQuery', query=COUNT)]
        for label, args, text in (
            (
                'indices-and-predict',
                {'indices': [1], 'predict': whole('users')},
                'not both',
            ),
            ('neither-indices-nor-predict', {}, 'needs the rows to score'),
            (
                'tabular-model-over-three-tables',
                {'indices': [1], 'model': TAB},
                'answers from a single table',
            ),
            (
                'predict-over-three-tables',
                {'predict': whole('users')},
                'requires a single-table schema',
            ),
            ('indices-a-string', {'indices': '1'}, 'sequence of entity keys'),
            ('indices-a-number', {'indices': 1}, 'sequence of entity keys'),
            (
                'indices-of-records',
                {'indices': [{'user_id': 1}, [2]]},
                'cannot be one',
            ),
            (
                'num-context-samples-a-float',
                {'indices': [1], 'num_context_samples': 2.5},
                'whole number of labelled',
            ),
            (
                'num-context-samples-a-bool',
                {'indices': [1], 'num_context_samples': True},
                'whole number of labelled',
            ),
            (
                'num-context-samples-zero',
                {'indices': [1], 'num_context_samples': 0},
                'at least 1',
            ),
            (
                'task-beside-a-relational-query',
                {'indices': [1], 'task': 'regression'},
                'settles what a single-table question',
            ),
            (
                'candidates-without-a-ranking',
                {'indices': [1], 'candidates': [1, 2]},
                'does not rank anything',
            ),
        ):
            flow(
                g,
                label,
                [*materialize, call('PredictiveQuery.materialize', **args)],
                want=Raises(text),
            )
        one = {'people': PEOPLE['people']}
        churn = 'PREDICT people.churn FOR EACH people.user_id'
        for label, args, text in (
            (
                'tabular-model-and-relationships',
                {'predict': scored, 'model': TAB},
                'reads no relationships',
            ),
            (
                'relational-model-with-predict-and-a-relationship',
                {'predict': scored, 'model': REL},
                'drop the relationship',
            ),
        ):
            flow(
                g,
                label,
                [
                    schema(
                        tables=['people'],
                        relationships=[
                            ['people', 'user_id', 'people', 'user_id']
                        ],
                    ),
                    call('PredictiveQuery', query=churn),
                    call('PredictiveQuery.materialize', **args),
                ],
                data=one,
                want=Raises(text),
            )
        for label, args, text in (
            (
                'options-for-the-tabular-model',
                {'predict': scored, 'num_hops': 2},
                'gathering options apply only',
            ),
            (
                'relational-model-with-predict',
                {'predict': scored, 'model': REL},
                'predict=<frame> is the single-table form',
            ),
            (
                'num-context-samples-with-predict',
                {'predict': scored, 'num_context_samples': 5},
                'cannot be used with predict',
            ),
        ):
            flow(
                g,
                label,
                [
                    schema(tables=['people']),
                    call('PredictiveQuery', query=churn),
                    call('PredictiveQuery.materialize', **args),
                ],
                data=one,
                want=Raises(text),
            )
        # A schema over tables in their source.
        for label, args, text in (
            (
                'predict-over-a-source',
                {'predict': whole('users'), 'model': REL},
                'predict= carries rows',
            ),
            (
                'tabular-model-over-a-source',
                {'indices': [1], 'model': TAB},
                'sampled with SQL',
            ),
        ):
            flow(
                g,
                label,
                [
                    call('Schema.from_duckdb'),
                    call('PredictiveQuery', query=COUNT),
                    call('PredictiveQuery.materialize', **args),
                ],
                want=Raises(text),
            )
        # Rankings and forecasts asked in parts.
        rank = [schema(), call('PredictiveQuery', query=RANKING)]
        for label, args, text in (
            (
                'ranking-with-a-statistic',
                {'candidates': [1, 2], 'prediction_statistic': 'mean'},
                'prediction_statistic applies only',
            ),
            ('ranking-without-candidates', {}, 'have to be named'),
            (
                'ranking-no-candidates',
                {'candidates': []},
                'candidates is empty',
            ),
            (
                'ranking-too-many-candidates',
                {'candidates': list(range(1, 202))},
                'calls, one for each',
            ),
            (
                'ranking-a-missing-candidate',
                {'candidates': [1, None]},
                'missing',
            ),
            (
                'ranking-a-boolean-candidate',
                {'candidates': [True]},
                'True and False',
            ),
            (
                'ranking-a-fractional-candidate',
                {'candidates': [1.5]},
                'cannot name a row',
            ),
            (
                'ranking-a-non-finite-candidate',
                {'candidates': [{'$float': 'inf'}]},
                'cannot name a row',
            ),
            (
                'ranking-a-decimal-candidate',
                {'candidates': [{'$decimal': '1.5'}]},
                'names no row of a column',
            ),
            (
                'ranking-a-quoted-candidate',
                {'candidates': ["it's"]},
                'cannot be written into a query',
            ),
        ):
            flow(
                g,
                label,
                [
                    *rank,
                    call(
                        'PredictiveQuery.materialize',
                        indices=[1],
                        anchor_time=ANCHOR,
                        **args,
                    ),
                ],
                want=Raises(text),
            )
        flow(
            g,
            'ranking-over-a-filter',
            [
                schema(),
                call(
                    'PredictiveQuery',
                    query='PREDICT LIST_DISTINCT(orders.product_id WHERE '
                    'orders.quantity > 1, 0, 30, days) RANK TOP 2 FOR EACH '
                    'users.user_id',
                ),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    candidates=[1, 2],
                    anchor_time=ANCHOR,
                ),
            ],
            want=Raises('chosen by a filter'),
        )
        texts = tables(
            {
                **shop(),
                'products': shop()['products'].astype({'product_id': str}),
                'orders': shop()['orders'].astype({'product_id': str}),
            }
        )
        flow(
            g,
            'ranking-a-number-against-text-keys',
            [
                *rank,
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    candidates=[1, {'$float': 'inf'}],
                    anchor_time=ANCHOR,
                ),
            ],
            data=texts,
            want=Raises('cannot name a row to rank'),
        )
        flow(
            g,
            'forecast-with-top-k',
            [
                schema(),
                call('PredictiveQuery', query=FORECAST),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    anchor_time=ANCHOR,
                    top_k=2,
                ),
            ],
            want=Raises('a forecast ranks nothing'),
        )
        flow(
            g,
            'forecast-too-many-steps',
            [
                schema(),
                call(
                    'PredictiveQuery',
                    query='PREDICT SUM(orders.quantity, 0, 1, days) FORECAST '
                    '201 TIMEFRAMES FOR EACH products.product_id',
                ),
                call('PredictiveQuery.materialize', indices=[1]),
            ],
            want=Raises('forecast steps means'),
        )
        # PredictionResult.merge.
        flow(
            g,
            'merging-nothing',
            [call('PredictionResult.merge', parts=[])],
            want=Raises('no parts to merge'),
        )
        tabular(
            g,
            'merging-answers-of-two-models',
            [
                direct(),
                tab_predict,
                schema(),
                call('PredictiveQuery', query=churn),
                call('PredictiveQuery.materialize', indices=[1], model=REL),
                call('SDMClient.predict', model=REL),
                call('PredictionResult.merge'),
            ],
            want=Raises('different models'),
        )
        tabular(
            g,
            'merging-answers-to-different-outputs',
            [
                direct(),
                tab_predict,
                call('SDMClient.predict', model=TAB, outputs=['prediction']),
                call('PredictionResult.merge'),
            ],
            want=Raises('different outputs'),
        )
        tabular(
            g,
            'merging-batches',
            [
                direct(predict=rows('people', 0, 4, drop=['churn'])),
                tab_predict,
                direct(predict=rows('people', 4, 12, drop=['churn'])),
                tab_predict,
                call('PredictionResult.merge'),
                to_pandas,
            ],
        )
        # Requests the model cannot answer.
        tabular(
            g,
            'tabular-request-to-the-relational-model',
            [direct(), call('SDMClient.predict', model=REL)],
            want=Raises('has to be gathered first'),
        )
        flow(
            g,
            'gathered-request-to-the-tabular-model',
            [
                *gathered(COUNT, indices=[1]),
                call('SDMClient.predict', model=TAB),
            ],
        )
        return g

    def variant(**changed: pd.DataFrame) -> dict[str, Any]:
        r"""The shop with the tables ``changed`` names replaced or added."""
        return tables({**shop(), **changed})

    def messages_gathering() -> Any:
        r"""Gathering a relational request: the materializer's checks of its
        options, the schema and the graph, and what it warns of."""
        g = Group('messages-gathering')
        users, products, orders = (
            shop()['users'],
            shop()['products'],
            shop()['orders'],
        )
        count = [schema(), call('PredictiveQuery', query=COUNT)]
        for label, options, text in (
            (
                'an-unknown-option',
                {'colour': 'blue'},
                'unknown materialize option',
            ),
            ('run-mode', {'run_mode': 'best'}, 'unknown materialize option'),
            ('no-indices', {'indices': []}, 'indices names no entity'),
            (
                'too-many-entities',
                {'indices': list(range(1, 1002))},
                'entities were asked for',
            ),
            (
                'too-many-context-samples',
                {'num_context_samples': 10_001},
                'accepts at most',
            ),
            ('num-neighbors-a-number', {'num_neighbors': 8}, 'num_neighbors'),
            ('num-neighbors-empty', {'num_neighbors': []}, 'no hop at all'),
            ('num-hops-a-float', {'num_hops': 1.5}, 'num_hops must be'),
            ('num-hops-zero', {'num_hops': 0}, 'no hop at all'),
            ('num-hops-seven', {'num_hops': 7}, 'hops and the endpoint'),
            ('random-seed-a-float', {'random_seed': 1.5}, 'random_seed must'),
            ('random-seed-negative', {'random_seed': -1}, 'not negative'),
            (
                'exclude-cols-not-a-mapping',
                {'exclude_cols_dict': ['users']},
                'must map a table name',
            ),
            (
                'exclude-cols-a-string',
                {'exclude_cols_dict': {'users': 'age'}},
                'must be a sequence of column names',
            ),
            (
                'statistic-unknown',
                {'prediction_statistic': 'mode'},
                'a regression reports',
            ),
            (
                'statistic-of-a-count-classification',
                {'prediction_statistic': 'mean'},
                '',
            ),
            (
                'anchor-before-the-data',
                {'anchor_time': {'$timestamp': '2020-01-01T00:00:00'}},
                'before the earliest',
            ),
        ):
            options = {'indices': [1], **options}
            flow(
                g,
                label,
                [*count, call('PredictiveQuery.materialize', **options)],
                want=Raises(text) if text else None,
            )
        flow(
            g,
            'statistic-of-a-classification',
            [
                schema(),
                call('PredictiveQuery', query=CATEGORY),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    prediction_statistic='mean',
                ),
            ],
            want=Raises('applies only to a regression'),
        )
        rank = [schema(), call('PredictiveQuery', query=RANKING)]
        for label, top_k, text in (
            ('top-k-a-string', '2', 'whole number of candidates'),
            ('top-k-zero', 0, 'ranks between 1 and'),
        ):
            flow(
                g,
                label,
                [
                    *rank,
                    call(
                        'PredictiveQuery.materialize',
                        indices=[1],
                        candidates=[1, 2],
                        anchor_time=ANCHOR,
                        top_k=top_k,
                    ),
                ],
                want=Raises(text),
            )
        daily = 'PREDICT COUNT(orders.*, 0, 1, days) FOR EACH users.user_id'
        flow(
            g,
            'a-default-context-at-its-cap',
            [
                schema(),
                call('PredictiveQuery', query=daily),
                call('PredictiveQuery.materialize', indices=[1], random_seed=7),
            ],
        )
        for label, renamed in (
            ('a-column-the-wire-reserves', {'age': 'instance_id'}),
            (
                'two-columns-the-wire-reserves',
                {'age': 'instance_id', 'country': '__kumo_anchor_time'},
            ),
        ):
            flow(
                g,
                label,
                [*count, call('PredictiveQuery.materialize', indices=[1])],
                data=variant(users=users.rename(columns=renamed)),
                want=Raises('this client needs for the request it builds'),
            )
        flagged = users.assign(vip=pd.Series([0, 1, 0, 1, 0, 1], dtype='int64'))
        flow(
            g,
            'two-whole-numbers-asked-as-a-quantity',
            [
                schema(),
                call(
                    'PredictiveQuery',
                    query='PREDICT users.vip FOR EACH users.user_id',
                ),
                call('PredictiveQuery.materialize', indices=[1, 2]),
                predict,
                to_pandas,
            ],
            data=variant(users=flagged),
        )
        flow(
            g,
            'a-link-to-a-column-not-the-key',
            [
                schema(),
                call(
                    'Schema.link', link=['orders', 'quantity', 'users', 'age']
                ),
                *count[1:],
                call('PredictiveQuery.materialize', indices=[1]),
            ],
            want=Raises('does not join to a primary key'),
        )
        flow(
            g,
            'a-link-whose-target-stops-being-the-key',
            [
                schema(),
                call(
                    'Schema.set_table',
                    table='users',
                    attribute='primary_key',
                    value='age',
                ),
                call(
                    'PredictiveQuery',
                    query='PREDICT users.country FOR EACH users.age',
                ),
                call('PredictiveQuery.materialize', indices=[25]),
            ],
        )
        from decimal import Decimal

        decimal_prices = products.assign(
            price=pd.Series(
                [
                    Decimal('10.5'),
                    Decimal('20'),
                    Decimal('7.25'),
                    Decimal('12'),
                ],
                dtype=object,
            )
        )
        flow(
            g,
            'a-declared-integer-holding-fractions',
            data=variant(products=decimal_prices),
            calls=[
                schema(),
                call(
                    'Schema.set',
                    table='products',
                    column='price',
                    attribute='dtype',
                    value='int64',
                ),
                call('PredictiveQuery', query=SUM),
                call('PredictiveQuery.materialize', indices=[1]),
            ],
            want=Raises('which is not a whole number'),
        )
        undated = orders.copy()
        undated.loc[[3, 7], 'ordered_at'] = pd.NaT
        flow(
            g,
            'rows-the-graph-drops',
            [*count, call('PredictiveQuery.materialize', indices=[1])],
            data=variant(orders=undated),
        )
        notes = pd.DataFrame(
            {'note_id': pd.Series([1, 2], dtype='int64'), 'text': ['a', 'b']}
        )
        flow(
            g,
            'a-table-nothing-links',
            [*count, call('PredictiveQuery.materialize', indices=[1])],
            data=variant(notes=notes),
        )
        flow(
            g,
            'a-table-the-hops-do-not-reach',
            [
                *count,
                call('PredictiveQuery.materialize', indices=[1], num_hops=1),
            ],
        )
        strangers = users.assign(user_id=users['user_id'] + 100)
        flow(
            g,
            'a-relationship-that-joins-nothing',
            [
                schema(
                    relationships=[
                        ['orders', 'user_id', 'users', 'user_id'],
                        ['orders', 'product_id', 'products', 'product_id'],
                    ]
                ),
                call('PredictiveQuery', query=SUM),
                call('PredictiveQuery.materialize', indices=[1]),
            ],
            data=variant(users=strangers),
        )
        flow(
            g,
            'a-text-column',
            [
                schema(),
                call(
                    'Schema.set',
                    table='users',
                    column='country',
                    attribute='stype',
                    value='text',
                ),
                *count[1:],
                call('PredictiveQuery.materialize', indices=[1]),
            ],
        )
        flow(
            g,
            'an-entity-filter',
            [
                schema(),
                call(
                    'PredictiveQuery',
                    query='PREDICT COUNT(orders.*, 0, 30, days) FOR EACH '
                    'users.user_id WHERE users.age > 30',
                ),
                call('PredictiveQuery.materialize', indices=[1, 4]),
            ],
        )
        children = {
            f'visits_{i}': pd.DataFrame(
                {
                    f'visit_{i}_id': pd.Series([1, 2, 3], dtype='int64'),
                    'user_id': pd.Series([1, 2, 3], dtype='int64'),
                    'visited_at': pd.Series(
                        pd.to_datetime(
                            ['2024-02-01', '2024-03-01', '2024-04-01']
                        ),
                        dtype='datetime64[us]',
                    ),
                }
            )
            for i in range(14)
        }
        flow(
            g,
            'more-tables-than-the-endpoint-reads',
            [*count, call('PredictiveQuery.materialize', indices=[1])],
            data=variant(**children),
        )
        # Sources.
        flow(
            g,
            'a-source-with-no-tables',
            [call('Schema.from_duckdb', load=[])],
            want=Raises('holds no table to describe'),
        )
        flow(
            g,
            'a-source-without-the-table',
            [call('Schema.from_duckdb', tables={'users': 'customers'})],
        )
        # The tabular route.
        churn = 'PREDICT people.churn FOR EACH people.user_id'
        one = [schema(tables=['people']), call('PredictiveQuery', query=churn)]
        tabular(
            g,
            'tabular-two-whole-numbers',
            [*one, call('PredictiveQuery.materialize', predict=scored)],
        )
        tabular(
            g,
            'tabular-indices-not-in-the-table',
            [*one, call('PredictiveQuery.materialize', indices=[1, 99, 98])],
        )
        tabular(
            g,
            'tabular-too-many-context-samples',
            [
                *one,
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    num_context_samples=10_001,
                ),
            ],
        )
        dated = people().assign(
            joined=pd.Series(
                pd.date_range('2024-01-01', periods=12), dtype='datetime64[us]'
            )
        )
        tabular(
            g,
            'tabular-a-timestamp-target',
            [
                schema(tables=['people']),
                call(
                    'PredictiveQuery',
                    query='PREDICT people.joined FOR EACH people.user_id',
                ),
                call('PredictiveQuery.materialize', predict=scored),
            ],
            data=tables({'people': dated}),
        )
        return g

    def messages_answers() -> Any:
        r"""Sending a request and reading the answer: the adapters' checks
        before sending, the endpoint's refusals, and answers that do not
        match what was asked."""
        g = Group('messages-answers')
        count = gathered(COUNT, indices=[1, 2], num_context_samples=12)
        products = shop()['products']
        for label, outputs, options, text in (
            (
                'explaining-two-entities',
                ['prediction', 'explanation'],
                {'indices': [1, 2]},
                'an explanation covers exactly',
            ),
            (
                'explaining-with-a-large-context',
                ['prediction', 'explanation'],
                {'indices': [1], 'num_context_samples': 1001},
                'smaller context budget',
            ),
        ):
            flow(
                g,
                label,
                [
                    *gathered(COUNT, **options),
                    call('SDMClient.predict', model=REL, outputs=outputs),
                ],
                want=Raises(text),
            )
        flow(
            g,
            'an-inference-option-the-model-does-not-read',
            [
                *gathered(
                    SUM, indices=[1], inference_config={'output_type': 'mean'}
                ),
                predict,
            ],
        )
        eleven = pd.DataFrame(
            {
                'product_id': pd.Series(range(1, 13), dtype='int64'),
                'category': [f'c{i}' for i in range(12)],
                'price': [float(i) for i in range(12)],
            }
        )
        flow(
            g,
            'more-classes-than-the-model-classifies',
            [*gathered(CATEGORY, indices=[1]), predict],
            data=variant(products=eleven),
            want=Raises('classifies at most'),
        )
        flow(
            g,
            'scoring-every-row-of-a-static-question',
            [*gathered(CATEGORY, indices=[1, 2, 3, 4]), predict],
            want=Raises('no context rows are left'),
        )
        same = products.assign(category='books')
        flow(
            g,
            'one-class-in-the-context',
            [*gathered(CATEGORY, indices=[1]), predict],
            data=variant(products=same),
            want=Raises('a classification needs at least two classes'),
        )
        priced = products.assign(price=9.5)
        flow(
            g,
            'one-value-in-the-context',
            [
                *gathered(
                    'PREDICT products.price FOR EACH products.product_id',
                    indices=[1],
                ),
                predict,
                to_pandas,
            ],
            data=variant(products=priced),
        )
        # The endpoint refuses.
        for label, status, body in (
            (
                'the-endpoint-refuses-a-text-column',
                422,
                {
                    'title': 'Unprocessable Entity',
                    'status': 422,
                    'code': 'UNSUPPORTED_FEATURE',
                    'detail': "semantic type 'text' is unsupported",
                },
            ),
            (
                'the-endpoint-refuses-a-cardinality',
                422,
                {
                    'title': 'Unprocessable Entity',
                    'status': 422,
                    'code': 'UNSUPPORTED_FEATURE',
                    'detail': 'categorical cardinality 13812 exceeds limit 10000',
                },
            ),
            (
                'the-endpoint-fails',
                500,
                {'title': 'Internal Server Error', 'status': 500},
            ),
            (
                'the-endpoint-answers-what-is-not-a-response',
                200,
                {'predictions': 'none'},
            ),
        ):
            flow(
                g,
                label,
                [*count, predict],
                stand_in={'fail': [{'at': 0, 'status': status, 'body': body}]},
            )
        texted = [
            schema(),
            call(
                'Schema.set',
                table='users',
                column='country',
                attribute='stype',
                value='text',
            ),
            call('PredictiveQuery', query=COUNT),
            call('PredictiveQuery.materialize', indices=[1, 2]),
            predict,
        ]
        flow(
            g,
            'the-endpoint-refuses-a-text-column-it-names',
            texted,
            stand_in={
                'fail': [
                    {
                        'at': 0,
                        'status': 422,
                        'body': {
                            'title': 'Unprocessable Entity',
                            'status': 422,
                            'code': 'UNSUPPORTED_FEATURE',
                            'detail': "semantic type 'text' is unsupported",
                        },
                    }
                ]
            },
        )
        # Answers that do not match what was asked.
        for label, items in (
            (
                'an-answer-for-a-row-not-asked',
                [{'from': 0, 'set': {'row_index': 9}}],
            ),
            (
                'an-answer-with-a-step',
                [{'from': 0, 'set': {'forecast_step': 1}}, {'from': 1}],
            ),
            ('an-answer-twice', [{'from': 0}, {'from': 0}, {'from': 1}]),
            (
                'an-answer-without-an-id',
                [{'from': 0, 'drop': ['id']}, {'from': 1}],
            ),
            (
                'an-answer-under-another-id',
                [{'from': 0, 'set': {'id': 'nobody'}}, {'from': 1}],
            ),
            ('a-row-unanswered', [{'from': 0}]),
        ):
            flow(
                g,
                label,
                [*count, predict],
                stand_in={'items_at': [{'at': 0, 'items': items}]},
            )
        # The tabular adapter.
        for label, items in (
            (
                'tabular-an-answer-for-a-row-not-asked',
                [{'from': 0, 'set': {'row_index': 7}}],
            ),
            ('tabular-an-answer-twice', [{'from': 0}, {'from': 0}]),
            (
                'tabular-an-answer-under-another-key',
                [{'from': 0, 'set': {'id': '[99]'}}, {'from': 1}, {'from': 2}],
            ),
            ('tabular-a-row-unanswered', [{'from': 0}, {'from': 1}]),
        ):
            tabular(
                g,
                label,
                [
                    schema(tables=['people']),
                    direct(schema={'$schema': None}, entity_table='people'),
                    tab_predict,
                ],
                stand_in={'items_at': [{'at': 0, 'items': items}]},
            )
        tabular(
            g,
            'tabular-the-endpoint-answers-what-is-not-a-response',
            [direct(), tab_predict],
            stand_in={'fail': [{'at': 0, 'status': 200, 'body': {'id': 3}}]},
        )
        tabular(
            g,
            'tabular-a-text-column',
            [
                schema(tables=['people']),
                call(
                    'Schema.set',
                    table='people',
                    column='segment',
                    attribute='stype',
                    value='text',
                ),
                direct(schema={'$schema': None}, entity_table='people'),
                tab_predict,
                to_pandas,
            ],
        )
        tabular(
            g,
            'tabular-a-column-the-schema-does-not-describe',
            [
                schema(tables=['people']),
                call(
                    'Request',
                    context={'$frame': {'table': 'wider'}},
                    predict={
                        '$frame': {
                            'table': 'wider',
                            'rows': [0, 2],
                            'drop': ['churn'],
                        }
                    },
                    target='churn',
                    task='classification',
                    schema={'$schema': None},
                    entity_table='people',
                ),
                tab_predict,
            ],
            data=tables(
                {'people': people(), 'wider': people().assign(extra=1.0)}
            ),
        )
        tabular(
            g,
            'tabular-a-regression-to-the-relational-session',
            [direct(), call('SDMClient.fit', model=REL)],
        )
        return g

    def messages_wire() -> Any:
        r"""Values the wire has no form for, in the rows sent and in the
        options beside them."""
        g = Group('messages-wire')
        from decimal import Decimal

        for label, column, values in (
            ('a-bytes-column', 'photo', [b'\x89PNG'] * 12),
            ('a-list-column', 'tags', [['a', 'b']] * 12),
            ('a-mapping-column', 'tags', [{'a': 1}] * 12),
        ):
            frame_ = people()
            frame_[column] = pd.Series(values, dtype=object)
            tabular(
                g,
                label,
                [direct(), tab_predict],
                data=tables({'people': frame_}),
            )
        nested: Any = 1
        for _ in range(102):
            nested = {'deeper': nested}
        for label, inference in (
            ('inference-referring-to-itself', {'options': {'$cyclic': 'self'}}),
            ('inference-nested-too-deep', nested),
        ):
            tabular(g, label, [direct(inference=inference), tab_predict])
        infinite = people().assign(
            age=[float('inf')] + [20.0 + i for i in range(1, 12)]
        )
        tabular(
            g,
            'an-infinite-feature',
            [direct(), tab_predict],
            data=tables({'people': infinite}),
        )
        for label, value, dtype in (
            ('a-fraction-as-an-integer', 1.5, 'int64'),
            ('an-infinite-integer', float('inf'), 'int64'),
            ('a-decimal-past-float64', Decimal('1e400'), 'float64'),
            ('an-integer-past-float64', 2**1024, 'float64'),
        ):
            g.add(
                label,
                'wire.serialize_cell',
                {'value': encode_value(value), 'dtype': dtype},
            )
        # The tabular model asked what it does not serve.
        crowd = pd.DataFrame(
            {
                'user_id': pd.Series(range(10_005), dtype='int64'),
                'churn': pd.Series(
                    [i % 2 for i in range(10_005)], dtype='int64'
                ),
            }
        )
        tabular(
            g,
            'tabular-a-context-past-the-cap',
            [
                call(
                    'Request',
                    context={'$frame': {'table': 'crowd'}},
                    predict={
                        '$frame': {
                            'table': 'crowd',
                            'rows': [0, 2],
                            'drop': ['churn'],
                        }
                    },
                    target='churn',
                    task='classification',
                ),
                tab_predict,
            ],
            data=tables({'crowd': crowd}),
        )
        tabular(
            g,
            'tabular-more-rows-than-a-context-holds',
            [
                schema(tables=['crowd']),
                call(
                    'PredictiveQuery',
                    query='PREDICT crowd.churn FOR EACH crowd.user_id',
                ),
                call('PredictiveQuery.materialize', indices=[0, 1]),
            ],
            data=tables({'crowd': crowd}),
        )
        age = 'PREDICT people.age FOR EACH people.user_id'
        tabular(
            g,
            'tabular-model-sent-a-gathered-request',
            [
                schema(tables=['people']),
                call('PredictiveQuery', query=age),
                call('PredictiveQuery.materialize', indices=[1], model=REL),
                tab_predict,
            ],
        )
        tabular(
            g,
            'tabular-model-sent-a-gathered-classification',
            [
                schema(tables=['people']),
                call(
                    'PredictiveQuery',
                    query='PREDICT people.segment FOR EACH people.user_id',
                ),
                call('PredictiveQuery.materialize', indices=[1], model=REL),
                tab_predict,
            ],
        )
        tabular(
            g,
            'tabular-a-text-target',
            [
                schema(tables=['people']),
                call(
                    'Schema.set',
                    table='people',
                    column='segment',
                    attribute='stype',
                    value='text',
                ),
                call(
                    'PredictiveQuery',
                    query='PREDICT people.segment FOR EACH people.user_id',
                ),
                call('PredictiveQuery.materialize', indices=[1]),
            ],
        )
        return g

    def messages_engine() -> Any:
        r"""The engine's refusals and warnings that reach users: through
        gathering, where the client re-raises each refusal as its own, and
        through the relational wire's value encoding, run directly."""
        g = Group('messages-engine')
        orders_query = 'PREDICT orders.quantity FOR EACH orders.order_id'
        for label, query, options, text in (
            (
                'anchor-a-string',
                COUNT,
                {'anchor_time': '2024-09-01'},
                'must be a pandas.Timestamp',
            ),
            (
                'max-pq-iterations-zero',
                COUNT,
                {'max_pq_iterations': 0},
                'greater than zero',
            ),
            (
                'context-anchor-entity-beside-a-timestamp',
                COUNT,
                {'anchor_time': ANCHOR, 'context_anchor_time': 'entity'},
                'needs to be shared',
            ),
            (
                'anchor-entity-for-a-window',
                COUNT,
                {'anchor_time': 'entity'},
                'only valid for static',
            ),
            (
                'anchor-entity-without-a-time-column',
                CATEGORY,
                {'anchor_time': 'entity'},
                'to have a time column',
            ),
            (
                'anchor-entity-beside-a-context-timestamp',
                orders_query,
                {'anchor_time': 'entity', 'context_anchor_time': ANCHOR},
                'needs to be shared',
            ),
            (
                'indices-not-of-the-key',
                COUNT,
                {'indices': ['first']},
                'must hold values matching',
            ),
            (
                'lag-timesteps-negative',
                COUNT,
                {'lag_timesteps': -1},
                'cannot be negative',
            ),
            (
                'context-anchor-before-the-data',
                COUNT,
                {
                    'anchor_time': ANCHOR,
                    'context_anchor_time': {
                        '$timestamp': '2020-01-01T00:00:00'
                    },
                },
                'Context anchor timestamp is too early',
            ),
            (
                'context-anchor-after-the-anchor',
                COUNT,
                {
                    'anchor_time': ANCHOR,
                    'context_anchor_time': {
                        '$timestamp': '2024-10-01T00:00:00'
                    },
                },
                '',
            ),
            (
                'context-anchor-whose-window-overlaps',
                COUNT,
                {
                    'anchor_time': ANCHOR,
                    'context_anchor_time': {
                        '$timestamp': '2024-08-20T00:00:00'
                    },
                },
                '',
            ),
            (
                'context-anchor-whose-window-starts-before-the-data',
                FORECAST,
                {
                    'anchor_time': ANCHOR,
                    'context_anchor_time': {
                        '$timestamp': '2024-02-01T00:00:00'
                    },
                },
                '',
            ),
            (
                'anchor-after-the-data',
                COUNT,
                {'anchor_time': {'$timestamp': '2026-01-01T00:00:00'}},
                '',
            ),
            (
                'num-neighbors-beside-num-hops',
                COUNT,
                {'num_neighbors': [4, 4], 'num_hops': 1},
                '',
            ),
            ('lag-timesteps', COUNT, {'lag_timesteps': 2}, ''),
            (
                'lag-timesteps-of-a-static-question',
                CATEGORY,
                {'lag_timesteps': 2},
                '',
            ),
            (
                'link-prediction-one-hop',
                LINK,
                {'anchor_time': ANCHOR, 'num_neighbors': [16]},
                'without any historical target entities',
            ),
            (
                'an-entity-filter-no-entity-passes',
                'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id '
                'WHERE users.age > 100',
                {},
                '',
            ),
        ):
            options = {'indices': [1], 'random_seed': 7, **options}
            flow(
                g,
                label,
                [
                    schema(),
                    call('PredictiveQuery', query=query),
                    call('PredictiveQuery.materialize', **options),
                ],
                want=Raises(text) if text else None,
            )
        # A foreign key linked to two tables: the engine cannot tell which
        # one a LIST_DISTINCT ranks and raises NotImplementedError, which
        # the client reports as INTERNAL_ERROR.
        catalog = pd.DataFrame(
            {
                'sku': pd.Series([1, 2, 3, 4], dtype='int64'),
                'supplier': ['acme', 'globex', 'acme', 'globex'],
            }
        )
        flow(
            g,
            'link-prediction-over-a-key-linked-to-two-tables',
            [
                schema(primary_keys={'catalog': 'sku'}),
                call(
                    'Schema.link',
                    link=['orders', 'product_id', 'catalog', 'sku'],
                ),
                call('PredictiveQuery', query=LINK),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    anchor_time=ANCHOR,
                    random_seed=7,
                ),
            ],
            data=tables({**shop(), 'catalog': catalog}),
            # Phase 9's fix: refused by name, where today's engine ended in
            # NotImplementedError as INTERNAL_ERROR.
            want=Raises('cannot say which table'),
        )
        from decimal import Decimal

        for label, task_type, value in (
            ('link-target-null-item', 'temporal_link_prediction', ['a', None]),
            (
                'link-target-non-finite-item',
                'temporal_link_prediction',
                ['a', Decimal('Infinity')],
            ),
            ('binary-target-a-word', 'binary_classification', 'yes'),
        ):
            g.add(
                label,
                'payload.target_json_value',
                {'task_type': task_type, 'value': encode_value(value)},
            )
        g.add(
            'stringlist-a-non-finite-item',
            'payload.dtype_name',
            {
                'series': {
                    'dtype': 'object',
                    'values': [encode_value(['a', Decimal('Infinity')])],
                }
            },
        )
        g.add(
            'decimal-past-the-safe-integers',
            'payload.json_value',
            {'value': encode_value(Decimal('9007199254740993.5'))},
        )
        for label, value, dtype in (
            ('stringlist-a-scalar', 5, 'stringlist'),
            ('float-an-integer-past-float64', 2**1024, 'float64'),
            ('float-a-decimal-past-float64', Decimal('1e400'), 'float64'),
            ('a-mapping-under-a-dtype-the-wire-lacks', {'a': 1}, 'date32'),
        ):
            g.add(
                label,
                'payload.cell_json_value',
                {'value': encode_value(value), 'dtype': dtype},
            )
        return g

    def request_paths() -> Any:
        r"""The paths through building, sending and reading a request that no
        message fixture or transcript takes: sessions on the tabular model,
        empty batches, reading a gathered request's halves, and the values
        the wire is asked to type and read back."""
        g = Group('request-paths', 'scripts/parity/_fixture_messages.py')
        keyed = [schema(tables=['people'])]

        def keyed_request(**args: Any) -> dict[str, Any]:
            return direct(
                schema={'$schema': None}, entity_table='people', **args
            )

        fit_tab = call('SDMClient.fit', model=TAB)
        tabular(
            g,
            'tabular-session',
            [
                *keyed,
                keyed_request(predict=rows('people', 0, 2, drop=['churn'])),
                fit_tab,
                keyed_request(predict=rows('people', 2, 5, drop=['churn'])),
                call('FittedModel.predict'),
                to_pandas,
                call('FittedModel.close'),
                call('FittedModel.close'),
            ],
        )
        tabular(
            g,
            'tabular-session-scoring-nothing',
            [
                *keyed,
                keyed_request(),
                fit_tab,
                keyed_request(predict=rows('people', 0, 0, drop=['churn'])),
                call('FittedModel.predict'),
                to_pandas,
            ],
        )
        worded = people().astype({'age': object})
        worded['age'] = ['old'] * 12
        tabular(
            g,
            'tabular-session-scoring-rows-its-schema-cannot-hold',
            [
                *keyed,
                keyed_request(),
                fit_tab,
                call(
                    'Request',
                    context=context,
                    predict={'$frame': {'table': 'worded', 'rows': [0, 2]}},
                    target='churn',
                    task='classification',
                ),
                call('FittedModel.predict'),
            ],
            data=tables(
                {'people': people(), 'worded': worded.drop(columns=['churn'])}
            ),
        )
        tabular(
            g,
            'tabular-session-asked-over-another-context',
            [
                direct(),
                fit_tab,
                direct(context=rows('people', 0, 8)),
                call('FittedModel.predict'),
            ],
        )
        for label, calls in (
            (
                'an-empty-batch',
                [direct(predict=rows('people', 0, 0, drop=['churn']))],
            ),
            (
                'an-empty-batch-of-keyed-rows',
                [
                    *keyed,
                    keyed_request(predict=rows('people', 0, 0, drop=['churn'])),
                ],
            ),
        ):
            tabular(
                g,
                label,
                [
                    *calls,
                    call(
                        'SDMClient.predict', model=TAB, outputs=['prediction']
                    ),
                    to_pandas,
                ],
            )
        flow(
            g,
            'releasing-a-session-fails',
            [
                *gathered(COUNT, indices=[1], num_context_samples=6),
                call('SDMClient.fit', model=REL),
            ],
            stand_in={
                'fail': [
                    {
                        'at': 1,
                        'status': 500,
                        'body': {
                            'title': 'Internal Server Error',
                            'status': 500,
                        },
                    }
                ]
            },
        )
        for label, status, detail in (
            ('fitting-where-the-endpoint-fails', 500, 'Internal Server Error'),
            (
                'fitting-a-text-column-the-endpoint-refuses',
                422,
                "semantic type 'text' is unsupported",
            ),
        ):
            flow(
                g,
                label,
                [
                    schema(),
                    call(
                        'Schema.set',
                        table='users',
                        column='country',
                        attribute='stype',
                        value='text',
                    ),
                    call('PredictiveQuery', query=COUNT),
                    call(
                        'PredictiveQuery.materialize',
                        indices=[1],
                        num_context_samples=6,
                    ),
                    call('SDMClient.fit', model=REL),
                ],
                stand_in={
                    'fail': [
                        {
                            'at': 0,
                            'status': status,
                            'body': {'status': status, 'detail': detail},
                        }
                    ]
                },
            )
        flow(
            g,
            'the-endpoint-refuses-columns-with-too-many-values',
            [*gathered(COUNT, indices=[1], num_context_samples=6), predict],
            stand_in={
                'fail': [
                    {
                        'at': 0,
                        'status': 422,
                        'body': {
                            'status': 422,
                            'code': 'UNSUPPORTED_FEATURE',
                            'detail': 'categorical cardinality 3 exceeds limit 2',
                        },
                    }
                ]
            },
        )
        flow(
            g,
            'the-endpoint-names-the-invalid-parameters',
            [*gathered(COUNT, indices=[1], num_context_samples=6), predict],
            stand_in={
                'fail': [
                    {
                        'at': 0,
                        'status': 422,
                        'body': {
                            'status': 422,
                            'title': 'Request validation failed.',
                            'invalid_params': [
                                {'name': 'task.kind', 'reason': 'unknown'},
                                {'name': 'output'},
                                {'reason': 'too large'},
                                'context',
                                {},
                            ],
                        },
                    }
                ]
            },
        )
        rank = gathered(
            RANKING,
            indices=[1],
            candidates=[1.0, 2.0],
            anchor_time=ANCHOR,
            num_context_samples=6,
        )
        flow(g, 'ranking-whole-float-candidates', [*rank, predict, to_pandas])
        flow(
            g,
            'ranking-asked-of-the-tabular-model',
            [*rank, call('SDMClient.predict', model=TAB)],
        )
        flow(
            g,
            'forecast-asked-of-the-tabular-model',
            [
                *gathered(
                    FORECAST,
                    indices=[1],
                    anchor_time=ANCHOR,
                    num_context_samples=6,
                ),
                call('SDMClient.predict', model=TAB),
            ],
        )
        flow(
            g,
            'a-model-handle',
            [
                *gathered(COUNT, indices=[1], num_context_samples=6),
                call('SDMClient.predict', model={'$model': 'KumoRelational'}),
                to_pandas,
            ],
        )
        referred = shop()['orders'].assign(
            referrer_id=pd.Series(
                [(i + 1) % 6 + 1 for i in range(40)], dtype='int64'
            )
        )
        flow(
            g,
            'a-question-over-two-links',
            [
                schema(
                    relationships=[
                        ['orders', 'user_id', 'users', 'user_id'],
                        ['orders', 'referrer_id', 'users', 'user_id'],
                        ['orders', 'product_id', 'products', 'product_id'],
                    ]
                ),
                call('PredictiveQuery', query=COUNT),
                call('PredictiveQuery.materialize', indices=[1]),
            ],
            data=variant(orders=referred),
        )
        flow(
            g,
            'a-text-target',
            [
                schema(),
                call(
                    'Schema.set',
                    table='users',
                    column='country',
                    attribute='stype',
                    value='text',
                ),
                call(
                    'PredictiveQuery',
                    query='PREDICT users.country FOR EACH users.user_id',
                ),
                call('PredictiveQuery.materialize', indices=[1]),
            ],
        )
        for label, query, options in (
            ('the-halves-of-a-gathered-count', COUNT, {'indices': [1, 2]}),
            (
                'the-halves-of-a-gathered-classification',
                CATEGORY,
                {'indices': [1]},
            ),
            (
                'the-halves-of-a-gathered-forecast',
                FORECAST,
                {'indices': [1], 'anchor_time': ANCHOR},
            ),
        ):
            flow(
                g,
                label,
                [
                    *gathered(
                        query,
                        num_context_samples=4,
                        num_neighbors=[2],
                        **options,
                    ),
                    call('Request.halves'),
                ],
            )
        tabular(
            g,
            'the-halves-of-a-direct-request',
            [direct(), call('Request.halves')],
        )
        flow(
            g,
            'holdout-keys-of-every-kind',
            [
                *gathered(COUNT, indices=[1], num_context_samples=6),
                call(
                    'SDMClient.fit',
                    model=REL,
                    holdout=[None, {'$float': 'inf'}, 1.5, 2.0, [3]],
                ),
            ],
        )
        whole_ages = people().assign(
            age=pd.Series([20 + i for i in range(12)], dtype='int64')
        )
        fractional = pd.DataFrame(
            {
                'user_id': pd.Series([20, 21], dtype='int64'),
                'age': [1.5, 2.5],
                'segment': ['north', 'south'],
            }
        )
        tabular(
            g,
            'tabular-session-scoring-a-value-its-schema-cannot-hold',
            [
                *keyed,
                keyed_request(),
                fit_tab,
                keyed_request(predict={'$frame': {'table': 'fractional'}}),
                call('FittedModel.predict'),
            ],
            data=tables({'people': whole_ages, 'fractional': fractional}),
        )
        flow(
            g,
            'a-question-over-tables-in-their-source',
            [
                call(
                    'Schema.from_duckdb',
                    tables={'users': 'users', 'orders': 'orders'},
                ),
                call('PredictiveQuery', query=COUNT),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1, 2],
                    num_context_samples=6,
                    random_seed=7,
                ),
                predict,
                to_pandas,
            ],
        )
        churn = 'PREDICT people.churn FOR EACH people.user_id'
        for label, args in (
            (
                'tabular-a-chosen-context-size',
                {'indices': [1], 'num_context_samples': 5},
            ),
            (
                'tabular-rows-to-score-without-their-key',
                {'predict': rows('people', 0, 2, drop=['churn', 'user_id'])},
            ),
            (
                'tabular-rows-to-score-with-their-key',
                {'predict': rows('people', 0, 2, drop=['churn'])},
            ),
        ):
            tabular(
                g,
                label,
                [
                    schema(tables=['people']),
                    call('PredictiveQuery', query=churn),
                    call('PredictiveQuery.materialize', **args),
                    tab_predict,
                    to_pandas,
                ],
            )
        tabular(
            g,
            'assigning-a-half-of-a-direct-request',
            [
                direct(),
                call(
                    'Request.assign',
                    half='predict',
                    value=rows('people', 3, 5, drop=['churn']),
                ),
                tab_predict,
                to_pandas,
            ],
        )
        tabular(
            g,
            'a-composite-key',
            [
                schema(
                    tables=['people'],
                    primary_keys={'people': ['user_id', 'segment']},
                ),
                keyed_request(),
                tab_predict,
                to_pandas,
            ],
        )
        tabular(
            g,
            'a-schema-without-the-entity-table',
            [
                schema(tables=['people']),
                direct(schema={'$schema': None}),
                tab_predict,
                to_pandas,
            ],
        )
        flow(
            g,
            'scores-and-embeddings',
            [
                *gathered(COUNT, indices=[1], num_context_samples=6),
                call(
                    'SDMClient.predict',
                    model=REL,
                    outputs=['prediction', 'scores', 'embeddings'],
                ),
                to_pandas,
            ],
        )
        unkeyed = tables({'people': people().drop(columns=['user_id'])})
        tabular(
            g,
            'an-empty-batch-of-rows-without-a-key',
            [
                direct(predict=rows('people', 0, 0, drop=['churn'])),
                call('SDMClient.predict', model=TAB, outputs=['prediction']),
                to_pandas,
            ],
            data=unkeyed,
        )
        tabular(
            g,
            'rows-to-score-without-a-feature',
            [
                call(
                    'Request',
                    context=whole('people', drop=['user_id', 'age', 'segment']),
                    predict=rows(
                        'people',
                        0,
                        2,
                        drop=['user_id', 'age', 'segment', 'churn'],
                    ),
                    target='churn',
                    task='classification',
                ),
                tab_predict,
                to_pandas,
            ],
        )
        flagged = people().assign(churn=[bool(i % 2) for i in range(12)])
        for label, data, outputs in (
            (
                'probabilities-of-whole-number-classes',
                PEOPLE,
                ['probabilities'],
            ),
            (
                'probabilities-of-boolean-classes',
                tables({'people': flagged}),
                ['prediction', 'probabilities'],
            ),
        ):
            tabular(
                g,
                label,
                [
                    direct(),
                    call('SDMClient.predict', model=TAB, outputs=outputs),
                    to_pandas,
                ],
                data=data,
            )
        for label, context_values, predict_values, dtypes in (
            (
                'widening-whole-numbers-to-float32',
                [1, 2, 3, 4],
                [1.5, 2.5],
                ('int64', 'float32'),
            ),
            (
                'widening-text-and-numbers',
                ['a', 'b', 'c', 'd'],
                [1, 2],
                ('object', 'int64'),
            ),
            (
                'widening-nothing-but-nulls',
                [None] * 4,
                [None, None],
                ('object', 'object'),
            ),
            (
                'arrow-backed-features',
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0],
                ('double[pyarrow]', 'double[pyarrow]'),
            ),
        ):
            labelled = pd.DataFrame(
                {
                    'score': pd.Series(context_values, dtype=dtypes[0]),
                    'churn': pd.Series([0, 1, 0, 1], dtype='int64'),
                }
            )
            asked = pd.DataFrame(
                {'score': pd.Series(predict_values, dtype=dtypes[1])}
            )
            tabular(
                g,
                label,
                [
                    call(
                        'Request',
                        context={'$frame': {'table': 'labelled'}},
                        predict={'$frame': {'table': 'asked'}},
                        target='churn',
                        task='classification',
                    ),
                    tab_predict,
                    to_pandas,
                ],
                data=tables({'labelled': labelled, 'asked': asked}),
            )
        # The values the wire types and reads back.
        from decimal import Decimal

        for label, values, dtype in (
            ('booleans', [True, False], 'bool'),
            ('nothing-but-nulls', [None, None], 'object'),
            ('whole-numbers-as-objects', [1, 2, None], 'object'),
            ('decimals-as-objects', [Decimal('1.5'), Decimal('2')], 'object'),
            (
                'whole-decimals-as-objects',
                [Decimal('1'), Decimal('20')],
                'object',
            ),
            ('text-in-arrow', ['a', 'b'], 'string[pyarrow]'),
            ('integers-in-arrow', [1, None], 'int64[pyarrow]'),
            ('text-as-objects', ['a', None], 'object'),
        ):
            g.add(
                label,
                'wire.infer_dtype',
                {
                    'series': {
                        'dtype': dtype,
                        'values': [encode_value(v) for v in values],
                    }
                },
            )
        for label, value, dtype in (
            ('a-null-under-int64', None, 'int64'),
            ('a-boolean-under-int64', True, 'int64'),
            ('a-whole-number-under-float64', 2, 'float64'),
            ('text-under-float64', '1.5', 'float64'),
            ('a-null-dtype', 3, None),
            ('true-under-bool', 'true', 'bool'),
            ('false-under-bool', 'false', 'bool'),
            ('one-under-bool', 1, 'bool'),
            ('two-under-bool', 2, 'bool'),
            ('a-word-under-bool', 'maybe', 'bool'),
            ('a-boolean-under-bool', True, 'bool'),
            ('text-under-string', 'x', 'string'),
        ):
            g.add(
                label,
                'wire.as_declared',
                {'value': encode_value(value), 'dtype': dtype},
            )
        for label, value, dtype in (
            (
                'a-zoned-timestamp',
                pd.Timestamp('2024-01-01T08:00:00+01:00'),
                'timestamp[us]',
            ),
            ('a-boolean', True, 'bool'),
            ('a-list-under-string', [1, 2], 'string'),
        ):
            g.add(
                label,
                'wire.serialize_cell',
                {'value': encode_value(value), 'dtype': dtype},
            )
        return g

    def request_shapes() -> Any:
        r"""The data shapes, options and endpoint answers no transcript
        holds: values the wire has to keep exact, refusals explained from the
        schema, options given in their other forms, and the value readers of
        the relational payload asked directly. Where what is sent is the
        point, a case records the bodies posted (``show_bodies``)."""
        import datetime as dt
        from decimal import Decimal

        import numpy as np
        import pyarrow as pa

        g = Group('request-shapes', 'scripts/parity/_fixture_messages.py')

        def sent(
            label: str,
            calls: list[dict[str, Any]],
            data: dict[str, Any] | None = None,
            stand_in: dict[str, Any] | None = None,
        ) -> None:
            r"""A ``client.run`` case that records the bodies it posts."""
            g.flow(
                label,
                {
                    'tables': SHOP if data is None else data,
                    'calls': calls,
                    'show_bodies': True,
                },
                stand_in=stand_in,
            )

        def asked(query: str, **options: Any) -> list[dict[str, Any]]:
            options.setdefault('num_context_samples', 6)
            return gathered(query, **options)

        def refusing(status: int, body: dict[str, Any]) -> dict[str, Any]:
            return {'fail': [{'at': 0, 'status': status, 'body': body}]}

        # A string label of more than 64 values is inferred `text`, and the
        # refusal explains that, differently for each model.
        crowd = shop()['users'].iloc[[0] * 70].reset_index(drop=True)
        crowd = crowd.assign(
            user_id=pd.Series(range(1, 71), dtype='int64'),
            label=[f'label {i:02d}' for i in range(70)],
            **{'my label': [f'label {i:02d}' for i in range(70)]},
        )
        for label, query in (
            (
                'a-many-valued-label-asked-of-the-relational-model',
                'PREDICT users.label FOR EACH users.user_id',
            ),
            (
                'a-many-valued-label-named-in-backticks',
                'PREDICT users.`my label` FOR EACH users.user_id',
            ),
            (
                'a-many-valued-label-in-backticks-and-a-literal-naming-a-target',
                'PREDICT users.`my label` FOR EACH users.user_id '
                "WHERE users.country = 'PREDICT nosuch.col '",
            ),
        ):
            flow(
                g,
                label,
                [
                    schema(),
                    call('PredictiveQuery', query=query),
                    call('PredictiveQuery.materialize', indices=[1]),
                ],
                data=variant(users=crowd),
            )
        many = people(70).assign(label=[f'label {i:02d}' for i in range(70)])
        tabular(
            g,
            'a-many-valued-label-asked-of-the-tabular-model',
            [
                schema(tables=['people']),
                call(
                    'PredictiveQuery',
                    query='PREDICT people.label FOR EACH people.user_id',
                ),
                call('PredictiveQuery.materialize', indices=[1]),
            ],
            data=tables({'people': many}),
        )
        # A refusal that only quotes the words the link advice looks for.
        doubled = shop()['orders'].assign(
            referrer_id=pd.Series(
                [(i + 1) % 6 + 1 for i in range(40)], dtype='int64'
            ),
            alt_product_id=pd.Series(
                [(i + 1) % 4 + 1 for i in range(40)], dtype='int64'
            ),
        )
        for label, links, data in (
            ('a-literal-saying-a-link-is-not-unique', None, None),
            (
                'a-literal-saying-a-link-is-not-unique-over-two-doubled-links',
                [
                    ['orders', 'user_id', 'users', 'user_id'],
                    ['orders', 'referrer_id', 'users', 'user_id'],
                    ['orders', 'product_id', 'products', 'product_id'],
                    ['orders', 'alt_product_id', 'products', 'product_id'],
                ],
                variant(orders=doubled),
            ),
        ):
            flow(
                g,
                label,
                [
                    schema(
                        **({} if links is None else {'relationships': links})
                    ),
                    call(
                        'PredictiveQuery',
                        query="PREDICT users.age = 'is not unique' "
                        'FOR EACH users.user_id',
                    ),
                    call('PredictiveQuery.materialize', indices=[1]),
                ],
                data=data,
            )
        # What the endpoint refuses, read back against the payload.
        lean = tables(
            {
                'users': shop()['users'].drop(columns=['age']),
                'orders': shop()['orders'].drop(
                    columns=['product_id', 'quantity']
                ),
            }
        )
        flow(
            g,
            'the-endpoint-refuses-a-type-only-the-entity-rows-carry',
            [*asked(COUNT, indices=[1]), predict],
            data=lean,
            stand_in=refusing(
                422,
                {
                    'status': 422,
                    'detail': "semantic type 'numerical' is unsupported",
                },
            ),
        )
        flow(
            g,
            'the-endpoint-refuses-many-values-in-a-text-column',
            [
                schema(),
                call(
                    'Schema.set',
                    table='users',
                    column='country',
                    attribute='stype',
                    value='text',
                ),
                call('PredictiveQuery', query=COUNT),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    num_context_samples=6,
                    random_seed=7,
                ),
                predict,
            ],
            stand_in=refusing(
                422,
                {
                    'status': 422,
                    'code': 'UNSUPPORTED_FEATURE',
                    'detail': 'categorical cardinality 3 exceeds limit 2',
                },
            ),
        )
        flow(
            g,
            'the-endpoint-names-no-usable-invalid-parameter',
            [*asked(COUNT, indices=[1]), predict],
            stand_in=refusing(
                422,
                {
                    'status': 422,
                    'title': 'Request validation failed.',
                    'invalid_params': [{}, 'context', {'name': ''}],
                },
            ),
        )
        # A session over a question holding a literal: the scoring call's
        # question is compared with the pinned one, literals spelled alike.
        for label, literal in (
            ('a-session-over-a-question-holding-a-literal', "'US'"),
            (
                'a-session-over-a-question-holding-a-double-quoted-literal',
                '"US"',
            ),
        ):
            flow(
                g,
                label,
                [
                    *asked(
                        COUNT + f' WHERE users.country = {literal}',
                        indices=[1],
                    ),
                    call('SDMClient.fit', model=REL),
                    call('FittedModel.predict'),
                    to_pandas,
                ],
            )
        # A graph with no time column at all, so no row is anchored in time.
        sent(
            'a-graph-with-no-time-column',
            [
                *asked(
                    CATEGORY.replace('products', 'users')
                    .replace('category', 'country')
                    .replace('product_id', 'user_id'),
                    indices=[1],
                ),
                predict,
                to_pandas,
            ],
            data=tables(
                {
                    'users': shop()['users'],
                    'profiles': pd.DataFrame(
                        {
                            'profile_id': pd.Series(range(1, 7), dtype='int64'),
                            'user_id': pd.Series(range(1, 7), dtype='int64'),
                            'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        }
                    ),
                }
            ),
        )
        # Data columns named as the engine names its own.
        sent(
            'data-columns-named-as-the-engine-names-its-own',
            [*asked(COUNT, indices=[1]), predict, to_pandas],
            data=variant(
                orders=shop()['orders']
                .drop(columns=['order_id'])
                .assign(
                    __node_id=[i // 2 for i in range(40)],
                    __node_id_1=[i // 2 for i in range(40)],
                ),
                users=shop()['users'].assign(__kumo_entity_ref_0=range(6)),
            ),
        )
        tabular(
            g,
            'a-one-valued-target-asked-as-a-forecast',
            [
                call(
                    'Request',
                    context=context,
                    predict=rows('people', 0, 2, drop=['churn']),
                    target='churn',
                    task='forecasting',
                ),
                tab_predict,
            ],
            data=tables({'people': people().assign(churn=1)}),
        )
        for label, values in (
            ('two-words', ['a', 'b']),
            ('two-booleans', [True, False]),
        ):
            g.add(
                label,
                'tabular.two_whole_numbers',
                {
                    'series': frame({'x': pd.Series(values, dtype=object)})[
                        'columns'
                    ]['x']
                },
            )
        # A session's scoring call refused as its fit would have been.
        for label, body in (
            (
                'a-session-scoring-call-refused-for-too-many-values',
                {
                    'status': 422,
                    'code': 'UNSUPPORTED_FEATURE',
                    'detail': 'categorical cardinality 3 exceeds limit 2',
                },
            ),
            (
                'a-session-scoring-call-refused-for-a-type',
                {
                    'status': 422,
                    'detail': "semantic type 'categorical' is unsupported",
                },
            ),
        ):
            flow(
                g,
                label,
                [
                    *asked(COUNT, indices=[1]),
                    call('SDMClient.fit', model=REL),
                    call('FittedModel.predict'),
                ],
                stand_in={'fail': [{'at': 1, 'status': 422, 'body': body}]},
            )
        # Inference options in their other forms.
        flow(
            g,
            'an-inference-option-this-client-does-not-forward',
            [
                *asked(
                    SUM,
                    indices=[1],
                    inference_config={'target_transforms': [None]},
                ),
                predict,
            ],
        )
        for label, query, cls, fields in (
            (
                'a-default-regression-config',
                SUM,
                'RegressionInferenceConfig',
                {},
            ),
            (
                'a-regression-config-choosing-the-mean',
                SUM,
                'RegressionInferenceConfig',
                {'output_type': 'mean'},
            ),
            (
                'a-regression-config-of-two-estimators',
                SUM,
                'RegressionInferenceConfig',
                {'num_estimators': 2},
            ),
            (
                'a-classification-config-shuffling-classes',
                CATEGORY,
                'ClassificationInferenceConfig',
                {'class_shuffle': True},
            ),
        ):
            sent(
                f'an-inference-config-dataclass-{label}',
                [
                    *asked(
                        query,
                        indices=[1],
                        inference_config={
                            '$inference_config': {
                                'class': cls,
                                'fields': fields,
                            }
                        },
                    ),
                    predict,
                    to_pandas,
                ],
            )
        sent(
            'an-inference-config-of-none',
            [
                *asked(SUM, indices=[1], inference_config=None),
                predict,
                to_pandas,
            ],
        )
        sent(
            'columns-left-out-with-exclude-cols-dict',
            [
                *asked(
                    COUNT,
                    indices=[1],
                    exclude_cols_dict={
                        'users': ['country'],
                        'orders': ['quantity', 'quantity'],
                    },
                ),
                predict,
                to_pandas,
            ],
        )
        # Decimals, as pandas and as Arrow read them, on the relational route.
        users = shop()['users']
        held = users.assign(
            balance=pd.Series(
                [
                    Decimal(v) if v else None
                    for v in ('1.5', '2.25', '', '3', '4.75', '5')
                ],
                dtype=object,
            ),
            points=pd.Series(
                [
                    Decimal(v) if v else None
                    for v in ('1', '2', '', '3', '4', '5')
                ],
                dtype=object,
            ),
            visits=pd.Series(
                [
                    Decimal(v)
                    for v in ('1', '2', '7', '3', '4', '9007199254740993')
                ],
                dtype=object,
            ),
            serial=pd.Series(
                [
                    Decimal(v)
                    for v in ('9223372036854775808', '2', '7', '3', '4', '5')
                ],
                dtype=object,
            ),
            credit=pd.Series(
                [
                    Decimal(v) if v else None
                    for v in ('1.50', '2.00', '', '3.25', '4.00', '5.75')
                ],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            ),
        )
        sent(
            'decimal-columns-as-pandas-and-arrow-read-them',
            [*asked(COUNT, indices=[1, 2]), predict, to_pandas],
            data=variant(users=held),
        )
        # Time columns of text, of objects, and of mixed offsets.
        lines = shop()['orders']
        for label, written in (
            (
                'a-time-column-of-text',
                lines['ordered_at'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            ),
            ('a-time-column-of-text-that-is-no-date', ['n/a'] * 40),
        ):
            sent(
                label,
                [
                    schema(),
                    call(
                        'Schema.set_table',
                        table='orders',
                        attribute='time_column',
                        value='ordered_at',
                    ),
                    call('PredictiveQuery', query=COUNT),
                    call(
                        'PredictiveQuery.materialize',
                        indices=[1],
                        num_context_samples=6,
                        random_seed=7,
                    ),
                    predict,
                    to_pandas,
                ],
                data=variant(orders=lines.assign(ordered_at=list(written))),
            )
        zoned = pd.Series(
            [
                pd.Timestamp(t).tz_localize('Europe/Berlin')
                for t in lines['ordered_at']
            ],
            dtype=object,
        )
        signed_up = pd.Series(
            [
                pd.Timestamp('2023-03-20T10:00:00+01:00'),
                pd.Timestamp('2023-04-02T10:00:00+02:00'),
            ]
            * 3,
            dtype=object,
        )
        sent(
            'timestamps-of-mixed-offsets-held-as-objects',
            [*asked(COUNT, indices=[1]), predict, to_pandas],
            data=variant(
                orders=lines.assign(ordered_at=zoned),
                users=users.assign(signed_up=signed_up),
            ),
        )
        flow(
            g,
            'a-table-with-no-rows',
            [*asked(COUNT, indices=[1]), predict],
            data=variant(
                reviews=pd.DataFrame(
                    {
                        'review_id': pd.Series([], dtype='int64'),
                        'user_id': pd.Series([], dtype='int64'),
                        'stars': pd.Series([], dtype='int64'),
                    }
                )
            ),
        )
        sent(
            'a-column-left-out-of-a-table-in-its-source',
            [
                call(
                    'Schema.from_duckdb',
                    tables={'users': 'users', 'orders': 'orders'},
                ),
                call('Schema.delete', table='users', column='country'),
                call('PredictiveQuery', query=COUNT),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1, 2],
                    num_context_samples=6,
                    random_seed=7,
                ),
                predict,
                to_pandas,
            ],
        )
        # The tabular model: the dtypes a column is sent as.
        whole_ages = people().assign(
            age=pd.Series(range(20, 32), dtype='int64')
        )
        for label, ages in (
            ('whole-numbers-and-a-null', [21.0, None]),
            ('fractions-and-a-null', [21.5, None]),
            ('nothing-but-nulls', [None, None]),
        ):
            scored_rows = pd.DataFrame(
                {
                    'user_id': pd.Series([20, 21], dtype='int64'),
                    'age': pd.Series(ages, dtype='float64'),
                    'segment': ['north', 'south'],
                }
            )
            data = tables({'people': whole_ages, 'scored': scored_rows})
            sent(
                f'an-integer-feature-scored-as-{label}',
                [
                    direct(predict={'$frame': {'table': 'scored'}}),
                    tab_predict,
                    to_pandas,
                ],
                data=data,
            )
            sent(
                f'an-integer-feature-scored-in-a-session-as-{label}',
                [
                    schema(tables=['people']),
                    call(
                        'Request',
                        schema={'$schema': None},
                        entity_table='people',
                        context=context,
                        predict=rows('people', 0, 2, drop=['churn']),
                        target='churn',
                        task='classification',
                    ),
                    call('SDMClient.fit', model=TAB),
                    call(
                        'Request',
                        schema={'$schema': None},
                        entity_table='people',
                        context=context,
                        predict={'$frame': {'table': 'scored'}},
                        target='churn',
                        task='classification',
                    ),
                    call('FittedModel.predict'),
                    to_pandas,
                ],
                data=data,
            )
        unknown = pd.DataFrame(
            {
                'when': pd.Series([None] * 4, dtype=object),
                'churn': pd.Series([0, 1, 0, 1], dtype='int64'),
            }
        )
        for label, values in (
            ('timestamps', pd.to_datetime(['2024-01-01', '2024-02-01'])),
            ('fractions', pd.Series([1.5, 2.5], dtype='float64')),
            ('booleans', pd.Series([True, False], dtype='bool')),
        ):
            sent(
                f'a-column-with-no-context-values-scored-as-{label}',
                [
                    call(
                        'Request',
                        context={'$frame': {'table': 'unknown'}},
                        predict={'$frame': {'table': 'scored'}},
                        target='churn',
                        task='classification',
                    ),
                    tab_predict,
                    to_pandas,
                ],
                data=tables(
                    {
                        'unknown': unknown,
                        'scored': pd.DataFrame({'when': pd.Series(values)}),
                    }
                ),
            )
        # A regression over a column of labels is sent, its target typed by
        # what its dtype can mean.
        sent(
            'a-regression-over-a-column-of-labels',
            [
                call(
                    'Request',
                    context=context,
                    predict=rows('people', 0, 2, drop=['segment']),
                    target='segment',
                    task='regression',
                ),
                tab_predict,
                to_pandas,
            ],
            data=PEOPLE,
        )
        # A scoring row the pinned schema describes and the wire cannot hold.
        unshaped = people().drop(columns=['segment'])
        tabular(
            g,
            'tabular-a-session-scoring-a-value-the-wire-cannot-hold',
            [
                schema(tables=['people']),
                call(
                    'Request',
                    schema={'$schema': None},
                    entity_table='people',
                    context=context,
                    predict=rows('people', 0, 2, drop=['churn']),
                    target='churn',
                    task='classification',
                ),
                call('SDMClient.fit', model=TAB),
                call(
                    'Request',
                    schema={'$schema': None},
                    entity_table='people',
                    context=context,
                    predict={'$frame': {'table': 'infinite'}},
                    target='churn',
                    task='classification',
                ),
                call('FittedModel.predict'),
            ],
            data=tables(
                {
                    'people': unshaped,
                    'infinite': pd.DataFrame(
                        {
                            'user_id': pd.Series([20, 21], dtype='int64'),
                            'age': [float('inf'), 1.0],
                        }
                    ),
                }
            ),
        )
        keyed_people = [
            schema(tables=['people']),
            call(
                'Request',
                schema={'$schema': None},
                entity_table='people',
                context=context,
                predict=rows('people', 0, 3, drop=['churn']),
                target='churn',
                task='classification',
            ),
        ]
        # Ids the endpoint may answer under: its own token, a key list that
        # is not one key, and the bare key, right or wrong.
        for label, first in (
            ('an-id-longer-than-the-contract-allows', {'id': 'x' * 1025}),
            ('an-id-listing-two-keys', {'id': '[0, 1]'}),
            ('the-bare-key-as-the-id', {'id': '0'}),
            ('another-rows-bare-key-as-the-id', {'id': '1'}),
        ):
            tabular(
                g,
                label,
                [*keyed_people, tab_predict, to_pandas],
                stand_in={
                    'items_at': [
                        {
                            'at': 0,
                            'items': [
                                {'from': 0, 'set': first},
                                {'from': 1},
                                {'from': 2},
                            ],
                        }
                    ]
                },
            )
        tabular(
            g,
            'tabular-a-session-scoring-rows-without-their-key',
            [
                *keyed_people,
                call('SDMClient.fit', model=TAB),
                call(
                    'Request',
                    schema={'$schema': None},
                    entity_table='people',
                    context=context,
                    predict=rows('people', 3, 5, drop=['churn', 'user_id']),
                    target='churn',
                    task='classification',
                ),
                call('FittedModel.predict'),
                to_pandas,
            ],
        )
        tabular(
            g,
            'tabular-a-session-scoring-a-column-it-never-saw',
            [
                *keyed_people,
                call('SDMClient.fit', model=TAB),
                call(
                    'Request',
                    schema={'$schema': None},
                    entity_table='people',
                    context=context,
                    predict={'$frame': {'table': 'extra'}},
                    target='churn',
                    task='classification',
                ),
                call('FittedModel.predict'),
            ],
            data=tables(
                {
                    'people': people(),
                    'extra': people()
                    .iloc[3:5]
                    .reset_index(drop=True)
                    .drop(columns=['churn'])
                    .assign(more=[1, 2]),
                }
            ),
        )
        tabular(
            g,
            'tabular-keys-held-in-lists',
            [
                schema(tables=['people']),
                call(
                    'Request',
                    schema={'$schema': None},
                    entity_table='people',
                    context=context,
                    predict={'$frame': {'table': 'listed'}},
                    target='churn',
                    task='classification',
                ),
                tab_predict,
            ],
            data=tables(
                {
                    'people': people(),
                    'listed': pd.DataFrame(
                        {
                            'user_id': pd.Series([[1], [2]], dtype=object),
                            'age': [1.0, 2.0],
                            'segment': ['north', 'south'],
                        }
                    ),
                }
            ),
        )
        tabular(
            g,
            'tabular-an-empty-frame-to-score-with-its-key',
            [
                schema(tables=['people']),
                call(
                    'PredictiveQuery',
                    query='PREDICT people.churn FOR EACH people.user_id',
                ),
                call(
                    'PredictiveQuery.materialize',
                    predict=rows('people', 0, 0, drop=['churn']),
                ),
                tab_predict,
                to_pandas,
            ],
        )
        # Halves assigned after the request was built.
        # Sent as it stands; the endpoint is the one to refuse it.
        sent(
            'assigning-a-context-without-the-target',
            [
                direct(),
                call(
                    'Request.assign',
                    half='context',
                    value=whole('people', drop=['churn']),
                ),
                tab_predict,
            ],
            data=PEOPLE,
            stand_in=refusing(
                422,
                {
                    'status': 422,
                    'title': 'Request validation failed.',
                    'invalid_params': [
                        {'name': 'task.target', 'reason': 'not in context'}
                    ],
                },
            ),
        )
        unanswered = people().assign(
            churn=pd.Series([None] * 12, dtype='float64')
        )
        tabular(
            g,
            'assigning-a-context-with-no-answered-row',
            [
                *keyed_people[:1],
                call(
                    'Request',
                    schema={'$schema': None},
                    entity_table='people',
                    context=context,
                    predict=rows('people', 0, 3, drop=['churn']),
                    target='churn',
                    task='regression',
                ),
                call(
                    'Request.assign',
                    half='context',
                    value={'$frame': {'table': 'unanswered'}},
                ),
                tab_predict,
                to_pandas,
            ],
            data=tables({'people': people(), 'unanswered': unanswered}),
        )
        # The relational payload's value readers, asked directly: the engine
        # takes frames from callers of its own, not only from this client.
        from predictive_query.schema.task import TaskType

        for label, value, task in (
            (
                'a-decimal-regression-target',
                Decimal('1.25'),
                TaskType.REGRESSION,
            ),
            ('a-binary-target-of-one', 1, TaskType.BINARY_CLASSIFICATION),
            ('a-binary-target-of-zero', 0, TaskType.BINARY_CLASSIFICATION),
        ):
            g.add(
                label,
                'payload.target_json_value',
                {'task_type': task.value, 'value': encode_value(value)},
            )
        for label, values, dtype in (
            ('arrow-fractional-decimals', ['1.50', '2.25', None], (10, 2)),
            ('arrow-narrow-whole-decimals', ['1', '20', None], (18, 0)),
            ('arrow-wide-whole-decimals-that-fit', ['1', '20'], (38, 0)),
            (
                'arrow-wide-whole-decimals-too-wide',
                ['1', '92233720368547758080'],
                (38, 0),
            ),
        ):
            g.add(
                label,
                'payload.decimal_dtype_name',
                {
                    'series': frame(
                        {
                            'x': pd.Series(
                                [
                                    None if v is None else Decimal(v)
                                    for v in values
                                ],
                                dtype=pd.ArrowDtype(pa.decimal128(*dtype)),
                            )
                        }
                    )['columns']['x']
                },
            )
        g.add(
            'a-non-finite-decimal-among-whole-ones',
            'payload.decimal_dtype_name',
            {
                'series': frame(
                    {
                        'x': pd.Series(
                            [Decimal('NaN'), Decimal('1'), Decimal('Infinity')],
                            dtype=object,
                        )
                    }
                )['columns']['x']
            },
        )
        for label, value in (
            ('an-array', np.array([1, 2])),
            ('a-list', [Decimal('1.5'), None, 'a']),
            ('a-tuple', (1, 2)),
        ):
            g.add(label, 'payload.json_value', {'value': encode_value(value)})
        # The fixture encoder turns a numpy datetime into a Timestamp, so the
        # value is spelled out. The unit matters: `.item()` gives a datetime
        # for `us` and an int of nanoseconds for `ns`.
        for unit in ('us', 'ns'):
            g.add(
                f'a-numpy-datetime-in-{unit}',
                'payload.json_value',
                {
                    'value': {
                        '$np': {
                            'type': f'datetime64[{unit}]',
                            'value': '2024-01-01T08:00:00',
                        }
                    }
                },
            )
        for label, value, dtype in (
            (
                'a-datetime-under-timestamp',
                dt.datetime(2024, 1, 1, 8),
                'timestamp[us]',
            ),
            (
                'an-epoch-under-timestamp',
                1_700_000_000_000_000,
                'timestamp[us]',
            ),
            ('text-under-an-unnamed-dtype', 'x', 'date32'),
            ('a-dict-under-an-unnamed-dtype', {'a': 1}, 'date32'),
            ('an-array-under-stringlist', np.array(['a', 'b']), 'stringlist'),
            (
                'a-tuple-under-stringlist',
                ('a', Decimal('1.5'), None),
                'stringlist',
            ),
            ('text-under-stringlist', 'a', 'stringlist'),
            ('nan-text-under-float64', 'NaN', 'float64'),
            ('infinite-text-under-float64', 'Infinity', 'float64'),
        ):
            g.add(
                label,
                'payload.cell_json_value',
                {'value': encode_value(value), 'dtype': dtype},
            )
        for label, values in (
            ('lists-of-tokens', [['a', 'b'], ['b'], None, ['c', 'a']] * 2),
            ('arrays-of-tokens', [np.array(['a', 'b']), np.array(['c'])] * 4),
        ):
            g.add(
                f'a-feature-of-{label}',
                'payload.feature_column',
                {
                    'feature': frame({'x': pd.Series(values, dtype=object)})[
                        'columns'
                    ]['x']
                },
            )
        for label, value, dtype in (
            ('a-boolean-class', True, 'bool'),
            ('a-decimal-class', Decimal('2.50'), 'float64'),
            ('a-whole-number-class', np.int64(3), 'int64'),
        ):
            g.add(
                label,
                'payload.class_label',
                {'value': encode_value(value), 'dtype': dtype},
            )
        for label, values in (
            ('nothing-but-nulls', [None, None]),
            ('nothing-that-reads-as-a-date', ['n/a', 'unknown']),
        ):
            g.add(label, 'time_column.read', {'values': values})
        return g

    def request_branches() -> Any:
        r"""The branches of the request path that Phase 8.5's coverage review
        found reachable and untaken: options given in their other forms,
        refusals a part, a key or a column of an unusual type gives, answers
        spelled otherwise than the column, and the payload's value readers
        asked directly. Each case names what it takes."""
        from decimal import Decimal

        import numpy as np
        from sdm_relational_engine.rfm.base import composite_key

        g = Group('request-branches', 'scripts/parity/_fixture_messages.py')

        def sent(
            label: str,
            calls: list[dict[str, Any]],
            data: dict[str, Any] | None = None,
            stand_in: dict[str, Any] | None = None,
        ) -> None:
            g.flow(
                label,
                {
                    'tables': SHOP if data is None else data,
                    'calls': calls,
                    'show_bodies': True,
                },
                stand_in=stand_in,
            )

        # rfm.py `_resolve_task_request_options`: an inference config given
        # as a mapping, for a classification.
        sent(
            'a-classification-given-an-inference-config-mapping',
            [
                schema(),
                call(
                    'PredictiveQuery',
                    query='PREDICT COUNT(orders.*, 0, 30, days)=0 '
                    'FOR EACH users.user_id',
                ),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1, 2],
                    random_seed=7,
                    inference_config={'num_estimators': 4},
                ),
                predict,
                to_pandas,
            ],
        )
        # decompose.py `about_the_part`: a forecast step whose window reaches
        # back before the data, which the engine says in words the
        # materializer records as `window_outran_the_data`.
        flow(
            g,
            'a-forecast-step-whose-window-outran-the-data',
            [
                schema(),
                call('PredictiveQuery', query=FORECAST),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    anchor_time={'$timestamp': '2024-02-15T00:00:00'},
                    num_context_samples=12,
                    random_seed=7,
                ),
            ],
            want=Raises('ask for fewer than 2 steps'),
        )
        # decompose.py `_decimal_as_typed`: a decimal candidate with a
        # fraction, which a column of decimals sends as float64.
        g.add(
            'a-decimal-candidate-with-a-fraction',
            'decompose.as_literal',
            {'value': encode_value(Decimal('1.5'))},
        )
        # request.py `_warn_of_rows_scored_from_their_own_answers`: context
        # keys the key's dtype cannot write, so no answered key is read and
        # nothing is compared; the request is then refused for the same value.
        users = people()
        too_wide = users.astype({'user_id': object})
        too_wide.loc[11, 'user_id'] = 2**70
        tabular(
            g,
            'context-keys-the-key-dtype-cannot-write',
            [
                schema(tables=['people']),
                call(
                    'Request',
                    context=whole('too_wide'),
                    predict=rows('people', 0, 3, drop=['churn']),
                    target='churn',
                    task='classification',
                    schema={'$schema': None},
                    entity_table='people',
                ),
                tab_predict,
            ],
            data=tables({'people': users, 'too_wide': too_wide}),
            want=Raises('outside the range int64 represents'),
        )
        # wire/answers.py `as_declared_target`: a boolean target's
        # probabilities keyed as the endpoint spells the classes.
        flags = users.assign(churn=users['churn'].astype(bool))
        spelled = {'false': 0.25, 'true': 0.75}
        tabular(
            g,
            'probabilities-keyed-otherwise-than-a-boolean-column',
            [
                call(
                    'Request',
                    context=whole('flags'),
                    predict=rows('flags', 0, 3, drop=['churn']),
                    target='churn',
                    task='classification',
                ),
                call(
                    'SDMClient.predict',
                    model=TAB,
                    outputs=['prediction', 'probabilities'],
                ),
                to_pandas,
            ],
            data=tables({'flags': flags}),
            stand_in={
                'items_at': [
                    {
                        'at': 0,
                        'items': [
                            {'from': i, 'set': {'probabilities': spelled}}
                            for i in range(3)
                        ],
                    }
                ]
            },
        )
        # base/composite_key.py `refuse_unfoldable_dtypes`: a composite key
        # over a float column.
        flow(
            g,
            'a-composite-key-over-a-float-column',
            [
                schema(primary_keys={'products': ['product_id', 'price']}),
                call('PredictiveQuery', query=COUNT),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    num_context_samples=6,
                    random_seed=7,
                ),
            ],
            want=Raises("composite key column 'price' is float"),
        )
        # backend/local/table.py `_materialize_derived_key`: a frame that
        # already holds a column of the name a composite key folds into.
        folded = composite_key.DERIVED_PREFIX + composite_key.encode_identity(
            ['order_id', 'product_id']
        )
        holding = shop()
        holding['orders'] = holding['orders'].assign(**{folded: 'x'})
        flow(
            g,
            'a-frame-holding-the-column-a-composite-key-folds-into',
            [
                schema(primary_keys={'orders': ['order_id', 'product_id']}),
                call('PredictiveQuery', query=COUNT),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    num_context_samples=6,
                    random_seed=7,
                ),
            ],
            data=tables(holding),
            # Phase 9's fix: the column is refused by name, where today's
            # gather refused a key the caller never named.
            want=Raises('uses a name this client needs'),
        )
        # materializers/relational.py `_engine_errors`: a column the engine
        # cannot type, which it refuses with a RuntimeError.
        # Complex numbers have no fixture value, so the column is written as
        # whole numbers that `build_series` makes complex.
        unusual = tables(shop())
        unusual['users']['columns']['signal'] = {
            'dtype': 'complex128',
            'values': list(range(6)),
        }
        flow(
            g,
            'a-column-the-engine-cannot-type',
            [
                schema(),
                call('PredictiveQuery', query=COUNT),
                call(
                    'PredictiveQuery.materialize',
                    indices=[1],
                    num_context_samples=6,
                    random_seed=7,
                ),
            ],
            data=unusual,
            # Phase 9's fix: UNSUPPORTED_REQUEST naming the column, where
            # today's was the engine's failure as INTERNAL_ERROR.
            want=Raises('a type no model reads'),
        )
        # payload.py `_timestamp_json_value`: a zoned timestamp, and the
        # integer pandas reads as NaT, under a timestamp column.
        g.add(
            'a-zoned-timestamp',
            'payload.json_value',
            {'value': encode_value(pd.Timestamp('2024-03-01T10:00:00+02:00'))},
        )
        g.add(
            'the-integer-pandas-reads-as-nat-under-a-timestamp-column',
            'payload.cell_json_value',
            {
                'value': encode_value(np.iinfo(np.int64).min),
                'dtype': 'timestamp[us]',
            },
        )
        # client.py `_ranked`: a NaN score among enough candidates that
        # timsort merges runs, galloping, rather than inserting into one.
        candidates = list(range(1, 151))
        g.add(
            'a-nan-score-among-150-candidates',
            'client.ranked',
            {
                'candidates': candidates,
                'scores': [
                    [{'$float': 'nan'} if c % 7 == 0 else (c * 37 % 10) / 10]
                    for c in candidates
                ],
                'top': 20,
            },
        )
        return g

    def request_gaps() -> Any:
        r"""The branches of the request path that Phase 8.5's coverage-gap
        verifier found ordinary public calls reach and no recorded case
        took: options and question kinds no other case asks for, a fit
        refused for a column's cardinality, a tabular fit with held-out
        keys, a target typed as an ID, a name a result does not know,
        rankings over names that need backticks, frames changed in place
        after their schema read them, and a ``SourceTable`` put into a
        schema's tables. Each case names what it takes."""
        g = Group('request-gaps', 'scripts/parity/_fixture_messages.py')

        def sent(
            label: str,
            calls: list[dict[str, Any]],
            data: dict[str, Any] | None = None,
            stand_in: dict[str, Any] | None = None,
            want: Any = None,
            original: bool = False,
        ) -> None:
            if original:
                assert want is None
                observation = subprocess.run(
                    [
                        sys.executable,
                        '-c',
                        'import json, sys; '
                        'from make_fixtures import Group; '
                        'spec = json.load(sys.stdin); '
                        'group = Group(spec["group"], spec["source"]); '
                        'group.flow(spec["label"], spec["input"], '
                        'stand_in=spec["stand_in"]); '
                        'json.dump(group.cases[0], sys.stdout)',
                    ],
                    input=json.dumps(
                        {
                            'group': g.name,
                            'source': g.source,
                            'label': label,
                            'input': {
                                'tables': SHOP if data is None else data,
                                'calls': calls,
                                'show_bodies': True,
                            },
                            'stand_in': stand_in,
                        }
                    ),
                    env={
                        **os.environ,
                        'SDM_CLIENT_IMPLEMENTATION': 'python',
                        'PYTHONPATH': os.pathsep.join(sys.path),
                    },
                    capture_output=True,
                    text=True,
                    check=True,
                )
                case = json.loads(observation.stdout)
                assert case['id'] not in g._ids
                g._ids.add(case['id'])
                g.cases.append(case)
                return
            g.flow(
                label,
                {
                    'tables': SHOP if data is None else data,
                    'calls': calls,
                    'show_bodies': True,
                },
                stand_in=stand_in,
                **({} if want is None else {'want': want}),
            )

        asked = {'indices': [1, 2], 'num_context_samples': 6, 'random_seed': 7}

        # rfm/inference/payload.py `_inference_config_to_json`: quantiles
        # asked for in the inference config, as the gathered envelope
        # carries them (the relational model then refuses the option).
        sent(
            'quantile-outputs',
            [
                schema(),
                call('PredictiveQuery', query=SUM),
                call(
                    'PredictiveQuery.materialize',
                    **asked,
                    inference_config={'output_type': 'quantiles'},
                ),
                call('Request.payloads'),
            ],
        )
        # payload.py `_inference_config_to_json`: embeddings asked for.
        sent(
            'return-embeddings',
            [
                schema(),
                call('PredictiveQuery', query=SUM),
                call(
                    'PredictiveQuery.materialize',
                    **asked,
                    return_embeddings=True,
                ),
                call(
                    'SDMClient.predict',
                    model=REL,
                    outputs=['prediction', 'embeddings'],
                ),
                to_pandas,
            ],
        )
        # payload.py `_dtype_name`: a bool target, which a static question
        # gathers as a multiclass classification.
        active = shop()
        active['users'] = active['users'].assign(
            active=[True, False, True, True, False, True]
        )
        sent(
            'a-bool-target',
            [
                schema(),
                call(
                    'PredictiveQuery',
                    query='PREDICT users.active FOR EACH users.user_id',
                ),
                call('PredictiveQuery.materialize', **asked),
                predict,
                to_pandas,
            ],
            data=tables(active),
        )
        # payload.py `_entity_values`: a plain link prediction whose sampled
        # context leaves some instances' entities out, so their references
        # are null (the default number of context rows does).
        sent(
            'a-link-prediction',
            [
                schema(),
                call('PredictiveQuery', query=LINK),
                call(
                    'PredictiveQuery.materialize', indices=[1, 2], random_seed=7
                ),
                predict,
                to_pandas,
            ],
        )
        # payload.py `_payload_tables_with_names`: a fit refused for a
        # column's cardinality, whose session body has no predict half.
        sent(
            'a-fit-refused-for-a-categorical-cardinality',
            [
                *gathered(COUNT, indices=[1, 2], num_context_samples=6),
                call('SDMClient.fit', model=REL),
            ],
            stand_in={
                'fail': [
                    {
                        'at': 0,
                        'status': 400,
                        'body': {
                            'error': {
                                'code': 'INVALID_REQUEST',
                                'message': 'categorical cardinality 99 '
                                'exceeds limit 1',
                            }
                        },
                    }
                ]
            },
            want=Raises('cardinality'),
        )
        # wire/keys.py `keys_in`, `without_the_keys`: a tabular fit with
        # held-out keys, then a prediction from its session, with and
        # without a key column.
        keyless = people().drop(columns=['user_id'])
        for label, table, holdout, scored_rows in (
            ('a-tabular-fit-with-holdout-keys', 'people', [5, 6], [4, 7]),
            (
                'a-tabular-fit-with-holdout-keys-and-no-key-column',
                'keyless',
                [1, 2],
                [0, 3],
            ),
        ):
            sent(
                label,
                [
                    call(
                        'Request',
                        context=rows(table, 4, 12),
                        predict=rows(table, 0, 2, drop=['churn']),
                        target='churn',
                        task='classification',
                    ),
                    call('SDMClient.fit', model=TAB, holdout=holdout),
                    call(
                        'Request',
                        context=rows(table, 4, 12),
                        predict=rows(
                            table,
                            scored_rows[0],
                            scored_rows[1],
                            drop=['churn'],
                        ),
                        target='churn',
                        task='classification',
                    ),
                    call('FittedModel.predict'),
                    to_pandas,
                ],
                data=tables({'people': people(), 'keyless': keyless}),
            )
        # materializers/tabular.py `_task_kind`: a tabular target whose
        # semantic type was set to ID.
        sent(
            'a-tabular-target-typed-as-an-id',
            [
                schema(tables=['people']),
                call(
                    'Schema.set',
                    table='people',
                    column='churn',
                    attribute='stype',
                    value='ID',
                ),
                call(
                    'PredictiveQuery',
                    query='PREDICT people.churn FOR EACH people.user_id',
                ),
                call('PredictiveQuery.materialize', indices=[1, 2], model=TAB),
            ],
            data=PEOPLE,
            want=Raises("has semantic type 'ID'"),
        )
        # result.py `PredictionResult._carries`: a name the result does not
        # know, which it reports as carried.
        sent(
            'absent-asked-of-a-name-the-result-does-not-know',
            [
                call(
                    'Request',
                    context=rows('people', 4, 12),
                    predict=rows('people', 0, 2, drop=['churn']),
                    target='churn',
                    task='classification',
                ),
                call('SDMClient.predict', model=TAB),
                call(
                    'PredictionResult.absent',
                    requested=['row_index', 'nonsense', 'explanation'],
                ),
            ],
            data=PEOPLE,
        )
        # decompose.py `_how_the_column_travels`: a ranking over a table, and
        # over a column, whose name needs backticks. The schema is asked
        # with the backticks, so the dtype lookup finds nothing and each
        # candidate is written as its value says.
        for label, table, column in (
            (
                'a-ranking-over-a-table-whose-name-needs-backticks',
                'order lines',
                'product_id',
            ),
            (
                'a-ranking-over-a-column-whose-name-needs-backticks',
                'orders',
                'product id',
            ),
        ):
            renamed = shop()
            lines = renamed.pop('orders').rename(columns={'product_id': column})
            renamed[table] = lines
            sent(
                label,
                [
                    schema(
                        relationships=[
                            [table, 'user_id', 'users', 'user_id'],
                            [table, column, 'products', 'product_id'],
                        ]
                    ),
                    call(
                        'PredictiveQuery',
                        query=f'PREDICT LIST_DISTINCT(`{table}`.`{column}`, 0, '
                        f'30, days) RANK TOP 2 FOR EACH users.user_id',
                    ),
                    call(
                        'PredictiveQuery.materialize',
                        **asked,
                        candidates=[1, 2, 3],
                    ),
                    predict,
                    to_pandas,
                ],
                data=tables(renamed),
            )
        # rfm/backend/local/table.py `LocalTable.__init__`: a frame relabelled
        # in place after Schema.from_data read it.
        for label, labels in (
            (
                'a-frame-given-a-multi-index-after-its-schema',
                [['user_id', ''], ['age', ''], ['country', '']],
            ),
            (
                'a-frame-given-repeated-column-names-after-its-schema',
                ['user_id', 'extra', 'extra'],
            ),
            (
                'a-frame-given-an-empty-column-name-after-its-schema',
                ['user_id', 'age', ''],
            ),
        ):
            sent(
                label,
                [
                    schema(),
                    call('PredictiveQuery', query=COUNT),
                    call('Frame.set_columns', table='users', columns=labels),
                    call('PredictiveQuery.materialize', **asked),
                    predict,
                ],
                want=Raises('Data frame must'),
            )
        # rfm/base/table.py `_set_composite_primary_key`: a part of a
        # composite key dropped from its frame after Schema.from_data.
        lined = shop()
        lined['orders'] = lined['orders'].assign(
            line=lined['orders']['order_id'] % 3
        )
        sent(
            'a-composite-key-part-dropped-after-its-schema',
            [
                schema(primary_keys={'orders': ['order_id', 'line']}),
                call('PredictiveQuery', query=COUNT),
                call('Frame.drop', table='orders', columns=['line']),
                call('PredictiveQuery.materialize', **asked),
                predict,
            ],
            data=tables(lined),
            want=Raises('are not present'),
        )
        # materializers/relational.py `_build_graph`, `_as_typed`: a column,
        # and a column a relationship joins on, dropped from its frame after
        # Schema.from_data read it, which the schema still says.
        for label, table, column in (
            ('a-column-dropped-after-its-schema', 'users', 'age'),
            ('a-foreign-key-dropped-after-its-schema', 'orders', 'user_id'),
        ):
            sent(
                label,
                [
                    schema(),
                    call('PredictiveQuery', query=COUNT),
                    call('Frame.drop', table=table, columns=[column]),
                    call('PredictiveQuery.materialize', **asked),
                    predict,
                    to_pandas,
                ],
            )
        # materializers/tabular.py `table`: the target dropped from its frame
        # after Schema.from_data read it, asked with rows to score.
        sent(
            'a-tabular-target-dropped-after-its-schema',
            [
                schema(tables=['people']),
                call(
                    'PredictiveQuery',
                    query='PREDICT people.churn FOR EACH people.user_id',
                ),
                call('Frame.drop', table='people', columns=['churn']),
                call(
                    'PredictiveQuery.materialize',
                    predict=rows('scored', 0, 3),
                    model=TAB,
                ),
            ],
            data=tables(
                {'people': people(), 'scored': people().drop(columns=['churn'])}
            ),
            original=True,
        )
        # materializers/tabular.py `_rows`: a SourceTable put into the tables
        # of a schema over frames, asked on the tabular route.
        sent(
            'a-source-table-put-into-a-frame-schema',
            [
                schema(tables=['people']),
                call('Schema.put_source_table', table='people'),
                call(
                    'PredictiveQuery',
                    query='PREDICT people.churn FOR EACH people.user_id',
                ),
                call('PredictiveQuery.materialize', indices=[1, 2], model=TAB),
            ],
            data=PEOPLE,
            want=Raises('stays in its source'),
        )
        return g

    return (
        messages_client,
        messages_request,
        messages_gathering,
        messages_answers,
        messages_wire,
        messages_engine,
        request_paths,
        request_shapes,
        request_branches,
        request_gaps,
    )
