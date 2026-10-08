# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Every scenario the corpus holds, as data.

A scenario is what a transcript starts as: a backend, how its data is made,
and the public API calls to make. Recording runs the calls and adds the
steps, payloads and result; replay runs the calls again from the transcript.

* The offline scenarios ask the shop the six questions the the Databricks validation workspace notebooks ask,
  on SQLite, DuckDB, PostgreSQL and in memory, and are recorded.
* The synthetic scenarios ask a generated marketplace what the shop never
  makes the sampler do (see :data:`SYNTHETIC`), in memory and on DuckDB.
* The tabular scenarios are the tabular cases of
  ``scripts/snapshot_wire_payloads.py``, under the names the golden file
  gives them, with its relational case beside them.
* The live scenarios ask the same six questions of Databricks (Unity Catalog
  tables, and a metric view over them), Lakebase, and Snowflake (tables, and
  a semantic view over them). They are ``pending-live`` until someone with
  credentials records them (see :mod:`live`).
* The session-zone scenarios ask a question as of a midday anchor in a
  session whose time zone isn't UTC, over a time column stored zoned or
  naive. Each names its twin: the same question over naive columns in UTC,
  whose payloads and answer it must equal.
* The safety-net scenarios (:func:`safety_net`) are what a port of request
  building and answer assembly is checked against: rankings with an
  unscored candidate, with every candidate cold, with tied scores and with
  no probability answered; an absent field; a forecast explained by step;
  the endpoint's 4xx and 5xx refusals; results of several calls merged; and
  sessions, with and without a holdout, on every offline backend.
* The link-prediction scenarios (:func:`link_prediction`) ask the shop a
  plain ``LIST_DISTINCT`` question, which reaches the engine as a temporal
  link prediction, on every offline backend; sampled one hop deep, the engine
  refuses it for finding no target entity. Its variants
  (:func:`link_prediction_variants`) ask it over a foreign key linked to two
  tables, which the engine refuses as multilabel, and over decimal keys.
* The decimal-entity-key scenario (:func:`decimal_entity_keys`) asks a
  question for each product when the product ids are DuckDB decimals, which
  the engine refuses and the core answers; its variant
  (:func:`decimal_mixed_indices`) asks it for indices mixing an int and a
  text, which the engine refuses first.
* The declared-key scenarios (:func:`sqlite_declared_keys`) ask the count
  question on SQLite when the user ids are declared a type the engine reads
  no dtype off.
* The self-link scenario (:func:`self_link`) asks a count of users for each
  user over a schema linking the users to themselves, which the engine
  refuses.
* The categorical-key scenario (:func:`categorical_keys`) asks the count
  question in memory when the user ids are a pandas ``category``, for
  indices written as text, which the engine refuses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TASK_KINDS = (
    'count',
    'classification',
    'regression',
    'forecast',
    'ranking',
    'inactivity',
)
OFFLINE_BACKENDS = ('sqlite', 'duckdb', 'postgres', 'memory')
LIVE_BACKENDS = ('databricks', 'lakebase', 'snowflake')

#: The seed of the shop's data, and the seed every question samples with.
SHOP_SEED = 0
SAMPLE_SEED = 7
#: A moment inside the shop's year, for questions asked as of a past date.
ANCHOR = {'$timestamp': '2024-10-01T00:00:00'}
#: The notebooks' ensemble: four estimators over shuffled columns.
ENSEMBLE = {'num_estimators': 4, 'column_shuffle': True}
#: Small contexts keep the stored payloads small; the paths are the same.
CONTEXT = 30

#: The six questions, as the the Databricks validation workspace notebooks ask them of their own data.
QUESTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    'count': (
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        {'indices': [1, 2, 3], 'num_context_samples': CONTEXT},
    ),
    'classification': (
        'PREDICT products.category FOR EACH products.product_id',
        {'indices': [1, 2]},
    ),
    'regression': (
        'PREDICT SUM(orders.quantity, 0, 30, days) '
        'FOR EACH products.product_id',
        {
            'indices': [1, 2, 3],
            'num_context_samples': CONTEXT,
            'inference_config': ENSEMBLE,
        },
    ),
    'forecast': (
        'PREDICT SUM(orders.quantity, 0, 30, days) '
        'FORECAST 3 TIMEFRAMES FOR EACH products.product_id',
        {'indices': [1], 'anchor_time': ANCHOR, 'num_context_samples': CONTEXT},
    ),
    'ranking': (
        'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) '
        'RANK TOP 3 FOR EACH users.user_id',
        {
            'indices': [1],
            'candidates': [1, 2, 3, 4, 5],
            'anchor_time': ANCHOR,
            'num_context_samples': CONTEXT,
        },
    ),
    'inactivity': (
        'PREDICT COUNT(orders.*, 0, 90, days)=0 FOR EACH users.user_id',
        {'indices': [1, 2, 3, 4], 'num_context_samples': CONTEXT},
    ),
}

#: The shop's tables, named the same in every database.
SHOP_TABLES = {'users': 'users', 'products': 'products', 'orders': 'orders'}


@dataclass(frozen=True)
class Scenario:
    name: str
    backend: str
    task_kind: str
    data: dict[str, Any]
    calls: list[dict[str, Any]]
    live: bool = False
    #: How the stand-in model misbehaves for this scenario
    #: (:class:`nim.SyntheticSession`); ``None`` for a well-behaved one.
    stand_in: dict[str, Any] | None = None

    def skeleton(self) -> dict[str, Any]:
        r"""The transcript this scenario starts as."""
        from _parity import FORMAT_VERSION

        skeleton = {
            'format_version': FORMAT_VERSION,
            'name': self.name,
            'backend': self.backend,
            'task_kind': self.task_kind,
            'data': dict(self.data),
            'calls': [dict(call) for call in self.calls],
        }
        if self.stand_in is not None:
            skeleton['stand_in'] = dict(self.stand_in)
        return skeleton


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {'call': name, 'args': args}


def _relational_calls(
    kind: str, first: dict[str, Any], upper: bool = False
) -> list[dict[str, Any]]:
    r"""The calls asking ``kind``'s question; with ``upper``, of a semantic
    model whose tables and columns are named in upper case."""
    query, options = QUESTIONS[kind]
    if upper:
        query = re.sub(
            r'\b[a-z_]+\.(?:[a-z_]+|\*)', lambda m: m[0].upper(), query
        )
    return [
        first,
        _call('PredictiveQuery', query=query),
        _call(
            'PredictiveQuery.materialize',
            **options,
            random_seed=SAMPLE_SEED,
        ),
        _call('SDMClient.predict', model='kumo-relational', outputs=None),
        _call('PredictionResult.to_pandas'),
    ]


def _shop_data(**extra: Any) -> dict[str, Any]:
    return {'generator': 'shop', 'seed': SHOP_SEED, **extra}


def offline() -> list[Scenario]:
    out = []
    for backend in OFFLINE_BACKENDS:
        for kind in TASK_KINDS:
            if backend == 'memory':
                first = _call('Schema.from_data', tables=list(SHOP_TABLES))
                data = _shop_data()
            else:
                first = _call('Schema.from_source', tables=dict(SHOP_TABLES))
                data = _shop_data(
                    namespace={'schema': 'parity_shop'}
                    if backend == 'postgres'
                    else {}
                )
            out.append(
                Scenario(
                    name=f'{backend}/{kind}',
                    backend=backend,
                    task_kind=kind,
                    data=data,
                    calls=_relational_calls(kind, first),
                )
            )
    return out


def _frame(table: str, head: int | None = None, drop: list[str] | None = None):
    spec: dict[str, Any] = {'table': table}
    if head is not None:
        spec['head'] = head
    if drop:
        spec['drop'] = drop
    return {'$frame': spec}


def golden() -> list[Scenario]:
    r"""The cases ``tests/golden/wire_payloads.json`` holds, by its names.

    A ``#default-outputs`` case asks for the model's default fields, the
    others for ``prediction`` alone, as the snapshot script does.
    """
    data = {'generator': 'wire-snapshot', 'seed': 0}
    people = _frame('people')
    unlabelled = _frame('people', head=3, drop=['churn'])
    cases = {
        'direct-classification': (
            'classification',
            [
                _call(
                    'Request',
                    context=people,
                    predict=unlabelled,
                    target='churn',
                    task='classification',
                )
            ],
        ),
        'direct-regression': (
            'regression',
            [
                _call(
                    'Request',
                    context=people,
                    predict=_frame('people', head=3, drop=['age']),
                    target='age',
                    task='regression',
                )
            ],
        ),
        'pql-predict-frame': (
            'classification',
            [
                _call(
                    'PredictiveQuery',
                    query='PREDICT people.churn FOR EACH people.user_id',
                    schema={'$frames': ['people']},
                ),
                _call('PredictiveQuery.materialize', predict=unlabelled),
            ],
        ),
        'pql-derived-regression': (
            'regression',
            [
                _call(
                    'PredictiveQuery',
                    query='PREDICT people.age FOR EACH people.user_id',
                    schema={'$frames': ['people']},
                ),
                _call(
                    'PredictiveQuery.materialize',
                    predict=_frame('people', head=3, drop=['age']),
                ),
            ],
        ),
        'pql-derived-classification': (
            'classification',
            [
                _call(
                    'PredictiveQuery',
                    query='PREDICT people.segment FOR EACH people.user_id',
                    schema={'$frames': ['people']},
                ),
                _call(
                    'PredictiveQuery.materialize',
                    predict=_frame('people', head=3, drop=['segment']),
                ),
            ],
        ),
    }
    # The cases whose default fields are more than `prediction`.
    with_defaults = {
        'direct-classification',
        'pql-predict-frame',
        'pql-derived-classification',
    }
    out = []
    for case, (kind, calls) in cases.items():
        variants = [('', ['prediction'])]
        if case in with_defaults:
            variants.append(('#default-outputs', None))
        for label, outputs in variants:
            out.append(
                Scenario(
                    name=f'tabular/{case}{label}',
                    backend='tabular',
                    task_kind=kind,
                    data=data,
                    calls=[
                        *calls,
                        _call(
                            'SDMClient.predict',
                            model='kumo-tabular',
                            outputs=outputs,
                        ),
                        _call('PredictionResult.to_pandas'),
                    ],
                )
            )
    out.append(
        Scenario(
            name='relational/pql-count-aggregation',
            backend='memory',
            task_kind='count',
            data=data,
            calls=[
                _call('Schema.from_data', tables=['people', 'orders']),
                _call(
                    'PredictiveQuery',
                    query='PREDICT COUNT(orders.*, 0, 30, days) '
                    'FOR EACH people.user_id',
                ),
                _call('PredictiveQuery.materialize', indices=[0, 1, 2]),
                _call(
                    'SDMClient.predict',
                    model='kumo-relational',
                    outputs=['prediction'],
                ),
                _call('PredictionResult.to_pandas'),
            ],
        )
    )
    return out


#: A midday anchor, so windows start and end among the day's orders, and a
#: window moved by a zone's offset holds other orders.
MIDDAY = {'$timestamp': '2024-10-01T12:00:00'}

#: The session-zone scenarios: name, its twin, the session's zone, and the
#: type ``orders.ordered_at`` is stored as.
SESSION_ZONES: dict[str, tuple[str, str, str]] = {
    'postgres/count#timestamptz-new-york': (
        'postgres/count#midday',
        'America/New_York',
        'timestamptz',
    ),
    'postgres/inactivity#timestamptz-kolkata': (
        'postgres/inactivity#midday',
        'Asia/Kolkata',
        'timestamptz',
    ),
    'postgres/count#timestamp-kolkata': (
        'postgres/count#midday',
        'Asia/Kolkata',
        'timestamp',
    ),
    'duckdb/count#timestamptz-new-york': (
        'duckdb/count#midday',
        'America/New_York',
        'TIMESTAMPTZ',
    ),
    'duckdb/count#timestamp-new-york': (
        'duckdb/count#midday',
        'America/New_York',
        'TIMESTAMP',
    ),
}


def _midday(name: str) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    r"""The backend, kind, calls and data of a midday scenario ``name``."""
    backend, rest = name.split('/', 1)
    kind = rest.split('#', 1)[0]
    query, options = QUESTIONS[kind]
    first = _call('Schema.from_source', tables=dict(SHOP_TABLES))
    calls = [
        first,
        _call('PredictiveQuery', query=query),
        _call(
            'PredictiveQuery.materialize',
            **{**options, 'anchor_time': MIDDAY},
            random_seed=SAMPLE_SEED,
        ),
        _call('SDMClient.predict', model='kumo-relational', outputs=None),
        _call('PredictionResult.to_pandas'),
    ]
    data = _shop_data(
        namespace={'schema': 'parity_shop'} if backend == 'postgres' else {}
    )
    return backend, kind, calls, data


def session_zones() -> list[Scenario]:
    r"""Each session-zone scenario, and each twin once."""
    out = []
    twins: dict[str, None] = {}
    for name, (twin, zone, column_type) in SESSION_ZONES.items():
        backend, kind, calls, data = _midday(name)
        out.append(
            Scenario(
                name=name,
                backend=backend,
                task_kind=kind,
                data={
                    **data,
                    'session': {'TimeZone': zone},
                    'column_types': {'orders.ordered_at': column_type},
                    'twin': twin,
                },
                calls=calls,
            )
        )
        twins[twin] = None
    for twin in twins:
        backend, kind, calls, data = _midday(twin)
        out.append(
            Scenario(
                name=twin,
                backend=backend,
                task_kind=kind,
                data=data,
                calls=calls,
            )
        )
    return out


def live() -> list[Scenario]:
    r"""The live cells: defined here, recorded with credentials later.

    Databricks reads Unity Catalog tables, and its metric view; Snowflake
    reads tables, and its semantic view; Lakebase reads PostgreSQL tables.
    Every one asks the six questions.
    """
    out = []
    for backend, namespace, views in (
        (
            'databricks',
            {'catalog': '$PARITY_DATABRICKS_CATALOG', 'schema': 'parity_shop'},
            ('metric-view',),
        ),
        ('lakebase', {'schema': 'parity_shop'}, ()),
        (
            'snowflake',
            {'database': '$PARITY_SNOWFLAKE_DATABASE', 'schema': 'PARITY_SHOP'},
            ('semantic-view',),
        ),
    ):
        for kind in TASK_KINDS:
            first = _call('Schema.from_source', tables=dict(SHOP_TABLES))
            out.append(
                Scenario(
                    name=f'{backend}/{kind}',
                    backend=backend,
                    task_kind=kind,
                    data=_shop_data(namespace=namespace),
                    calls=_relational_calls(kind, first),
                    live=True,
                )
            )
            for view in views:
                # A semantic model names the tables and how they join; the
                # schema is read from it rather than from the tables. A
                # Snowflake semantic view names them in upper case, as
                # Snowflake spells unquoted names.
                first = _call(
                    'Schema.from_source',
                    semantic_model='parity_shop_metrics'
                    if view == 'metric-view'
                    else 'PARITY_SHOP_SEMANTICS',
                )
                out.append(
                    Scenario(
                        name=f'{backend}/{view}/{kind}',
                        backend=backend,
                        task_kind=kind,
                        data=_shop_data(namespace=namespace, view=view),
                        calls=_relational_calls(
                            kind, first, upper=view == 'semantic-view'
                        ),
                        live=True,
                    )
                )
    return out


#: The questions the synthetic scenarios ask of the marketplace, each the
#: shop never asks: a sum of fractional amounts over items with more reviews
#: than a hop samples, while two tables have a time column; a static
#: question anchored at each customer's own, often shared, signup time; and a
#: count compared with a constant, for the customers an entity filter keeps,
#: with lagged targets and a second hop of no neighbors. The rest reach what
#: the sampler does only for some questions: windows in every unit, one of
#: them open at the start, aggregating one table five times, with a context
#: anchor whose window leaks into the prediction's and few PQ iterations; a
#: static numerical column whose context is drawn after a prediction anchor
#: past the data; a context anchor so early its window passes the data's
#: start; entities keyed by text; and a graph without a time column.
SYNTHETIC: dict[str, tuple[str, str, dict[str, Any]]] = {
    'marketplace-sum': (
        'regression',
        'PREDICT SUM(purchases.amount, 0, 30, days) FOR EACH items.item_id',
        {'indices': [1, 2], 'num_context_samples': CONTEXT},
    ),
    'marketplace-entity': (
        'classification',
        'PREDICT customers.segment FOR EACH customers.customer_id',
        {
            'indices': [1, 2],
            'anchor_time': 'entity',
            'num_context_samples': CONTEXT,
        },
    ),
    'marketplace-filter': (
        'classification',
        'PREDICT COUNT(purchases.*, 0, 30, days) > 1 '
        "FOR EACH customers.customer_id WHERE customers.segment = 'retail'",
        {
            'indices': [1, 2],
            'num_context_samples': CONTEXT,
            'lag_timesteps': 2,
            'num_neighbors': [8, 0],
        },
    ),
    'marketplace-windows': (
        'regression',
        'PREDICT SUM(purchases.amount, 0, 36, hours) '
        'FOR EACH customers.customer_id '
        'WHERE COUNT(purchases.*, -1, 0, months) > 0 '
        'AND COUNT(purchases.*, -30, -1, days) >= 0 '
        'AND COUNT(purchases.*, -2, -1, weeks) >= 0 '
        'AND COUNT(purchases.*, -INF, -90, minutes) >= 0 '
        'AND COUNT(purchases.*, -7200, -60, seconds) >= 0 '
        "AND customers.segment = 'retail' "
        'AND customers.signup_at IS NOT NULL',
        {
            'indices': [1, 2],
            'lag_timesteps': 1,
            'anchor_time': {'$timestamp': '2024-11-01T00:00:00'},
            'context_anchor_time': {'$timestamp': '2024-10-31T00:00:00'},
            'max_pq_iterations': 1,
            'num_context_samples': CONTEXT,
        },
    ),
    'marketplace-static': (
        'regression',
        'PREDICT purchases.amount FOR EACH purchases.purchase_id',
        {
            'indices': [1, 2],
            'anchor_time': {'$timestamp': '2025-03-01T00:00:00'},
            'context_anchor_time': {'$timestamp': '2025-04-01T00:00:00'},
            'num_context_samples': CONTEXT,
        },
    ),
    'marketplace-early': (
        'regression',
        'PREDICT COUNT(purchases.*, 0, 7, days) FOR EACH customers.customer_id',
        {
            'indices': [1, 2],
            'anchor_time': {'$timestamp': '2024-10-01T00:00:00'},
            'context_anchor_time': {'$timestamp': '2024-01-04T00:00:00'},
            'num_context_samples': CONTEXT,
        },
    ),
    'marketplace-stores': (
        'regression',
        'PREDICT COUNT(purchases.*, 0, 30, days) FOR EACH stores.store_code',
        {'indices': ['S01', 'S02'], 'num_context_samples': CONTEXT},
    ),
    'marketplace-timeless': (
        'classification',
        'PREDICT items.category FOR EACH items.item_id',
        {'indices': [1, 2], 'num_context_samples': CONTEXT},
    ),
    'marketplace-garbled': (
        'count',
        'PREDICT COUNT(purchases.*, 0, 30, days) FOR EACH customers.customer_id',
        {'indices': [1, 2], 'num_context_samples': CONTEXT},
    ),
}

#: The synthetic scenarios over other data than the marketplace's own: a
#: time column stored as text, some of whose values pandas cannot read, so
#: the sampler warns as it reads them.
SYNTHETIC_DATA: dict[str, str] = {'marketplace-garbled': 'marketplace-garbled'}

#: The time columns a synthetic scenario names, where inference finds none.
SYNTHETIC_TIME_COLUMNS: dict[str, dict[str, str]] = {
    'marketplace-garbled': {'purchases': 'purchased_at'},
}

#: The synthetic scenarios asked only of some backends. In memory, the client
#: refuses a time column with values that do not read
#: (sdm_client/materializers/relational.py, _refuse_values_that_do_not_read)
#: before the engine sees it.
SYNTHETIC_ONLY_ON: dict[str, tuple[str, ...]] = {
    'marketplace-garbled': ('duckdb', 'sqlite', 'postgres'),
}

#: The synthetic scenarios asked of only some of the marketplace's tables:
#: the items and their reviews, neither of which has a time column.
SYNTHETIC_TABLES: dict[str, tuple[str, ...]] = {
    'marketplace-timeless': ('items', 'reviews'),
}

#: The backends the synthetic scenarios run on: in memory and on DuckDB, as
#: the acceptance checks ask, and SQLite and PostgreSQL besides.
SYNTHETIC_BACKENDS = ('memory', 'duckdb', 'sqlite', 'postgres')

#: The marketplace's one key that is text, which inference cannot tell from
#: a foreign key that happens not to repeat.
MARKETPLACE_KEYS = {'stores': 'store_code'}

#: The marketplace's tables, named the same in every database.
MARKETPLACE_TABLES = {
    'customers': 'customers',
    'items': 'items',
    'reviews': 'reviews',
    'stores': 'stores',
    'purchases': 'purchases',
}


def synthetic() -> list[Scenario]:
    r"""The synthetic scenarios: what the shop never makes the sampler do
    (``tests/rust_core/sampling/driver/CONTRACT.md``, "Synthetic
    scenarios")."""
    out = []
    for backend in SYNTHETIC_BACKENDS:
        for name, (kind, query, options) in SYNTHETIC.items():
            if backend not in SYNTHETIC_ONLY_ON.get(name, (backend,)):
                continue
            asked = SYNTHETIC_TABLES.get(name, tuple(MARKETPLACE_TABLES))
            tables = {
                table: MARKETPLACE_TABLES[table]
                for table in MARKETPLACE_TABLES
                if table in asked
            }
            keys = {
                table: key
                for table, key in MARKETPLACE_KEYS.items()
                if table in tables
            }
            schema_args: dict[str, Any] = {'primary_keys': keys}
            if name in SYNTHETIC_TIME_COLUMNS:
                schema_args['time_columns'] = SYNTHETIC_TIME_COLUMNS[name]
            if backend == 'memory':
                first = _call(
                    'Schema.from_data', tables=list(tables), **schema_args
                )
            else:
                first = _call(
                    'Schema.from_source', tables=tables, **schema_args
                )
            data: dict[str, Any] = {
                'generator': SYNTHETIC_DATA.get(name, 'marketplace'),
                'seed': 0,
            }
            if backend == 'postgres':
                data['namespace'] = {'schema': 'parity_marketplace'}
            out.append(
                Scenario(
                    name=f'{backend}/{name}',
                    backend=backend,
                    task_kind=kind,
                    data=data,
                    calls=[
                        first,
                        _call('PredictiveQuery', query=query),
                        _call(
                            'PredictiveQuery.materialize',
                            **options,
                            random_seed=SAMPLE_SEED,
                        ),
                        _call(
                            'SDMClient.predict',
                            model='kumo-relational',
                            outputs=None,
                        ),
                        _call('PredictionResult.to_pandas'),
                    ],
                )
            )
    return out


#: Problem bodies the stand-in answers a failed call with, as a NIM does.
UNPROCESSABLE = {
    'type': 'about:blank',
    'title': 'Unprocessable Entity',
    'status': 422,
    'code': 'UNSUPPORTED_FEATURE',
    'detail': "semantic type 'text' is unsupported",
}
SERVER_ERROR = {
    'type': 'about:blank',
    'title': 'Internal Server Error',
    'status': 500,
    'code': 'INTERNAL_ERROR',
    'detail': (
        'Prediction failed unexpectedly with ValueError: failed to collect '
        'any context examples for the requested entities'
    ),
}


def _memory(
    name: str,
    kind: str,
    *,
    options: dict[str, Any] | None = None,
    outputs: list[str] | None = None,
    stand_in: dict[str, Any] | None = None,
    fails: bool = False,
) -> Scenario:
    r"""``kind``'s question over the shop in memory, with ``options`` over
    the question's own, asking for ``outputs``. One that ``fails`` ends at
    the prediction, whose error is what it records."""
    query, asked = QUESTIONS[kind]
    first = _call('Schema.from_data', tables=list(SHOP_TABLES))
    return Scenario(
        name=name,
        backend='memory',
        task_kind=kind,
        data=_shop_data(),
        calls=[
            first,
            _call('PredictiveQuery', query=query),
            _call(
                'PredictiveQuery.materialize',
                **{**asked, **(options or {})},
                random_seed=SAMPLE_SEED,
            ),
            _call(
                'SDMClient.predict', model='kumo-relational', outputs=outputs
            ),
            *([] if fails else [_call('PredictionResult.to_pandas')]),
        ],
        stand_in=stand_in,
    )


def _session(backend: str, holdout: list[int] | None) -> Scenario:
    r"""A count question pinned with ``fit`` and scored on other users with
    ``FittedModel.predict``; with ``holdout``, those users are held back."""
    query, asked = QUESTIONS['count']
    if backend == 'memory':
        first = _call('Schema.from_data', tables=list(SHOP_TABLES))
        data = _shop_data()
    else:
        first = _call('Schema.from_source', tables=dict(SHOP_TABLES))
        data = _shop_data(
            namespace={'schema': 'parity_shop'} if backend == 'postgres' else {}
        )
    scored = [4, 5, 6]
    fit: dict[str, Any] = {'model': 'kumo-relational', 'outputs': None}
    if holdout is not None:
        fit['holdout'] = holdout
    return Scenario(
        name=f'{backend}/session' + ('#holdout' if holdout else ''),
        backend=backend,
        task_kind='count',
        data=data,
        calls=[
            first,
            _call('PredictiveQuery', query=query),
            _call(
                'PredictiveQuery.materialize', **asked, random_seed=SAMPLE_SEED
            ),
            _call('SDMClient.fit', **fit),
            _call(
                'PredictiveQuery.materialize',
                **{**asked, 'indices': scored},
                random_seed=SAMPLE_SEED,
            ),
            _call('FittedModel.predict'),
            _call('PredictionResult.to_pandas'),
        ],
    )


def safety_net() -> list[Scenario]:
    r"""The scenarios a port of request building and answer assembly is
    checked against: what the stand-in answers awkwardly, what the client
    refuses before sending, results put together from several calls, and
    sessions."""
    ranking = QUESTIONS['ranking'][1]
    orders = 'orders'
    tabular_data = _shop_data()
    batches = [
        _call(
            'Request',
            context={'$frame': {'table': orders, 'rows': [0, 120]}},
            predict={
                '$frame': {
                    'table': orders,
                    'rows': [start, stop],
                    'drop': ['quantity'],
                }
            },
            target='quantity',
            task='regression',
        )
        for start, stop in ((120, 700), (700, 1183))
    ]
    out = [
        _memory(
            'memory/ranking#unscored-candidate',
            'ranking',
            options={'candidates': [*ranking['candidates'], 99]},
        ),
        _memory(
            'memory/ranking#all-candidates-cold',
            'ranking',
            options={'candidates': [98, 99]},
            fails=True,
        ),
        _memory(
            'memory/ranking#tied-scores',
            'ranking',
            stand_in={'seed': 'row'},
        ),
        _memory(
            'memory/ranking#no-positive-probability',
            'ranking',
            stand_in={'omit': ['probabilities']},
            fails=True,
        ),
        _memory(
            'memory/classification#absent-field',
            'classification',
            outputs=['prediction', 'probabilities'],
            stand_in={'omit': ['probabilities']},
        ),
        _memory(
            'memory/forecast#explanations-by-step',
            'forecast',
            outputs=['prediction', 'explanation'],
        ),
        _memory(
            'memory/regression#nim-4xx',
            'regression',
            stand_in={
                'fail': [{'at': 0, 'status': 422, 'body': UNPROCESSABLE}]
            },
            fails=True,
        ),
        _memory(
            'memory/count#nim-5xx',
            'count',
            stand_in={'fail': [{'at': 0, 'status': 500, 'body': SERVER_ERROR}]},
            fails=True,
        ),
        Scenario(
            name='tabular/batches',
            backend='tabular',
            task_kind='regression',
            data=tabular_data,
            calls=[
                call
                for request in batches
                for call in (
                    request,
                    _call(
                        'SDMClient.predict',
                        model='kumo-tabular',
                        outputs=['prediction'],
                    ),
                )
            ]
            + [
                _call('PredictionResult.merge'),
                _call('PredictionResult.to_pandas'),
            ],
        ),
    ]
    for backend in OFFLINE_BACKENDS:
        out.append(_session(backend, None))
        out.append(_session(backend, [4, 5, 6]))
    return out


#: A link prediction: the products each user orders in the next 30 days.
#: Only a ranked LIST_DISTINCT is asked one candidate at a time
#: (``sdm_client/decompose.py``); this one reaches the engine whole, over the
#: foreign key ``orders.product_id``, as a ``temporal_link_prediction`` task.
LINK_PREDICTION = (
    'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) '
    'FOR EACH users.user_id',
    {
        'indices': [1, 2, 3],
        'anchor_time': ANCHOR,
        'num_context_samples': CONTEXT,
    },
)

#: How each link-prediction scenario asks it: its label, its options over
#: the question's own, the backends it is asked on, and whether it fails.
#: One hop reaches a user's orders but not the products they name, so the
#: subgraph holds no target entity and the engine refuses before any
#: payload. ``top_k`` cuts the ranking the request asks for.
LINK_VARIANTS: tuple[tuple[str, dict[str, Any], tuple[str, ...], bool], ...] = (
    ('', {}, OFFLINE_BACKENDS, False),
    ('#no-target-entities', {'num_neighbors': [16]}, OFFLINE_BACKENDS, True),
    ('#top-k', {'top_k': 3}, ('memory',), False),
)


def link_prediction() -> list[Scenario]:
    r"""The link-prediction scenarios (:data:`LINK_VARIANTS`)."""
    query, options = LINK_PREDICTION
    out = []
    for backend in OFFLINE_BACKENDS:
        if backend == 'memory':
            first = _call('Schema.from_data', tables=list(SHOP_TABLES))
            data = _shop_data()
        else:
            first = _call('Schema.from_source', tables=dict(SHOP_TABLES))
            data = _shop_data(
                namespace={'schema': 'parity_shop'}
                if backend == 'postgres'
                else {}
            )
        for label, extra, backends, fails in LINK_VARIANTS:
            if backend not in backends:
                continue
            asked = _call(
                'PredictiveQuery.materialize',
                **{**options, **extra},
                random_seed=SAMPLE_SEED,
            )
            predicted = [
                _call(
                    'SDMClient.predict', model='kumo-relational', outputs=None
                ),
                _call('PredictionResult.to_pandas'),
            ]
            out.append(
                Scenario(
                    name=f'{backend}/link-prediction{label}',
                    backend=backend,
                    task_kind='link-prediction',
                    data=data,
                    calls=[
                        first,
                        _call('PredictiveQuery', query=query),
                        asked,
                        *([] if fails else predicted),
                    ],
                )
            )
    return out


#: The shop's relationships, written out, with ``orders.product_id`` also
#: linked to the supplier's ``catalog`` (``data.shop_catalog``): one column
#: with an edge to two tables.
MULTILABEL_RELATIONSHIPS = [
    {'$relationship': ['orders', 'user_id', 'users', 'user_id']},
    {'$relationship': ['orders', 'product_id', 'products', 'product_id']},
    {'$relationship': ['orders', 'product_id', 'catalog', 'sku']},
]

#: The types the decimal-key scenarios store the product ids as, on the
#: backends that have a decimal type: the listed foreign key and the key it
#: links to. Each is a backend, a label and the type; the scaled one keeps
#: two places, so every id is a decimal with a fraction of zero.
DECIMAL_KEYS = (
    ('duckdb', '', 'DECIMAL(10,0)'),
    ('duckdb', '-scaled', 'DECIMAL(12,2)'),
    ('postgres', '', 'NUMERIC(10,0)'),
)


def link_prediction_variants() -> list[Scenario]:
    r"""The link-prediction question over other schemas.

    * ``#multilabel``: ``orders.product_id`` links to ``products`` and to a
      supplier's ``catalog``, so the engine cannot tell which table the
      question ranks and refuses it as multilabel, before any payload. The
      engine refuses it before sampling anything, so the run first gathers
      the shop's count question over the same schema, which samples it: the
      question is written, a count is gathered, then the question is asked.
    * ``#decimal-keys``: the product ids are stored as decimals (DuckDB's
      ``DECIMAL``, PostgreSQL's ``NUMERIC``), both the listed foreign key and
      the key it links to.
    """
    query, options = LINK_PREDICTION
    asked = _call(
        'PredictiveQuery.materialize', **options, random_seed=SAMPLE_SEED
    )
    predicted = [
        _call('SDMClient.predict', model='kumo-relational', outputs=None),
        _call('PredictionResult.to_pandas'),
    ]
    count, count_options = QUESTIONS['count']
    tables = {**SHOP_TABLES, 'catalog': 'catalog'}
    out = []
    # In memory only: on a SQL source the sampler cannot sample a schema
    # whose column links to two tables at all; it selects the column twice
    # and fails building its key frame (base/sql_sampler.py:569), before the
    # question is asked.
    out.append(
        Scenario(
            name='memory/link-prediction#multilabel',
            backend='memory',
            task_kind='link-prediction',
            data={'generator': 'shop-catalog', 'seed': SHOP_SEED},
            calls=[
                _call(
                    'Schema.from_data',
                    tables=list(tables),
                    primary_keys={'catalog': 'sku'},
                    relationships=MULTILABEL_RELATIONSHIPS,
                ),
                _call('PredictiveQuery', query=query),
                _call('PredictiveQuery', query=count),
                _call(
                    'PredictiveQuery.materialize',
                    **count_options,
                    random_seed=SAMPLE_SEED,
                ),
                _call('PredictiveQuery', query=query),
                asked,
            ],
        )
    )
    for backend, label, column_type in DECIMAL_KEYS:
        data = _shop_data(
            namespace={'schema': 'parity_shop'}
            if backend == 'postgres'
            else {},
            column_types={
                'orders.product_id': column_type,
                'products.product_id': column_type,
            },
        )
        out.append(
            Scenario(
                name=f'{backend}/link-prediction#decimal-keys{label}',
                backend=backend,
                task_kind='link-prediction',
                data=data,
                calls=[
                    _call('Schema.from_source', tables=dict(SHOP_TABLES)),
                    _call('PredictiveQuery', query=query),
                    asked,
                    *predicted,
                ],
            )
        )
    return out


def decimal_entity_keys() -> list[Scenario]:
    r"""A question for each product when the product ids are decimals.

    DuckDB's sampler reads a ``DECIMAL`` key as a ``decimal128`` column, and
    the engine's task table refuses an entity column of that type
    (``rfm/inference/task.py``, ``add_columns``), so the regression question
    for each product ends in an error before any payload. The refusal comes
    after the targets are sampled but before the subgraph is, so the run
    first gathers the shop's count question for each user over the same
    data, which samples both. The core reads the whole decimals as the
    integers they hold (brief task 7b), so there the run goes on to predict.
    """
    count, count_options = QUESTIONS['count']
    regression, regression_options = QUESTIONS['regression']
    data = _shop_data(
        column_types={
            'orders.product_id': 'DECIMAL(10,0)',
            'products.product_id': 'DECIMAL(10,0)',
        },
    )
    return [
        Scenario(
            name='duckdb/regression#decimal-entity-keys',
            backend='duckdb',
            task_kind='regression',
            data=data,
            calls=[
                _call('Schema.from_source', tables=dict(SHOP_TABLES)),
                _call('PredictiveQuery', query=count),
                _call(
                    'PredictiveQuery.materialize',
                    **count_options,
                    random_seed=SAMPLE_SEED,
                ),
                _call('PredictiveQuery', query=regression),
                _call(
                    'PredictiveQuery.materialize',
                    **regression_options,
                    random_seed=SAMPLE_SEED,
                ),
                _call(
                    'SDMClient.predict', model='kumo-relational', outputs=None
                ),
                _call('PredictionResult.to_pandas'),
            ],
        )
    ]


#: The types the SQLite declared-key scenarios declare ``users.user_id``
#: and ``orders.user_id`` as, by label. SQLite gives each NUMERIC affinity
#: (an empty type, BLOB affinity), and the engine reads no dtype off any.
SQLITE_DECLARED_KEYS = (
    ('decimal', 'DECIMAL(10,0)'),
    ('numeric', 'NUMERIC'),
    ('untyped', ''),
)


def sqlite_declared_keys() -> list[Scenario]:
    r"""The count question for each user, on SQLite, when the user ids are
    declared a type the engine reads no dtype off (``SQLiteTable._to_dtype``),
    so a seeded draw of users looks the key's dtype up in the sampled data
    instead (``_seeded_order_by``)."""
    first = _call('Schema.from_source', tables=dict(SHOP_TABLES))
    out = []
    for label, column_type in SQLITE_DECLARED_KEYS:
        out.append(
            Scenario(
                name=f'sqlite/count#key-{label}',
                backend='sqlite',
                task_kind='count',
                data=_shop_data(
                    column_types={
                        'orders.user_id': column_type,
                        'users.user_id': column_type,
                    },
                ),
                calls=_relational_calls('count', first),
            )
        )
    return out


def decimal_mixed_indices() -> list[Scenario]:
    r"""A question for each product when the product ids are DuckDB decimals,
    asked for indices mixing an ``int`` and a ``str``.

    Each index casts to the key's ``decimal128`` alone, but the two do not
    cast together, so the engine refuses the indices without naming one
    (``rfm.py``, ``_first_uncastable``). As in
    ``duckdb/regression#decimal-entity-keys``, the run first gathers the
    count question for each user, which samples the targets and the
    subgraph.
    """
    count, count_options = QUESTIONS['count']
    regression, regression_options = QUESTIONS['regression']
    data = _shop_data(
        column_types={
            'orders.product_id': 'DECIMAL(10,0)',
            'products.product_id': 'DECIMAL(10,0)',
        },
    )
    return [
        Scenario(
            name='duckdb/regression#decimal-mixed-indices',
            backend='duckdb',
            task_kind='regression',
            data=data,
            calls=[
                _call('Schema.from_source', tables=dict(SHOP_TABLES)),
                _call('PredictiveQuery', query=count),
                _call(
                    'PredictiveQuery.materialize',
                    **count_options,
                    random_seed=SAMPLE_SEED,
                ),
                _call('PredictiveQuery', query=regression),
                _call(
                    'PredictiveQuery.materialize',
                    **{**regression_options, 'indices': [1, '2']},
                    random_seed=SAMPLE_SEED,
                ),
            ],
        )
    ]


#: The shop's relationships, written out, with each user linked to the user
#: who referred them (``data.shop_referrals``).
SELF_LINK_RELATIONSHIPS = [
    {'$relationship': ['orders', 'user_id', 'users', 'user_id']},
    {'$relationship': ['orders', 'product_id', 'products', 'product_id']},
    {'$relationship': ['users', 'referrer_id', 'users', 'user_id']},
]


def self_link() -> list[Scenario]:
    r"""A count of users for each user, over a schema linking ``users`` to
    itself.

    The link gives two edge types from ``users`` to ``users``, one each
    way, so the sampler finds no unique foreign key to aggregate the users
    over and refuses the question (``base/sampler.py``,
    ``_get_query_time_offset_dict``). The run first gathers the shop's count
    of orders for each user over the same schema, which samples the targets
    and the subgraph.
    """
    count, count_options = QUESTIONS['count']
    return [
        Scenario(
            name='memory/count#self-link',
            backend='memory',
            task_kind='count',
            data={'generator': 'shop-referrals', 'seed': SHOP_SEED},
            calls=[
                _call(
                    'Schema.from_data',
                    tables=list(SHOP_TABLES),
                    relationships=SELF_LINK_RELATIONSHIPS,
                ),
                _call('PredictiveQuery', query=count),
                _call(
                    'PredictiveQuery.materialize',
                    **count_options,
                    random_seed=SAMPLE_SEED,
                ),
                _call(
                    'PredictiveQuery',
                    query='PREDICT COUNT(users.*, 0, 30, days) '
                    'FOR EACH users.user_id',
                ),
                _call(
                    'PredictiveQuery.materialize',
                    **count_options,
                    random_seed=SAMPLE_SEED,
                ),
            ],
        )
    ]


def categorical_keys() -> list[Scenario]:
    r"""The count question for each user, in memory, when the user ids are a
    pandas ``category``, asked for indices written as text.

    The engine casts the indices to the key's dtype, which turns each text
    into a missing value without an error, so the task table is built; the
    model then finds no user it names (``backend/local/graph_store.py``,
    ``validate_entity_references``) and refuses the question. The run first
    gathers the question for indices that are ids, which samples the targets
    and the subgraph.
    """
    query, options = QUESTIONS['count']
    return [
        Scenario(
            name='memory/count#categorical-keys',
            backend='memory',
            task_kind='count',
            data={'generator': 'shop-categorical-keys', 'seed': SHOP_SEED},
            calls=[
                _call('Schema.from_data', tables=list(SHOP_TABLES)),
                _call('PredictiveQuery', query=query),
                _call(
                    'PredictiveQuery.materialize',
                    **options,
                    random_seed=SAMPLE_SEED,
                ),
                _call(
                    'PredictiveQuery.materialize',
                    **{**options, 'indices': ['1', '2']},
                    random_seed=SAMPLE_SEED,
                ),
            ],
        )
    ]


def every() -> list[Scenario]:
    return [
        *offline(),
        *session_zones(),
        *synthetic(),
        *golden(),
        *safety_net(),
        *link_prediction(),
        *link_prediction_variants(),
        *decimal_entity_keys(),
        *sqlite_declared_keys(),
        *decimal_mixed_indices(),
        *self_link(),
        *categorical_keys(),
        *live(),
    ]


def by_name() -> dict[str, Scenario]:
    return {scenario.name: scenario for scenario in every()}
