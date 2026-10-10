# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: deleted in Phase 5, with SDM_CORE_ROUTES (decision B-D7).

r"""The routing seam: which public paths run on the old code or the core.

Private, and not API. Every core path of ``scope-python.yaml`` (operation,
mode and backend) is routed ``'old'`` or ``'core'``, and every default is
``'old'`` until the Phase 3 switch, which a test holds. A test forces paths
onto the core with the ``SDM_CORE_ROUTES`` environment variable, path IDs
separated by commas, read at every call, so a change takes effect at the
next call. A forced path that is unknown, or that has no core route in this
build, fails the call with :class:`RouteError`; nothing falls back.

A call that may end on several paths, such as validation, which accepts or
refuses, routes them together: forcing some of them and not the others
fails too.

Taking the old route imports nothing native; only ``sdm_client._rust``
imports ``sdm_client._native``, and only on a core route.
"""

from __future__ import annotations

import os

#: The environment variable that forces paths onto the core, for tests.
ENVIRONMENT = 'SDM_CORE_ROUTES'

#: The core paths of ``scope-python.yaml``, in its order.
PATHS = (
    'query.validate/accept/any',
    'query.validate/refuse/any',
    'schema.from-source/tables/duckdb',
    'schema.from-source/catalog-keys/duckdb',
    'schema.from-source/all-tables/duckdb',
    'schema.from-source/tables/sqlite',
    'schema.from-source/catalog-keys/sqlite',
    'schema.from-source/all-tables/sqlite',
    'schema.from-source/tables/postgres',
    'schema.from-source/catalog-keys/postgres',
    'schema.from-source/all-tables/postgres',
    'schema.from-source/tables/lakebase',
    'schema.from-source/catalog-keys/lakebase',
    'schema.from-source/all-tables/lakebase',
    'schema.from-source/tables/databricks-sql',
    'schema.from-source/catalog-keys/databricks-sql',
    'schema.from-source/all-tables/databricks-sql',
    'schema.from-source/tables/databricks-spark',
    'schema.from-source/catalog-keys/databricks-spark',
    'schema.from-source/all-tables/databricks-spark',
    'schema.from-source/tables/snowflake',
    'schema.from-source/catalog-keys/snowflake',
    'schema.from-source/all-tables/snowflake',
    'schema.from-source/semantic-model/databricks-metric-view',
    'schema.from-source/semantic-model/snowflake-semantic-view',
    'schema.from-data/frames/frames',
    'schema.from-data/mapping/frames',
    'schema.construct/source-tables/any',
    'schema.construct/frame-tables/frames',
    'schema.edit/set-time-column/any',
    'schema.edit/set-primary-key/any',
    'schema.edit/set-stype/any',
    'schema.edit/drop-column/any',
    'schema.edit/edit-relationships/any',
    'schema.edit/snapshot/any',
    'schema.inspect/attributes/any',
    'schema.inspect/check-consistent/any',
    'state.pickle/schema/any',
    'state.pickle/request/any',
    'state.pickle/reference-pickle/any',
    'query.materialize/indices-binary/duckdb',
    'query.materialize/indices-multiclass/duckdb',
    'query.materialize/indices-regression/duckdb',
    'query.materialize/indices-link-prediction/duckdb',
    'query.materialize/indices-ranking/duckdb',
    'query.materialize/indices-forecast/duckdb',
    'query.materialize/indices-binary/sqlite',
    'query.materialize/indices-multiclass/sqlite',
    'query.materialize/indices-regression/sqlite',
    'query.materialize/indices-link-prediction/sqlite',
    'query.materialize/indices-ranking/sqlite',
    'query.materialize/indices-forecast/sqlite',
    'query.materialize/indices-binary/postgres',
    'query.materialize/indices-multiclass/postgres',
    'query.materialize/indices-regression/postgres',
    'query.materialize/indices-link-prediction/postgres',
    'query.materialize/indices-ranking/postgres',
    'query.materialize/indices-forecast/postgres',
    'query.materialize/indices-binary/lakebase',
    'query.materialize/indices-multiclass/lakebase',
    'query.materialize/indices-regression/lakebase',
    'query.materialize/indices-link-prediction/lakebase',
    'query.materialize/indices-ranking/lakebase',
    'query.materialize/indices-forecast/lakebase',
    'query.materialize/indices-binary/databricks-sql',
    'query.materialize/indices-multiclass/databricks-sql',
    'query.materialize/indices-regression/databricks-sql',
    'query.materialize/indices-link-prediction/databricks-sql',
    'query.materialize/indices-ranking/databricks-sql',
    'query.materialize/indices-forecast/databricks-sql',
    'query.materialize/indices-binary/databricks-spark',
    'query.materialize/indices-multiclass/databricks-spark',
    'query.materialize/indices-regression/databricks-spark',
    'query.materialize/indices-link-prediction/databricks-spark',
    'query.materialize/indices-ranking/databricks-spark',
    'query.materialize/indices-forecast/databricks-spark',
    'query.materialize/indices-binary/snowflake',
    'query.materialize/indices-multiclass/snowflake',
    'query.materialize/indices-regression/snowflake',
    'query.materialize/indices-link-prediction/snowflake',
    'query.materialize/indices-ranking/snowflake',
    'query.materialize/indices-forecast/snowflake',
    'query.materialize/indices-binary/databricks-metric-view',
    'query.materialize/indices-multiclass/databricks-metric-view',
    'query.materialize/indices-regression/databricks-metric-view',
    'query.materialize/indices-link-prediction/databricks-metric-view',
    'query.materialize/indices-ranking/databricks-metric-view',
    'query.materialize/indices-forecast/databricks-metric-view',
    'query.materialize/indices-binary/snowflake-semantic-view',
    'query.materialize/indices-multiclass/snowflake-semantic-view',
    'query.materialize/indices-regression/snowflake-semantic-view',
    'query.materialize/indices-link-prediction/snowflake-semantic-view',
    'query.materialize/indices-ranking/snowflake-semantic-view',
    'query.materialize/indices-forecast/snowflake-semantic-view',
    'query.materialize/indices-binary/frames',
    'query.materialize/indices-multiclass/frames',
    'query.materialize/indices-regression/frames',
    'query.materialize/indices-link-prediction/frames',
    'query.materialize/indices-ranking/frames',
    'query.materialize/indices-forecast/frames',
    'query.materialize/predict-frame/frames',
    'request.tabular/classification/frames',
    'request.tabular/regression/frames',
    'request.from-payloads/payloads/any',
    'request.inspect/attributes/any',
    'client.capabilities/capabilities/any',
    'client.predict/binary/duckdb',
    'client.predict/multiclass/duckdb',
    'client.predict/regression/duckdb',
    'client.predict/link-prediction/duckdb',
    'client.predict/ranking/duckdb',
    'client.predict/forecast/duckdb',
    'client.predict/binary/sqlite',
    'client.predict/multiclass/sqlite',
    'client.predict/regression/sqlite',
    'client.predict/link-prediction/sqlite',
    'client.predict/ranking/sqlite',
    'client.predict/forecast/sqlite',
    'client.predict/binary/postgres',
    'client.predict/multiclass/postgres',
    'client.predict/regression/postgres',
    'client.predict/link-prediction/postgres',
    'client.predict/ranking/postgres',
    'client.predict/forecast/postgres',
    'client.predict/binary/lakebase',
    'client.predict/multiclass/lakebase',
    'client.predict/regression/lakebase',
    'client.predict/link-prediction/lakebase',
    'client.predict/ranking/lakebase',
    'client.predict/forecast/lakebase',
    'client.predict/binary/databricks-sql',
    'client.predict/multiclass/databricks-sql',
    'client.predict/regression/databricks-sql',
    'client.predict/link-prediction/databricks-sql',
    'client.predict/ranking/databricks-sql',
    'client.predict/forecast/databricks-sql',
    'client.predict/binary/databricks-spark',
    'client.predict/multiclass/databricks-spark',
    'client.predict/regression/databricks-spark',
    'client.predict/link-prediction/databricks-spark',
    'client.predict/ranking/databricks-spark',
    'client.predict/forecast/databricks-spark',
    'client.predict/binary/snowflake',
    'client.predict/multiclass/snowflake',
    'client.predict/regression/snowflake',
    'client.predict/link-prediction/snowflake',
    'client.predict/ranking/snowflake',
    'client.predict/forecast/snowflake',
    'client.predict/binary/databricks-metric-view',
    'client.predict/multiclass/databricks-metric-view',
    'client.predict/regression/databricks-metric-view',
    'client.predict/link-prediction/databricks-metric-view',
    'client.predict/ranking/databricks-metric-view',
    'client.predict/forecast/databricks-metric-view',
    'client.predict/binary/snowflake-semantic-view',
    'client.predict/multiclass/snowflake-semantic-view',
    'client.predict/regression/snowflake-semantic-view',
    'client.predict/link-prediction/snowflake-semantic-view',
    'client.predict/ranking/snowflake-semantic-view',
    'client.predict/forecast/snowflake-semantic-view',
    'client.predict/binary/frames',
    'client.predict/multiclass/frames',
    'client.predict/regression/frames',
    'client.predict/link-prediction/frames',
    'client.predict/ranking/frames',
    'client.predict/forecast/frames',
    'client.predict/tabular/frames',
    'client.fit/binary/duckdb',
    'client.fit/multiclass/duckdb',
    'client.fit/regression/duckdb',
    'client.fit/binary/sqlite',
    'client.fit/multiclass/sqlite',
    'client.fit/regression/sqlite',
    'client.fit/binary/postgres',
    'client.fit/multiclass/postgres',
    'client.fit/regression/postgres',
    'client.fit/binary/lakebase',
    'client.fit/multiclass/lakebase',
    'client.fit/regression/lakebase',
    'client.fit/binary/databricks-sql',
    'client.fit/multiclass/databricks-sql',
    'client.fit/regression/databricks-sql',
    'client.fit/binary/databricks-spark',
    'client.fit/multiclass/databricks-spark',
    'client.fit/regression/databricks-spark',
    'client.fit/binary/snowflake',
    'client.fit/multiclass/snowflake',
    'client.fit/regression/snowflake',
    'client.fit/binary/databricks-metric-view',
    'client.fit/multiclass/databricks-metric-view',
    'client.fit/regression/databricks-metric-view',
    'client.fit/binary/snowflake-semantic-view',
    'client.fit/multiclass/snowflake-semantic-view',
    'client.fit/regression/snowflake-semantic-view',
    'client.fit/binary/frames',
    'client.fit/multiclass/frames',
    'client.fit/regression/frames',
    'client.fit/tabular/frames',
    'fitted.predict/binary/duckdb',
    'fitted.predict/multiclass/duckdb',
    'fitted.predict/regression/duckdb',
    'fitted.predict/binary/sqlite',
    'fitted.predict/multiclass/sqlite',
    'fitted.predict/regression/sqlite',
    'fitted.predict/binary/postgres',
    'fitted.predict/multiclass/postgres',
    'fitted.predict/regression/postgres',
    'fitted.predict/binary/lakebase',
    'fitted.predict/multiclass/lakebase',
    'fitted.predict/regression/lakebase',
    'fitted.predict/binary/databricks-sql',
    'fitted.predict/multiclass/databricks-sql',
    'fitted.predict/regression/databricks-sql',
    'fitted.predict/binary/databricks-spark',
    'fitted.predict/multiclass/databricks-spark',
    'fitted.predict/regression/databricks-spark',
    'fitted.predict/binary/snowflake',
    'fitted.predict/multiclass/snowflake',
    'fitted.predict/regression/snowflake',
    'fitted.predict/binary/databricks-metric-view',
    'fitted.predict/multiclass/databricks-metric-view',
    'fitted.predict/regression/databricks-metric-view',
    'fitted.predict/binary/snowflake-semantic-view',
    'fitted.predict/multiclass/snowflake-semantic-view',
    'fitted.predict/regression/snowflake-semantic-view',
    'fitted.predict/binary/frames',
    'fitted.predict/multiclass/frames',
    'fitted.predict/regression/frames',
    'fitted.predict/tabular/frames',
    'result.inspect/absent/any',
    'result.inspect/explanation/any',
    'result.inspect/unscored-candidates/any',
    'result.merge/merge/any',
)

#: Where each path runs by default: the old code, until Phase 3.
ROUTES: dict[str, str] = dict.fromkeys(PATHS, 'old')

#: S0: validation in ``PredictiveQuery.materialize``, accepting or refusing.
S0 = ('query.validate/accept/any', 'query.validate/refuse/any')

#: ``Schema.from_source`` over a DuckDB source with no tables named: every
#: table, and the keys its catalog declares, routed together (T1.11b).
FROM_SOURCE_DUCKDB = (
    'schema.from-source/all-tables/duckdb',
    'schema.from-source/catalog-keys/duckdb',
)

#: ``Schema.from_source`` over a DuckDB source with tables named, which has
#: no core route yet: forcing the keys path alone fails such a call.
FROM_SOURCE_DUCKDB_TABLES = (
    'schema.from-source/tables/duckdb',
    'schema.from-source/catalog-keys/duckdb',
)

#: ``PredictiveQuery.materialize`` with ``indices`` over a schema of a DuckDB
#: source: one call ends on the path of the task kind the query resolves to,
#: so its six paths are routed together (T1.11c). The core materializes a
#: regression and refuses the other kinds by name (register entry P1B-3).
MATERIALIZE_DUCKDB = (
    'query.materialize/indices-binary/duckdb',
    'query.materialize/indices-multiclass/duckdb',
    'query.materialize/indices-regression/duckdb',
    'query.materialize/indices-link-prediction/duckdb',
    'query.materialize/indices-ranking/duckdb',
    'query.materialize/indices-forecast/duckdb',
)

#: The paths this build can run on the core.
CORE_ROUTED = frozenset(S0 + FROM_SOURCE_DUCKDB + MATERIALIZE_DUCKDB)


class RouteError(RuntimeError):
    r"""A path cannot take the route it was given.

    Raised for an unknown forced path, a path with no core route, a call
    whose paths were forced apart, or a core route without
    ``sdm_client._native``. It is never a :class:`StructuredDataError`:
    a runner counts it as its own failure, not as the call's outcome.
    """


def forced() -> frozenset[str]:
    r"""The paths ``SDM_CORE_ROUTES`` forces onto the core, checked."""
    setting = os.environ.get(ENVIRONMENT, '')
    named = {part.strip() for part in setting.split(',')} - {''}
    for path in sorted(named):
        if path not in ROUTES:
            raise RouteError(f'{ENVIRONMENT} names an unknown path {path!r}')
        if path not in CORE_ROUTED:
            raise RouteError(
                f'{ENVIRONMENT} forces {path!r} onto the core, which has no '
                f'route for it in this build'
            )
    return frozenset(named)


def route(paths: tuple[str, ...]) -> str:
    r"""``'old'`` or ``'core'``: where a call that may end on ``paths`` runs.

    Raises:
        RouteError: ``SDM_CORE_ROUTES`` names an unknown path or one with
            no core route, or forces some of ``paths`` and not the others.
    """
    onto = forced()
    taken = {'core' if path in onto else ROUTES[path] for path in paths}
    if len(taken) != 1:
        raise RouteError(
            f'{ENVIRONMENT} forces some of {list(paths)} onto the core and '
            f'not the others; one call runs them on one route'
        )
    return taken.pop()
