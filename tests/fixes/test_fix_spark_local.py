# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.tz-shape, end to end on a local Spark (opt in).

With ``SDM_SPARK_LOCAL=1``, pyspark 4.2 or later and a JVM: the shop of
``_warehouse_stub.py`` is written to a local Spark as parquet tables, once
with ``orders.ordered_at`` a ``TIMESTAMP`` (LTZ) and once a
``TIMESTAMP_NTZ``, holding the same instants. A notebook-like session
answers the catalog's questions from the stub and runs every other statement
on Spark, under each session zone. Both column types, under every zone,
must send the payloads the ``TIMESTAMP_NTZ`` column sends under UTC.
Skipped otherwise.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from typing import Any

import pytest

# A local Spark, end to end (opt in) #########################################


def _spark_wanted() -> None:
    if os.getenv('SDM_SPARK_LOCAL') != '1':
        pytest.skip('set SDM_SPARK_LOCAL=1 to run against a local Spark')
    pyspark = pytest.importorskip('pyspark')
    major, minor = (int(p) for p in pyspark.__version__.split('.')[:2])
    if (major, minor) < (4, 2):
        pytest.fail(
            f"pyspark {pyspark.__version__} cannot run the engine's "
            'Databricks SQL (QUALIFY); use 4.2 or later'
        )


@pytest.fixture(scope='module')
def local_spark() -> Any:
    _spark_wanted()
    import datetime as dt

    import _warehouse_stub as stub
    from pyspark.sql import SparkSession
    from pyspark.sql import types as T

    # Spark's workers run the interpreter running these checks.
    os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
    spark = (
        SparkSession.builder.master('local[1]')
        # A Databricks driver's JVM runs in UTC; only the session zone moves.
        .config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC')
        .config('spark.ui.enabled', 'false')
        .config('spark.sql.session.timeZone', 'UTC')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')
    spark_types = {
        'int64': T.LongType(),
        'double': T.DoubleType(),
        'string': T.StringType(),
    }
    views = []
    for kind in ('ltz', 'ntz'):
        for name, table in stub.shop().items():
            fields = []
            for field in table.schema:
                if field.name == 'ordered_at':
                    zoned = kind == 'ltz'
                    spark_type = (
                        T.TimestampType() if zoned else T.TimestampNTZType()
                    )
                else:
                    spark_type = spark_types[str(field.type)]
                fields.append(T.StructField(field.name, spark_type))
            rows = []
            for row in table.to_pylist():
                at = row.get('ordered_at')
                if at is not None and kind == 'ltz':
                    row['ordered_at'] = at.replace(tzinfo=dt.timezone.utc)
                rows.append(tuple(row[f.name] for f in fields))
            view = f'sdm_fix_{kind}_{name}'
            # Rows held by the plan, so no session zone reads them again.
            spark.createDataFrame(
                rows, T.StructType(fields)
            ).createOrReplaceTempView(view)
            views.append(view)
    yield spark
    for view in views:
        spark.catalog.dropTempView(view)
    spark.stop()


def _local_payloads(spark: Any, kind: str, zone: str) -> Any:
    r"""The questions over a session answering the catalog from the stub and
    reading rows from Spark, with the session zone set to ``zone``."""
    import _warehouse_stub as stub

    time_type = 'timestamp' if kind == 'ltz' else 'timestamp_ntz'
    catalog = stub.Warehouse(kind='spark', time_type=time_type)

    class SparkSession:
        version = spark.version

        def sql(self, sqlQuery: str, args: Any = None, **kwargs: Any) -> Any:
            flat = ' '.join(sqlQuery.split())
            if (
                'information_schema' in flat
                or flat.startswith('SELECT current_catalog()')
                or flat.upper().startswith(('DESCRIBE', 'SHOW'))
            ):
                return stub._Frame(catalog.answer(sqlQuery, args))
            mapped = re.sub(
                r'`main`\.`shop`\.`(\w+)`', rf'sdm_fix_{kind}_\1', sqlQuery
            )
            return spark.sql(mapped, args=args, **kwargs)

    SparkSession.__module__ = 'pyspark.local'
    from sdm_connectors import DatabricksSource

    spark.conf.set('spark.sql.session.timeZone', zone)
    try:
        source = DatabricksSource(SparkSession(), catalog='main', schema='shop')
        from sdm_client import PredictiveQuery, Schema
        from sdm_client.adapters.relational import RelationalAdapter

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            schema = Schema.from_source(source)
            out = {}
            for name, (query, options) in stub.QUESTIONS.items():
                request = PredictiveQuery(query, schema).materialize(**options)
                out[name] = [
                    stub.normalize(p)
                    for p in RelationalAdapter().prediction_plan(
                        request, ('prediction',)
                    )
                ]
        return out
    finally:
        spark.conf.set('spark.sql.session.timeZone', 'UTC')


@pytest.mark.parametrize(
    'zone', ('UTC', 'America/Los_Angeles', 'Asia/Kolkata', 'Pacific/Chatham')
)
def test_local_spark_ltz_predicts_as_ntz_under_utc(
    local_spark: Any, zone: str
) -> None:
    want = _local_payloads(local_spark, 'ntz', 'UTC')
    assert _local_payloads(local_spark, 'ltz', zone) == want
    assert _local_payloads(local_spark, 'ntz', zone) == want
