# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 7 labels checks share: the seeded generators of P7.labels-gen,
today's executor's answers, the Rust core's through ``labels_op`` and
``labels_execute``, and the comparison.

The interface these rely on is ``../sampling/CONTRACT.md``, section Labels.
"""

from __future__ import annotations

import json
import math
import random
import struct
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PQL_CORPUS = REPO / 'tests' / 'parity' / 'pql' / 'valid.json'

for _path in (
    HERE,
    HERE.parent / 'sampling',
    HERE.parent,
    REPO / 'tests' / 'parity',
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _sampling import as_list, error_of, loads, native, short  # noqa: E402

LABELS_SEED = 20261015
NUM_GROUPS = 100_000
AGGREGATIONS = ('SUM', 'AVG', 'MIN', 'MAX', 'COUNT', 'LIST_DISTINCT')

F32_MAX = 3.4028234663852886e38


def f32(value: float) -> float:
    r"""``value`` rounded to the nearest ``float32``, as a Python float."""
    return struct.unpack('<f', struct.pack('<f', value))[0]


def _f32_bits(value: float) -> int:
    if math.isnan(value):
        return 0x7FC00000
    return struct.unpack('<I', struct.pack('<f', value))[0]


# --- Generated aggregation groups ----------------------------------------------


def _float(rng: random.Random, single: bool) -> Any:
    r"""A float: ordinary, large, tiny, signed zero, NaN or an infinity."""
    kind = rng.random()
    if kind < 0.04:
        return {'$float': 'nan'}
    if kind < 0.06:
        return {'$float': rng.choice(('inf', '-inf'))}
    if kind < 0.08:
        return rng.choice((0.0, -0.0))
    if kind < 0.12:
        value = rng.choice(
            (F32_MAX, -F32_MAX, 1.4e-45, -1.4e-45, 1.1754944e-38)
        )
    elif kind < 0.6:
        value = rng.uniform(-1000, 1000)
    elif kind < 0.8:
        value = rng.gauss(0, 1) * 10 ** rng.randint(-8, 8)
    else:
        value = rng.gauss(0, 1) * 10 ** rng.randint(-45, 38)
        if not single and rng.random() < 0.3:
            value = rng.gauss(0, 1) * 10 ** rng.randint(-320, 307)
    if single:
        value = (
            f32(value)
            if abs(value) <= F32_MAX
            else math.copysign(F32_MAX, value)
        )
    return value


def _int(rng: random.Random, edge: bool) -> int:
    if not edge:
        return rng.randint(-1000, 1000)
    if rng.random() < 0.5:
        return 2**63 - 1 - rng.randint(0, 2**20)
    return -(2**63) + rng.randint(0, 2**20)


def _timestamp(rng: random.Random) -> dict[str, str]:
    import pandas as pd

    ns = pd.Timestamp('2024-01-01').value + rng.randint(-(10**17), 10**17)
    return {'$timestamp': pd.Timestamp(ns).isoformat()}


FEATURE_KINDS = (
    'float32',
    'float32',
    'float64',
    'int64',
    'int64-edge',
    'Int64',
    'Float64',
    'datetime64[ns]',
)


def feature_series(rng: random.Random, kind: str, size: int) -> dict[str, Any]:
    r"""A feature column of ``kind`` as fixture JSON."""
    if kind == 'float32':
        return {
            'dtype': 'float32',
            'values': [_float(rng, True) for _ in range(size)],
        }
    if kind == 'float64':
        return {
            'dtype': 'float64',
            'values': [_float(rng, False) for _ in range(size)],
        }
    if kind in ('int64', 'int64-edge'):
        edge = kind == 'int64-edge'
        return {
            'dtype': 'int64',
            'values': [
                _int(rng, edge and rng.random() < 0.8) for _ in range(size)
            ],
        }
    if kind == 'Int64':
        return {
            'dtype': 'Int64',
            'values': [
                {'$na': None}
                if rng.random() < 0.15
                else _int(rng, rng.random() < 0.3)
                for _ in range(size)
            ],
        }
    if kind == 'Float64':
        return {
            'dtype': 'Float64',
            'values': [
                {'$na': None} if rng.random() < 0.15 else _float(rng, False)
                for _ in range(size)
            ],
        }
    return {
        'dtype': 'datetime64[ns]',
        'values': [
            {'$nat': None} if rng.random() < 0.1 else _timestamp(rng)
            for _ in range(size)
        ],
    }


def aggregation_type_inputs() -> list[dict[str, Any]]:
    r"""``labels.aggregation_type`` inputs: at least ``NUM_GROUPS`` groups
    in all, of every aggregation and feature kind, some with thousands of
    rows, some empty."""
    rng = random.Random(LABELS_SEED)
    out = []
    groups = 0
    while groups < NUM_GROUPS:
        kind = rng.choice(FEATURE_KINDS)
        if kind == 'datetime64[ns]':
            op = rng.choice(('MIN', 'MAX', 'COUNT', 'LIST_DISTINCT'))
        else:
            op = rng.choice(AGGREGATIONS)
        batch_size = rng.randint(1, 100)
        rows = rng.choice((0, 1, 5, 50, 300, rng.randint(0, 3000)))
        if op == 'LIST_DISTINCT':
            rows = min(rows, 300)
        out.append(
            {
                'op': op,
                'feat': feature_series(rng, kind, rows),
                'batch': [rng.randrange(batch_size) for _ in range(rows)],
                'batch_size': batch_size,
                'filter_na': rng.random() < 0.5,
            }
        )
        groups += batch_size
    return out


UNITS = ('seconds', 'minutes', 'hours', 'days', 'weeks', 'months')


def aggregation_inputs(count: int = 3000) -> list[dict[str, Any]]:
    r"""``labels.aggregation`` inputs: a target column, its times and
    batches, anchors, and a window of any unit, times on and around the
    window's edges and month ends."""
    import pandas as pd

    rng = random.Random(LABELS_SEED + 1)
    out = []
    for _ in range(count):
        kind = rng.choice(FEATURE_KINDS[:-1])
        op = rng.choice(AGGREGATIONS)
        batch_size = rng.randint(1, 30)
        rows = rng.randint(0, 120)
        unit = rng.choice(UNITS)
        start = rng.choice((0, 0, 0, rng.randint(-10, 10)))
        end = start + rng.randint(1, 40)
        anchors = []
        for _ in range(batch_size):
            day = pd.Timestamp(
                rng.choice(
                    ('2024-01-31', '2024-02-29', '2023-03-31', '2024-06-15')
                )
            )
            second = rng.choice((0, 0, rng.randint(0, 86_399)))
            anchors.append(day.value + second * 10**9)
        times = []
        batches = []
        for _ in range(rows):
            batch = rng.randrange(batch_size)
            step = {
                'seconds': 1,
                'minutes': 60,
                'hours': 3600,
                'days': 86_400,
                'weeks': 7 * 86_400,
                'months': 30 * 86_400,
            }[unit]
            delta = rng.randint(start - 2, end + 2) * step + rng.choice(
                (0, 0, 0, -1, 1, rng.randint(-step, step))
            )
            if unit == 'months' and rng.random() < 0.5:
                moved = pd.Timestamp(anchors[batch]) + pd.DateOffset(
                    months=rng.randint(start - 1, end + 1)
                )
                times.append(moved.value + rng.choice((0, -1, 1)) * 10**9)
            else:
                times.append(anchors[batch] + delta * 10**9)
            batches.append(batch)
        time_unit = rng.choice(('ns', 'us'))
        out.append(
            {
                'aggregation': {
                    'target': 'fact.v',
                    'aggr': op,
                    'start': start,
                    'end': end,
                    'unit': unit,
                },
                'tables': {
                    'fact': {'columns': {'v': feature_series(rng, kind, rows)}}
                },
                'times': {
                    'fact': {
                        'dtype': f'datetime64[{time_unit}]',
                        'values': [
                            {'$timestamp': pd.Timestamp(t).isoformat()}
                            for t in times
                        ],
                    }
                },
                'batches': {'fact': batches},
                'anchor_time': {
                    'dtype': f'datetime64[{time_unit}]',
                    'values': [
                        {'$timestamp': pd.Timestamp(t).isoformat()}
                        for t in anchors
                    ],
                },
                'filter_na': rng.random() < 0.5,
            }
        )
    return out


def groups_of(inputs: list[dict[str, Any]]) -> int:
    return sum(
        i.get('batch_size', len(i.get('anchor_time', {}).get('values', [])))
        for i in inputs
    )


def _aggregation_type(given: dict[str, Any]) -> Any:
    import numpy as np
    from _fixture_ops import build_series, series_out
    from predictive_query.schema.typing import AggregationType
    from sdm_relational_engine.rfm.query.pandas_executor import (
        PQueryPandasExecutor,
    )

    out, mask = PQueryPandasExecutor().execute_aggregation_type(
        op=AggregationType(given['op']),
        feat=build_series(given['feat']),
        batch=np.array(given['batch'], dtype='int64'),
        batch_size=given['batch_size'],
        filter_na=given['filter_na'],
    )
    return {'out': series_out(out), 'mask': [bool(m) for m in mask]}


def _aggregation(given: dict[str, Any]) -> Any:
    import numpy as np
    from _fixture_ops import build_frame, build_series, series_out
    from predictive_query.ast import Aggregation, Column, DateOffsetRange
    from predictive_query.schema.typing import AggregationType, TimeUnit
    from sdm_relational_engine.rfm.query.pandas_executor import (
        PQueryPandasExecutor,
    )

    spec = given['aggregation']
    aggr = Aggregation(
        target=Column(fqn=spec['target']),
        aggr=AggregationType(spec['aggr']),
        aggr_time_range=DateOffsetRange(
            spec['start'], spec['end'], TimeUnit(spec['unit'])
        ),
    )
    out, mask = PQueryPandasExecutor().execute_aggregation(
        aggr=aggr,
        feat_dict={n: build_frame(f) for n, f in given['tables'].items()},
        time_dict={n: build_series(s) for n, s in given['times'].items()},
        batch_dict={
            n: np.array(b, dtype='int64') for n, b in given['batches'].items()
        },
        anchor_time=build_series(given['anchor_time']),
        filter_na=given['filter_na'],
    )
    return {'out': series_out(out), 'mask': [bool(m) for m in mask]}


#: Today's answers to the generated operations, from the executor itself
#: rather than the fixture runner, which this phase extends.
_REFERENCE = {
    'labels.aggregation_type': _aggregation_type,
    'labels.aggregation': _aggregation,
}


def python_op(operation: str, given: dict[str, Any]) -> dict[str, Any]:
    r"""Today's answer, as ``{"output"}`` or, where the executor raises,
    ``{"error"}`` with the error's ``code`` where it has one, otherwise its
    class name, as the fixtures write errors."""
    try:
        return {'output': json.loads(json.dumps(_REFERENCE[operation](given)))}
    except Exception as error:  # noqa: BLE001 - every refusal is an outcome
        code = getattr(error, 'code', None)
        if not isinstance(code, str) or not code:
            code = type(error).__name__
        message = getattr(error, 'message', None)
        if not isinstance(message, str):
            message = str(error)
        return {'error': {'code': code, 'message': message}}


def rust_op(operation: str, given: dict[str, Any]) -> Any:
    module = native('labels_op')
    return loads(module.labels_op(operation, json.dumps(given)), 'labels_op')


def same_answer(want: dict[str, Any], got: Any) -> str | None:
    r"""How the core's answer differs from today's: outputs by value and
    JSON type (every NaN is ``{"$float": "nan"}``), errors by code and
    message."""
    from _fixture_ops import same

    if (
        not isinstance(got, dict)
        or len(got) != 1
        or not ({'output', 'error'} & set(got))
    ):
        return f'gave {short(got)}, not one of output or error'
    if 'output' in want:
        if 'output' not in got:
            return f'refused with {short(got["error"])}, today gives {short(want["output"])}'
        if not same(want['output'], got['output']):
            return f'gave {short(got["output"])}, today gives {short(want["output"])}'
        return None
    if got.get('error') != want['error']:
        return f'gave {short(got)}, today refuses with {short(want["error"])}'
    return None


# --- Generated queries ----------------------------------------------------------

#: The schema of the PQL corpus's shop, as ``tests/parity/pql/valid.json``
#: writes it, and the templates the generated questions are made of.
TEMPLATES = (
    'PREDICT {agg}(orders.{num}, {start}, {end}, {unit}) FOR EACH users.user_id',
    'PREDICT {agg}(orders.{num}, {start}, {end}, {unit}) FOR EACH products.product_id',
    'PREDICT COUNT(orders.*, {start}, {end}, {unit}) {op} {const} FOR EACH users.user_id',
    'PREDICT {agg}(orders.{num}, {start}, {end}, {unit}) {op} {const} FOR EACH users.user_id',
    'PREDICT COUNT(orders.* WHERE orders.status {op} {status}, {start}, {end}, {unit}) FOR EACH users.user_id',
    'PREDICT {agg}(orders.{num} WHERE orders.{num} {op} {const}, {start}, {end}, {unit}) FOR EACH users.user_id',
    'PREDICT COUNT(orders.*, {start}, {end}, {unit}) {op} {const} AND {agg}(orders.{num}, {start}, {end}, {unit}) {op} {const} FOR EACH users.user_id',
    'PREDICT COUNT(orders.*, {start}, {end}, {unit}) = 0 OR COUNT(reviews.*, {start}, {end}, {unit}) = 0 FOR EACH users.user_id',
    'PREDICT NOT COUNT(reviews.*, {start}, {end}, {unit}) {op} {const} FOR EACH users.user_id',
    'PREDICT {agg}(reviews.rating, {start}, {end}, {unit}) FOR EACH users.user_id WHERE users.age {op} {const}',
    'PREDICT COUNT(orders.*, {start}, {end}, {unit}) FOR EACH users.user_id WHERE users.country = {country}',
    'PREDICT COUNT(orders.*, {start}, {end}, {unit}) FOR EACH users.user_id ASSUMING COUNT(reviews.*, {start}, {end}, {unit}) {op} {const}',
    'PREDICT {agg}(orders.{num}, {start}, {end}, {unit}) IS NULL FOR EACH users.user_id',
    'PREDICT LIST_DISTINCT(orders.product_id, {start}, {end}, {unit}) RANK TOP 3 FOR EACH users.user_id',
    'PREDICT users.{ucol} FOR EACH users.user_id',
    'PREDICT users.age {op} {const} FOR EACH users.user_id',
    'PREDICT users.country = {country} FOR EACH users.user_id',
    'PREDICT products.price FOR EACH products.product_id WHERE products.category = {country}',
    'PREDICT products.tags FOR EACH products.product_id',
)


def corpus_cases() -> list[dict[str, Any]]:
    r"""Every relational question of the PQL corpus that resolves."""
    data = json.loads(PQL_CORPUS.read_text(encoding='utf-8'))
    return [
        c
        for c in data['cases']
        if c.get('profile', 'RELATIONAL') == 'RELATIONAL'
    ]


def shop_schema() -> dict[str, Any]:
    for case in corpus_cases():
        if set(case['schema']['tables']) == {
            'users',
            'products',
            'orders',
            'reviews',
        }:
            return case['schema']
    raise AssertionError('the PQL corpus has no shop schema')


def generated_questions(count: int) -> list[str]:
    rng = random.Random(LABELS_SEED + 2)
    out = []
    while len(out) < count:
        template = rng.choice(TEMPLATES)
        start = rng.choice((0, 0, 0, rng.randint(-10, 10)))
        out.append(
            template.format(
                agg=rng.choice(('SUM', 'AVG', 'MIN', 'MAX', 'COUNT')),
                num=rng.choice(('quantity', 'amount')),
                start=start,
                end=start + rng.randint(1, 40),
                unit=rng.choice(
                    ('days', 'hours', 'weeks', 'months', 'minutes', 'seconds')
                ),
                op=rng.choice(('>', '<', '>=', '<=', '=', '!=')),
                const=rng.choice(('0', '1', '2', '2.5', '-3', '100', '0.1')),
                status=rng.choice(("'shipped'", "'returned'", 'NULL')),
                country=rng.choice(("'DE'", "'US'")),
                ucol=rng.choice(('age', 'country', 'vip')),
            )
        )
    return out


def resolve(query: str, schema: dict[str, Any]) -> Any:
    r"""Today's ``ValidatedPredictiveQuery`` for ``query``, or ``None``."""
    import warnings

    import predictive_query
    from predictive_query.schema import json_serde
    from predictive_query.schema.definition import SchemaDefinition

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        resolved, _ = predictive_query.validate(
            query,
            json_serde.from_json(schema, SchemaDefinition),
            predictive_query.RELATIONAL,
        )
    return resolved


def _column_values(
    rng: random.Random, column: dict[str, Any], size: int
) -> Any:
    import numpy as np
    import pandas as pd

    dtype, name = column['dtype'], column['name']
    if dtype == 'int' and name.endswith('_id'):
        return pd.Series(
            [rng.randint(0, 6) for _ in range(size)], dtype='int64'
        )
    if dtype in ('int', 'float'):
        kind = rng.choice(('float32', 'float64', 'int64', 'Int64'))
        if kind in ('float32', 'float64'):
            values = []
            for _ in range(size):
                r = rng.random()
                if r < 0.1:
                    values.append(np.nan)
                elif r < 0.13:
                    values.append(rng.choice((np.inf, -np.inf)))
                elif r < 0.8:
                    values.append(
                        rng.choice((0.0, 1.0, 2.0, 2.5, 3.0, 100.0, -3.0, 0.1))
                    )
                else:
                    values.append(rng.uniform(-200, 200))
            return pd.Series(values, dtype=kind)
        if kind == 'int64':
            return pd.Series(
                [_int(rng, rng.random() < 0.05) for _ in range(size)],
                dtype='int64',
            )
        return pd.Series(
            [
                pd.NA if rng.random() < 0.1 else rng.randint(-5, 120)
                for _ in range(size)
            ],
            dtype='Int64',
        )
    if dtype == 'string':
        pool = ('shipped', 'returned', 'delivered', 'DE', 'US', 'x', None)
        return pd.Series([rng.choice(pool) for _ in range(size)], dtype=object)
    if dtype == 'bool':
        if rng.random() < 0.5:
            return pd.Series(
                [rng.random() < 0.5 for _ in range(size)], dtype='bool'
            )
        return pd.Series(
            [
                pd.NA if rng.random() < 0.1 else rng.random() < 0.5
                for _ in range(size)
            ],
            dtype='boolean',
        )
    if dtype == 'time':
        base = pd.Timestamp('2024-02-29').value
        return pd.Series(
            [
                pd.NaT
                if rng.random() < 0.03
                else pd.Timestamp(base + rng.randint(-90, 90) * 86_400 * 10**9)
                for _ in range(size)
            ],
            dtype='datetime64[ns]',
        )
    if dtype == 'stringlist':
        cells = np.empty(size, dtype=object)
        for i in range(size):
            cells[i] = (
                None
                if rng.random() < 0.1
                else [rng.choice('abc') for _ in range(rng.randint(0, 3))]
            )
        return pd.Series(cells, dtype=object)
    raise AssertionError(f'no generator for dtype {dtype}')


def sample_data(
    rng: random.Random, schema: dict[str, Any], entity_table: str
) -> dict[str, Any]:
    r"""What the sampler hands the executor for one batch: frames, the
    times of tables with a time column, each row's example, and anchors."""
    import numpy as np
    import pandas as pd

    size = rng.randint(1, 40)
    anchor_base = pd.Timestamp(
        rng.choice(('2024-02-29', '2024-01-31', '2023-12-31'))
    ).value
    anchors = pd.Series(
        [
            anchor_base
            + rng.choice((0, 0, rng.randint(-30, 30) * 86_400)) * 10**9
            for _ in range(size)
        ],
        dtype='int64',
    ).astype('datetime64[ns]')
    feat, times, batches = {}, {}, {}
    for name, table in schema['tables'].items():
        rows = size if name == entity_table else rng.randint(0, 200)
        frame = pd.DataFrame(
            {
                column['name']: _column_values(rng, column, rows)
                for column in table['cols']
            }
        )
        feat[name] = frame
        if name == entity_table:
            batches[name] = np.arange(size, dtype=np.int64)
        else:
            batches[name] = np.array(
                [rng.randrange(size) for _ in range(rows)], dtype=np.int64
            )
        time_col = table.get('time_col')
        if time_col:
            if name != entity_table and rows:
                near = anchors.to_numpy()[batches[name]].astype('int64')
                moved = near + np.array(
                    [
                        rng.choice((0, 1, -1, 86_400, -86_400)) * 10**9
                        + rng.randint(-45, 45) * 86_400 * 10**9
                        for _ in range(rows)
                    ],
                    dtype=np.int64,
                )
                frame[time_col] = pd.Series(moved).astype('datetime64[ns]')
            times[name] = frame[time_col]
    return {'feat': feat, 'times': times, 'batches': batches, 'anchor': anchors}


def python_execute(query: Any, data: dict[str, Any]) -> dict[str, Any]:
    from sdm_relational_engine.rfm.query.pandas_executor import (
        PQueryPandasExecutor,
    )

    try:
        out, mask = PQueryPandasExecutor().execute_query(
            query=query,
            feat_dict=data['feat'],
            time_dict=data['times'],
            batch_dict=data['batches'],
            anchor_time=data['anchor'],
        )
    except Exception as error:  # noqa: BLE001 - compared, not judged
        return {'error': error_of(error)}
    return {
        'dtype': str(out.dtype),
        'values': [plain(v) for v in out.tolist()],
        'mask': [bool(m) for m in mask],
    }


def arrow_args(data: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    import pyarrow as pa

    feat = {
        k: pa.Table.from_pandas(v, preserve_index=False)
        for k, v in data['feat'].items()
    }
    times = {k: pa.Array.from_pandas(v) for k, v in data['times'].items()}
    batches = {
        k: pa.array(v, type=pa.int64()) for k, v in data['batches'].items()
    }
    anchor = pa.Array.from_pandas(data['anchor'])
    return feat, times, batches, anchor


def plain(value: Any) -> Any:
    r"""A label as comparable data: NaN as ``"nan"``, a timestamp as its
    nanoseconds, a list item by item.

    A value can only be a pandas object if pandas is already imported, so a
    process that blocks pandas (``_jobs.py no-engine``) still gets answers."""
    pd = sys.modules.get('pandas')

    if value is None:
        return None
    if pd is not None:
        if value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, pd.Timestamp):
            return ['ts', int(value.value)]
    if hasattr(value, 'item') and not isinstance(value, (list, str)):
        return plain(value.item())
    if isinstance(value, float):
        return 'nan' if math.isnan(value) else value
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return [plain(v) for v in value.tolist()]
    return value


#: The Arrow types a label of each pandas dtype may come back as.
def arrow_type_fits(dtype: str, arrow_type: Any) -> bool:
    import pyarrow as pa

    if dtype == 'float32':
        return arrow_type == pa.float32()
    if dtype == 'float64':
        return arrow_type == pa.float64()
    if dtype == 'int64':
        return arrow_type == pa.int64()
    if dtype == 'bool':
        return arrow_type == pa.bool_()
    if dtype.startswith('datetime64['):
        return pa.types.is_timestamp(arrow_type) and arrow_type.tz is None
    if dtype in ('str', 'string', 'object'):
        return (
            pa.types.is_string(arrow_type)
            or pa.types.is_large_string(arrow_type)
            or pa.types.is_list(arrow_type)
            or pa.types.is_large_list(arrow_type)
            or pa.types.is_null(arrow_type)
        )
    return False


def rust_execute(query_json: str, data: dict[str, Any]) -> dict[str, Any]:
    return rust_execute_arrow(query_json, *arrow_args(data))


def rust_execute_arrow(
    query_json: str, feat: Any, times: Any, batches: Any, anchor: Any
) -> dict[str, Any]:
    r"""``labels_execute`` over Arrow inputs, its labels as plain data."""
    import pyarrow as pa

    module = native('labels_execute')
    try:
        got = module.labels_execute(query_json, feat, times, batches, anchor)
    except Exception as error:  # noqa: BLE001 - compared, not judged
        return {'error': error_of(error)}
    assert isinstance(got, tuple) and len(got) == 2, (
        f'labels_execute gave {type(got).__name__}, not (values, mask)'
    )
    values, mask = got
    assert isinstance(values, (pa.Array, pa.ChunkedArray)), (
        f'labels_execute gave values of {type(values).__name__}, not a pyarrow array'
    )
    if pa.types.is_timestamp(values.type):
        per = {'s': 10**9, 'ms': 10**6, 'us': 10**3, 'ns': 1}[values.type.unit]
        out_values = [
            None if v is None else ['ts', int(v) * per]
            for v in values.cast(pa.int64()).to_pylist()
        ]
    else:
        out_values = values.to_pylist()
    return {
        'type': values.type,
        'values': [plain(v) for v in out_values],
        'mask': [bool(m) for m in as_list(mask)],
    }


def float32_bits(values: list[Any]) -> list[Any]:
    return [
        None if v is None else _f32_bits(float('nan') if v == 'nan' else v)
        for v in values
    ]


def execute_problem(want: dict[str, Any], got: dict[str, Any]) -> str | None:
    if 'error' in want:
        if 'error' not in got:
            return f'gave labels, today raises {short(want["error"])}'
        if got['error']['code'] != want['error']['code']:
            return f'raised {short(got["error"])}, today raises {short(want["error"])}'
        return None
    if 'error' in got:
        return f'raised {short(got["error"])}, today gives {short(want)}'
    if got['mask'] != want['mask']:
        return f'mask {got["mask"]}, today {want["mask"]}'
    if not arrow_type_fits(want['dtype'], got['type']):
        return f'labels of type {got["type"]}, today of dtype {want["dtype"]}'
    if want['dtype'] == 'float32':
        if float32_bits(got['values']) != float32_bits(want['values']):
            return (
                f'labels {short(got["values"])}, today {short(want["values"])}'
            )
        return None
    if got['values'] != want['values']:
        return f'labels {short(got["values"])}, today {short(want["values"])}'
    return None
