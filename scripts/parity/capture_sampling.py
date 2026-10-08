#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Record every ``sample_target`` and ``sample_subgraph`` call of the parity
corpus at the sampler's boundary, as the Rust sampling driver is checked
against.

    capture_sampling.py [--out DIR]

Every transcript of the corpus, the live and synthetic ones too, is replayed
with an interceptor on the samplers (``tests/rust_core/sampling/driver``).
For each call it keeps what the core's run of the call starts from
(:func:`describe`), every statement today's method ran with its result, and
what the method gave, and writes one ``boundary.json`` per transcript, with
the Arrow files it names, under ``DIR`` (by default
``tests/parity/sampling/``), as ``tests/rust_core/sampling/driver/CONTRACT.md``
describes. It needs no network, database or credentials, and writes the same
bytes on every run.

:func:`describe` reads a call's sampler, graph and arguments into the inputs
the core's run takes; :func:`to_python` turns the run's ``done`` value back
into the objects today's method returns.

The inputs are JSON, of this form (times are integer nanoseconds since the
epoch, naive UTC):

- ``backend``: the sampler's ``DataBackend`` value.
- ``tables``: the graph's tables in order, each with its name, primary key,
  time and end time columns, and columns (name, dtype, stype, expression and
  whether it is a source column); for a SQL sampler also its quoted source
  name, its source columns and what its backend reads per table (see
  :func:`_sql_table`).
- ``edges``: the graph's links, ``[src, fkey, dst]``.
- ``state``: what the sampler holds from earlier calls: the time ranges it
  read, the SQLite time spellings it probed, the tables whose source enforces
  the key, the row counts, and whether the coverage warning was given.
- ``arguments``: the call's arguments, the query as its ``json_serde`` tree;
  an argument that is a ``Series`` is a table of ``tables``, named by
  ``argument_tables``, with its pandas dtype under ``arguments``.

:func:`describe` also describes a call of ``sample_lagged_target``, and of
``KumoRelational._get_task_table`` and ``_get_context`` (given the model's
sampler and graph), whose runs ``sdm_core._native.sample_lagged_target``,
``task_table`` and ``context`` start from it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DRIVER = REPO / 'tests' / 'rust_core' / 'sampling' / 'driver'
for _path in (HERE, DRIVER, REPO / 'tests' / 'rust_core' / 'dialects'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _driver import RECORDINGS, capture_corpus  # noqa: E402
from _old_python import fixed_scenarios, use_old_python  # noqa: E402

__all__ = ['describe', 'main', 'to_python']

#: The columns of the tables holding the arguments that are a ``Series``.
ARGUMENT_COLUMN = 'value'
#: The pandas dtype tag of a column the core's run gives (field metadata).
DTYPE_KEY = b'sdm.pandas_dtype'


# --- Describing a call -------------------------------------------------------


def _ns(value: Any) -> int:
    r"""A naive ``pd.Timestamp`` as nanoseconds since the epoch."""
    import pandas as pd

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        raise ValueError(f'a zoned time {stamp!r} where a naive one is kept')
    return int(stamp.value)


def _enum(value: Any) -> Any:
    return None if value is None else str(getattr(value, 'value', value))


def _column(column: Any) -> dict[str, Any]:
    return {
        'name': column.name,
        'dtype': _enum(column.dtype),
        'stype': _enum(column.stype),
        'expr': None if column.expr is None else str(column.expr),
        'is_source': bool(column.is_source),
    }


def _source_column(column: Any) -> dict[str, Any]:
    return {
        'name': column.name,
        'dtype': _enum(column.dtype),
        'is_primary_key': bool(column.is_primary_key),
        'is_unique_key': bool(column.is_unique_key),
        'is_nullable': bool(column.is_nullable),
    }


def _sql_table(sampler: Any, table: Any) -> dict[str, Any]:
    r"""What a SQL sampler reads of one of its tables: its source name and
    columns, and what its backend's statements depend on."""
    from _codec import encode

    name = table.name
    out: dict[str, Any] = {
        'source_name': sampler.source_name_dict[name],
        'unquoted_name': table.source_name,
        'source_columns': [
            _source_column(column)
            for column in sampler.source_table_dict[name].values()
        ],
    }
    backend = _enum(sampler.backend)
    if backend == 'duckdb':
        out['zoned'] = [
            column.name
            for column in table.columns
            if sampler._graph_tables[name]._is_zoned_time(column.name)
        ]
    elif backend == 'postgres':
        out['postgres_types'] = dict(sampler._postgres_type_dict[name])
    elif backend == 'snowflake':
        snow = sampler._table_dict[name]
        out['snowflake_types'] = dict(sampler._snowflake_type_dict[name])
        out['zoned_expr_columns'] = sorted(snow._zoned_expr_columns)
        # `_listed` is read once and kept; the run reads it where it isn't.
        listed = vars(snow).get('_listed')
        out['listed'] = None if listed is None else encode(listed)
        out['identity'] = {
            'database': snow._database,
            'schema': snow._schema,
            'source_name': snow._source_name,
        }
    return out


def _state(sampler: Any) -> dict[str, Any]:
    from sdm_relational_engine.rfm.base import sampler as base

    state: dict[str, Any] = {
        'min_time': {k: _ns(v) for k, v in sampler._min_time_dict.items()},
        'max_time': {k: _ns(v) for k, v in sampler._max_time_dict.items()},
        'min_time_unit': {k: v.unit for k, v in sampler._min_time_dict.items()},
        'max_time_unit': {k: v.unit for k, v in sampler._max_time_dict.items()},
        'coverage_warned': bool(base._coverage_warned),
    }
    if hasattr(sampler, '_enforced_key_tables'):
        state['enforced_key_tables'] = sorted(sampler._enforced_key_tables)
    if hasattr(sampler, '_num_rows_dict'):
        state['num_rows'] = dict(sampler._num_rows_dict)
    if hasattr(sampler, '_time_spelling_dict'):
        state['time_spellings'] = {
            name: [spelling.separator, spelling.digits, spelling.suffix]
            for name, spelling in sampler._time_spelling_dict.items()
        }
    return state


def _seed(value: Any) -> str | None:
    r"""A seed as decimal text, which holds one of any size."""
    return None if value is None else str(int(value))


def _anchor(value: Any) -> Any:
    if isinstance(value, str):
        return value
    return {'ns': _ns(value), 'unit': value.unit}


def _optional_anchor(value: Any) -> Any:
    r"""An anchor argument that may be absent, made naive UTC first as
    ``_check_anchor_time`` makes it."""
    from sdm_relational_engine.rfm.base.utils import to_naive_utc

    if value is None:
        return None
    return _anchor(to_naive_utc(value))


def _plain(value: Any) -> Any:
    r"""An ``indices`` value as JSON holds it: a numpy scalar as Python's."""
    item = getattr(value, 'item', None)
    value = item() if callable(item) else value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f'an index {value!r} that JSON does not hold')


def _describe_task(
    arguments: dict[str, Any],
    tables: dict[str, Any],
    taken: set[str],
    argument_tables: dict[str, str],
) -> dict[str, Any]:
    r"""What ``_get_context`` reads of its ``TaskTable`` and arguments."""
    import pandas as pd
    import pyarrow as pa

    task = arguments['task']
    entity = pd.concat(
        [
            task._context_df[task._entity_column],
            task._pred_df[task._entity_column],
        ],
        axis=0,
        ignore_index=True,
    )
    name = _free_name(taken, '#entity_pkey')
    tables[name] = _series_table(entity)
    argument_tables['entity_pkey'] = name
    anchor_dtype = None
    if task._time_column is not None and not task.use_entity_time:
        anchor = pd.concat(
            [
                task._context_df[task._time_column],
                task._pred_df[task._time_column],
            ],
            axis=0,
            ignore_index=True,
        )
        name = _free_name(taken, '#anchor_time')
        tables[name] = _series_table(anchor)
        argument_tables['anchor_time'] = name
        anchor_dtype = str(anchor.dtype)
    features = [c.name for c in task.feature_columns]
    if features:
        frame = pd.concat(
            [task._context_df[features], task._pred_df[features]],
            axis=0,
            ignore_index=True,
        )
        name = _free_name(taken, '#features')
        tables[name] = pa.Table.from_pandas(frame, preserve_index=False)
        argument_tables['features'] = name
    num_neighbors = arguments['num_neighbors']
    exclude = arguments['exclude_cols_dict']
    return {
        'task_type': _enum(task.task_type),
        'entity_table_names': list(task.entity_table_names),
        'entity_pkey_dtype': str(entity.dtype),
        'num_context': len(task._context_df),
        'time_column': task._time_column,
        'anchor_time_dtype': anchor_dtype,
        'run_mode': _enum(arguments['run_mode']),
        'num_neighbors': None
        if num_neighbors is None
        else [int(n) for n in num_neighbors],
        'exclude_cols_dict': None
        if exclude is None
        else {k: list(v) for k, v in exclude.items()},
        'random_seed': _seed(arguments['random_seed']),
        'validate_references': bool(arguments['_validate_references']),
    }


def _series_table(series: Any) -> Any:
    import pyarrow as pa

    frame = series.reset_index(drop=True).to_frame(name=ARGUMENT_COLUMN)
    return pa.Table.from_pandas(frame, preserve_index=False)


def _free_name(taken: set[str], wanted: str) -> str:
    name = wanted
    while name in taken:
        name = f'#{name}'
    taken.add(name)
    return name


def describe(
    sampler: Any, graph: Any, method: str, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    r"""The inputs and tables the core's run of this call starts from.

    Reads the sampler, its graph and the call's arguments, and nothing that
    samples or runs a statement.
    """
    import pyarrow as pa
    from predictive_query.schema import json_serde

    backend = _enum(sampler.backend)
    tables_out = []
    tables: dict[str, Any] = {}
    for table in graph.tables.values():
        entry: dict[str, Any] = {
            'name': table.name,
            'primary_key': table._primary_key,
            'time_column': table._time_column,
            'end_time_column': table._end_time_column,
            'columns': [_column(column) for column in table.columns],
        }
        if backend != 'local':
            entry['sql'] = _sql_table(sampler, table)
        tables_out.append(entry)
        if backend == 'local':
            tables[table.name] = pa.Table.from_pandas(
                table._data, preserve_index=False
            )
    taken = set(tables)
    argument_tables: dict[str, str] = {}
    given: dict[str, Any] = {}
    if method == 'sample_target':
        given = {
            'query': json_serde.to_json_dict(arguments['query']),
            'num_train_examples': int(arguments['num_train_examples']),
            'train_anchor_time': _anchor(arguments['train_anchor_time']),
            'num_train_trials': int(arguments['num_train_trials']),
            'num_test_examples': int(arguments['num_test_examples']),
            'test_anchor_time': _anchor(arguments['test_anchor_time']),
            'num_test_trials': int(arguments['num_test_trials']),
            'random_seed': _seed(arguments['random_seed']),
        }
    elif method == 'sample_lagged_target':
        for argument in ('entity_pkey', 'anchor_time'):
            series = arguments[argument]
            name = _free_name(taken, f'#{argument}')
            tables[name] = _series_table(series)
            argument_tables[argument] = name
        given = {
            'query': json_serde.to_json_dict(arguments['query']),
            'entity_pkey_dtype': str(arguments['entity_pkey'].dtype),
            'anchor_time_dtype': str(arguments['anchor_time'].dtype),
            'lag_timesteps': int(arguments['lag_timesteps']),
        }
    elif method == '_get_task_table':
        import pandas as pd

        indices = arguments['indices']
        given = {
            'query': json_serde.to_json_dict(arguments['query']),
            'indices': None
            if indices is None
            else [_plain(value) for value in indices],
            'anchor_time': _optional_anchor(arguments['anchor_time']),
            'context_anchor_time': _optional_anchor(
                arguments['context_anchor_time']
            ),
            'run_mode': _enum(arguments['run_mode']),
            'lag_timesteps': int(arguments['lag_timesteps']),
            'max_pq_iterations': int(arguments['max_pq_iterations']),
            'random_seed': _seed(arguments['random_seed']),
        }
        units = ['s', 'ms', 'us', 'ns']
        inferred = [
            str(pd.Series([pd.Timestamp(0).as_unit(unit)]).dtype)
            .removeprefix('datetime64[')
            .removesuffix(']')
            for unit in units
        ]
        if inferred != units:
            given['timestamp_series_units'] = inferred
    elif method == '_get_context':
        given = _describe_task(arguments, tables, taken, argument_tables)
    elif method in ('_get_missing_index_dict', '_warn_missing_indices'):
        # The checks a DuckDB or SQLite sampler makes as it is built.
        given = {'optimize': False}
    else:
        pkey = arguments['entity_pkey']
        name = _free_name(taken, '#entity_pkey')
        tables[name] = _series_table(pkey)
        argument_tables['entity_pkey'] = name
        anchor = arguments['anchor_time']
        anchor_dtype = None
        if not isinstance(anchor, str):
            name = _free_name(taken, '#anchor_time')
            tables[name] = _series_table(anchor)
            argument_tables['anchor_time'] = name
            anchor_dtype = str(anchor.dtype)
        exclude = arguments['exclude_cols_dict']
        scored = arguments['scored']
        given = {
            'entity_table_names': list(arguments['entity_table_names']),
            'entity_pkey_dtype': str(pkey.dtype),
            'anchor_time': anchor if isinstance(anchor, str) else None,
            'anchor_time_dtype': anchor_dtype,
            'num_neighbors': [int(n) for n in arguments['num_neighbors']],
            'exclude_cols_dict': None
            if exclude is None
            else {k: list(v) for k, v in exclude.items()},
            'random_seed': _seed(arguments['random_seed']),
            'scored': None if scored is None else [int(i) for i in scored],
        }
    inputs = {
        'backend': backend,
        'tables': tables_out,
        'edges': [list(edge) for edge in graph.edges],
        'state': _state(sampler),
        'arguments': given,
        'argument_tables': argument_tables,
    }
    text = json.dumps(inputs, ensure_ascii=False, allow_nan=False)
    return text, tables


# --- Back to today's objects -------------------------------------------------


def _series(array: Any, dtype: str) -> Any:
    r"""A pandas ``Series`` of ``dtype`` (the run's tag) holding ``array``."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if dtype == 'arrow':
        return pd.Series(pd.arrays.ArrowExtensionArray(array))
    if dtype == 'object':
        return pd.Series(array.to_pylist(), dtype=object)
    if dtype == 'category':
        # A dictionary's values are the categories, in their order.
        return pd.Series(array.to_pandas())
    if dtype.startswith('datetime64'):
        values = array.cast(pa.int64()).to_numpy(zero_copy_only=False)
        unit = dtype[len('datetime64[') : -1]
        return pd.Series(values.astype(f'datetime64[{unit}]'))
    if dtype in ('Int64', 'Float64', 'boolean', 'string', 'str'):
        return pd.Series(pd.array(array.to_pylist(), dtype=dtype))
    if pa.types.is_floating(array.type):
        return pd.Series(
            array.to_numpy(zero_copy_only=False).astype(np.dtype(dtype))
        )
    return pd.Series(array.to_numpy(zero_copy_only=False).astype(dtype))


def _frame(table: Any) -> Any:
    import pandas as pd

    data = {}
    for field, column in zip(table.schema, table.columns, strict=True):
        dtype = (field.metadata or {}).get(DTYPE_KEY, b'arrow').decode()
        data[field.name] = _series(column, dtype)
    return pd.DataFrame(data, index=pd.RangeIndex(table.num_rows))


def _ints(values: Any) -> Any:
    import numpy as np

    if values is None:
        return None
    if hasattr(values, 'to_numpy'):
        return values.to_numpy(zero_copy_only=False).astype(np.int64)
    return np.asarray(values, dtype=np.int64)


def to_python(
    sampler: Any,
    graph: Any,
    method: str,
    arguments: dict[str, Any],
    inputs_json: str,
    value: Any,
) -> Any:
    r"""What today's method returns, made from the run's ``done`` value."""
    if method == 'sample_target':
        from sdm_relational_engine.rfm.base.sampler import TargetOutput

        out = []
        for part in ('train', 'test'):
            given = value[part]
            dtypes = given['dtypes']
            out.append(
                TargetOutput(
                    _series(given['entity_pkey'], dtypes['entity_pkey']),
                    _series(given['anchor_time'], 'datetime64[ns]'),
                    _series(given['target'], dtypes['target']),
                )
            )
        return tuple(out)
    from predictive_query.schema.typing import Stype
    from sdm_relational_engine.rfm.inference.context import (
        SampledEdgeLayout,
        SampledLink,
        SampledSubgraph,
        SampledTable,
    )

    subgraph = SampledSubgraph(
        anchor_time=_ints(value['anchor_time']), tables={}, links={}
    )
    for name, table in value['tables'].items():
        subgraph.tables[name] = SampledTable(
            df=_frame(table['df']),
            row=_ints(table['row']),
            batch=_ints(table['batch']),
            num_sampled_nodes=[int(n) for n in table['num_sampled_nodes']],
            stype_dict={
                column: Stype(stype)
                for column, stype in table['stype_dict'].items()
            },
            primary_key=table['primary_key'],
        )
    for edge_type, link in value['links'].items():
        subgraph.links[tuple(edge_type)] = SampledLink(
            layout=SampledEdgeLayout(link['layout']),
            row=_ints(link['row']),
            col=_ints(link['col']),
            num_sampled_edges=[int(n) for n in link['num_sampled_edges']],
        )
    return subgraph


# --- The command line --------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--out', default=str(RECORDINGS))
    args = parser.parse_args(argv)
    use_old_python()
    capture_corpus(Path(args.out), describe, fixed_scenarios())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
