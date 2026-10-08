# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 7.5 acceptance checks share: the boundary recordings'
format, the canonical form outputs are compared in, driving a sampling run,
and the interceptor that sits on ``Sampler.sample_target`` and
``Sampler.sample_subgraph`` while the corpus replays.

The interface these rely on is ``CONTRACT.md``, beside this file. The
capture script may import this module, as ``capture_dialects.py`` imports
``tests/rust_core/dialects/_codec.py``.

This module imports neither pandas nor the engine at import time, so the
recording-driven runs (``_driver_run.py``) can use it in a process where
those can't be imported.
"""

from __future__ import annotations

import base64
import contextlib
import datetime as dt
import importlib
import importlib.util
import inspect
import json
import math
import operator
import struct
import sys
import uuid
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SAMPLING = HERE.parent
REPO = HERE.parents[3]
TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'
RECORDINGS = REPO / 'tests' / 'parity' / 'sampling'
CAPTURE = REPO / 'scripts' / 'parity' / 'capture_sampling.py'
BENCH = REPO / 'scripts' / 'parity' / 'bench_sampling.py'
BASELINE = REPO / 'scripts' / 'parity' / 'sampling_baseline.json'
GAPS = REPO / 'scripts' / 'parity' / 'sampling_coverage_gaps.json'

for _path in (
    HERE,
    SAMPLING,
    SAMPLING.parent,
    REPO / 'tests' / 'parity',
    REPO / 'scripts' / 'parity',
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

#: The sampler methods a recording holds calls of, and the binding's runs.
METHODS = ('sample_target', 'sample_subgraph')
#: The recording format this module writes and reads.
FORMAT = 1
#: The file of a scenario's recording, in its directory.
RECORDING_FILE = 'boundary.json'

ENGINE = 'sdm_relational_engine.rfm'
#: The modules holding the samplers, imported before any is patched.
SAMPLER_MODULES = (
    f'{ENGINE}.base.sampler',
    f'{ENGINE}.base.sql_sampler',
    f'{ENGINE}.backend.local.sampler',
    f'{ENGINE}.backend.sqlite.sampler',
    f'{ENGINE}.backend.duckdb.sampler',
    f'{ENGINE}.backend.postgres.sampler',
    f'{ENGINE}.backend.snow.sampler',
    f'{ENGINE}.backend.databricks.sampler',
)
#: What a describer or converter may not call: every sampler method that
#: samples, reads the database or computes a time range.
SAMPLING_METHODS = (
    'sample_target',
    'sample_subgraph',
    'sample_lagged_target',
    'get_min_time',
    'get_max_time',
    'validate_entity_references',
    '_get_min_max_time_dict',
    '_sample_subgraph',
    '_sample_entity_table',
    '_sample_query_data',
    '_sample_target',
    '_sample_target_set',
    '_by_pkey',
    '_by_fkey',
    '_by_time',
    '_time_spelling',
    '_probe_time_spelling',
    '_run',
    '_rows',
)


# --- The binding ----------------------------------------------------------------


def native(*names: str) -> Any:
    r"""``sdm_core._native``, built from this checkout (Phase 7's helper)."""
    from _sampling import native as build

    return build(*names)


# --- Canonical form ---------------------------------------------------------------
#
# Outputs are compared as JSON in one canonical form, whichever side made
# them: an Arrow array becomes {"type": T, "values": [...]}, with T its type
# named by `type_name` and each value as `_values` writes it.


def type_name(t: Any) -> str:
    r"""An Arrow type's name, with representations that hold the same values
    named alike: a dictionary by its values, any string as ``string``, any
    binary as ``binary``, any list as ``list<item>``."""
    import pyarrow as pa

    types = pa.types
    if types.is_dictionary(t):
        return type_name(t.value_type)
    if types.is_string(t) or types.is_large_string(t):
        return 'string'
    if getattr(types, 'is_string_view', lambda _: False)(t):
        return 'string'
    if types.is_binary(t) or types.is_large_binary(t):
        return 'binary'
    if getattr(types, 'is_binary_view', lambda _: False)(t):
        return 'binary'
    if (
        types.is_list(t)
        or types.is_large_list(t)
        or types.is_fixed_size_list(t)
        or getattr(types, 'is_list_view', lambda _: False)(t)
        or getattr(types, 'is_large_list_view', lambda _: False)(t)
    ):
        return f'list<{type_name(t.value_type)}>'
    if types.is_struct(t):
        fields = ', '.join(
            f'{t.field(i).name}: {type_name(t.field(i).type)}'
            for i in range(t.num_fields)
        )
        return f'struct<{fields}>'
    return str(t)


def _float(value: Any, width: int) -> str:
    r"""A float's IEEE bits in hex; NaN, and a null, as ``nan``.

    A null and a NaN are one value here: pandas holds a missing float as NaN
    in a numpy column, so today's Python can't tell them apart either.
    """
    if value is None or math.isnan(value):
        return 'nan'
    code = {16: '>e', 32: '>f', 64: '>d'}[width]
    return struct.pack(code, value).hex()


def _plain(array: Any) -> Any:
    r"""An array, unchunked and without dictionary encoding."""
    import pyarrow as pa

    if isinstance(array, pa.ChunkedArray):
        array = (
            array.combine_chunks()
            if array.num_chunks
            else pa.array([], type=array.type)
        )
    if pa.types.is_dictionary(array.type):
        array = array.dictionary_decode()
    return array


def _values(array: Any) -> list[Any]:
    import pyarrow as pa
    import pyarrow.compute as pc

    array = _plain(array)
    t = array.type
    types = pa.types
    if types.is_null(t):
        return [None] * len(array)
    if types.is_floating(t):
        width = t.bit_width
        return [_float(v, width) for v in array.to_pylist()]
    if (
        types.is_timestamp(t)
        or types.is_date(t)
        or types.is_time(t)
        or types.is_duration(t)
    ):
        storage = pa.int64() if t.bit_width == 64 else pa.int32()
        return pc.cast(array.view(storage), pa.int64()).to_pylist()
    if types.is_decimal(t):
        return [None if v is None else str(v) for v in array.to_pylist()]
    if type_name(t) == 'binary':
        return [
            None if v is None else base64.b64encode(v).decode('ascii')
            for v in array.to_pylist()
        ]
    if type_name(t).startswith('list<'):
        out: list[Any] = []
        for i in range(len(array)):
            item = array[i]
            out.append(_values(item.values) if item.is_valid else None)
        return out
    if types.is_struct(t):
        columns = [_values(array.field(i)) for i in range(t.num_fields)]
        valid = array.is_valid().to_pylist()
        return [
            {t.field(j).name: columns[j][i] for j in range(t.num_fields)}
            if valid[i]
            else None
            for i in range(len(array))
        ]
    if types.is_boolean(t) or types.is_integer(t) or type_name(t) == 'string':
        return array.to_pylist()
    raise AssertionError(f'the Arrow type {t} is not in the recording format')


def canon_array(array: Any) -> dict[str, Any]:
    r"""An Arrow array (or chunked array) in canonical form."""
    import pyarrow as pa

    if not isinstance(array, (pa.Array, pa.ChunkedArray)):
        raise AssertionError(
            f'expected a pyarrow array, got {type(array).__name__}'
        )
    plain = _plain(array)
    return {'type': type_name(plain.type), 'values': _values(plain)}


def canon_frame(table: Any) -> dict[str, Any]:
    r"""An Arrow table in canonical form: its column names and columns."""
    import pyarrow as pa

    if not isinstance(table, pa.Table):
        raise AssertionError(
            f'expected a pyarrow.Table, got {type(table).__name__}'
        )
    return {
        'names': list(table.column_names),
        'columns': [canon_array(column) for column in table.columns],
    }


def canon_ints(values: Any, what: str) -> list[int] | None:
    r"""``None``, or a sequence of integers as a list.

    A list or tuple of ``int``, a numpy integer array, or a pyarrow integer
    array without nulls.
    """
    if values is None:
        return None
    if hasattr(values, 'dtype') and hasattr(values, 'tolist'):
        kind = getattr(values.dtype, 'kind', None)
        if kind not in ('i', 'u'):
            raise AssertionError(f'{what}: numpy dtype {values.dtype}')
        out = values.tolist()
    elif hasattr(values, 'to_pylist') and hasattr(values, 'type'):
        import pyarrow as pa

        plain = _plain(values)
        if not pa.types.is_integer(plain.type) or plain.null_count:
            raise AssertionError(f'{what}: Arrow {plain.type} with nulls')
        out = plain.to_pylist()
    elif isinstance(values, (list, tuple)):
        out = list(values)
    else:
        raise AssertionError(f'{what}: {type(values).__name__} is not ints')
    if any(isinstance(v, bool) or not hasattr(v, '__index__') for v in out):
        raise AssertionError(f'{what}: not all integers')
    return [operator.index(v) for v in out]


def canon_times(values: Any, what: str) -> list[int | None]:
    r"""Times as integer nanoseconds since the epoch, a null as ``None``.

    A pyarrow timestamp array of any unit, or integers as ``canon_ints``
    takes them, which are nanoseconds already.
    """
    import pyarrow as pa

    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        plain = _plain(values)
        if pa.types.is_timestamp(plain.type):
            scale = {'s': 10**9, 'ms': 10**6, 'us': 10**3, 'ns': 1}
            factor = scale[plain.type.unit]
            return [None if v is None else v * factor for v in _values(plain)]
        if pa.types.is_integer(plain.type):
            return plain.to_pylist()
    if hasattr(values, 'dtype') and str(values.dtype).startswith('datetime64'):
        import numpy as np

        nat = np.isnat(values)
        ints = values.astype('datetime64[ns]').astype('int64').tolist()
        return [
            None if n else v for v, n in zip(ints, nat.tolist(), strict=True)
        ]
    out = canon_ints(values, what)
    assert out is not None, f'{what}: no times'
    return list(out)


def canon_warnings(given: Any, what: str) -> list[dict[str, str]]:
    if not isinstance(given, (list, tuple)):
        raise AssertionError(f'{what}: warnings are not a list')
    out = []
    for item in given:
        if not (
            isinstance(item, Mapping)
            and isinstance(item.get('category'), str)
            and isinstance(item.get('message'), str)
        ):
            raise AssertionError(f'{what}: a warning is {item!r}')
        out.append({'category': item['category'], 'message': item['message']})
    return out


def warning_of(caught: warnings.WarningMessage) -> dict[str, str]:
    category = caught.category
    return {
        'category': f'{category.__module__}.{category.__qualname__}',
        'message': str(caught.message),
    }


def error_of(error: BaseException) -> dict[str, str]:
    r"""``{"code", "message"}``: the error's class name and its text."""
    return {'code': type(error).__name__, 'message': str(error)}


# --- Today's Python, in canonical form -----------------------------------------


def series_array(series: Any) -> Any:
    r"""A pandas Series as Arrow, as ``pa.Table.from_pandas`` makes it."""
    import pyarrow as pa

    frame = series.reset_index(drop=True).to_frame(name='value')
    return pa.Table.from_pandas(frame, preserve_index=False).column(0)


def frame_table(frame: Any) -> Any:
    import pyarrow as pa

    return pa.Table.from_pandas(
        frame.reset_index(drop=True), preserve_index=False
    )


def _enum(value: Any) -> str:
    return str(getattr(value, 'value', value))


def python_outputs(
    method: str,
    result: Any,
    error: BaseException | None,
    caught: list[dict[str, str]],
) -> dict[str, Any]:
    r"""What today's sampler gave, in canonical form."""
    if error is not None:
        return {'error': error_of(error)}
    if method == 'sample_target':
        train, test = result
        return {
            'train': _python_part(train),
            'test': _python_part(test),
            'warnings': caught,
        }
    return {
        'anchor_time': canon_times(result.anchor_time, 'anchor_time'),
        'tables': [
            {
                'name': name,
                'df': canon_frame(frame_table(table.df)),
                'row': canon_ints(table.row, f'{name}.row'),
                'batch': canon_ints(table.batch, f'{name}.batch'),
                'num_sampled_nodes': canon_ints(
                    list(table.num_sampled_nodes), f'{name}.num_sampled_nodes'
                ),
                'stype_dict': [
                    [column, _enum(stype)]
                    for column, stype in table.stype_dict.items()
                ],
                'primary_key': table.primary_key,
            }
            for name, table in result.tables.items()
        ],
        'links': [
            {
                'edge_type': list(edge_type),
                'layout': _enum(link.layout),
                'row': canon_ints(link.row, f'{edge_type}.row'),
                'col': canon_ints(link.col, f'{edge_type}.col'),
                'num_sampled_edges': canon_ints(
                    list(link.num_sampled_edges), f'{edge_type}.edges'
                ),
            }
            for edge_type, link in result.links.items()
        ],
        'warnings': caught,
    }


def _python_part(part: Any) -> dict[str, Any]:
    import pyarrow as pa

    anchor = series_array(part.anchor_time)
    if not pa.types.is_timestamp(_plain(anchor).type):
        raise AssertionError(f'an anchor time of type {anchor.type}')
    return {
        'entity_pkey': canon_array(series_array(part.entity_pkey)),
        'anchor_time': canon_times(anchor, 'anchor_time'),
        'target': canon_array(series_array(part.target)),
    }


# --- The core's runs, in canonical form ----------------------------------------


def _key(given: Mapping[str, Any], key: str, what: str) -> Any:
    if not isinstance(given, Mapping) or key not in given:
        raise AssertionError(f'{what} has no {key!r}')
    return given[key]


def rust_outputs(method: str, value: Any) -> dict[str, Any]:
    r"""The value of a run's ``done`` step, in canonical form
    (``CONTRACT.md``, "What a run gives")."""
    caught = canon_warnings(_key(value, 'warnings', 'done'), 'done')
    if method == 'sample_target':
        return {
            'train': _rust_part(_key(value, 'train', 'done'), 'train'),
            'test': _rust_part(_key(value, 'test', 'done'), 'test'),
            'warnings': caught,
        }
    tables = _key(value, 'tables', 'done')
    links = _key(value, 'links', 'done')
    if not isinstance(tables, Mapping) or not isinstance(links, Mapping):
        raise AssertionError('done: tables and links are not dicts')
    out_tables = []
    for name, table in tables.items():
        stypes = _key(table, 'stype_dict', name)
        if not isinstance(stypes, Mapping):
            raise AssertionError(f'{name}.stype_dict is not a dict')
        out_tables.append(
            {
                'name': name,
                'df': canon_frame(_key(table, 'df', name)),
                'row': canon_ints(_key(table, 'row', name), f'{name}.row'),
                'batch': canon_ints(
                    _key(table, 'batch', name), f'{name}.batch'
                ),
                'num_sampled_nodes': canon_ints(
                    _key(table, 'num_sampled_nodes', name),
                    f'{name}.num_sampled_nodes',
                ),
                'stype_dict': [
                    [column, _enum(stype)] for column, stype in stypes.items()
                ],
                'primary_key': _key(table, 'primary_key', name),
            }
        )
    out_links = []
    for edge_type, link in links.items():
        what = f'link {edge_type}'
        if not (isinstance(edge_type, tuple) and len(edge_type) == 3):
            raise AssertionError(f'{what}: the key is not a 3-tuple')
        out_links.append(
            {
                'edge_type': list(edge_type),
                'layout': _enum(_key(link, 'layout', what)),
                'row': canon_ints(_key(link, 'row', what), f'{what}.row'),
                'col': canon_ints(_key(link, 'col', what), f'{what}.col'),
                'num_sampled_edges': canon_ints(
                    _key(link, 'num_sampled_edges', what), f'{what}.edges'
                ),
            }
        )
    return {
        'anchor_time': canon_times(
            _key(value, 'anchor_time', 'done'), 'anchor_time'
        ),
        'tables': out_tables,
        'links': out_links,
        'warnings': caught,
    }


def _rust_part(part: Any, what: str) -> dict[str, Any]:
    return {
        'entity_pkey': canon_array(_key(part, 'entity_pkey', what)),
        'anchor_time': canon_times(
            _key(part, 'anchor_time', what), f'{what}.anchor_time'
        ),
        'target': canon_array(_key(part, 'target', what)),
    }


def difference(expected: Any, got: Any, path: str = '$') -> str | None:
    r"""Where two canonical outputs differ; ``None`` if they don't.

    Values compare by JSON type as well as value. The one leniency: an empty
    ``entity_pkey`` or ``target`` matches an empty one of any type.
    """
    if isinstance(expected, dict) and isinstance(got, dict):
        if (
            path.endswith(('.entity_pkey', '.target'))
            and set(expected) == set(got) == {'type', 'values'}
            and expected['values'] == []
            and got['values'] == []
        ):
            return None
        if list(expected) != list(got):
            return f'{path}: keys {list(got)} instead of {list(expected)}'
        for key in expected:
            found = difference(expected[key], got[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return (
                f'{path}: {len(got)} items instead of {len(expected)}: '
                f'{_short(got)} instead of {_short(expected)}'
            )
        for i, (a, b) in enumerate(zip(expected, got, strict=True)):
            found = difference(a, b, f'{path}[{i}]')
            if found:
                return found
        return None
    if type(expected) is not type(got) or expected != got:
        return f'{path}: {got!r} instead of {expected!r}'
    return None


def _short(value: Any, limit: int = 300) -> str:
    text = json.dumps(value, default=repr)
    return text if len(text) <= limit else text[:limit] + '...'


# --- Parameters -----------------------------------------------------------------


def encode_params(params: Any) -> Any:
    r"""A statement's parameters as a transcript step encodes them
    (``scripts/parity/_parity/values.py``), ``None`` for none or an empty
    list.

    The values the core gives are plain Python ones, which this encodes
    without pandas, as ``values.encode`` does; anything else goes to
    ``values.encode`` itself.
    """
    if params is None:
        return None
    items = list(params)
    if not items:
        return None
    return [_encode(item) for item in items]


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if type(value) is float:
        if math.isnan(value):
            return {'$float': 'nan'}
        if math.isinf(value):
            return {'$float': 'inf' if value > 0 else '-inf'}
        return value
    if type(value) is dt.datetime:
        return {'$datetime': value.isoformat()}
    if type(value) is dt.date:
        return {'$date': value.isoformat()}
    if type(value) is dt.time:
        return {'$time': value.isoformat()}
    if type(value) is dt.timedelta:
        micros = (
            value.days * 86_400_000_000
            + value.seconds * 1_000_000
            + value.microseconds
        )
        return {'$timedelta_us': micros}
    if isinstance(value, Decimal):
        return {'$decimal': str(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'$bytes': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, uuid.UUID):
        return {'$uuid': str(value)}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        out = {key: _encode(item) for key, item in value.items()}
        if len(out) == 1 and next(iter(out)).startswith('$'):
            return {'$object': out}
        return out
    from _parity.values import encode

    return encode(value)


# --- Driving a run ------------------------------------------------------------------


class Mismatch(AssertionError):
    r"""A run asked for a statement other than the next recorded one."""


def outcome_of(module: Any, step: Mapping[str, Any], result: Any) -> Any:
    r"""The ``Outcome`` a host submits for a recorded step: its table, or its
    error as ``CONTRACT.md`` says."""
    error = step.get('error')
    if error is None:
        return module.Outcome.data(result)
    return module.Outcome.error(
        error.get('sqlstate'), error.get('vendor_code'), error['message']
    )


def drive(
    module: Any,
    run: Any,
    steps: Sequence[Mapping[str, Any]],
    result_of: Callable[[int], Any],
) -> tuple[Any, list[dict[str, Any]]]:
    r"""Drive ``run`` to its end, answering each query with the next of
    ``steps``; the ``done`` value, and the queries it asked.

    Each query must be the next step, its SQL byte for byte and its
    parameters encoded alike, in the order the run lists them; the run must
    ask for every step. Raises :class:`Mismatch` where it doesn't, and what
    ``run.next()`` raises where it raises.
    """
    asked: list[dict[str, Any]] = []
    position = 0
    for _ in range(100_000):
        step = run.next()
        kind = getattr(step, 'kind', None)
        if kind == 'done':
            if position != len(steps):
                raise Mismatch(
                    f'the run finished after {position} of {len(steps)} '
                    f'statements; the next was '
                    f'{_short(steps[position]["sql"], 200)}'
                )
            return step.value, asked
        if kind != 'run_queries':
            raise AssertionError(f'a step of kind {kind!r}')
        queries = list(step.queries)
        if not queries:
            raise AssertionError('a run_queries step with no queries')
        for query in queries:
            sql = query.sql
            params = encode_params(query.params)
            asked.append({'sql': sql, 'params': params})
            if position >= len(steps):
                raise Mismatch(
                    f'statement {position}: the run asks for '
                    f'{_short(sql, 200)}, after every recorded one'
                )
            want = steps[position]
            if sql != want['sql']:
                at = next(
                    (
                        i
                        for i, (a, b) in enumerate(
                            zip(sql, want['sql'], strict=False)
                        )
                        if a != b
                    ),
                    min(len(sql), len(want['sql'])),
                )
                raise Mismatch(
                    f'statement {position}: SQL differs at character {at}: '
                    f'{sql[max(0, at - 40) : at + 60]!r} instead of '
                    f'{want["sql"][max(0, at - 40) : at + 60]!r}'
                )
            if json.dumps(params) != json.dumps(want['params']):
                raise Mismatch(
                    f'statement {position}: parameters {_short(params)} '
                    f'instead of {_short(want["params"])}'
                )
            run.submit(query.id, outcome_of(module, want, result_of(position)))
            position += 1
    raise AssertionError('the run took more than 100,000 steps')


# --- Recordings -----------------------------------------------------------------


def scenario_dir(root: Path, name: str) -> Path:
    r"""Where a scenario's files go, as for transcripts: ``#`` as ``--``."""
    return root.joinpath(*name.replace('#', '--').split('/'))


def write_arrow(table: Any, path: Path) -> None:
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    path.write_bytes(sink.getvalue().to_pybytes())


def read_arrow(path: Path) -> Any:
    import pyarrow as pa

    return pa.ipc.open_file(pa.BufferReader(path.read_bytes())).read_all()


def dumps(document: Any) -> str:
    r"""A recording as text: its header, then one call per line."""

    def one(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    calls = document['calls']
    lines = [
        '{',
        f' "format": {one(document["format"])},',
        f' "scenario": {one(document["scenario"])},',
        ' "calls": [',
    ]
    for i, call in enumerate(calls):
        lines.append(f'  {one(call)}' + (',' if i < len(calls) - 1 else ''))
    lines += [' ]', '}', '']
    return '\n'.join(lines)


def strict_json(text: Any, what: str) -> Any:
    r"""``text`` parsed as JSON without NaN or infinities."""
    if not isinstance(text, str):
        raise AssertionError(f'{what} is {type(text).__name__}, not str')

    def refuse(constant: str) -> Any:
        raise AssertionError(f'{what} holds {constant}, which JSON lacks')

    try:
        return json.loads(text, parse_constant=refuse)
    except ValueError as error:
        raise AssertionError(f'{what} is not JSON: {error}') from error


def write_recording(
    directory: Path, scenario: str, calls: list[dict[str, Any]]
) -> None:
    r"""Write one scenario's recording into ``directory``, afresh.

    Each call is ``{"method", "inputs_json", "tables", "steps", "outputs"}``:
    ``tables`` maps names to ``pyarrow.Table``s and each step holds its
    result as a ``pyarrow.Table`` or ``None``, as :class:`Interceptor`
    records them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.iterdir():
        if stale.is_file():
            stale.unlink()
    document: dict[str, Any] = {
        'format': FORMAT,
        'scenario': scenario,
        'calls': [],
    }
    for i, call in enumerate(calls):
        tables = {}
        for k, (name, table) in enumerate(call['tables'].items()):
            file = f'call-{i:04d}-input-{k:02d}.arrow'
            write_arrow(table, directory / file)
            tables[name] = file
        steps = []
        for j, step in enumerate(call['steps']):
            file = None
            if step['result'] is not None:
                file = f'call-{i:04d}-step-{j:04d}.arrow'
                write_arrow(step['result'], directory / file)
            steps.append(
                {
                    'sql': step['sql'],
                    'params': step['params'],
                    'result': file,
                    'error': step['error'],
                }
            )
        document['calls'].append(
            {
                'method': call['method'],
                'inputs': strict_json(call['inputs_json'], 'inputs_json'),
                'tables': tables,
                'steps': steps,
                'outputs': call['outputs'],
            }
        )
    (directory / RECORDING_FILE).write_text(dumps(document), encoding='utf-8')


def read_recording(directory: Path) -> dict[str, Any]:
    document = json.loads((directory / RECORDING_FILE).read_text('utf-8'))
    assert document.get('format') == FORMAT, (
        f'{directory}: format {document.get("format")!r}, not {FORMAT}'
    )
    return document


def recorded_tables(directory: Path, call: Mapping[str, Any]) -> dict:
    return {
        name: read_arrow(directory / file)
        for name, file in call['tables'].items()
    }


def transcripts() -> list[Path]:
    return sorted(TRANSCRIPTS.rglob('transcript.json'))


def recording_dirs(root: Path = RECORDINGS) -> list[Path]:
    return sorted(path.parent for path in root.rglob(RECORDING_FILE))


# --- The hooks the checks call ----------------------------------------------------


def capture_module() -> Any:
    r"""``scripts/parity/capture_sampling.py``, imported."""
    assert CAPTURE.is_file(), f'{CAPTURE} does not exist'
    spec = importlib.util.spec_from_file_location('capture_sampling', CAPTURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('capture_sampling', module)
    spec.loader.exec_module(module)
    for name in ('describe', 'to_python'):
        assert callable(getattr(module, name, None)), (
            f'{CAPTURE.name} defines no {name}()'
        )
    return module


# --- Guards -------------------------------------------------------------------------


class Tripped(BaseException):
    r"""A describer or converter reached what it may not: a database, or
    today's sampling. A ``BaseException``, so no ``except Exception`` in the
    engine hides it."""


class _Tripwire:
    def __init__(self, what: str, guard: Guard) -> None:
        object.__setattr__(self, '_what', what)
        object.__setattr__(self, '_guard', guard)

    def __getattr__(self, name: str) -> Any:
        self._guard.trip(f'{self._what}.{name}')

    def __setattr__(self, name: str, value: Any) -> None:
        self._guard.trip(f'{self._what}.{name}')


class Guard:
    r"""While active, the samplers' sampling methods, every ``Source._run``,
    today's label executor, and a local sampler's graph store and neighbor
    sampler trip it."""

    def __init__(self) -> None:
        self.active = False
        self.tripped: list[str] = []

    def trip(self, what: str) -> Any:
        self.tripped.append(what)
        raise Tripped(f'{what} was reached while describing or converting')

    @contextlib.contextmanager
    def on(self, sampler: Any) -> Iterator[None]:
        from sdm_relational_engine.relationallib import NeighborSampler
        from sdm_relational_engine.rfm.backend.local import LocalGraphStore

        swapped = {
            name: value
            for name, value in vars(sampler).items()
            if isinstance(value, (LocalGraphStore, NeighborSampler))
        }
        for name in swapped:
            vars(sampler)[name] = _Tripwire(f'sampler.{name}', self)
        self.active = True
        before = len(self.tripped)
        try:
            yield
        finally:
            self.active = False
            vars(sampler).update(swapped)
        if len(self.tripped) > before:
            raise Tripped(
                f'{self.tripped[before]} was reached while describing or '
                f'converting'
            )


# --- The interceptor -----------------------------------------------------------------


def _sampler_classes() -> list[type]:
    for name in SAMPLER_MODULES:
        importlib.import_module(name)
    from sdm_relational_engine.rfm.base.sampler import Sampler

    found, todo = [], [Sampler]
    while todo:
        cls = todo.pop()
        found.append(cls)
        todo.extend(cls.__subclasses__())
    return found


def _source_classes() -> list[type]:
    import sdm_connectors.source as source

    return [
        cls
        for cls in vars(source).values()
        if inspect.isclass(cls)
        and issubclass(cls, source.Source)
        and '_run' in vars(cls)
    ]


class SwapFailed(BaseException):
    r"""The core's run did not stand in for today's call. A
    ``BaseException``, so the replay reports it rather than the engine or the
    client turning it into an error of their own."""


class Interceptor:
    r"""Sits on ``Sampler.sample_target`` and ``Sampler.sample_subgraph``
    while transcripts replay.

    For each call (the outermost, on any sampler) it asks
    ``describe(sampler, graph, method, arguments)`` for the run's inputs,
    under a :class:`Guard`; runs today's method, keeping the statements it
    ran through ``Source._run``, what it gave and the warnings it gave; and
    keeps all of that in :attr:`calls`.

    With ``core`` (``sdm_core._native``) and ``to_python``, it then also
    drives the core's run of the same name, answering each query from the
    statements today's method ran, and returns
    ``to_python(sampler, graph, method, arguments, inputs_json, value)`` of
    the run's ``done`` value, under the guard, with the run's warnings,
    instead of what today's method gave. Where today's method raised, the
    run must raise alike, and today's error is raised.
    """

    def __init__(
        self,
        describe: Callable[..., Any],
        core: Any = None,
        to_python: Callable[..., Any] | None = None,
    ) -> None:
        self.describe = describe
        self.core = core
        self.to_python = to_python
        self.calls: list[dict[str, Any]] = []
        self.guard = Guard()
        self.graphs: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._depth = 0
        self._steps: list[dict[str, Any]] | None = None
        self._run_depth = 0

    # Installing.

    @contextlib.contextmanager
    def installed(self) -> Iterator[Interceptor]:
        from sdm_relational_engine.rfm.base.sampler import Sampler
        from sdm_relational_engine.rfm.query.pandas_executor import (
            PQueryPandasExecutor,
        )

        undo: list[tuple[Any, str, Any]] = []

        def patch(owner: Any, name: str, value: Any) -> None:
            undo.append((owner, name, vars(owner)[name]))
            setattr(owner, name, value)

        try:
            patch(Sampler, '__init__', self._init(vars(Sampler)['__init__']))
            for method in METHODS:
                patch(
                    Sampler, method, self._call(method, vars(Sampler)[method])
                )
            for cls in _sampler_classes():
                for name in SAMPLING_METHODS:
                    if name in vars(cls) and name not in METHODS:
                        patch(cls, name, self._guarded(name, vars(cls)[name]))
            for cls in _source_classes():
                patch(cls, '_run', self._source_run(vars(cls)['_run']))
            patch(
                PQueryPandasExecutor,
                'execute_query',
                self._guarded(
                    'execute_query', vars(PQueryPandasExecutor)['execute_query']
                ),
            )
            yield self
        finally:
            for owner, name, original in reversed(undo):
                setattr(owner, name, original)

    def _init(self, original: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(original)
        graphs = self.graphs

        def init(sampler: Any, *args: Any, **kwargs: Any) -> None:
            bound = signature.bind(sampler, *args, **kwargs)
            graphs[sampler] = bound.arguments['graph']
            original(sampler, *args, **kwargs)

        return init

    def _guarded(self, name: str, original: Any) -> Any:
        guard = self.guard
        if isinstance(original, property):
            getter = original.fget

            def get(this: Any) -> Any:
                if guard.active:
                    guard.trip(name)
                return getter(this)

            return property(get, original.fset, original.fdel, original.__doc__)
        if isinstance(original, (staticmethod, classmethod)):
            return original

        def guarded(*args: Any, **kwargs: Any) -> Any:
            if guard.active:
                guard.trip(name)
            return original(*args, **kwargs)

        return guarded

    def _source_run(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def run(source: Any, sql: Any, params: Any = None) -> Any:
            if self.guard.active:
                self.guard.trip('Source._run')
            if self._steps is None or self._run_depth:
                return original(source, sql, params)
            step: dict[str, Any] = {
                'sql': sql,
                'params': encode_params(params),
                'result': None,
                'error': None,
            }
            self._run_depth += 1
            try:
                step['result'] = original(source, sql, params)
                return step['result']
            except Exception as error:
                sqlstate = getattr(error, 'sqlstate', None)
                errno = getattr(error, 'errno', None)
                step['error'] = {
                    'type': f'{type(error).__module__}.'
                    f'{type(error).__qualname__}',
                    'message': str(error),
                    'sqlstate': sqlstate if isinstance(sqlstate, str) else None,
                    'vendor_code': errno
                    if isinstance(errno, int) and not isinstance(errno, bool)
                    else None,
                }
                raise
            finally:
                self._run_depth -= 1
                self._steps.append(step)

        return run

    def _call(self, method: str, original: Any) -> Callable[..., Any]:
        signature = inspect.signature(original)

        def call(sampler: Any, *args: Any, **kwargs: Any) -> Any:
            if self.guard.active:
                self.guard.trip(method)
            if self._depth:
                return original(sampler, *args, **kwargs)
            self._depth += 1
            try:
                return self._intercept(
                    method, original, signature, sampler, args, kwargs
                )
            finally:
                self._depth -= 1

        return call

    # One call.

    def _intercept(
        self,
        method: str,
        original: Any,
        signature: inspect.Signature,
        sampler: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        bound = signature.bind(sampler, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop('self', None)
        graph = self.graphs.get(sampler)
        with self.guard.on(sampler):
            described = self.describe(sampler, graph, method, arguments)
        inputs_json, tables = _check_described(described)

        self._steps = []
        result: Any = None
        error: BaseException | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = original(sampler, *args, **kwargs)
            except Exception as raised:
                error = raised
        steps, self._steps = self._steps, None
        given = [w for w in caught if issubclass(w.category, UserWarning)]
        python = python_outputs(
            method, result, error, [warning_of(w) for w in given]
        )
        self.calls.append(
            {
                'method': method,
                'backend': _enum(sampler.backend),
                'sampler': sampler,
                'graph': graph,
                'arguments': arguments,
                'inputs_json': inputs_json,
                'tables': tables,
                'steps': steps,
                'outputs': python,
            }
        )
        if self.core is None:
            for w in given:
                warnings.warn_explicit(
                    w.message, w.category, w.filename, w.lineno
                )
            if error is not None:
                raise error
            return result
        return self._swap(
            method, sampler, graph, arguments, inputs_json, tables, steps,
            error,
        )  # fmt: skip

    def _swap(
        self,
        method: str,
        sampler: Any,
        graph: Any,
        arguments: dict[str, Any],
        inputs_json: str,
        tables: dict[str, Any],
        steps: list[dict[str, Any]],
        error: BaseException | None,
    ) -> Any:
        assert self.to_python is not None
        start = getattr(self.core, method)
        try:
            run = start(inputs_json, tables)
            value, _ = drive(
                self.core, run, steps, lambda i: steps[i]['result']
            )
        except Mismatch as mismatch:
            raise SwapFailed(f'{method}: {mismatch}') from None
        except Exception as raised:
            if error is not None and error_of(raised) == error_of(error):
                raise error from None
            raise SwapFailed(
                f'{method}: the run raised {error_of(raised)}'
                + (
                    f' where today raised {error_of(error)}'
                    if error is not None
                    else ''
                )
            ) from None
        if error is not None:
            raise SwapFailed(
                f'{method}: the run finished where today raised '
                f'{error_of(error)}'
            )
        try:
            caught = canon_warnings(_key(value, 'warnings', 'done'), 'done')
        except AssertionError as bad:
            raise SwapFailed(f'{method}: {bad}') from None
        with self.guard.on(sampler):
            converted = self.to_python(
                sampler, graph, method, arguments, inputs_json, value
            )
        for item in caught:
            warnings.warn(
                item['message'], _category(item['category']), stacklevel=2
            )
        return converted


def _category(name: str) -> type[Warning]:
    module, _, qualname = name.rpartition('.')
    try:
        found: Any = importlib.import_module(module)
        for part in qualname.split('.'):
            found = getattr(found, part)
    except (ImportError, AttributeError):
        raise SwapFailed(f'no warning class {name}') from None
    if not (inspect.isclass(found) and issubclass(found, Warning)):
        raise SwapFailed(f'{name} is not a warning class')
    return found


def _check_described(described: Any) -> tuple[str, dict[str, Any]]:
    import pyarrow as pa

    if not (isinstance(described, tuple) and len(described) == 2):
        raise AssertionError('describe() did not return (inputs_json, tables)')
    inputs_json, tables = described
    strict_json(inputs_json, 'inputs_json')
    if not isinstance(tables, dict) or not all(
        isinstance(k, str) and isinstance(v, pa.Table)
        for k, v in tables.items()
    ):
        raise AssertionError(
            'describe() tables are not a dict of str to pyarrow.Table'
        )
    return inputs_json, tables


@contextlib.contextmanager
def replaying(interceptor: Interceptor) -> Iterator[Callable[[Path], Any]]:
    r"""Install ``interceptor``; yields a function replaying one transcript
    under it, which returns ``(reason, calls)``: ``None`` or why the replay
    failed (``_parity.corpus.check``), and the calls it intercepted."""
    from _parity.corpus import check

    with interceptor.installed():

        def replay(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
            interceptor.calls = []
            reason = check(path)
            return reason, interceptor.calls

        yield replay


def scenario_name(path: Path) -> str:
    return json.loads(path.read_text('utf-8'))['name']


def capture_corpus(
    out: Path,
    describe: Callable[..., Any],
    fixed: frozenset[str] = frozenset(),
) -> dict[str, list[dict[str, Any]]]:
    r"""Replay every transcript under an :class:`Interceptor` without a core
    and write each scenario's recording under ``out``; the calls, by
    scenario. Listed fixes require the exact authorized old refusal and
    complete governed native replay before their earlier calls are kept.
    Every other replay failure is an error."""
    found: dict[str, list[dict[str, Any]]] = {}
    changed = []
    with replaying(Interceptor(describe)) as replay:
        for path in transcripts():
            name = scenario_name(path)
            reason, calls = replay(path)
            if name in fixed:
                changed.append((path, reason))
            elif reason is not None:
                raise AssertionError(f'{name} does not replay: {reason}')
            found[name] = calls
    from _fixed_contracts import fixed_replay

    for path, reason in changed:
        fixed_replay(path, reason)
    for name, calls in found.items():
        write_recording(scenario_dir(out, name), name, calls)
    return found
