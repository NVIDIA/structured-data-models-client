#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's cast of the entities to the key's type into
``contract/key-cast-vectors.json`` (T1.32a's rule-table vectors).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/key_cast_record.py

Written by T1.32a's independent test author before the implementation.
Each vector is one key type crossed with one list of entities:

- the key type is ``contract/bases.yaml``'s ``shop`` with ``users.user_id``
  and ``orders.user_id`` changed to a DuckDB type (``KEYS``), the type the
  source hands over as the key's Arrow type;
- the entities are Python values a caller passes as ``indices`` (``INPUTS``
  and each key's own ``extra``): integers in and out of every width's
  range, integral and non-integral floats, digit and other text, mixed
  lists, booleans, None, NaN and ``pd.NA``, NumPy scalars, decimals,
  timestamps and dates, bytes.

For each, the reference runs ``Schema.from_source`` and
``PredictiveQuery.materialize(indices=..., anchor_time=2024-03-01)`` and
this script keeps:

- ``reference.public``: what the public call gave, a success with the
  entities as the payload's predicted rows carry them, or the refusal's
  class, ``StructuredDataError.code`` and stage (no message text, H17);
  this is what both hosts are held to;
- ``reference.cast``: the reference's own cast where the engine makes it
  (``rfm.py``'s ``pd.Series(indices, dtype=train_pkey.dtype)``), observed
  in place: the key's dtype and the values cast, or the exception it
  raised. It documents the rule table; no host is compared on it;
- ``host_arrow``: the entities as one Arrow column by the host's natural
  inference (``pa.array(values)``) or, when that fails (mixed kinds, an
  integer past int64), as text: ``str()`` of each value, None, NaN,
  ``pd.NA`` and ``pd.NaT`` as null, a column of them alone Arrow's null
  type (design, "Host-held semantics", the ledger's row for this cast).
  ``rust-host`` hands the core this column; ``python-core`` hands the
  bridge the Python values themselves;
- ``register``: whether a host may differ from the reference only under
  a register entry naming the vector (``REGISTER_REF``), and why. It is
  derived from the reference's answer, never chosen: the reference fails
  with ``INTERNAL_ERROR`` (E66(b)'s class, a clearer answer is a change),
  or accepts an entity by changing its kind (text read as a number, a
  number or a time written as text, a boolean or a time read as a number)
  or its value (``1.5`` cast to ``1``). Every other vector is held to the
  reference exactly. This marks where the core is expected to differ; it
  decides no policy.

Nothing here is written by hand but the inputs: every expected value is
the reference's. ``observe``, ``decode`` and ``canonical`` are imported by
``contract/tests/test_t132a_key_cast.py``, so both routes are observed by
the same code.
"""

from __future__ import annotations

import base64
import json
import math
import platform
import struct
import subprocess
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/key-cast-vectors.json'
FORMAT = 'sdm-key-cast-vectors/1'
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = '2024-03-01'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix and a vector's id, or a glob of ids (``fnmatch``), for example
#: ``contract/key-cast-vectors.json#decimal128-18-0.*``.
REGISTER_REF = 'contract/key-cast-vectors.json#'


def _alter(sql_type: str, using: str | None = None) -> list[str]:
    tail = f' USING {using}' if using else ''
    return [
        f'ALTER TABLE users ALTER user_id TYPE {sql_type}'
        + tail.replace('X', 'user_id'),
        f'ALTER TABLE orders ALTER user_id TYPE {sql_type}'
        + tail.replace('X', 'user_id'),
    ]


#: The key types: ``shop``'s ``user_id`` (0 to 7, BIGINT) changed to each
#: DuckDB type a key may have in the reference, the Arrow type it hands
#: over named by ``id``. Each ``extra`` holds entities of the table's own
#: values in that type's natural Python form. DATE, TIMESTAMP and BOOLEAN
#: are not here: the reference refuses them as keys before any cast
#: (``Schema.from_source``: "incompatible semantic type").
KEYS: list[dict[str, Any]] = [
    {'id': 'int8', 'setup': _alter('TINYINT')},
    {'id': 'int16', 'setup': _alter('SMALLINT')},
    {'id': 'int32', 'setup': _alter('INTEGER')},
    {'id': 'int64', 'setup': []},
    {'id': 'uint8', 'setup': _alter('UTINYINT')},
    {'id': 'uint16', 'setup': _alter('USMALLINT')},
    {'id': 'uint32', 'setup': _alter('UINTEGER')},
    {'id': 'uint64', 'setup': _alter('UBIGINT')},
    {
        'id': 'uint64-past-int64',
        'setup': _alter('UBIGINT', 'X::UBIGINT + 9223372036854775808::UBIGINT'),
        'extra': {
            'own.past-int64': [2**63 + 1, 2**63 + 2],
            'own.past-int64-text': [str(2**63 + 1)],
            'own.past-int64-float': [float(2**63)],
        },
    },
    {'id': 'float32', 'setup': _alter('FLOAT')},
    {
        'id': 'float64',
        'setup': _alter('DOUBLE'),
        'extra': {'own.half-key': [0.5]},
    },
    {'id': 'utf8', 'setup': _alter('VARCHAR', 'X::VARCHAR')},
    {
        'id': 'large-utf8',
        'setup': [
            'SET arrow_large_buffer_size = true',
            *_alter('VARCHAR', 'X::VARCHAR'),
        ],
    },
    {'id': 'decimal128-18-0', 'setup': _alter('DECIMAL(18, 0)')},
    {
        'id': 'decimal128-38-10',
        'setup': _alter('DECIMAL(38, 10)', 'X + 0.5'),
        'extra': {
            'own.decimal': [Decimal('1.5')],
            'own.text': ['1.5'],
            'own.float': [1.5],
        },
    },
    {'id': 'decimal128-38-0', 'setup': _alter('HUGEINT')},
]


def _inputs() -> dict[str, list[Any]]:
    r"""The entities every key type is crossed with, built afresh for each
    call (so no run sees another's objects)."""
    import numpy as np
    import pandas as pd

    nan = float('nan')
    return {
        # integers: present, absent, and at each width's ends
        'int.present': [1, 2],
        'int.absent': [8],
        'int.negative': [-1],
        'int.128': [128],
        'int.256': [256],
        'int.32768': [32768],
        'int.65536': [65536],
        'int.2pow31': [2**31],
        'int.2pow32': [2**32],
        'int.2pow53-plus-1': [2**53 + 1],
        'int.2pow63-minus-1': [2**63 - 1],
        'int.2pow63': [2**63],
        'int.2pow64': [2**64],
        'int.minus-2pow63-minus-1': [-(2**63) - 1],
        # floats: integral and not
        'float.integral': [1.0, 2.0],
        'float.half': [1.5],
        'float.negative-zero': [-0.0],
        'float.1e20': [1e20],
        'float.inf': [math.inf],
        'float.2pow63': [float(2**63)],
        # text
        'text.digits': ['1', '2'],
        'text.decimal-point': ['1.0'],
        'text.half': ['1.5'],
        'text.spaced': [' 1 '],
        'text.leading-zero': ['01'],
        'text.letters': ['x'],
        'text.empty': [''],
        'text.exponent': ['1e0'],
        'text.negative': ['-1'],
        'text.hex': ['0x1'],
        'text.fullwidth-digit': ['１'],
        'text.2pow64': [str(2**64)],
        # mixed lists
        'mixed.int-text': [1, '2'],
        'mixed.int-float': [1, 2.0],
        'mixed.int-half': [1, 2.5],
        'mixed.float-text': [1.0, '2'],
        'mixed.bool-int': [True, 1],
        'mixed.int-none': [1, None],
        'mixed.text-none': ['1', None],
        'mixed.int-nan': [1, nan],
        # booleans
        'bool.both': [True, False],
        # missing values
        'null.none': [None],
        'null.nan': [nan],
        'null.pd-na': [pd.NA],
        # NumPy scalars
        'numpy.int64': [np.int64(1), np.int64(2)],
        'numpy.float64-half': [np.float64(1.5)],
        # decimals
        'decimal.integral': [Decimal('1')],
        'decimal.half': [Decimal('1.5')],
        # times
        'time.timestamp': [pd.Timestamp('2024-01-01')],
        'time.timestamp-1us': [pd.Timestamp(1, unit='us')],
        'time.timestamp-utc': [pd.Timestamp('2024-01-01', tz='UTC')],
        'time.date': [date(2024, 1, 1)],
        # bytes
        'bytes.digit': [b'1'],
    }


def inputs(key: dict[str, Any]) -> dict[str, list[Any]]:
    r"""``key``'s entities: the shared inputs and its own."""
    return {**_inputs(), **key.get('extra', {})}


# --- values as JSON ----------------------------------------------------------


def encode(value: Any) -> dict[str, Any]:
    r"""A Python entity as tagged JSON that ``decode`` gives back exactly."""
    import numpy as np
    import pandas as pd

    if value is None:
        return {'t': 'None'}
    if value is pd.NA:
        return {'t': 'pd.NA'}
    if value is pd.NaT:
        return {'t': 'pd.NaT'}
    if isinstance(value, np.generic):
        return {'t': f'numpy.{value.dtype}', 'v': repr(value.item())}
    if isinstance(value, bool):
        return {'t': 'bool', 'v': value}
    if isinstance(value, int):
        return {'t': 'int', 'v': str(value)}
    if isinstance(value, float):
        return {'t': 'float', 'v': repr(value)}
    if isinstance(value, str):
        return {'t': 'str', 'v': value}
    if isinstance(value, bytes):
        return {'t': 'bytes', 'v': base64.b64encode(value).decode()}
    if isinstance(value, Decimal):
        return {'t': 'Decimal', 'v': str(value)}
    if isinstance(value, pd.Timestamp):
        return {
            't': 'pd.Timestamp',
            'v': value.isoformat(),
            'unit': value.unit,
            'tz': None if value.tz is None else str(value.tz),
        }
    if isinstance(value, datetime):
        return {'t': 'datetime', 'v': value.isoformat()}
    if isinstance(value, date):
        return {'t': 'date', 'v': value.isoformat()}
    raise TypeError(f'no encoding for {type(value).__name__}')


def decode(tagged: dict[str, Any]) -> Any:
    r"""The Python entity ``encode`` wrote."""
    import numpy as np
    import pandas as pd

    kind = tagged['t']
    if kind == 'None':
        return None
    if kind == 'pd.NA':
        return pd.NA
    if kind == 'pd.NaT':
        return pd.NaT
    if kind.startswith('numpy.'):
        dtype = np.dtype(kind[len('numpy.') :])
        text = tagged['v']
        return dtype.type(float(text) if dtype.kind == 'f' else int(text))
    if kind == 'bool':
        return bool(tagged['v'])
    if kind == 'int':
        return int(tagged['v'])
    if kind == 'float':
        return float(tagged['v'])
    if kind == 'str':
        return tagged['v']
    if kind == 'bytes':
        return base64.b64decode(tagged['v'])
    if kind == 'Decimal':
        return Decimal(tagged['v'])
    if kind == 'pd.Timestamp':
        stamp = pd.Timestamp(tagged['v']).as_unit(tagged['unit'])
        return stamp if tagged['tz'] is None else stamp.tz_convert(tagged['tz'])
    if kind == 'datetime':
        return datetime.fromisoformat(tagged['v'])
    if kind == 'date':
        return date.fromisoformat(tagged['v'])
    raise ValueError(f'unknown tag {kind!r}')


def _bits(value: float) -> str:
    return struct.pack('>d', value).hex()


def canonical(value: Any) -> Any:
    r"""A value as both hosts can spell it: null, ``["b", bool]``,
    ``["i", digits]``, ``["f", the double's IEEE-754 bits in hex]``,
    ``["s", text]``, ``["d", decimal text]``, or ``["?", type name]``."""
    import numpy as np

    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return ['b', value]
    if isinstance(value, int):
        return ['i', str(value)]
    if isinstance(value, float):
        return ['f', _bits(value)]
    if isinstance(value, str):
        return ['s', value]
    if isinstance(value, Decimal):
        return ['d', str(value)]
    return ['?', type(value).__name__]


# --- the host's natural Arrow column ------------------------------------------


def _missing(value: Any) -> bool:
    import pandas as pd

    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    )


def _scalar(value: Any, arrow_type: Any) -> dict[str, Any]:
    r"""One value of an Arrow column as the contract's scalar
    (``sdm-contract``'s ``Scalar``, its ``t``-tagged JSON)."""
    import pyarrow as pa

    if value is None:
        return {'t': 'null'}
    integers = {
        pa.int8(): 'i8',
        pa.int16(): 'i16',
        pa.int32(): 'i32',
        pa.int64(): 'i64',
        pa.uint8(): 'u8',
        pa.uint16(): 'u16',
        pa.uint32(): 'u32',
        pa.uint64(): 'u64',
    }
    if arrow_type in integers:
        return {'t': integers[arrow_type], 'v': str(value)}
    if pa.types.is_boolean(arrow_type):
        return {'t': 'bool', 'v': value}
    if pa.types.is_floating(arrow_type):
        text = repr(float(value))
        text = {'nan': 'NaN', 'inf': 'inf', '-inf': '-inf'}.get(text, text)
        return {'t': 'f64' if arrow_type == pa.float64() else 'f32', 'v': text}
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return {'t': 'utf8', 'v': value}
    if pa.types.is_binary(arrow_type):
        return {'t': 'binary', 'v': base64.b64encode(value).decode()}
    if pa.types.is_decimal128(arrow_type):
        unscaled = int(value.scaleb(arrow_type.scale))
        return {
            't': 'd128',
            'p': arrow_type.precision,
            's': arrow_type.scale,
            'v': str(unscaled),
        }
    if pa.types.is_date32(arrow_type):
        return {'t': 'date32', 'v': str((value - date(1970, 1, 1)).days)}
    raise TypeError(f'no scalar for {arrow_type}')


def host_arrow(values: list[Any]) -> dict[str, Any]:
    r"""The entities as one Arrow column by the host's natural inference,
    or as text when it fails: the column's type and each value as the
    contract's scalar. A timestamp is carried as its count of its column's
    unit (``us`` or ``ns``, the contract's units)."""
    import pyarrow as pa

    try:
        column = pa.array(values)
        rule = 'inferred'
    except (pa.ArrowException, OverflowError, TypeError, ValueError):
        if all(_missing(v) for v in values):
            column = pa.nulls(len(values))
        else:
            column = pa.array(
                [None if _missing(v) else str(v) for v in values], pa.string()
            )
        rule = 'text'
    kind = column.type
    if pa.types.is_timestamp(kind):
        unit = kind.unit if kind.unit in ('us', 'ns') else 'us'
        counts = column.cast(pa.timestamp(unit, kind.tz)).cast(pa.int64())
        scalars = [
            {'t': 'null'}
            if v is None
            else {'t': 'ts', 'unit': unit, 'zone': kind.tz, 'v': str(v)}
            for v in counts.to_pylist()
        ]
        return {
            'rule': rule,
            'type': str(pa.timestamp(unit, kind.tz)),
            'values': scalars,
        }
    return {
        'rule': rule,
        'type': str(kind),
        'values': [_scalar(v, kind) for v in column.to_pylist()],
    }


# --- the public call -----------------------------------------------------------


def shop() -> list[str]:
    r"""``contract/bases.yaml``'s ``shop`` statements."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def observe(key: dict[str, Any], values: list[Any]) -> dict[str, Any]:
    r"""``Schema.from_source`` and ``materialize`` over ``key``'s source
    with ``values`` as the entities, on the route the environment selects:
    the outcome, and on success the entities as the payload's predicted
    rows carry them, in order."""
    import duckdb
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    for statement in shop() + key['setup']:
        connection.execute(statement)
    stage = 'from_source'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            stage = 'materialize'
            request = PredictiveQuery(QUERY, schema).materialize(
                indices=values, anchor_time=pd.Timestamp(ANCHOR)
            )
    except Exception as error:  # noqa: BLE001 - the outcome is recorded
        return {
            'outcome': 'refusal',
            'stage': stage,
            'error': type(error).__name__,
            'code': getattr(error, 'code', None),
        }
    finally:
        connection.close()
    table = request._payloads[0]['predict']['instance_table']
    at = next(
        i
        for i, name in enumerate(table['columns'])
        if name.startswith('__kumo_entity_ref')
    )
    return {
        'outcome': 'success',
        'entities': [canonical(row[at]) for row in table['rows']],
    }


# --- the reference's own cast, observed in place ------------------------------


@contextmanager
def _watching_the_cast(values: list[Any]) -> Iterator[dict[str, Any]]:
    r"""Observes the reference engine's ``pd.Series(indices, dtype=...)``
    of these entities (the same objects, element by element) as it runs:
    the dtype, and the values cast or the exception raised."""
    import pandas as pd
    import pyarrow as pa

    seen: dict[str, Any] = {}
    original = pd.Series.__init__

    def init(self: Any, data: Any = None, *args: Any, **kwargs: Any) -> None:
        ours = (
            kwargs.get('dtype') is not None
            and isinstance(data, (list, tuple))
            and len(data) == len(values)
            and all(a is b for a, b in zip(data, values, strict=True))
        )
        if not ours:
            original(self, data, *args, **kwargs)
            return
        seen['dtype'] = str(kwargs['dtype'])
        try:
            original(self, data, *args, **kwargs)
        except Exception as error:
            seen['raised'] = type(error).__name__
            raise
        seen['values'] = [canonical(v) for v in pa.array(self).to_pylist()]

    pd.Series.__init__ = init  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        pd.Series.__init__ = original  # type: ignore[method-assign]


def _kind(value: Any) -> str:
    r"""What an entity is, for ``register``'s kind rule."""
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, (int, float, Decimal)):
        return 'number'
    if isinstance(value, str):
        return 'text'
    if isinstance(value, (date, datetime)):
        return 'time'
    return type(value).__name__


def _number(cell: Any) -> Decimal | None:
    if not isinstance(cell, list):
        return None
    if cell[0] in ('i', 'd'):
        return Decimal(cell[1])
    if cell[0] == 'f':
        value = struct.unpack('>d', bytes.fromhex(cell[1]))[0]
        return Decimal(value) if math.isfinite(value) else None
    return None


def register(values: list[Any], public: dict[str, Any]) -> list[str]:
    r"""Why a host may differ from the reference on this vector only under a
    register entry: the reference's answer, never a choice. Empty: held to
    the reference exactly."""
    why = []
    if public['outcome'] == 'refusal' and public['code'] == 'INTERNAL_ERROR':
        why.append(
            'the reference fails with INTERNAL_ERROR (E66(b): a named '
            'refusal or a success in its place is a change to register)'
        )
    if public['outcome'] == 'success':
        for value, cell in zip(values, public['entities'], strict=True):
            if _missing(value) or cell is None:
                continue
            kind = _kind(value)
            got = {'i': 'number', 'f': 'number', 'd': 'number'}.get(
                cell[0], 'text' if cell[0] == 's' else cell[0]
            )
            if kind != got:
                why.append(
                    f'the reference accepts a {kind} entity as a {got} key '
                    "(pandas' cast changes its kind)"
                )
                break
            if kind == 'number':
                plain = value.item() if hasattr(value, 'item') else value
                finite = not isinstance(plain, float) or math.isfinite(plain)
                wanted = Decimal(plain) if finite else None
                if wanted is None or wanted != _number(cell):
                    why.append(
                        'the reference accepts a number as a different '
                        "number (pandas' cast changes its value)"
                    )
                    break
    return why


def record(key: dict[str, Any], name: str, values: list[Any]) -> dict[str, Any]:
    r"""One vector, observed on the reference."""
    encoded = [encode(v) for v in values]
    arrow = host_arrow(values)
    with _watching_the_cast(values) as seen:
        public = observe(key, values)
    cast: dict[str, Any] | None
    if not seen:
        cast = None  # the call ended before the engine's cast
    elif 'raised' in seen:
        cast = {'dtype': seen['dtype'], 'raised': seen['raised']}
    else:
        cast = {'dtype': seen['dtype'], 'values': seen['values']}
    why = register(values, public)
    return {
        'id': f'{key["id"]}.{name}',
        'key': key['id'],
        'input': encoded,
        'host_arrow': arrow,
        'reference': {'public': public, 'cast': cast},
        'register': {'required': bool(why), 'why': why},
    }


def alike(vectors: list[dict[str, Any]]) -> None:
    r"""Marks the vectors the core cannot tell apart: on one key, entities
    the host hands over as the same Arrow column (``host_arrow``'s type and
    values), which the reference answers differently (``[1, '2']`` is
    refused where ``['1', '2']`` is taken). The core sees one column, so it
    gives them one answer, and each that differs needs its entry."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for vector in vectors:
        column = vector['host_arrow']
        same = json.dumps(
            [vector['key'], column['type'], column['values']], sort_keys=True
        )
        groups.setdefault(same, []).append(vector)
    for group in groups.values():
        answers = {
            json.dumps(v['reference']['public'], sort_keys=True) for v in group
        }
        if len(answers) < 2:
            continue
        for vector in group:
            others = ', '.join(v['id'] for v in group if v is not vector)
            vector['register']['why'].append(
                f'the host gives the core the same column as {others}, '
                'which the reference answers differently'
            )
            vector['register']['required'] = True


def main() -> int:
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import sdm_relational_engine  # noqa: F401 - the reference's engine

    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    import sdm_client

    if not Path(sdm_client.__file__).resolve().is_relative_to(reference):
        print("sdm_client is not the reference checkout's", file=sys.stderr)
        return 1
    vectors = []
    for key in KEYS:
        for name, values in inputs(key).items():
            vector = record(key, name, values)
            vectors.append(vector)
            public = vector['reference']['public']
            print(
                vector['id'],
                public['outcome'],
                public.get('code') or '',
                '(register)' if vector['register']['required'] else '',
                file=sys.stderr,
            )
    alike(vectors)
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/key_cast_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'query': QUERY,
        'anchor_time': ANCHOR,
        'register_ref': REGISTER_REF,
        'keys': [{'id': key['id'], 'setup': key['setup']} for key in KEYS],
        'vectors': vectors,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    required = sum(v['register']['required'] for v in vectors)
    print(
        f'{len(vectors)} vectors written to {OUT.relative_to(ROOT)}, '
        f'{required} marked register entry required',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
