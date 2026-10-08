# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The sdm-core wheel's smoke test.

Run as ``python smoke.py`` wherever a wheel is installed: the phase venv, a
manylinux build container, Python 3.10 with pyarrow 8.0.0. It needs nothing
but pyarrow and the wheel. It exits non-zero on the first failure, and on
success prints one line starting ``SDM_CORE_SMOKE_OK`` with the SHA-256 of
this file, so a log holding that line proves this test ran.

It checks:
- ``version()`` is the installed distribution's version;
- ``roundtrip_arrow`` gives back a table, and a stream, unchanged and without
  copying: every buffer at the same address;
- ``roundtrip_payload`` gives back a request parsed-equal, and refuses one
  that is not a request.
"""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    print(f'SDM_CORE_SMOKE_FAILED {message}', file=sys.stderr)
    sys.exit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def check_version(native: Any) -> str:
    from importlib.metadata import version

    got = native.version()
    check(isinstance(got, str) and got, f'version() gave {got!r}')
    installed = version('sdm-core')
    if got != installed:
        try:
            from packaging.version import Version
        except ImportError:
            fail(f'version() is {got!r}, the wheel is {installed!r}')
        check(
            Version(got) == Version(installed),
            f'version() is {got!r}, the wheel is {installed!r}',
        )
    return got


# --- Arrow ---------------------------------------------------------------------


def make_batches(pa: Any) -> tuple[Any, list[Any]]:
    r"""Three batches over most of the types a sampler reads.

    A column either has a null in every batch or no validity buffer at all,
    so that a buffer pyarrow allocated is one a faithful round trip keeps.
    """
    schema = pa.schema(
        [
            pa.field(
                'id', pa.int64(), nullable=False, metadata={'role': 'key'}
            ),
            pa.field('amount', pa.float64()),
            pa.field('name', pa.string()),
            pa.field('flag', pa.bool_()),
            pa.field('at', pa.timestamp('us', tz='UTC')),
            pa.field('day', pa.date32()),
            pa.field('price', pa.decimal128(12, 2)),
            pa.field('tags', pa.list_(pa.string())),
            pa.field('segment', pa.dictionary(pa.int32(), pa.string())),
            pa.field(
                'point',
                pa.struct([('x', pa.int32()), ('label', pa.string())]),
            ),
        ],
        metadata={'origin': 'sdm-core smoke'},
    )
    batches = []
    for b in range(3):
        n = 4 + b
        base = 1000 * b
        arrays = [
            pa.array([base + i for i in range(n)], pa.int64()),
            pa.array(
                [None] + [base + i + 0.5 for i in range(1, n)], pa.float64()
            ),
            pa.array([f'name-{base + i}' for i in range(n - 1)] + [None]),
            pa.array([i % 2 == 0 for i in range(n)], pa.bool_()),
            pa.array(
                [
                    dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
                    + dt.timedelta(hours=base + i)
                    for i in range(n)
                ],
                pa.timestamp('us', tz='UTC'),
            ),
            pa.array(
                [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)],
                pa.date32(),
            ),
            pa.array(
                [decimal.Decimal(f'{base + i}.25') for i in range(n)],
                pa.decimal128(12, 2),
            ),
            pa.array(
                [[f't{j}' for j in range(i % 3)] for i in range(n)],
                pa.list_(pa.string()),
            ),
            pa.DictionaryArray.from_arrays(
                pa.array([i % 2 for i in range(n)], pa.int32()),
                pa.array(['north', 'south']),
            ),
            pa.StructArray.from_arrays(
                [
                    pa.array(list(range(n)), pa.int32()),
                    pa.array([f'p{i}' for i in range(n)]),
                ],
                names=['x', 'label'],
            ),
        ]
        batches.append(pa.RecordBatch.from_arrays(arrays, schema=schema))
    return schema, batches


def addresses(array: Any) -> list[Any]:
    out = [None if b is None else b.address for b in array.buffers()]
    dictionary = getattr(array, 'dictionary', None)
    if dictionary is not None:
        out.append(addresses(dictionary))
    return out


def batch_addresses(batch: Any) -> list[Any]:
    return [addresses(column) for column in batch.columns]


def check_table(pa: Any, native: Any, schema: Any, batches: list[Any]) -> None:
    table = pa.Table.from_batches(batches, schema=schema)
    expected = [batch_addresses(b) for b in table.to_batches()]
    out = native.roundtrip_arrow(table)
    check(isinstance(out, pa.Table), f'a table came back as {type(out)}')
    check(
        out.schema.equals(schema, check_metadata=True),
        f'the schema changed:\n{out.schema}\nwas\n{schema}',
    )
    check(out.equals(table), 'the table came back with different values')
    got = [batch_addresses(b) for b in out.to_batches()]
    check(
        len(got) == len(expected),
        f'{len(expected)} chunks came back as {len(got)}',
    )
    check(got == expected, 'a table came back copied: buffer addresses moved')


def check_reader(pa: Any, native: Any, schema: Any, batches: list[Any]) -> None:
    expected = [batch_addresses(b) for b in batches]
    reader = pa.RecordBatchReader.from_batches(schema, batches)
    out = native.roundtrip_arrow(reader)
    check(
        isinstance(out, pa.RecordBatchReader),
        f'a stream came back as {type(out)}',
    )
    check(
        out.schema.equals(schema, check_metadata=True),
        'the stream schema changed',
    )
    got_batches = []
    while True:
        try:
            got_batches.append(out.read_next_batch())
        except StopIteration:
            break
    check(
        len(got_batches) == len(batches),
        f'{len(batches)} batches came back as {len(got_batches)}',
    )
    for got, want in zip(got_batches, batches, strict=True):
        check(got.equals(want), 'a batch came back with different values')
    got = [batch_addresses(b) for b in got_batches]
    check(got == expected, 'a stream came back copied: buffer addresses moved')


def check_arrow(pa: Any, native: Any) -> None:
    schema, batches = make_batches(pa)
    check_table(pa, native, schema, batches)
    check_reader(pa, native, schema, batches)
    # No rows, no batches: still the same schema back.
    empty = native.roundtrip_arrow(pa.Table.from_batches([], schema=schema))
    check(
        isinstance(empty, pa.Table) and empty.num_rows == 0,
        'an empty table did not come back empty',
    )
    check(
        empty.schema.equals(schema, check_metadata=True),
        'an empty table lost its schema',
    )
    for wrong in ({'id': [1, 2]}, 42, None):
        try:
            native.roundtrip_arrow(wrong)
        except Exception:
            continue
        fail(f'roundtrip_arrow accepted {wrong!r}')


# --- Payloads ------------------------------------------------------------------

PAYLOAD = {
    'model': 'kumo-tabular',
    'task': {
        'kind': 'classification',
        'target': {'column_name': 'churn', 'dtype': 'int64'},
    },
    'schema': {
        'instance_table': {
            'columns': {
                'user_id': {'dtype': 'int64', 'stype': 'ID'},
                'age': {'dtype': 'float64', 'stype': 'numerical'},
                'segment': {'dtype': 'string', 'stype': 'categorical'},
                'churn': {'dtype': 'int64', 'stype': 'categorical'},
            },
            'primary_key': ['user_id'],
        }
    },
    'context': {
        'instance_table': {
            'format': 'arrays',
            'columns': ['user_id', 'age', 'segment', 'churn'],
            'rows': [
                [0, 20.0, 'south', 0],
                ['9223372036854775807', 21.5, None, 1],
            ],
        }
    },
    'predict': {
        'instance_table': {
            'format': 'arrays',
            'columns': ['user_id', 'age', 'segment'],
            'rows': [[2, 22.0, 'north']],
        }
    },
    'output': {'fields': ['prediction', 'probabilities']},
    'metadata': {'request_id': '<request-id>'},
}


def same_json(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(same_json, a, b))
    return type(a) is type(b) and a == b


def check_payload(native: Any) -> None:
    out = native.roundtrip_payload(json.dumps(PAYLOAD))
    check(isinstance(out, str), f'roundtrip_payload gave {type(out)}')
    check(
        same_json(json.loads(out), PAYLOAD),
        f'the payload came back different:\n{out}',
    )
    missing = {k: v for k, v in PAYLOAD.items() if k != 'task'}
    for wrong in ('{', json.dumps(missing), json.dumps({**PAYLOAD, 'x': 1})):
        try:
            native.roundtrip_payload(wrong)
        except ValueError:
            continue
        fail(f'roundtrip_payload accepted {wrong[:80]!r}')


def main() -> None:
    import pyarrow as pa
    from sdm_core import _native as native

    version = check_version(native)
    check_arrow(pa, native)
    check_payload(native)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(
        f'SDM_CORE_SMOKE_OK sha256={digest} machine={platform.machine()} '
        f'python={platform.python_version()} pyarrow={pa.__version__} '
        f'version={version} native={Path(native.__file__).resolve()}'
    )


if __name__ == '__main__':
    main()
