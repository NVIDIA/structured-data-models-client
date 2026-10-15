#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's handling of a float column inferred categorical
asked as a static target into ``contract/float-multiclass-vectors.json``
(T1.36 final-review fix group 1, P2-1a).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    TZ=UTC ~/code/sdmc-reference/.venv/bin/python -I contract/tools/float_multiclass_record.py

Written by T1.36's independent test author before the fix. A DuckDB
``DOUBLE`` or ``FLOAT`` column holding few distinct values (fewer than 20 in
its first 1,000 rows, or at most 5% of them) is described ``categorical``,
and ``PREDICT users.f FOR EACH users.user_id`` over it is a
``multiclass_classification``. The reference (``payload._task_spec``,
``_target_dtype``, ``_class_label``) sends the label column as ``float32``,
each label the float32's double as a JSON number, and the classes as the
text ``str`` gives that double, in the order ``pd.unique`` first meets them
in the context's training labels. The core refuses the question today (``UNSUPPORTED_REQUEST``,
rule ``target_type``, ``sdm-request/src/payload.rs``).

Each vector is ``contract/bases.yaml``'s ``shop`` with a column ``f`` added
to ``users`` (``shop``'s own 8 users) or ``users`` and ``orders`` replaced
by generated tables of 1,200 users (``large.*``; ``multicategorical_record
.generated``), asked :data:`QUESTION` for ``indices`` ``[0, 1]``:

- ``double.*`` and ``float.*``: the column as ``DOUBLE`` and as ``FLOAT``:
  whole values, tenths (float32 rounding), the spellings ``str`` gives
  small and large values, two doubles that round to one float32, the
  float32 limits (its largest, a subnormal, a double that underflows to
  zero), a null label in the context and on an entity asked for, and a
  negative zero alone;
- ``preview.*``: a NaN or both signed zeros in the first 1,000 rows, which
  the core's ``Schema.from_source`` refuses by name (``float_nan``,
  ``float_signed_zero``: E92's Phase 2 class, approved by the owner); kept
  so the reference's answer is on record for that lift;
- ``large.*``: 1,200 users: a NaN or the other signed zero only past the
  first 1,000 rows (so the description takes the column and the labels
  carry them), ten classes, and a control of 61 values, which the
  reference describes ``numerical`` and asks as a regression;
- ``refuse.*``: what the reference refuses: an infinite label in the
  context or on an entity asked for, a double past float32's range
  (``INVALID_REQUEST`` at materialize), one class in the context, none,
  or 40, past the relational model's ten (``UNSUPPORTED_REQUEST`` at
  predict, nothing sent).

Each vector is a case in ``contract/cases``' format
(``contract/cases/s1/base.multiclass.json``'s, its setup and question the
vector's), run by ``contract/runners/run_python.py --mode python-ref``
through the public calls, and this script keeps the reference's answer as
``project`` reads it from the observation: ``empty_null_record.project``'s
(the outcome, the public schema, the warning classes, each request's
digest as the ``payload`` rule compares it with its tables' rows and null
cells, the result); ``task``, each request's ``task.target`` (its dtype and
classes, in order) and ``task.kind``; and ``labels``, the label column
(``TARGET``) as it travels in each request
(``multicategorical_record.wire``: its declared dtype and stype, its
cells, kinds, distinct cells and digest). Beside it, what DuckDB holds:
``f``'s type, row count, nulls, NaNs and zeros of each sign in the first
1,000 rows and past them, the distinct values, and, for ``shop``'s 8
users, each value's ``repr``.

``register`` marks the vectors where the core is expected to differ, as
``empty_null_record.register`` derives it from the reference's answer
(the reference fails unnamed or with ``INTERNAL_ERROR``); none is marked.

Nothing here is written by hand but the inputs: every expected value is
the reference's. ``VECTORS``, ``case_for``, ``run`` and ``project`` are
imported by ``contract/tests/test_t136_float_multiclass.py``, so
``python-core`` and ``python-old`` are observed and projected by the same
code.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/float-multiclass-vectors.json'
TEMPLATE = ROOT / 'contract/cases/s1/base.multiclass.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-float-multiclass-vectors/1'
REGISTER_REF = 'contract/float-multiclass-vectors.json#'
CASE_PREFIX = 't136g1.'
QUESTION = 'PREDICT users.f FOR EACH users.user_id'
INDICES = [0, 1]
#: The label column's name in a request.
LABEL = 'TARGET'
#: The rows of the description's preview.
PREVIEW = 1000
#: The generated tables' users.
LARGE = 1200


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


EN = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')
MC = _module(
    'multicategorical_record',
    ROOT / 'contract/tools/multicategorical_record.py',
)


# --- the inputs ----------------------------------------------------------------


def cycle(values: list[str], sql_type: str, over: str = 'user_id') -> str:
    r"""A DuckDB expression taking ``values`` (SQL literals, each cast to
    ``sql_type``) in turn by ``over``: the first value on user 0."""
    items = ', '.join(
        'NULL' if v == 'NULL' else f'{v}::{sql_type}' for v in values
    )
    return f'([{items}])[1 + {over} % {len(values)}]'


def _shop(vid: str, expression: str) -> dict[str, Any]:
    return {'id': vid, 'rows': 8, 'setup': MC.on_shop(expression)}


def _large(vid: str, expression: str) -> dict[str, Any]:
    return {
        'id': vid,
        'rows': LARGE,
        'setup': MC.generated(expression, LARGE),
    }


def _typed() -> list[dict[str, Any]]:
    out = []
    for sql_type, prefix in (('DOUBLE', 'double'), ('FLOAT', 'float')):
        out += [
            # The finding's own: whole values, 1.0 to 5.0.
            _shop(f'{prefix}.one-to-five', f'(1 + user_id % 5)::{sql_type}'),
            # Tenths: the float32's double is not the decimal.
            _shop(
                f'{prefix}.tenths',
                cycle(['0.1', '0.2', '0.3'], sql_type),
            ),
        ]
    out += [
        _shop(
            'float.null-in-context',
            'CASE WHEN user_id = 3 THEN NULL ELSE (1 + user_id % 3)::FLOAT END',
        ),
        # Each spelling ``str`` gives: exponents below 1e-4, at and past
        # 1e16; the first two users (asked for) hold plain values.
        _shop(
            'double.small-spellings',
            cycle(
                [
                    '7.0',
                    '8.0',
                    '1e-7',
                    '0.0001',
                    '0.001',
                    '1e-5',
                    '-0.5',
                    '2.5',
                ],
                'DOUBLE',
            ),
        ),
        _shop(
            'double.large-spellings',
            cycle(
                [
                    '7.0',
                    '8.0',
                    '1e20',
                    '123456789.0',
                    '1e16',
                    '16777217.0',
                    '9999999999999998.0',
                    '-1e10',
                ],
                'DOUBLE',
            ),
        ),
        # 0.1 and 0.1000000000001 are one float32: one class.
        _shop(
            'double.one-float32',
            cycle(['0.1', '0.1000000000001', '0.2'], 'DOUBLE'),
        ),
        # float32's largest, a subnormal, a double that underflows to 0.
        _shop(
            'double.float32-limits',
            cycle(
                [
                    '1.0',
                    '2.0',
                    '3.4028234663852886e38',
                    '1e-40',
                    '1e-46',
                    '1.0',
                    '2.0',
                    '1e-40',
                ],
                'DOUBLE',
            ),
        ),
        _shop(
            'double.null-in-context',
            'CASE WHEN user_id = 3 THEN NULL '
            'ELSE (1 + user_id % 3)::DOUBLE END',
        ),
        _shop(
            'double.null-asked',
            'CASE WHEN user_id = 0 THEN NULL '
            'ELSE (1 + user_id % 3)::DOUBLE END',
        ),
        # A negative zero and no positive one: the description takes it.
        _shop(
            'double.negative-zero',
            cycle(["'-0.0'", '1.0', '2.0'], 'DOUBLE'),
        ),
        _shop(
            'float.negative-zero',
            cycle(["'-0.0'", '1.0', '2.0'], 'FLOAT'),
        ),
    ]
    return out


#: The vectors ``Schema.from_source`` refuses on the core by an approved
#: rule (E92's Phase 2 class), by the rule.
PREVIEW_RULES = {
    'preview.double-nan': 'float_nan',
    'preview.float-nan': 'float_nan',
    'preview.double-nan-asked': 'float_nan',
    'preview.double-signed-zeros': 'float_signed_zero',
    'preview.float-signed-zeros': 'float_signed_zero',
}


def _preview() -> list[dict[str, Any]]:
    return [
        _shop(
            'preview.double-nan',
            "CASE WHEN user_id = 3 THEN 'nan'::DOUBLE "
            'ELSE (1 + user_id % 3)::DOUBLE END',
        ),
        _shop(
            'preview.float-nan',
            "CASE WHEN user_id = 3 THEN 'nan'::FLOAT "
            'ELSE (1 + user_id % 3)::FLOAT END',
        ),
        _shop(
            'preview.double-nan-asked',
            "CASE WHEN user_id = 0 THEN 'nan'::DOUBLE "
            'ELSE (1 + user_id % 3)::DOUBLE END',
        ),
        _shop(
            'preview.double-signed-zeros',
            cycle(['0.0', "'-0.0'", '1.0'], 'DOUBLE'),
        ),
        _shop(
            'preview.float-signed-zeros',
            cycle(["'-0.0'", '0.0', '1.0'], 'FLOAT'),
        ),
    ]


def _past_preview() -> list[dict[str, Any]]:
    return [
        _large('large.double-whole', '(1 + i % 3)::DOUBLE'),
        _large(
            'large.double-nan-past-preview',
            f"CASE WHEN i >= {PREVIEW} AND i % 2 = 0 THEN 'nan'::DOUBLE "
            'ELSE (1 + i % 3)::DOUBLE END',
        ),
        _large(
            'large.float-nan-past-preview',
            f"CASE WHEN i >= {PREVIEW} AND i % 2 = 0 THEN 'nan'::FLOAT "
            'ELSE (1 + i % 3)::FLOAT END',
        ),
        # +0.0 in the preview, -0.0 past it: one class, spelled as the
        # context first holds it.
        _large(
            'large.double-negative-zero-past-preview',
            f"CASE WHEN i >= {PREVIEW} AND i % 2 = 0 THEN '-0.0'::DOUBLE "
            'WHEN i % 3 = 0 THEN 0.0::DOUBLE ELSE (1 + i % 3)::DOUBLE END',
        ),
        # -0.0 in the preview, +0.0 past it.
        _large(
            'large.double-positive-zero-past-preview',
            f'CASE WHEN i >= {PREVIEW} AND i % 2 = 0 THEN 0.0::DOUBLE '
            "WHEN i % 3 = 0 THEN '-0.0'::DOUBLE "
            'ELSE (1 + i % 3)::DOUBLE END',
        ),
        # Ten classes, the most the relational model takes.
        _large('large.double-ten-classes', '(i % 10)::DOUBLE / 8'),
        _large('large.float-ten-classes', '((i % 10)::DOUBLE / 8)::FLOAT'),
        # The control: 61 values (6.1%) are numerical, a regression.
        _large('large.double-sixty-one-values', '(i % 61)::DOUBLE / 8'),
    ]


def _refused() -> list[dict[str, Any]]:
    return [
        _shop(
            'refuse.double-inf-in-context',
            "CASE WHEN user_id = 3 THEN 'inf'::DOUBLE "
            'ELSE (1 + user_id % 3)::DOUBLE END',
        ),
        _shop(
            'refuse.float-negative-inf-in-context',
            "CASE WHEN user_id = 3 THEN '-inf'::FLOAT "
            'ELSE (1 + user_id % 3)::FLOAT END',
        ),
        _shop(
            'refuse.double-inf-asked',
            "CASE WHEN user_id = 0 THEN 'inf'::DOUBLE "
            'ELSE (1 + user_id % 3)::DOUBLE END',
        ),
        _shop(
            'refuse.double-past-float32',
            cycle(['1e39', '1.0', '2.0'], 'DOUBLE'),
        ),
        # 40 values in 1,000 rows (4%) are categorical: 40 classes, past
        # the relational model's ten.
        _large('refuse.double-forty-classes', '(i % 40)::DOUBLE / 8'),
        _shop('refuse.double-one-class', '2.0::DOUBLE'),
        _shop(
            'refuse.double-no-class',
            'CASE WHEN user_id > 1 THEN NULL ELSE 1.0::DOUBLE END',
        ),
    ]


VECTORS: list[dict[str, Any]] = [
    *_typed(),
    *_preview(),
    *_past_preview(),
    *_refused(),
]


# --- the cases -----------------------------------------------------------------


def case_for(vector: dict[str, Any], core: bool = False) -> dict[str, Any]:
    r"""The vector as a case: ``base.multiclass``'s, its setup the vector's
    and its question :data:`QUESTION`; with ``core``, forced onto the
    core's routes (``empty_null_record.core_routes``), and otherwise on
    none (the runner's own choice, as the reference has)."""
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    case = copy.deepcopy(template)
    case['id'] = CASE_PREFIX + vector['id']
    case['status'] = 'diagnostic'
    case['routes'] = EN.core_routes(vector) if core else []
    case['cells'] = []
    case['input']['setup'] = list(vector['setup'])
    case['input']['pql'] = QUESTION
    case['input']['materialize']['indices'] = list(INDICES)
    case['core']['describe_source']['source'] = EN.core_source(None)
    case['core']['materialize']['indices'] = EN.core_indices(INDICES)
    case['expect'] = {'outcome': 'success', 'stage': 'predict'}
    case['why'] = f'T1.36 fix group 1 vector {vector["id"]}'
    case['source'] = 'contract/tools/float_multiclass_record.py'
    case.pop('register', None)
    return case


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    directory: Path,
    *,
    core: bool = False,
    reference: Path | None = None,
) -> dict[str, dict[str, Any]]:
    r"""``vectors`` as cases through ``run_python.py --mode mode``, in this
    interpreter, from ``directory``, forced onto the core with ``core``:
    each observation by vector id."""
    cases = directory / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        case = case_for(vector, core)
        (cases / f'{case["id"]}.json').write_text(
            json.dumps(case, indent=1), encoding='utf-8'
        )
    out = directory / f'{mode}.jsonl'
    command = [sys.executable, '-I', str(RUNNER), '--mode', mode]
    if reference is not None:
        command += ['--reference', str(reference)]
    command += ['--out', str(out), str(cases)]
    done = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if done.returncode != 0:
        raise RuntimeError(f'{mode} run failed:\n{done.stdout}\n{done.stderr}')
    observed = {}
    for line in out.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        observed[observation['case'][len(CASE_PREFIX) :]] = observation
    return observed


# --- the public answer ---------------------------------------------------------


def task(payloads: Any) -> Any:
    r"""Each request's task as the classes decide it: ``task.kind`` and
    ``task.target`` (its column, dtype and classes, in order); null when
    no request was made."""
    if not payloads:
        return None
    out = []
    for each in payloads:
        parsed = each.get('parsed')
        spec = parsed.get('task') if isinstance(parsed, dict) else None
        if not isinstance(spec, dict):
            out.append(None)
            continue
        out.append({'kind': spec.get('kind'), 'target': spec.get('target')})
    return out


def project(
    observation: dict[str, Any], vector: dict[str, Any]
) -> dict[str, Any]:
    r"""The public answer of one run of ``vector``, as both Python runners
    record it: ``empty_null_record.project``'s, each request's ``task``
    and the label column's ``labels`` (``multicategorical_record.wire``)."""
    answer: dict[str, Any] = EN.project(observation, vector)
    payloads = (observation.get('public') or {}).get('payloads')
    answer['task'] = task(payloads)
    answer['labels'] = MC.wire(payloads, LABEL)
    return answer


outcome_class = EN.outcome_class
register = EN.register


# --- what DuckDB holds ---------------------------------------------------------


def _zero(value: Any, sign: float) -> bool:
    return (
        isinstance(value, float)
        and value == 0.0
        and math.copysign(1.0, value) == sign
    )


def _counts(values: list[Any]) -> dict[str, int]:
    return {
        'nulls': sum(v is None for v in values),
        'nans': sum(isinstance(v, float) and math.isnan(v) for v in values),
        'infinities': sum(
            isinstance(v, float) and math.isinf(v) for v in values
        ),
        'positive_zeros': sum(_zero(v, 1.0) for v in values),
        'negative_zeros': sum(_zero(v, -1.0) for v in values),
    }


def duckdb_holds(vector: dict[str, Any]) -> dict[str, Any]:
    r"""``users`` after the setup: its row count, ``f``'s catalog type,
    the nulls, NaNs, infinities and zeros of each sign ``f`` holds in the
    first :data:`PREVIEW` rows and past them (DuckDB's own order, the
    preview's), the count of its distinct values (NaN and the zeros' signs
    apart, by ``repr``) in the first :data:`PREVIEW` rows, and, up to 8
    rows, each value's ``repr``."""
    import duckdb

    connection = duckdb.connect()
    try:
        for statement in EN.shop() + vector['setup']:
            connection.execute(statement)
        found = connection.execute(
            'SELECT data_type FROM information_schema.columns '
            "WHERE table_name = 'users' AND column_name = 'f'"
        ).fetchone()
        values = [
            v for (v,) in connection.execute('SELECT f FROM users').fetchall()
        ]
        out: dict[str, Any] = {
            'f_type': found[0] if found else None,
            'rows': len(values),
            'preview': _counts(values[:PREVIEW]),
            'past_preview': _counts(values[PREVIEW:]),
            'distinct_in_preview': len(
                {repr(v) for v in values[:PREVIEW] if v is not None}
            ),
        }
        if len(values) <= 8:
            out['values'] = [None if v is None else repr(v) for v in values]
        return out
    finally:
        connection.close()


def main() -> int:
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa

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
    ids = [v['id'] for v in VECTORS]
    if len(set(ids)) != len(ids):
        print('a vector id twice', file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix='t136g1-record-') as tmp:
        observed = run('python-ref', VECTORS, Path(tmp), reference=reference)
    vectors = []
    for vector in VECTORS:
        answer = project(observed[vector['id']], vector)
        why = register(answer)
        entry = {key: copy.deepcopy(value) for key, value in vector.items()}
        entry['duckdb'] = duckdb_holds(vector)
        entry['reference'] = answer
        entry['register'] = {'required': bool(why), 'why': why}
        vectors.append(entry)
        outcome = answer['outcome']
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['error'] or '',
            outcome['code'] or '',
            [t and t['target'].get('classes') for t in answer['task'] or []],
            answer['warnings'],
            '(register)' if why else '',
            file=sys.stderr,
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/float_multiclass_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'template': 'contract/cases/s1/base.multiclass.json',
        'question': QUESTION,
        'indices': INDICES,
        'register_ref': REGISTER_REF,
        'preview_rules': PREVIEW_RULES,
        'payload_exclusions': EN.exclusions(),
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
