#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.32b's old-against-core probe (decision E49), a local pre-push probe,
not a CI gate: generated DuckDB sources holding decimals, asked through
``Schema.from_source``, ``materialize`` and ``predict`` on the reference and
on the core route, compared by their public answers.

    python -I contract/tools/decimal_probe.py run \
        --reference-python REF --candidate-python CAND [--seed 32] [--n 50]
    python -I contract/tools/decimal_probe.py run \
        --reference-python REF --candidate-python CAND --targeted

``REF`` is the reference checkout's environment (``ref-pandas3``), ``CAND``
one holding the candidate's client; each runs ``observe``, which hands the
sources to ``contract/tools/decimal_record.py``'s ``run`` as cases
(``python-ref`` from ``~/code/sdmc-reference``, ``python-core`` forced onto
every DuckDB route of the three calls) and keeps its ``project`` of each:
the outcome (result, stage, class and code), the public schema, the warning
classes, each request as the ``payload`` rule compares it, the result, and
each decimal column's dtype, stype and cells as sent.

Each generated source is ``contract/bases.yaml``'s ``shop`` with one
decimal of a random precision (1 to 38) and scale (0 to the precision),
its eight values drawn per source (small or at the precision's end, of
both signs, zero and null among them), in one of three places: a feature
``f`` of the entity or the fact table under the recorder's question (a
regression over ``SUM(orders.price)``); the fact table's ``price`` under a
windowed ``SUM``, ``AVG``, ``MIN``, ``MAX`` or ``COUNT``, some values near
the precision's end so a sum passes it; or a condition, a sum of that
price or a static ``users.f`` compared with an integer or a float.
``--targeted`` runs hand-picked sources instead: static conditions with
constants the column's type holds or not, at small and large scales and
past int64; sums at the declared precision's end; averages that round.

``run`` prints, per source, whether the answers agree and, where they do
not, the parts that differ with both outcomes; it exits 1 on any
difference not explained, or on no evidence. A difference is explained by
the register entries the package proposes, named in ``EXPLAINED``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path.home() / 'code/sdmc-reference'
AGGREGATIONS = ('SUM', 'AVG', 'MIN', 'MAX', 'COUNT')
OPS = ('>', '>=', '<', '<=', '=', '!=')
#: The smallest scale at which the reference's pyarrow reads a float
#: constant by its approximation (Arrow's ``Decimal128::FromReal``) where the
#: core reads it exactly (measured: exact through scale 27).
APPROXIMATE_FROM = 28
#: The labelled target, whose cells P1B-71's exact reading may change.
TARGET = 'instance_table.TARGET'


def _masked(answer: dict[str, Any]) -> dict[str, Any]:
    r"""``answer`` without what a label decides: the target's cells in each
    request and each request's digest."""
    answer = copy.deepcopy(answer)
    for request in answer.get('payload') or []:
        request.pop('digest', None)
    target = (answer.get('columns') or {}).get(TARGET) or {}
    for each in target.get('request') or []:
        for side in ('context', 'predict'):
            if isinstance(each.get(side), dict):
                each[side].pop('cells', None)
    return answer


def _p1b71_hint(
    vector: dict[str, Any], ref: dict[str, Any], core: dict[str, Any]
) -> bool:
    r"""Whether P1B-71 may be a difference's cause: a float compared with a
    static decimal of a scale of 28 or more, both answers successes alike
    in every part but the target's labels and the request's digest. Only a
    hint: the probe cannot verify which rows the reference's approximation
    labels otherwise, so it never takes this as an explanation (the
    orchestrator's decision after the bundle's review round 2)."""
    return bool(
        vector.get('static_condition')
        and vector.get('float_constant')
        and vector.get('scale', 0) >= APPROXIMATE_FROM
        and ref['outcome'] == core['outcome']
        and ref['outcome']['result'] == 'success'
        and _masked(ref) == _masked(core)
    )


#: The differences a register entry of the package explains: a predicate
#: on the source and on both answers, and the entry. None: P1B-71's exact
#: reading of a float constant may label rows otherwise than the
#: reference's approximation from scale 28, but the probe cannot verify
#: which, so such a difference is reported unexplained, its line naming
#: P1B-71 as a possible cause (:func:`_p1b71_hint`).
EXPLAINED: list[tuple[str, Any]] = []


def _is_float(constant: str) -> bool:
    return any(c in constant for c in '.eE')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def recorder() -> ModuleType:
    return _module('decimal_record', ROOT / 'contract/tools/decimal_record.py')


# --- the sources --------------------------------------------------------------


def _value(rng: random.Random, p: int, s: int, at_end: bool) -> Decimal:
    digits = p if at_end else rng.randint(1, min(p, 4))
    top = 10**digits - 1
    unscaled = top - rng.randint(0, 2) if at_end else rng.randint(0, top)
    unscaled = max(unscaled, 0)
    if rng.random() < 0.4:
        unscaled = -unscaled
    # Built from its digits: no context rounds it.
    return Decimal(f'{unscaled}E-{s}')


def _values(rng: random.Random, p: int, s: int) -> list[Decimal | None]:
    at_end = rng.random() < 0.3
    values: list[Decimal | None] = [
        _value(rng, p, s, at_end and rng.random() < 0.6) for _ in range(6)
    ]
    values += [Decimal(0), None]
    rng.shuffle(values)
    return values


def _constant(rng: random.Random) -> str:
    return rng.choice(
        ['0', '1', '-3', '5000', '0.5', '0.25', '0.1', '2.675', '-0.75', '1e3']
    )


def generated(seed: int, n: int) -> list[dict[str, Any]]:
    R = recorder()
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    target = [['instance_table', 'TARGET']]
    for i in range(n):
        p = rng.randint(1, 38)
        s = rng.randint(0, p)
        values = _values(rng, p, s)
        place = rng.choice(['feature', 'feature', 'target', 'condition'])
        tag = f'gen{i:02d}.{R.tag(p, s)}'
        if place == 'feature':
            table = rng.choice(['users', 'orders'])
            out.append(
                {
                    'id': f'{tag}.feature-{table}',
                    'columns': [*target, [table, 'f']],
                    'setup': [R.add_f(table, values, p, s)],
                }
            )
        elif place == 'target':
            aggregation = rng.choice(AGGREGATIONS)
            out.append(
                {
                    'id': f'{tag}.{aggregation.lower()}',
                    'columns': [*target, ['orders', 'price']],
                    'setup': [R.price_as(values, p, s)],
                    'pql': (
                        f'PREDICT {aggregation}(orders.price, 0, 30, days) '
                        'FOR EACH users.user_id'
                    ),
                }
            )
        elif rng.random() < 0.5:
            op, constant = rng.choice(OPS), _constant(rng)
            out.append(
                {
                    'id': f'{tag}.sum-condition',
                    'columns': [*target, ['orders', 'price']],
                    'setup': [R.price_as(values, p, s)],
                    'pql': (
                        f'PREDICT SUM(orders.price, 0, 30, days) {op} '
                        f'{constant} FOR EACH users.user_id'
                    ),
                }
            )
        else:
            op, constant = rng.choice(OPS), _constant(rng)
            out.append(
                {
                    'id': f'{tag}.static-condition',
                    'columns': [*target, ['users', 'f']],
                    'setup': [R.add_f('users', values, p, s)],
                    'pql': f'PREDICT users.f {op} {constant} FOR EACH users.user_id',
                    'anchor_time': None,
                    'static_condition': True,
                    'scale': s,
                    'float_constant': _is_float(constant),
                    'values': [None if v is None else str(v) for v in values],
                }
            )
    return out


def targeted() -> list[dict[str, Any]]:
    R = recorder()
    out: list[dict[str, Any]] = []
    target = [['instance_table', 'TARGET']]
    spread = ['-2.5', '-1', '0', '0.25', '1', '2', '3.75', None]

    def static(name: str, p: int, s: int, condition: str) -> None:
        values = [None if v is None else Decimal(v) for v in spread]
        if s == 0:
            values = [
                None if v is None else v.to_integral_value() for v in values
            ]
        if p - s < 1:
            values = [None if v is None else v / 10 for v in values]
        out.append(
            {
                'id': f'static.{name}.{R.tag(p, s)}',
                'columns': [*target, ['users', 'f']],
                'setup': [R.add_f('users', values, p, s)],
                'pql': f'PREDICT users.f {condition} FOR EACH users.user_id',
                'anchor_time': None,
                'static_condition': True,
                'scale': s,
                'float_constant': _is_float(condition.split()[-1]),
                'values': [None if v is None else str(v) for v in values],
            }
        )

    static('gt-int', 4, 2, '> 1')
    static('gt-int-not-held', 4, 2, '> 100')
    static('eq-int-held', 4, 0, '= 1000')
    static('gt-float-exact', 4, 2, '> 0.25')
    static('gt-float-inexact', 4, 2, '> 0.1')
    static('ge-float-rounds', 4, 2, '>= 0.255')
    static('lt-float-half', 4, 0, '< 0.5')
    static('gt-float-small-scale', 18, 18, '> 0.1')
    static('gt-float-scale-20', 38, 20, '> 0.1')
    static('gt-float-scale-22', 38, 22, '> 0.1')
    static('gt-float-scale-23', 38, 23, '> 0.1')
    static('gt-float-scale-23-exact', 38, 23, '> 0.25')
    static('gt-float-scale-30', 38, 30, '> 0.7')
    static('gt-int-large-scale', 38, 38, '> 0')
    static('gt-int-past-int64', 38, 0, '> 10000000000000000000')
    static('ne-null', 9, 2, '!= NULL')
    for p, s in [(4, 0), (4, 2), (9, 9), (18, 0), (19, 2)]:
        top = R.largest(p, s)
        values: list[Decimal | None] = [top, -top, R.ulp(s), None] * 2
        out.append(
            {
                'id': f'sum.at-end.{R.tag(p, s)}',
                'columns': [*target, ['orders', 'price']],
                'setup': [R.price_as(values, p, s)],
            }
        )
    for p, s, values in [
        (4, 0, ['1', '2', '2', '1', '1', '2', '2', '1']),
        (4, 2, ['0.10', '0.20', '0.20', '-0.15', '0.05', '0.01', '0.02', None]),
        (38, 10, ['1', '2', '-3', '0.0000000001', '5', '7', '8', None]),
    ]:
        out.append(
            {
                'id': f'avg.rounds.{R.tag(p, s)}',
                'columns': [*target, ['orders', 'price']],
                'setup': [
                    R.price_as(
                        [None if v is None else Decimal(v) for v in values],
                        p,
                        s,
                    )
                ],
                'pql': 'PREDICT AVG(orders.price, 0, 30, days) FOR EACH users.user_id',
            }
        )
    return out


# --- the runs -----------------------------------------------------------------


def observe(mode: str, vectors_path: Path, out: Path) -> None:
    R = recorder()
    vectors = json.loads(vectors_path.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='t132b-probe-') as tmp:
        observed = R.run(
            mode,
            vectors,
            Path(tmp),
            core=mode == 'python-core',
            reference=REFERENCE if mode == 'python-ref' else None,
        )
    answers = {v['id']: R.project(observed[v['id']], v) for v in vectors}
    if mode == 'python-core':
        for vector in vectors:
            answer = answers[vector['id']]
            if (
                vector.get('static_condition')
                and answer['outcome']['code'] == 'UNSUPPORTED_REQUEST'
            ):
                answer['core_rule'] = core_rule(vector)
    out.write_text(json.dumps(answers, indent=1), encoding='utf-8')


def core_rule(vector: dict[str, Any]) -> str | None:
    r"""The rule the core's refusal of ``vector`` names, from the same
    public calls in process (the runner's observation keeps no rule)."""
    import os
    import warnings

    from sdm_client import PredictiveQuery, Schema
    from sdm_client.errors import StructuredDataError
    from sdm_connectors import DuckDBSource

    runner = _module('run_python', ROOT / 'contract/runners/run_python.py')
    case = recorder().case_for(vector, core=True)
    connection = runner._connection(case, runner.load_bases())
    os.environ['SDM_CORE_ROUTES'] = ','.join(case['routes'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            PredictiveQuery(case['input']['pql'], schema).materialize(
                **runner._materialize_arguments(case['input']['materialize'])
            )
    except StructuredDataError as refused:
        return (getattr(refused, 'details', None) or {}).get('rule')
    finally:
        os.environ.pop('SDM_CORE_ROUTES', None)
    return None


def _outcome(answer: dict[str, Any]) -> str:
    o = answer['outcome']
    if o['result'] == 'success':
        return f'success at {o["stage"]}'
    return f'refusal {o["code"] or o["error"]} at {o["stage"]}'


def compare(
    vectors: list[dict[str, Any]],
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> int:
    if not vectors or set(reference) != set(candidate) or not reference:
        print('no evidence: the runs do not cover the sources')
        return 1
    agree = explained = 0
    unexplained: list[str] = []
    for vector in vectors:
        ref, core = reference[vector['id']], candidate[vector['id']]
        parts = sorted(k for k in ref if ref[k] != core.get(k))
        if not parts:
            agree += 1
            print(f'{vector["id"]}: agree, {_outcome(ref)}')
            continue
        line = (
            f'{vector["id"]}: {parts} differ: the reference {_outcome(ref)}, '
            f'the core {_outcome(core)} ({vector.get("pql", "")})'
        )
        entry = next(
            (e for e, holds in EXPLAINED if holds(vector, ref, core)), None
        )
        if entry is None:
            if _p1b71_hint(vector, ref, core):
                line += (
                    ', possibly P1B-71 (the exact reading of the constant), '
                    'unverified'
                )
            unexplained.append(line)
        else:
            explained += 1
            line += f', under {entry}'
        print(line)
    print(
        f'{len(vectors)} sources: {agree} agree, {explained} explained, '
        f'{len(unexplained)} unexplained'
    )
    return 1 if unexplained else 0


def run(args: argparse.Namespace) -> int:
    vectors = targeted() if args.targeted else generated(args.seed, args.n)
    with tempfile.TemporaryDirectory(prefix='t132b-probe-') as tmp:
        path = Path(tmp) / 'vectors.json'
        path.write_text(json.dumps(vectors), encoding='utf-8')
        answers = {}
        for mode, python in (
            ('python-ref', args.reference_python),
            ('python-core', args.candidate_python),
        ):
            out = Path(tmp) / f'{mode}.json'
            done = subprocess.run(
                [python, '-I', __file__, 'observe', mode, str(path), str(out)],
                capture_output=True,
                text=True,
                cwd=ROOT,
                check=False,
            )
            if done.returncode != 0:
                print(f'{mode} failed:\n{done.stdout}\n{done.stderr}')
                return 1
            answers[mode] = json.loads(out.read_text(encoding='utf-8'))
    return compare(vectors, answers['python-ref'], answers['python-core'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    go = sub.add_parser('run')
    go.add_argument('--reference-python', required=True)
    go.add_argument('--candidate-python', required=True)
    go.add_argument('--seed', type=int, default=32)
    go.add_argument('--n', type=int, default=50)
    go.add_argument('--targeted', action='store_true')
    seen = sub.add_parser('observe')
    seen.add_argument('mode', choices=('python-ref', 'python-core'))
    seen.add_argument('vectors', type=Path)
    seen.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.command == 'observe':
        observe(args.mode, args.vectors, args.out)
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
