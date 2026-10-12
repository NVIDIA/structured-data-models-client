#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The old-against-core differential (decision E49), a local pre-push
probe, not a CI gate: seeded DuckDB sources through the public calls
(``Schema.from_source``, schema edits, ``materialize``, ``predict``), on
the reference and on the core route, every difference classified against
the register.

It is T1.26b's and T1.24b's oracle made one tool (T1.27; checkpoint 7,
subagent P2-3): their generator of sources (entity, fact and lookup
tables, text keys, ties, null times, a second fact table, a table reached
at the second hop), grown with the labels' inputs (aggregations, windows
of days, weeks, hours and months across month ends and a leap day, float,
integer and decimal columns with nulls, conditions, static targets);
their probe of the public calls; and their comparison, which now ends in
a classification.

    python -I contract/tools/differential.py generate SEED N OUT.json
    REF/bin/python -I contract/tools/differential.py probe SPEC OUT --route old
    CAND/bin/python -I contract/tools/differential.py probe SPEC OUT --route core
    python -I contract/tools/differential.py compare OLD CORE [--register FILE]
    python -I contract/tools/differential.py run --reference-python REF \
        --candidate-python CAND [--seed 27] [--n 50] [--out DIR]

``REF`` is the reference checkout's environment (``rust-core-reference``,
pandas 3); ``CAND`` an environment holding the candidate's client with a
``test-ops`` build of ``sdm_client._native`` (``scripts/rust_core/
native_wheel.py --test-ops``): the core probe refuses another, since a
call the core route refuses after its labels (a classification's layout,
T1.28a's) is labelled by the test-only ``labels`` operation in its place,
so its labels are compared too.

What each probe records, per source: ``from_source``'s outcome; each
edit's; ``materialize``'s outcome (its class, ``StructuredDataError.code``
and the ``rule`` its details name), its payload, the sampler's warnings by
the words that name them, and the ``_by_time`` seeds (each example's key
and window ends, as the reference's ``DuckDBSampler._by_time`` makes them
and as the core's key set reaches its provider); on the core route, the
labels operation's examples; then ``predict``'s outcome and result
against the scripted endpoint.

How ``compare`` classifies a source's difference, each by a rule here
naming a register entry, checked against ``docs/design/rust-core/
register.yaml`` (the entry must exist, and for a refusal by rule name its
text must name the rule; its status decides the class):

- the same outcome, payload, seeds and result: agree;
- the core refuses by a rule (``UNSUPPORTED_REQUEST`` naming it) what the
  reference materializes: the entry ``RULES`` names for the rule (P1B-3's
  rules not yet ported, unless another entry is named), and the core's
  labels, where it gave them, compared with the reference payload's
  context exactly (a static target's context leaves out the entities
  predicted for, T1.28a's rule);
- both refuse with the same code: agree; the reference with an internal
  error (``INTERNAL_ERROR``, or an exception with no code) and the core by
  name: the owner's class E66(b), under the entry ``RULES`` names;
- anything else: UNCLASSIFIED, and the command exits 1.

An entry approved (by the owner or as E66(b)'s class) is an approved
class; a proposed one, a proposed entry the review sees. The tool approves
nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNERS = ROOT / 'contract/runners'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'

#: The register entry each core refusal rule is classified under, and for
#: a refusal where the reference fails internally, the entry naming it.
RULES = {
    'classification_task': 'P1B-3',
    'target_leakage': 'P1B-3',
    'wire_dtype': 'P1B-3',
    'condition': 'P1B-3',
    'condition_value': 'P1B-3',
    'logical_target': 'P1B-3',
    'target_filter': 'P1B-3',
    'target_stype': 'P1B-3',
    'aggregation': 'P1B-3',
    'open_window': 'P1B-3',
    'parent_table': 'P1B-3',
    'keyless_neighbors': 'P1B-3',
    'text_column': 'P1B-3',
    'target_type': 'P1B-18',
}
#: What the sampler's warnings are told apart by, in either route's words.
WARNINGS = (
    'Missing',
    'context examples',
    'leak',
    'before the data',
    'after the',
)


# --- generate ---------------------------------------------------------------


def _ts(expr: str) -> str:
    return f"TIMESTAMP '2023-11-20' + to_seconds(({expr})::BIGINT)"


def source(rng: random.Random, i: int) -> dict[str, Any]:
    r"""One generated source: T1.26b's tables, with the labels' inputs."""
    s: dict[str, Any] = {'id': f'g{i:02d}', 'setup': [], 'edits': []}
    key_text = rng.random() < 0.25
    n_e = rng.choice([4, 8, 12, 40, 200])
    eid = "'e' || i" if key_text else 'i::BIGINT'
    cols = [
        f'{eid} AS id',
        '(i % 7)::INTEGER AS size',
        "CASE WHEN i % 9 = 4 THEN NULL ELSE ['US', 'DE', 'IN'][1 + i % 3] END AS cat",
    ]
    ent_time = rng.random() < 0.25
    if ent_time:
        cols.append(f'{_ts("i * 86400")} AS born')
    s['setup'].append(
        f'CREATE TABLE e AS SELECT {", ".join(cols)} FROM range({n_e}) t(i)'
    )
    pks = {'e': 'id'}
    times: dict[str, str] = {'e': 'born'} if ent_time else {}
    rels = [('a', 'e_ref', 'e', 'id')]
    # a: the fact table the labels aggregate, its times across month ends
    # and 2024-02-29, every few hours or days.
    n_a = rng.choice([40, 120, 400, 900])
    step = rng.choice([3_600 * 7, 86_400, 86_400 * 2, 86_400 * 3])
    null_time = rng.random() < 0.3
    tcol = (
        f'CASE WHEN {"i % 11 = 0" if null_time else "FALSE"} THEN NULL '
        f'ELSE {_ts(f"(i // 2) * {step}")} END'
    )
    ekey = (
        f"'e' || (i % {min(n_e, 40)})"
        if key_text
        else f'(i % {min(n_e, 40)})::BIGINT'
    )
    nulls = rng.random() < 0.5
    acols = [
        'i::BIGINT AS aid',
        f'{ekey} AS e_ref',
        f'CASE WHEN {"i % 5 = 1" if nulls else "FALSE"} THEN NULL '
        'ELSE (1 + i * 0.37)::DOUBLE END AS v',
        f'CASE WHEN {"i % 4 = 2" if nulls else "FALSE"} THEN NULL '
        'ELSE (i % 13 - 4)::INTEGER END AS n',
        '((i % 17) * 0.1)::FLOAT AS f',
        f'{tcol} AS ts',
    ]
    decimal = rng.random() < 0.15
    if decimal:
        acols.append('(i * 0.000000000000000001 + 1.25)::DECIMAL(38, 18) AS d')
    s['setup'].append(
        f'CREATE TABLE a AS SELECT {", ".join(acols)} FROM range({n_a}) t(i)'
    )
    pks['a'] = 'aid'
    times['a'] = 'ts'
    if rng.random() < 0.5:
        n_c = rng.choice([3, 6, 50])
        s['setup'].append(
            'ALTER TABLE a ADD COLUMN c_ref BIGINT; '
            f'UPDATE a SET c_ref = aid % {n_c}'
        )
        s['setup'].append(
            "CREATE TABLE c AS SELECT i::BIGINT AS cid, 'Label ' || i AS label, "
            f'(i * 1.25)::DOUBLE AS cost FROM range({n_c}) t(i)'
        )
        pks['c'] = 'cid'
        rels.append(('a', 'c_ref', 'c', 'cid'))
    if rng.random() < 0.3:
        bkey = (
            f"'e' || (i % {min(n_e, 40)})"
            if key_text
            else f'(i % {min(n_e, 40)})::BIGINT'
        )
        s['setup'].append(
            f'CREATE TABLE b AS SELECT (1000 + i)::BIGINT AS bid, {bkey} AS e_ref, '
            f'(i % 13)::INTEGER AS w, {_ts("i * 7200")} AS ts FROM range(120) t(i)'
        )
        pks['b'] = 'bid'
        times['b'] = 'ts'
        rels.append(('b', 'e_ref', 'e', 'id'))
    s['from_source'] = {
        'primary_keys': pks,
        'time_columns': times,
        'relationships': [
            {
                'source_table': a,
                'source_column': b,
                'target_table': c,
                'target_column': d,
            }
            for a, b, c, d in rels
        ],
    }
    s['pql'] = _query(rng, decimal)
    s['indices'] = ['e0', 'e1'] if key_text else [0, 1]
    if rng.random() < 0.6:
        s['anchor_time'] = rng.choice(
            [
                '2024-02-29',
                '2024-01-31',
                '2024-03-31',
                '2023-12-31',
                '2024-03-01',
            ]
        )
    return s


def _query(rng: random.Random, decimal: bool) -> str:
    windows = [
        (0, 30, 'days'),
        (0, 7, 'days'),
        (0, 1, 'months'),
        (0, 2, 'months'),
        (-1, 1, 'months'),
        (0, 1, 'weeks'),
        (0, 48, 'hours'),
        (5, 20, 'days'),
    ]
    start, end, unit = rng.choice(windows)
    kind = rng.choice(['agg', 'agg', 'agg', 'condition', 'condition', 'static'])
    if kind == 'static':
        return rng.choice(
            [
                'PREDICT e.cat FOR EACH e.id',
                'PREDICT e.size FOR EACH e.id',
                'PREDICT e.size > 3 FOR EACH e.id',
            ]
        )
    aggregation = rng.choice(['SUM', 'COUNT', 'AVG', 'MIN', 'MAX'])
    columns = ['v', 'n', 'f'] + (['d'] if decimal else [])
    column = rng.choice(columns)
    if aggregation == 'COUNT' and rng.random() < 0.5:
        column = '*'
    target = f'{aggregation}(a.{column}, {start}, {end}, {unit})'
    if kind == 'agg':
        return f'PREDICT {target} FOR EACH e.id'
    op = rng.choice(['>', '>=', '<', '<=', '=', '!='])
    constant = rng.choice(['0', '1', '3', '0.5', '2.2', '10'])
    return f'PREDICT {target} {op} {constant} FOR EACH e.id'


def generate(seed: int, n: int) -> dict[str, Any]:
    rng = random.Random(seed)
    return {'seed': seed, 'sources': [source(rng, i) for i in range(n)]}


# --- probe --------------------------------------------------------------------


def _endpoint() -> Any:
    spec = importlib.util.spec_from_file_location(
        'scripted_endpoint', RUNNERS / 'scripted_endpoint.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['scripted_endpoint'] = module
    spec.loader.exec_module(module)
    return module


def _outcome(error: BaseException) -> list[Any]:
    details = getattr(error, 'details', None)
    rule = details.get('rule') if isinstance(details, dict) else None
    return [type(error).__name__, getattr(error, 'code', None), rule]


def _text(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _plain(value: Any) -> Any:
    r"""A key as JSON holds it: a numpy scalar as the Python value it is."""
    return value.item() if hasattr(value, 'item') else value


def probe(spec_path: Path, out: Path, route: str) -> None:
    import duckdb
    import pandas as pd
    import pyarrow as pa
    from sdm_client import (
        KumoRelational,
        PredictiveQuery,
        Relationship,
        Schema,
        SDMClient,
    )
    from sdm_connectors import DuckDBSource

    endpoint = _endpoint()
    seeds: list[list[Any]] = []
    labelled: dict[str, Any] = {}
    if route == 'core':
        from sdm_client import _native, _routing
        from sdm_client._rust import materialize as bridge
        from sdm_client._rust.providers import duckdb as providers

        if not getattr(_native, 'TEST_OPS', False):
            raise SystemExit(
                'the core probe needs a test-ops build of sdm_client._native'
            )
        os.environ['SDM_CORE_ROUTES'] = ','.join(
            dict.fromkeys(
                _routing.FROM_SOURCE_DUCKDB
                + _routing.FROM_SOURCE_DUCKDB_TABLES
                + _routing.MATERIALIZE_DUCKDB
                + _routing.PREDICT_DUCKDB
            )
        )
        real_call = providers.DuckDBProvider.__call__

        def call(self: Any, step: dict[str, Any], keys: Any = None) -> Any:
            window = step['request'].get('window') or {}
            if window.get('lower') is not None and keys:
                table = providers._arrow_table(next(iter(keys.values())))
                ends = table['end'].cast(pa.int64()).to_pylist()
                starts = table['start'].cast(pa.int64()).to_pylist()
                seeds.extend(
                    [_plain(k), e, s]
                    for k, e, s in zip(
                        table['key'].to_pylist(), ends, starts, strict=True
                    )
                )
            return real_call(self, step, keys)

        providers.DuckDBProvider.__call__ = call
        real_run = bridge.run

        def run_also_labels(
            operation: dict[str, Any], attachments: Any, **kw: Any
        ) -> Any:
            # The labels operation first, with the same inputs and
            # provider; then the call itself.
            materialize = {k: v for k, v in operation.items() if k != 'op'}
            try:
                # Its warnings are the call's own, which the call gives.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    done = real_run(
                        {
                            'op': 'test',
                            'test': 'labels',
                            'materialize': materialize,
                        },
                        dict(attachments),
                        provider=kw['provider'],
                    )
                stream = done.take(done.result['data'])
                table = pa.RecordBatchReader._import_from_c_capsule(
                    stream.__arrow_c_stream__()
                ).read_all()
                labelled['rows'] = [
                    [
                        _text(e),
                        pd.Timestamp(t, unit='ns').strftime(
                            '%Y-%m-%dT%H:%M:%S.%fZ'
                        ),
                        _text(float(v) if isinstance(v, float) else v),
                    ]
                    for e, t, v in zip(
                        table['entity'].to_pylist(),
                        table['anchor'].cast(pa.int64()).to_pylist(),
                        table['target'].to_pylist(),
                        strict=True,
                    )
                ]
            except Exception as error:  # noqa: BLE001
                labelled['rows'] = _outcome(error)
            seeds.clear()
            return real_run(operation, attachments, **kw)

        bridge.run = run_also_labels
    else:
        from sdm_relational_engine.rfm.backend.duckdb import sampler

        real_by_time = sampler.DuckDBSampler._by_time

        def by_time(
            self: Any,
            table_name: str,
            foreign_key: str,
            index: Any,
            anchor_time: Any,
            min_offset: Any,
            max_offset: Any,
            columns: Any,
        ) -> Any:
            ends = anchor_time + max_offset
            starts = anchor_time + min_offset
            seeds.extend(
                [_plain(k), int(e.value), int(s.value)]
                for k, e, s in zip(list(index), ends, starts, strict=True)
            )
            return real_by_time(
                self,
                table_name,
                foreign_key,
                index,
                anchor_time,
                min_offset,
                max_offset,
                columns,
            )

        sampler.DuckDBSampler._by_time = by_time

    spec = json.loads(spec_path.read_text())
    records = []
    for src in spec['sources']:
        rec: dict[str, Any] = {'id': src['id']}
        connection = duckdb.connect()
        for statement in src['setup']:
            connection.execute(statement)
        kw = dict(src.get('from_source', {}))
        kw['relationships'] = [
            Relationship(**r) for r in kw.get('relationships', [])
        ]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                schema = Schema.from_source(DuckDBSource(connection), **kw)
            rec['from_source'] = ['ok']
        except Exception as error:  # noqa: BLE001
            rec['from_source'] = _outcome(error)
            records.append(rec)
            continue
        args: dict[str, Any] = {'indices': src['indices']}
        if 'anchor_time' in src:
            args['anchor_time'] = pd.Timestamp(src['anchor_time'])
        seeds.clear()
        labelled.clear()
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter('always')
            try:
                request = PredictiveQuery(src['pql'], schema).materialize(
                    **args
                )
                rec['materialize'] = ['ok']
                rec['payload'] = _text(request._payloads[0])
            except Exception as error:  # noqa: BLE001
                request = None
                rec['materialize'] = _outcome(error)
        rec['warnings'] = sorted(
            word
            for message in shown
            for word in WARNINGS
            if word in str(message.message)
        )
        rec['seeds'] = list(seeds)
        if route == 'core':
            rec['labels'] = labelled.get('rows')
        if request is not None:
            with endpoint.serve() as (url, _received):
                try:
                    with SDMClient(url) as client:
                        result = client.predict(request, model=KumoRelational())
                    frame = result.to_pandas()
                    rec['predict'] = ['ok']
                    rec['result'] = [
                        [str(n), str(d)] for n, d in frame.dtypes.items()
                    ] + [hashlib.sha256(frame.to_json().encode()).hexdigest()]
                except Exception as error:  # noqa: BLE001
                    rec['predict'] = _outcome(error)
        records.append(rec)
    out.write_text(json.dumps(records, indent=0) + '\n')


# --- compare ----------------------------------------------------------------


def _register(path: Path) -> dict[str, dict[str, Any]]:
    import yaml

    return {e['id']: e for e in yaml.safe_load(path.read_text())['entries']}


def _cell(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {'number'}:
        text = value['number']
        try:
            return int(text)
        except ValueError:
            return float(text)
    return value


def label_problems(payload: Any, indices: list[Any], core: Any) -> list[str]:
    r"""How the core's labels depart from the reference payload's context,
    as the hosts' label tests read it; missing evidence is a problem."""
    if not isinstance(core, list) or not core or not isinstance(core[0], list):
        return [f'the core gave no labels: {core}']
    try:
        table = payload['context']['instance_table']
        columns = table['columns']
        at = [
            columns.index(c)
            for c in (
                'instance_id',
                '__kumo_entity_ref_0',
                '__kumo_anchor_time',
                'TARGET',
            )
        ]
        count = _cell(payload['metadata']['num_context_examples'])
        rows = table['rows']
    except (KeyError, TypeError, ValueError):
        return ['the reference payload holds no context']
    found = []
    if len(core) != count:
        found.append(f'{len(core)} labels, the reference {count}')
    seen = set()
    for row in rows:
        position = _cell(row[at[0]])
        if not isinstance(position, int) or not 0 <= position < len(core):
            found.append(f'a row names no example: {position}')
            continue
        seen.add(position)
        wanted = [_cell(row[at[1]]), row[at[2]], _cell(row[at[3]])]
        if core[position] != wanted or [type(v) for v in core[position]] != [
            type(v) for v in wanted
        ]:
            found.append(f'example {position}: {core[position]}, not {wanted}')
    for position, example in enumerate(core):
        if position not in seen and example[0] not in indices:
            found.append(
                f'example {position}: not the reference and not predicted for'
            )
    return found


def classify(
    old: dict[str, Any],
    core: dict[str, Any],
    indices: list[Any],
    register: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    r"""``(class, why)``: ``agree``, ``approved:<entry>``,
    ``proposed:<entry>`` or ``UNCLASSIFIED``."""

    def under(entry: str, rule: str | None, why: str) -> tuple[str, str]:
        found = register.get(entry)
        if found is None:
            return 'UNCLASSIFIED', f'{why}; {entry} is not in the register'
        if rule is not None and rule not in json.dumps(found):
            return 'UNCLASSIFIED', f'{why}; {entry} does not name {rule}'
        status = 'approved' if found.get('status') == 'approved' else 'proposed'
        return f'{status}:{entry}', why

    if old.get('from_source') != core.get('from_source'):
        if (
            old['from_source'][0] != 'ok'
            and old['from_source'][1:2] == core['from_source'][1:2]
        ):
            return 'agree', 'both refuse from_source with one code'
        return (
            'UNCLASSIFIED',
            f'from_source {old["from_source"]} | {core["from_source"]}',
        )
    if old['from_source'][0] != 'ok':
        return 'agree', 'both refuse from_source'
    om, cm = old['materialize'], core['materialize']
    if om[0] == 'ok' and cm[0] == 'ok':
        problems = []
        if old['payload'] != core['payload']:
            problems.append('payload')
        if old['seeds'] != core['seeds']:
            problems.append('_by_time seeds')
        if old['warnings'] != core['warnings']:
            problems.append(f'warnings {old["warnings"]} | {core["warnings"]}')
        if old.get('predict') != core.get('predict') or old.get(
            'result'
        ) != core.get('result'):
            problems.append(
                f'predict {old.get("predict")} | {core.get("predict")}'
            )
        if problems:
            return 'UNCLASSIFIED', '; '.join(problems)
        return 'agree', 'the same request and result'
    if om[0] == 'ok':
        rule = cm[2]
        if cm[1] != 'UNSUPPORTED_REQUEST' or rule not in RULES:
            return (
                'UNCLASSIFIED',
                f'the core refuses {cm} what the reference takes',
            )
        problems = []
        if old['seeds'] != core['seeds'] and core['seeds']:
            problems.append('_by_time seeds')
        labels = core.get('labels')
        if isinstance(labels, list) and labels and isinstance(labels[0], list):
            problems += label_problems(old['payload'], indices, labels)
        elif rule in ('classification_task', 'target_leakage'):
            problems.append(
                f'no labels where the core refuses after them: {labels}'
            )
        if problems:
            return 'UNCLASSIFIED', '; '.join(problems[:3])
        return under(RULES[rule], rule, f'{rule}, labels compared')
    if cm[0] == 'ok':
        return 'UNCLASSIFIED', f'the core takes what the reference refuses {om}'
    if om[1] == cm[1]:
        return 'agree', f'both refuse {om[1]}'
    if om[1] in (None, 'INTERNAL_ERROR') and cm[1] not in (
        None,
        'INTERNAL_ERROR',
    ):
        rule = cm[2]
        if rule in RULES:
            return under(
                RULES[rule], rule, f'E66(b): the reference {om}, the core {cm}'
            )
    return 'UNCLASSIFIED', f'the reference refuses {om}, the core {cm}'


def compare(
    old_path: Path, core_path: Path, spec_path: Path | None, register_path: Path
) -> int:
    old = {r['id']: r for r in json.loads(old_path.read_text())}
    core = {r['id']: r for r in json.loads(core_path.read_text())}
    spec = (
        {s['id']: s for s in json.loads(spec_path.read_text())['sources']}
        if spec_path is not None
        else {}
    )
    register = _register(register_path)
    if set(old) != set(core) or not old:
        print('the two probes are not of the same sources')
        return 1
    counts: dict[str, int] = {}
    unclassified = 0
    for name in sorted(old):
        indices = spec.get(name, {}).get('indices', [0, 1, 'e0', 'e1'])
        kind, why = classify(old[name], core[name], indices, register)
        counts[kind] = counts.get(kind, 0) + 1
        if kind != 'agree':
            print(f'{name}: {kind}: {why}')
        unclassified += kind == 'UNCLASSIFIED'
    print('classes', json.dumps(counts, sort_keys=True))
    return 1 if unclassified else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('seed', type=int)
    g.add_argument('n', type=int)
    g.add_argument('out', type=Path)
    p = sub.add_parser('probe')
    p.add_argument('spec', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--route', choices=('old', 'core'), required=True)
    c = sub.add_parser('compare')
    c.add_argument('old', type=Path)
    c.add_argument('core', type=Path)
    c.add_argument('--spec', type=Path)
    c.add_argument('--register', type=Path, default=REGISTER)
    r = sub.add_parser('run')
    r.add_argument('--reference-python', required=True)
    r.add_argument('--candidate-python', required=True)
    r.add_argument('--seed', type=int, default=27)
    r.add_argument('--n', type=int, default=50)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--register', type=Path, default=REGISTER)
    args = parser.parse_args(argv)
    if args.command == 'generate':
        args.out.write_text(
            json.dumps(generate(args.seed, args.n), indent=1) + '\n'
        )
        return 0
    if args.command == 'probe':
        probe(args.spec, args.out, args.route)
        return 0
    if args.command == 'compare':
        return compare(args.old, args.core, args.spec, args.register)
    args.out.mkdir(parents=True, exist_ok=True)
    spec = args.out / 'spec.json'
    spec.write_text(json.dumps(generate(args.seed, args.n), indent=1) + '\n')
    me = str(Path(__file__).resolve())
    for python, route in (
        (args.reference_python, 'old'),
        (args.candidate_python, 'core'),
    ):
        subprocess.run(
            [
                python,
                '-I',
                me,
                'probe',
                str(spec),
                str(args.out / f'{route}.json'),
                '--route',
                route,
            ],
            check=True,
        )
    return compare(
        args.out / 'old.json', args.out / 'core.json', spec, args.register
    )


if __name__ == '__main__':
    sys.exit(main())
