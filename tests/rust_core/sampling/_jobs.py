# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Questions for the Rust core that ``accept_hashseed.py`` and
``accept_independent.py`` ask in other processes, and the core's answers.

A job is JSON, with any Arrow input kept in an IPC file beside it, so a
process that can't import pandas or the engine can still ask it.

``python _jobs.py MODE JOBS OUT`` blocks the modules ``MODE`` names
(``BLOCKED``) before importing anything else, answers every job in the JSON
file ``JOBS`` through the binding, and writes the answers to ``OUT``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

#: What each mode makes unimportable.
BLOCKED = {
    'all': (),
    # JSON in and out: no Python numerics, dates or engine may answer.
    'no-python': (
        'numpy',
        'pandas',
        'dateutil',
        'pytz',
        'torch',
        'sdm_relational_engine',
        'sdm_client',
        'predictive_query',
    ),
    # Arrays in and out: numpy and pyarrow carry them, nothing else answers.
    'no-engine': (
        'pandas',
        'dateutil',
        'pytz',
        'torch',
        'sdm_relational_engine',
        'sdm_client',
        'predictive_query',
    ),
}


# --- Making jobs (in a process with everything importable) --------------------


def _write_table(table: Any, path: Path) -> str:
    import pyarrow as pa

    with pa.OSFile(str(path), 'wb') as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def _write_array(array: Any, path: Path) -> str:
    import pyarrow as pa

    return _write_table(pa.table({'values': array}), path)


def python_jobs(count: int) -> list[dict[str, Any]]:
    r"""Random numbers, offsets, text and the fixture operations: JSON only."""
    from _sampling import (
        RNG_SEED,
        fixture_cases,
        mt_jobs,
        offset_pairs,
        random_offset,
        rng_ops,
        seeds,
        texts,
    )

    rng = random.Random(RNG_SEED + 11)
    jobs: list[dict[str, Any]] = []
    for seed in seeds()[:count]:
        jobs.append({'kind': 'random_state', 'seed': seed, 'ops': rng_ops(rng)})
    for job in mt_jobs(seeds()[:count]):
        jobs.append({'kind': 'mt', 'job': job})
    for pair in offset_pairs()[:count]:
        jobs.append(
            {
                'kind': 'offset_apply',
                'timestamps': pair['timestamps'],
                'offset': pair['offset'],
                'sign': pair['sign'],
            }
        )
        for bound in ('earliest', 'latest'):
            jobs.append(
                {
                    'kind': 'offset_seconds',
                    'offset': pair['offset'],
                    'bound': bound,
                }
            )
        jobs.append({'kind': 'offset_ns', 'offset': pair['offset']})
    for _ in range(count // 4):
        group = [random_offset(rng) for _ in range(rng.randint(1, 4))]
        jobs.append(
            {
                'kind': 'offset_pool',
                'offsets': group,
                'earliest': rng.random() < 0.5,
            }
        )
    values = texts()
    for start in range(0, min(len(values), count * 4), 200):
        for max_words in (50, None):
            jobs.append(
                {
                    'kind': 'text',
                    'values': values[start : start + 200],
                    'max_words': max_words,
                }
            )
    for case in fixture_cases('labels.'):
        jobs.append(
            {
                'kind': 'labels_op',
                'op': case['operation'],
                'input': case['input'],
            }
        )
    for case in fixture_cases('sampling.'):
        jobs.append(
            {
                'kind': 'sampling_op',
                'op': case['operation'],
                'input': case['input'],
            }
        )
    sys.path.insert(0, str(HERE.parent / 'labels'))
    from _labels import aggregation_inputs, aggregation_type_inputs

    for given in aggregation_type_inputs()[:count]:
        jobs.append(
            {
                'kind': 'labels_op',
                'op': 'labels.aggregation_type',
                'input': given,
            }
        )
    for given in aggregation_inputs(count // 2):
        jobs.append(
            {'kind': 'labels_op', 'op': 'labels.aggregation', 'input': given}
        )
    return jobs


def array_jobs(count: int, tmp: Path) -> list[dict[str, Any]]:
    r"""The neighbor sampler, the graph store and ``labels_execute``: Arrow
    inputs written to ``tmp``."""
    import warnings

    from _sampling import (
        STORE_SEED,
        arrow_tables,
        build_graph,
        graph_json,
        graphs,
        random_store,
    )

    jobs: list[dict[str, Any]] = []
    for graph in graphs(count):
        jobs.append({'kind': 'sampler', 'graph': graph})
    rng = random.Random(STORE_SEED)
    made = 0
    while made < max(count // 10, 20):
        graph = build_graph(random_store(rng))
        if graph is None:
            continue
        tables = {
            name: _write_table(table, tmp / f'store-{made}-{index}.arrow')
            for index, (name, table) in enumerate(arrow_tables(graph).items())
        }
        jobs.append(
            {'kind': 'store', 'tables': tables, 'graph_json': graph_json(graph)}
        )
        made += 1

    sys.path.insert(0, str(HERE.parent / 'labels'))
    from _labels import (
        LABELS_SEED,
        arrow_args,
        corpus_cases,
        resolve,
        sample_data,
    )
    from predictive_query.schema import json_serde

    rng = random.Random(LABELS_SEED + 9)
    asked = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for case in corpus_cases():
            if asked >= max(count // 10, 30):
                break
            resolved = resolve(case['query'], case['schema'])
            if resolved is None:
                continue
            data = sample_data(rng, case['schema'], resolved.entity_table)
            feat, times, batches, anchor = arrow_args(data)
            stem = tmp / f'execute-{asked}'
            jobs.append(
                {
                    'kind': 'execute',
                    'query_json': json.dumps(json_serde.to_json_dict(resolved)),
                    'feat': {
                        name: _write_table(
                            table, Path(f'{stem}-feat-{i}.arrow')
                        )
                        for i, (name, table) in enumerate(feat.items())
                    },
                    'times': {
                        name: _write_array(
                            array, Path(f'{stem}-times-{i}.arrow')
                        )
                        for i, (name, array) in enumerate(times.items())
                    },
                    'batches': {
                        name: _write_array(
                            array, Path(f'{stem}-batches-{i}.arrow')
                        )
                        for i, (name, array) in enumerate(batches.items())
                    },
                    'anchor': _write_array(
                        anchor, Path(f'{stem}-anchor.arrow')
                    ),
                }
            )
            asked += 1
    return jobs


# --- Answering jobs (in any process) ------------------------------------------


def _read_table(path: str) -> Any:
    import pyarrow as pa

    with pa.memory_map(path, 'r') as source:
        return pa.ipc.open_file(source).read_all()


def _array(path: str) -> Any:
    import pyarrow as pa

    column = _read_table(path).column('values')
    if column.num_chunks == 0:
        # combine_chunks() would build the empty array with pa.array, which
        # asks pandas, and pandas may be unimportable here.
        return pa.nulls(0, column.type)
    return column.combine_chunks()


def answer(module: Any, job: dict[str, Any]) -> Any:
    r"""The core's answer to one job, as JSON-ready data."""
    from _sampling import (
        as_list,
        run_sampler,
        rust_draws,
        rust_mt_draws,
        rust_store_of,
    )

    kind = job['kind']
    if kind == 'random_state':
        return rust_draws(module.random_state(job['seed']), job['ops'])
    if kind == 'mt':
        return rust_mt_draws(module, job['job'])
    if kind == 'offset_apply':
        return json.loads(
            module.date_offset_apply(
                json.dumps(job['timestamps']),
                json.dumps(job['offset']),
                job['sign'],
            )
        )
    if kind == 'offset_seconds':
        return module.date_offset_to_seconds(
            json.dumps(job['offset']), job['bound']
        )
    if kind == 'offset_ns':
        return json.loads(module.date_offset_to_ns(json.dumps(job['offset'])))
    if kind == 'offset_pool':
        pooled = json.loads(
            module.date_offset_pool(json.dumps(job['offsets']), job['earliest'])
        )
        # pandas keeps an offset's keywords in an order that follows the
        # hash seed; the keywords, not their order, are the offset.
        if isinstance(pooled, dict) and isinstance(pooled.get('kwds'), dict):
            pooled['kwds'] = dict(sorted(pooled['kwds'].items()))
        return pooled

    if kind == 'text':
        return [
            as_list(w)
            for w in as_list(
                module.normalize_text(job['values'], job['max_words'])
            )
        ]
    if kind == 'labels_op':
        return json.loads(module.labels_op(job['op'], json.dumps(job['input'])))
    if kind == 'sampling_op':
        return json.loads(
            module.sampling_op(job['op'], json.dumps(job['input']))
        )
    if kind == 'sampler':
        return run_sampler(module.NeighborSampler, job['graph'])
    if kind == 'store':
        tables = {
            name: _read_table(path) for name, path in job['tables'].items()
        }
        return rust_store_of(module, tables, job['graph_json'])
    if kind == 'execute':
        sys.path.insert(0, str(HERE.parent / 'labels'))
        from _labels import rust_execute_arrow

        got = rust_execute_arrow(
            job['query_json'],
            {name: _read_table(path) for name, path in job['feat'].items()},
            {name: _array(path) for name, path in job['times'].items()},
            {name: _array(path) for name, path in job['batches'].items()},
            _array(job['anchor']),
        )
        if 'type' in got:
            got['type'] = str(got['type'])
        return got
    raise ValueError(f'no job of kind {kind!r}')


def answer_all(jobs: list[dict[str, Any]]) -> list[Any]:
    from importlib import import_module

    module = import_module('sdm_core._native')
    out = []
    for job in jobs:
        try:
            got = answer(module, job)
        except Exception as error:  # noqa: BLE001 - compared, not judged
            got = {'raised': type(error).__name__, 'text': str(error)}
        out.append(json.loads(json.dumps(got, default=str)))
    return out


def main() -> None:
    mode, jobs, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    for name in BLOCKED[mode]:
        sys.modules[name] = None  # type: ignore[assignment]
    for path in (HERE, HERE.parent, HERE.parent / 'labels'):
        sys.path.insert(0, str(path))
    out.write_text(json.dumps(answer_all(json.loads(jobs.read_text()))))


if __name__ == '__main__':
    main()
