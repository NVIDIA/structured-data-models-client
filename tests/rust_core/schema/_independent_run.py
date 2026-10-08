# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Ask the Rust core, through the binding, what ``accept_independent.py``
asks, in a process where today's Python can't be imported.

``python _independent_run.py MODE JOBS OUT`` blocks the modules ``MODE``
names (``BLOCKED``) before importing anything else, runs every job in the
JSON file ``JOBS`` with :func:`run_jobs`, and writes the answers to ``OUT``.
The checks run the same jobs in their own process and compare.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: What each mode makes unimportable.
BLOCKED = {
    # The describer and schema runs may use pandas, as the host does, but not
    # today's inference.
    'no-inference': ('sdm_client', 'sdm_relational_engine', 'predictive_query'),
    # Dates, composite keys and metric views are JSON in and out: nothing of
    # Python's date parsing or YAML may answer them.
    'no-python': (
        'sdm_client',
        'sdm_relational_engine',
        'predictive_query',
        'pandas',
        'numpy',
        'dateutil',
        'yaml',
        'pytz',
    ),
}


def run_jobs(jobs: list[dict[str, Any]]) -> list[Any]:
    r"""Each job's answer from the core, or what it raised."""
    from importlib import import_module

    native = import_module('sdm_core._native')
    out: list[Any] = []
    for job in jobs:
        kind = job['kind']
        try:
            if kind == 'dates':
                given = job['input']
                answer: Any = native.dates_op(
                    job['op'],
                    json.dumps(given.get('values')),
                    json.dumps(
                        {k: v for k, v in given.items() if k != 'values'}
                    ),
                )
            elif kind == 'composite':
                answer = native.composite_op(
                    job['op'], json.dumps(job['input'])
                )
            elif kind == 'metric':
                answer = native.metric_view_parse(job['yaml'], job['trust'])
            elif kind == 'from_data':
                from _fixture_ops import build_frame

                describe = import_module('sdm_core.describe')
                frames = {
                    name: build_frame(spec)
                    for name, spec in job['tables'].items()
                }
                tables, tags = describe.describe_frames(frames)
                answer = list(
                    native.schema_from_data(
                        tables, tags, json.dumps(job['options'])
                    )
                )
            elif kind == 'semantic':
                import _schema
                import _warehouse

                warehouse = _warehouse.Warehouse(
                    {job['qualified']: job['yaml']},
                    {tuple(key): columns for key, columns in job['tables']},
                )
                model = _warehouse.source(warehouse).semantic_model(
                    job['view'], trust_expressions=job['trust']
                )
                answer = list(_schema.rust_from_source(model, {}))
            else:
                raise ValueError(f'no job of kind {kind!r}')
        except Exception as error:  # noqa: BLE001 - compared, not judged
            answer = {'raised': type(error).__name__, 'text': str(error)}
        out.append(json.loads(json.dumps(answer, default=str)))
    return out


def main() -> None:
    mode, jobs, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    for name in BLOCKED[mode]:
        sys.modules[name] = None  # type: ignore[assignment]
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here.parent))
    sys.path.insert(0, str(here.parents[1] / 'parity'))
    out.write_text(json.dumps(run_jobs(json.loads(jobs.read_text()))))


if __name__ == '__main__':
    main()
