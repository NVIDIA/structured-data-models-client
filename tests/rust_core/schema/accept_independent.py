# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.independent: the Rust core answers on its own, not by calling today's
Python back.

The other checks compare the core's answers with today's Python's in one
process, where a binding could pass by asking Python. Here the same questions
go to the core in a process where Python's answer can't be reached, and the
answers must be the ones it gives with everything importable:

- schemas from frames (every ``schema.from_data`` fixture and describer
  shape) and from a semantic view over the stand-in warehouse (the quoted
  view of ``accept_semantic_quoting.py`` and the first generated ones), with
  ``sdm_client``, ``sdm_relational_engine`` and ``predictive_query``
  unimportable;
- dates (every ``dates`` fixture case and the first generated columns),
  composite keys (every fixture case) and metric views (the first generated
  ones), with pandas, numpy, dateutil, PyYAML and pytz unimportable too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from _dates import columns, inputs
from _independent_run import run_jobs
from _schema import HERE, INSTALLED, fixture_cases, native
from _views import metric_views, semantic_views
from _warehouse import tables_of

SCRIPT = HERE / '_independent_run.py'


def _run(mode: str, jobs: list[dict[str, Any]], tmp: Path) -> list[Any]:
    given, out = tmp / f'{mode}-jobs.json', tmp / f'{mode}-out.json'
    given.write_text(json.dumps(jobs))
    env = dict(os.environ)
    env[INSTALLED] = '1'
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), mode, str(given), str(out)],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, proc.stderr[-6000:]
    return json.loads(out.read_text())


def _compare(mode: str, jobs: list[dict[str, Any]], tmp: Path) -> None:
    native()
    here = run_jobs(jobs)
    alone = _run(mode, jobs, tmp)
    assert len(alone) == len(jobs)
    differ = [
        f'{json.dumps(job)[:200]}: {json.dumps(a)[:200]} alone, '
        f'{json.dumps(b)[:200]} with Python'
        for job, a, b in zip(jobs, alone, here, strict=True)
        if a != b
    ]
    assert not differ, (
        f'{len(differ)} of {len(jobs)} answers change when Python cannot '
        f'answer:\n' + '\n'.join(differ[:20])
    )
    raised = [b for b in here if isinstance(b, dict) and 'raised' in b]
    assert len(raised) < len(jobs) / 10, f'the core raised: {raised[:5]}'


def _from_data_jobs() -> list[dict[str, Any]]:
    import accept_describer

    jobs = []
    for name in ('time-and-keys', 'relationships'):
        for case in fixture_cases(name):
            given = case['input']
            jobs.append(
                {
                    'kind': 'from_data',
                    'tables': given['tables'],
                    'options': {
                        'primary_keys': given.get('primary_keys'),
                        'time_columns': given.get('time_columns'),
                        'relationships': given.get('relationships'),
                        'edits': given.get('edits', []),
                    },
                }
            )
    for values in accept_describer.ALL_SHAPES.values():
        jobs.append(
            {
                'kind': 'from_data',
                'tables': {
                    't': {
                        'columns': {
                            'id': {
                                'dtype': 'int64',
                                'values': list(range(len(values))),
                            },
                            'c': {'dtype': 'object', 'values': values},
                        }
                    }
                },
                'options': {
                    'primary_keys': None,
                    'time_columns': None,
                    'relationships': None,
                    'edits': [],
                },
            }
        )
    return jobs


def _semantic_jobs() -> list[dict[str, Any]]:
    import accept_semantic_quoting as quoting

    recorded = quoting._recorded_yaml()
    quoted = yaml.safe_dump(
        quoting._quote_names(yaml.safe_load(recorded)), sort_keys=False
    )
    views = [(quoted, tables_of(yaml.safe_load(quoted)), False)]
    views += list(semantic_views(100))
    return [
        {
            'kind': 'semantic',
            'qualified': 'TEST.PARITY_SHOP.SV',
            'view': 'SV',
            'yaml': text,
            'tables': [[list(key), columns] for key, columns in tables.items()],
            'trust': trust,
        }
        for text, tables, trust in views
    ]


def test_schemas_need_no_python_inference(tmp_path: Path) -> None:
    _compare('no-inference', _from_data_jobs() + _semantic_jobs(), tmp_path)


def test_dates_keys_and_metric_views_need_no_python(tmp_path: Path) -> None:
    jobs: list[dict[str, Any]] = []
    for case in fixture_cases('dates'):
        jobs.append(
            {'kind': 'dates', 'op': case['operation'], 'input': case['input']}
        )
    for generated in columns(2000):
        for operation, given in inputs(generated):
            jobs.append({'kind': 'dates', 'op': operation, 'input': given})
    for case in fixture_cases('composite-key'):
        jobs.append(
            {
                'kind': 'composite',
                'op': case['operation'],
                'input': case['input'],
            }
        )
    for definition, trust in metric_views(1000):
        jobs.append({'kind': 'metric', 'yaml': definition, 'trust': trust})
    _compare('no-python', jobs, tmp_path)
