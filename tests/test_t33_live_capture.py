# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""C-1's and E-0's templates in dry mode (T3.3): the plan of each leg
reaches nothing and names what it records, what the loader refuses and
the environment it needs, by name only; the serverless job template fills
and passes ``dbx_runs.py``'s check with seeded values, and names no
workspace value of its own."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / 'scripts/live'


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        f't33_live_{name}', LIVE / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture_run = _module('capture_run')
dbx_runs = _module('dbx_runs')


@pytest.mark.parametrize('leg', ['snowflake', 'lakebase'])
def test_a_leg_s_plan_reaches_nothing_and_names_its_needs(
    leg: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in capture_run.ENVIRONMENT[leg]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SDM_CAPTURE_SCHEMA', 'zq_private_schema')
    planned = capture_run.plan(leg)
    assert planned['campaign'] == 'C-1'
    assert len(planned['record']) > 200
    # The batching cases stay off the warehouse legs (KO3-45).
    assert not any(
        '100000' in path or '100001' in path for path in planned['record']
    )
    assert planned['environment']['SDM_CAPTURE_SCHEMA'] is True
    assert planned['environment']['SDM_LIVE_CONFIG'] is False
    assert 'zq_private_schema' not in json.dumps(planned)
    # The Databricks legs one after the other.
    order = planned['order']
    assert (
        abs(order.index('databricks-sql') - order.index('databricks-spark')) > 1
    )
    assert capture_run.run(leg, ROOT, ROOT / 'never-written') == 2
    assert not (ROOT / 'never-written').exists()


def test_the_job_template_fills_and_passes_the_runs_check() -> None:
    text = (LIVE / 'capture.job.template.json').read_text()
    template = json.loads(text)
    assert not re.search(
        r'cloud\.databricks\.com|snowflakecomputing|/Users/', text
    )
    values = {
        'FOLDER': '/Workspace/seeded/folder',
        'NOTEBOOK': 'capture_job',
        'LEG': 'lakebase',
        'CAMPAIGN': 'c1',
        'REPO_DIR': '/Volumes/seeded/repo',
        'REFERENCE_DIR': '/Volumes/seeded/reference',
        'VOLUME_DIR': '/Volumes/seeded/volume',
        'SHA': 'a' * 40,
        'SHA7': 'a' * 7,
        'UUID': '00000000-0000-0000-0000-000000000000',
        'ENVIRONMENT_VERSION': '4',
    }
    payload = dbx_runs.fill(
        {k: v for k, v in template.items() if not k.startswith('_')}, values
    )
    dbx_runs.check(payload, values['FOLDER'], 'unused-policy')
    parameters = payload['tasks'][0]['notebook_task']['base_parameters']
    assert parameters['leg'] == 'lakebase' and parameters['campaign'] == 'c1'
