# Databricks notebook source
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, F821
# (Databricks injects `dbutils`.)

"""A Databricks notebook: C-1 or E-0 on the Spark or Lakebase leg, in a
one-time serverless job (capture.job.template.json; design, the live-test
plan). It installs the reference's wheels from ``wheel_dir`` (the old
client the capture records), then runs ``capture_run.py run`` from
``repo_dir`` (a snapshot of the recording commit) for ``leg``, the
captures written under ``out_dir``. The live configuration, the
placeholders' private names and the write guard's test schema come from
the job's environment (``SDM_LIVE_CONFIG``, ``SDM_CAPTURE_*``,
``SDM_LIVE_TEST_*``), never from this file. It prints the planned case
count and the manifests written, never a value of the environment.

Run outside Databricks, it prints this text and exits.

Widgets (job base parameters): leg, campaign (c1 or e0), repo_dir,
reference_dir, wheel_dir, out_dir, commit.
"""

import sys

if 'dbutils' not in globals():
    print(__doc__)
    raise SystemExit(0 if {'-h', '--help'} & set(sys.argv) else 2)

for name in (
    'leg',
    'campaign',
    'repo_dir',
    'reference_dir',
    'wheel_dir',
    'out_dir',
    'commit',
):
    dbutils.widgets.text(name, '', name)

import glob
import subprocess
from pathlib import Path

leg = dbutils.widgets.get('leg')
campaign = dbutils.widgets.get('campaign')
repo = Path(dbutils.widgets.get('repo_dir'))
assert leg in ('databricks-spark', 'lakebase'), (
    'the jobs record Spark and Lakebase'
)
assert campaign in ('c1', 'e0'), campaign
wheels = sorted(
    glob.glob(dbutils.widgets.get('wheel_dir').rstrip('/') + '/*.whl')
)
assert wheels, 'no reference wheel in wheel_dir'
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', *wheels], check=True
)
command = [
    sys.executable,
    str(repo / 'scripts/live/capture_run.py'),
    'run',
    '--leg',
    leg,
    '--reference',
    dbutils.widgets.get('reference_dir'),
    '--out',
    dbutils.widgets.get('out_dir'),
]
if campaign == 'e0':
    command.append('--e0')
done = subprocess.run(command, check=False)
dbutils.notebook.exit(
    'passed' if done.returncode == 0 else f'failed {done.returncode}'
)
