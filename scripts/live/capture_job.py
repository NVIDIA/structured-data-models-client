# Databricks notebook source
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, F821
# (Databricks injects `dbutils` and `spark`, and the cell after
# restartPython imports again what it uses, since it clears the
# interpreter.)

"""A Databricks notebook: C-1 or E-0 on the Spark or Lakebase leg, in a
one-time serverless job (capture.job.template.json; design, the live-test
plan).

The first cell installs the reference's wheels from ``wheel_dir``, removes
its client wheel and names the reference checkout's client source in the
site directory (``capture_run.py bootstrap``), since the recording runs
the checkout's own client, then restarts Python so the import takes it.
The second cell sets the run's environment from ``environment_file`` on
the authorized volume (the live configuration, the placeholders' private
names and the write guard's test schema; ``capture_run.load_environment``,
never printed, never in the repository) and runs ``capture_run.run`` in
this process, so the Spark leg reads the notebook's own Spark session,
every write through the live write guard at ``guard_path``. It prints the
manifests written, never a value of the environment.

Run outside Databricks, it prints this text and exits.

Widgets (job base parameters): leg, campaign (c1 or e0), repo_dir,
reference_dir, wheel_dir, out_dir, guard_path, environment_file, commit.
"""

import sys

if 'dbutils' not in globals():
    print(__doc__)
    raise SystemExit(0 if {'-h', '--help'} & set(sys.argv) else 2)

WIDGETS = (
    'leg',
    'campaign',
    'repo_dir',
    'reference_dir',
    'wheel_dir',
    'out_dir',
    'guard_path',
    'environment_file',
    'commit',
)
for name in WIDGETS:
    dbutils.widgets.text(name, '', name)

import glob
import site
import subprocess
from pathlib import Path

wheels = sorted(
    glob.glob(dbutils.widgets.get('wheel_dir').rstrip('/') + '/*.whl')
)
assert wheels, 'no reference wheel in wheel_dir'
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', *wheels], check=True
)
subprocess.run(
    [
        sys.executable,
        '-m',
        'pip',
        'uninstall',
        '-q',
        '-y',
        'structured-data-models-client',
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(
            Path(dbutils.widgets.get('repo_dir'))
            / 'scripts/live/capture_run.py'
        ),
        'bootstrap',
        '--reference',
        dbutils.widgets.get('reference_dir'),
        '--site',
        site.getsitepackages()[0],
    ],
    check=True,
)
dbutils.library.restartPython()

# COMMAND ----------

import sys
from pathlib import Path

leg = dbutils.widgets.get('leg')
campaign = dbutils.widgets.get('campaign')
assert leg in ('databricks-spark', 'lakebase'), (
    'the jobs record Spark and Lakebase'
)
assert campaign in ('c1', 'e0'), campaign
sys.path.insert(0, str(Path(dbutils.widgets.get('repo_dir')) / 'scripts/live'))
import capture_run

names = capture_run.load_environment(dbutils.widgets.get('environment_file'))
print(f'the run reads {len(names)} variable(s) from its environment file')
status = capture_run.run(
    leg,
    Path(dbutils.widgets.get('reference_dir')),
    Path(dbutils.widgets.get('out_dir')),
    campaign == 'e0',
    dbutils.widgets.get('guard_path'),
)
dbutils.notebook.exit('passed' if status == 0 else f'failed {status}')
