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

The first cell installs the reference's wheels from ``wheel_dir``, those
of this host's platform (``capture_run.installable_wheels``; serverless
runs on x86_64 and aarch64), and removes its client wheel, then restarts
Python. The second cell puts the
reference checkout's client source first on ``sys.path``
(``capture_run.use_reference_client``), since the recording runs the
checkout's own client; a notebook's environment does not read a ``.pth``
file, so this is done in the process itself. It then sets the run's environment from the job's parameters
(the live configuration's path, the placeholders' private names and the
write guard's test schema; ``capture_run.job_environment``, filled when
the job is submitted, never printed, never in the repository) and runs ``capture_run.run`` in
this process, so the Spark leg reads the notebook's own Spark session,
every write through the repository's live write guard
(``scripts/live/write_guard.py`` in ``repo_dir``). It prints the
manifests written, never a value of the environment.

Both checkouts are copied off the volume to local disk with each file's
mode as its commit holds it (``capture_run.local_checkout``), so the run
records from clean trees, which it requires; the scripted endpoint is
served on a local address predict reaches here
(``capture_run.endpoint_host``: serverless refuses IPv4 loopback), found
before anything is recorded.

Run outside Databricks, it prints this text and exits.

Each case's progress is printed and written to the output folder as the
run goes (``<leg>.progress.jsonl``), each failure at once
(``<leg>.failed.json``), and a case past ``case_timeout`` seconds fails
alone (``capture_run.run``).

Widgets (job base parameters): leg, campaign (c1 or e0), repo_dir,
reference_dir, wheel_dir, out_dir, live_config,
capture_catalog, capture_schema, test_database, test_schema,
case_timeout, commit, capture_names (the other placeholders' names, JSON).
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
    'live_config',
    'capture_catalog',
    'capture_schema',
    'test_database',
    'test_schema',
    'case_timeout',
    'commit',
    'capture_names',
)
for name in WIDGETS:
    dbutils.widgets.text(name, '', name)

import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(dbutils.widgets.get('repo_dir')) / 'scripts/live'))
import capture_run

# The folder holds each platform's wheels; this host's only go to pip.
wheels = capture_run.installable_wheels(dbutils.widgets.get('wheel_dir'))
assert wheels, 'no reference wheel in wheel_dir for this host'
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
import capture_run as staging

# Both checkouts copied off the volume, each file's mode as its commit
# holds it, so the run records from clean trees (it refuses others).
repo = staging.local_checkout(dbutils.widgets.get('repo_dir'), '/tmp/sdm-repo')
reference = staging.local_checkout(
    dbutils.widgets.get('reference_dir'), '/tmp/sdm-reference'
)
sys.path.remove(str(Path(dbutils.widgets.get('repo_dir')) / 'scripts/live'))
del sys.modules['capture_run']
sys.path.insert(0, str(repo / 'scripts/live'))
import capture_run

capture_run.use_reference_client(reference)
# The scripted endpoint on an address predict reaches here (serverless
# refuses IPv4 loopback): found before anything is recorded.
print(f'the scripted endpoint answers on {capture_run.endpoint_host()}')
names = capture_run.job_environment(
    {
        name: dbutils.widgets.get(name)
        for name in (*capture_run.JOB_ENVIRONMENT, capture_run.JOB_NAMES)
    }
)
print(f'the run reads {len(names)} variable(s) from its job parameters')
status = capture_run.run(
    leg,
    reference,
    Path(dbutils.widgets.get('out_dir')),
    campaign == 'e0',
    case_timeout=float(
        dbutils.widgets.get('case_timeout') or capture_run.CASE_TIMEOUT
    ),
)
dbutils.notebook.exit('passed' if status == 0 else f'failed {status}')
