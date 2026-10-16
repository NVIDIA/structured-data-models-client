# Databricks notebook source
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, F811, F821
# (Databricks injects `dbutils` and `spark`, and the cells after
# restartPython import and define again what they use, since it clears the
# interpreter.)

"""A Databricks notebook: does docs/get-started/installation.md's "On
Databricks" install work on this runtime, from a given wheel set?

Run once per runtime as a one-time job run (dbx_runs.py and
dbr_install_check.job.template.json; dbr_install_check_prepare.sh builds
the wheels and prints the steps). Its "On Databricks" cells are the guide's,
verbatim, except that the install line adds ``--find-links <wheel_dir>`` and
pins the wheels' version, because a branch's wheels are not on an index. It
calls no model endpoint, reads no table, writes nothing outside the cluster,
and touches no other notebook or cluster.

Widgets (job base parameters):

    wheel_dir  absolute path of the folder holding the five wheels
    commit     the commit the wheels were built from (at least 7 characters)
    extras     the client's extras to install (default: the guide's)

It checks that NumPy is held where the guide's first cell puts it (the
runtime's own NumPy, which meets the client's floor of 1.23.5), that the client's
own floors for NumPy, pandas and pyarrow are met, that the client installed
is the native wheel from ``wheel_dir`` and its ``sdm_client._native`` loads
without test-only operations, that the runtime's own packages still import,
that Spark still answers, that the relational engine imports, and an offline
journey up to ``materialize``. Each line it prints for a caller to parse
starts with ``SDM_INSTALL_CHECK``; the last also is the notebook's exit
value, a JSON object whose ``passed`` is true when every check passed.

Run outside Databricks, it prints this text and exits.
"""

import sys

if 'dbutils' not in globals():
    print(__doc__)
    raise SystemExit(0 if {'-h', '--help'} & set(sys.argv) else 2)

dbutils.widgets.text('wheel_dir', '', 'Folder holding the wheel set')
dbutils.widgets.text('commit', '', 'Commit the wheels are built from')
dbutils.widgets.text(
    'extras', 'relational,databricks,databricks-serving', 'Client extras'
)

import json
import os
import platform
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version

wheel_dir = dbutils.widgets.get('wheel_dir').rstrip('/') + '/'
commit = dbutils.widgets.get('commit')
extras = dbutils.widgets.get('extras').replace(' ', '')
assert wheel_dir.startswith('/'), 'wheel_dir must be an absolute path'
assert len(commit) >= 7, 'name the commit the wheels were built from'
assert re.fullmatch(r'[a-z0-9,-]*', extras), extras

found = sorted(
    re.match(r'structured_data_models_client-([^-]+)-', name).group(1)
    for name in os.listdir(wheel_dir)
    if name.startswith('structured_data_models_client-')
    and name.endswith('.whl')
)
assert len(found) == 1, f'want one client wheel in wheel_dir, found {found}'
client_version = found[0]


def installed(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


RUNTIME_PACKAGES = (
    'numpy',
    'pandas',
    'pyarrow',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'pyspark',
    'mlflow-skinny',
    'mlflow',
    'databricks-sdk',
    'databricks-sql-connector',
)
before = {p: installed(p) for p in RUNTIME_PACKAGES}
print(
    'SDM_INSTALL_CHECK before '
    + json.dumps(
        {
            'runtime': os.environ.get('DATABRICKS_RUNTIME_VERSION'),
            'python': sys.version.split()[0],
            'machine': platform.machine(),
            'commit': commit,
            'client_version': client_version,
            **before,
        },
        sort_keys=True,
    )
)
# Kept on the driver's disk for the cells after restartPython.
with open(
    os.path.join(tempfile.gettempdir(), 'sdm_install_check.json'), 'w'
) as f:
    json.dump({'before': before, 'client_version': client_version}, f)

# COMMAND ----------

# The guide's first cell, verbatim.
import numpy

keep_numpy = f'numpy=={numpy.__version__}'

# Not in the guide: where the wheels are, since they are not on an index.
install = (
    f'--find-links {wheel_dir} '
    f'structured-data-models-client[{extras}]=={client_version}'
)
print('SDM_INSTALL_CHECK keep_numpy ' + keep_numpy)

# COMMAND ----------

# MAGIC %pip install --only-binary=:all: $install $keep_numpy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import platform
import re
import sys
import tempfile
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    requires,
    version,
)

commit = dbutils.widgets.get('commit')
wheel_dir = dbutils.widgets.get('wheel_dir').rstrip('/') + '/'
with open(os.path.join(tempfile.gettempdir(), 'sdm_install_check.json')) as f:
    saved = json.load(f)
before, client_version = saved['before'], saved['client_version']
checks = {}


def installed(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def check(name, thunk):
    try:
        checks[name] = {'ok': True, 'value': thunk()}
    except BaseException as error:  # every failure is the result
        checks[name] = {
            'ok': False,
            'error': f'{type(error).__name__}: {error}'[:500],
        }


def numbers(text):
    return tuple(int(x) for x in re.findall(r'\d+', text)[:3])


FIRST_PARTY = (
    'structured-data-models-client',
    'structured-data-models-api',
    'structured-data-models-pql',
    'structured-data-models-connectors',
    'structured-data-models-relational-engine',
)
after = {p: installed(p) for p in (*before, *FIRST_PARTY)}


def numpy_held():
    runtime, now = before['numpy'], after['numpy']
    assert now is not None, 'numpy missing'
    assert now == runtime, f'numpy moved from {runtime} to {now}'
    return now


check('numpy held where the guide puts it', numpy_held)

# The client's own floors, read from its metadata.
for requirement in requires('structured-data-models-client') or []:
    if ';' in requirement:
        continue
    floor = re.match(r'(numpy|pandas|pyarrow)\s*>=\s*([0-9.]+)', requirement)
    if floor:
        package, at_least = floor.groups()

        def meets(package=package, at_least=at_least):
            now = after[package]
            assert now is not None, f'{package} missing'
            assert numbers(now) >= numbers(at_least), f'{now} < {at_least}'
            return now

        check(f'{package} >= {at_least}', meets)


def all_one_version():
    versions = {p: after[p] for p in FIRST_PARTY}
    assert set(versions.values()) == {client_version}, versions
    return client_version


check('the five distributions at the wheels version', all_one_version)


def native_from_wheel():
    files = [
        str(f)
        for f in distribution('structured-data-models-client').files or []
    ]
    native = [f for f in files if f.startswith('sdm_client/_native')]
    assert any(f.endswith(('.so', '.pyd')) for f in native), files[:20]
    return native


check('client wheel carries sdm_client._native', native_from_wheel)


def native_loads():
    from sdm_client import _native

    assert _native.TEST_OPS is False, (
        'the shipped wheel has no test-only operations'
    )
    return {
        'contract': _native.CONTRACT_VERSION,
        'codes': len(_native.ERROR_CODES),
    }


check('sdm_client._native loads', native_loads)

# The runtime's own packages, built against its NumPy, still import (the
# guide: "within NumPy 1, a newer release still runs packages built against
# an older one"). Only those the runtime ships are tried.
RUNTIME_MODULES = {
    'scipy.stats': 'scipy',
    'sklearn.linear_model': 'scikit-learn',
    'matplotlib.pyplot': 'matplotlib',
    'pyspark.pandas': 'pyspark',
    'mlflow': 'mlflow-skinny',
}
for module, package in RUNTIME_MODULES.items():
    if before.get(package) or (
        package == 'mlflow-skinny' and before.get('mlflow')
    ):
        check(
            f'import {module}',
            lambda module=module: __import__(module, fromlist=['_']).__name__,
        )

if 'spark' in globals():
    check('spark still answers', lambda: spark.range(3).count())


def engine_imports():
    from sdm_relational_engine.rfm import Graph, KumoRelational  # noqa: F401

    return True


if 'relational' in dbutils.widgets.get('extras').split(','):
    check('relational engine imports', engine_imports)


def offline_journey():
    # The README's example up to materialize: no endpoint is called.
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema

    users = pd.DataFrame({'user_id': range(12)})
    orders = pd.DataFrame(
        {
            'order_id': range(120),
            'user_id': [i % 12 for i in range(120)],
            'price': [10.0 + i for i in range(120)],
            'ordered_at': pd.date_range('2026-01-01', periods=120, freq='D'),
        }
    )
    schema = Schema.from_data({'users': users, 'orders': orders})
    query = PredictiveQuery(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
    )
    request = query.materialize(indices=[1, 2, 3])
    return type(request).__name__


check('offline journey to materialize', offline_journey)

result = {
    'runtime': os.environ.get('DATABRICKS_RUNTIME_VERSION'),
    'python': sys.version.split()[0],
    'machine': platform.machine(),
    'commit': commit,
    'client_version': client_version,
    'before': before,
    'after': after,
    'checks': checks,
    'passed': all(c['ok'] for c in checks.values()),
}
print('SDM_INSTALL_CHECK after ' + json.dumps(result, sort_keys=True))
dbutils.notebook.exit(json.dumps(result, sort_keys=True))
