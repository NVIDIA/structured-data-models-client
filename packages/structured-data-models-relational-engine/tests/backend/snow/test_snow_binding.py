# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

import pytest

try:
    import snowflake.connector  # noqa: F401
    from sdm_relational_engine.rfm.backend.snow.binding import (  # noqa: F401
        paramstyle,
    )
except ImportError:
    pytest.skip("'snowflake' extension not installed", allow_module_level=True)

# Each scenario runs in a fresh interpreter. snowflake-connector-python 5.x
# warns about a deprecated attribute only on its first use in a process (it
# keeps its own record, whatever the `warnings` filters say), so in this
# process any earlier test that touched the attribute would hide the warning
# from these. A fresh interpreter makes the check independent of test order.
#
# The fake subclasses the real driver class, so the attributes it reads and
# writes are the installed driver's own, and skips the driver's `__init__`,
# which would open a connection. It sets and reads the style the way the
# installed driver supports publicly: 5.x has a settable `paramstyle`
# property, while 4.x has only the private `_paramstyle` attribute.
_SCRIPT = r"""
import json
import sys
import warnings

from snowflake.connector import SnowflakeConnection

from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.snow.binding import paramstyle

PUBLIC = isinstance(getattr(SnowflakeConnection, 'paramstyle', None), property)


def style_of(connection):
    style = connection.paramstyle if PUBLIC else connection._paramstyle
    return str(style)


def set_style(connection, style):
    if PUBLIC:
        connection.paramstyle = style
    else:
        connection._paramstyle = style


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, parameters=None):
        # What the driver would bind this statement with:
        self.connection.styles.append(style_of(self.connection))

    def fetchall(self):
        return []


class FakeConnection(SnowflakeConnection):
    def __init__(self):
        set_style(self, 'pyformat')
        self.styles = []

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        pass

    def __del__(self):
        pass


def run(scenario, connection):
    if scenario == 'success':
        with paramstyle(connection):
            connection.styles.append(style_of(connection))
        return None
    if scenario == 'error':
        try:
            with paramstyle(connection):
                connection.styles.append(style_of(connection))
                raise RuntimeError('query failed')
        except RuntimeError as error:
            return type(error).__name__
        return None
    if scenario == 'graph':
        # A public entry point that binds a value: discovering the tables of
        # a schema. Finding none raises once the discovery query is done.
        try:
            Graph.from_snowflake(
                connection=connection,
                database='DB',
                schema='SCH',
                verbose=False,
            )
        except Exception as error:
            return type(error).__name__
        return None
    raise AssertionError(scenario)


connection = FakeConnection()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    raised = run(sys.argv[1], connection)
    after = style_of(connection)

print(
    json.dumps(
        {
            'during': connection.styles,
            'after': after,
            'raised': raised,
            'warnings': [
                f'{w.category.__name__}: {w.message}' for w in caught
            ],
        }
    )
)
"""


def _run(scenario: str) -> dict:
    result = subprocess.run(
        [sys.executable, '-c', _SCRIPT, scenario],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_paramstyle_switches_and_restores_without_a_deprecation_warning():
    # Regression test: `paramstyle()` toggled the driver's private
    # `_paramstyle`, which snowflake-connector-python 5.0 keeps only for
    # backward compatibility and warns about, and this package turns
    # warnings into errors.
    outcome = _run('success')

    assert outcome['during'] == ['qmark']
    assert outcome['after'] == 'pyformat'
    assert outcome['warnings'] == []


def test_paramstyle_restores_after_an_error_without_a_deprecation_warning():
    # The connection is usually the caller's, so a failed query must not
    # leave it in 'qmark' mode, and the error itself must still propagate.
    outcome = _run('error')

    assert outcome['raised'] == 'RuntimeError'
    assert outcome['during'] == ['qmark']
    assert outcome['after'] == 'pyformat'
    assert outcome['warnings'] == []


def test_graph_discovery_binds_with_qmark_without_a_deprecation_warning():
    # The same through a public entry point: listing a schema's tables binds
    # the schema name, so the query must run under 'qmark', and the caller's
    # connection must be left as it was.
    outcome = _run('graph')

    assert outcome['raised'] == 'NoTablesFoundError'
    assert outcome['during'] == ['qmark']
    assert outcome['after'] == 'pyformat'
    assert outcome['warnings'] == []
