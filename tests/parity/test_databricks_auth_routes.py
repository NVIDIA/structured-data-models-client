# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK authentication children are observed without allowing client work."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
from routes import LegacyGuard, databricks_auth_process


@pytest.fixture
def cli(tmp_path, monkeypatch):
    sdk = pytest.importorskip('databricks.sdk.credentials_provider')
    path = tmp_path / 'databricks'
    path.write_bytes(b'controlled-cli-test-identity')
    monkeypatch.setattr('routes.shutil.which', lambda name: str(path))
    sdk.DatabricksCliTokenSource._probe_cli_version.cache_clear()
    return sdk, path


def observe(
    cli, monkeypatch, command, *, token=True, bound=True, audit_fault=None
):
    sdk, path = cli
    report = []

    def run(args, **kwargs):
        event_args = (
            str(path)
            if audit_fault != 'executable'
            else '/unrelated/databricks',
            args,
            None,
            {} if audit_fault == 'environment' else None,
        )
        report.append(databricks_auth_process('subprocess.Popen', event_args))
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    'Version': '0.300.0',
                    'access_token': 'private-token',
                    'token_type': 'Bearer',
                    'expiry': '2099-01-01T00:00:00Z',
                }
            ).encode()
        )

    monkeypatch.setattr(subprocess, 'run', run)
    if token:
        instance = object.__new__(sdk.DatabricksCliTokenSource)
        instance._cmd = command if bound else [str(path), 'unrelated']
        instance._access_token_field = 'access_token'
        instance._token_type_field = 'token_type'
        instance._expiry_field = 'expiry'
        instance._exec_cli_command(command)
    else:
        sdk.DatabricksCliTokenSource._probe_cli_version(str(path))
    return report[0]


def test_exact_sdk_version_and_bound_token_are_sanitized(cli, monkeypatch):
    sdk, path = cli
    for command, token in [
        ([str(path), 'version', '--output', 'json'], False),
        (
            [
                str(path),
                'auth',
                'token',
                '--profile',
                'private-profile',
                '--force-refresh',
            ],
            True,
        ),
    ]:
        report = observe(cli, monkeypatch, command, token=token)
        assert report['scope'] == 'authentication-only'
        assert (
            report['executable_sha256']
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
        serialized = json.dumps(report)
        assert 'private-profile' not in serialized
        assert 'private-token' not in serialized
        assert str(path) not in serialized


@pytest.mark.parametrize(
    'fault', ['command', 'binding', 'caller', 'executable', 'environment']
)
def test_other_children_remain_unobserved(cli, monkeypatch, fault):
    sdk, path = cli
    command = [str(path), 'auth', 'token', '--profile', 'private-profile']
    if fault == 'command':
        command = [str(path), 'jobs', 'run-now', '--job-id', '1']
        assert observe(cli, monkeypatch, command) is None
    elif fault == 'binding':
        assert observe(cli, monkeypatch, command, bound=False) is None
    elif fault in ('executable', 'environment'):
        assert observe(cli, monkeypatch, command, audit_fault=fault) is None
    else:
        with LegacyGuard() as guard:
            guard.audit('subprocess.Popen', (str(path), command, None, None))
        assert guard.report['unobserved_subprocesses'] == ['subprocess.Popen']
        assert not guard.report['observed_subprocesses']
