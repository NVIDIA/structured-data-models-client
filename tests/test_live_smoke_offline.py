# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live-model smoke script's offline parts (``scripts/live/rfm_smoke.py``,
decision E27, T1.11d): its argument handling, its redaction, its refusal to
run with no endpoint, and its dry run, which builds the request on the core
and shows what would be sent with every socket connection refused. Nothing
here reaches a model endpoint or reads Databricks configuration; the live
smoke itself is run by hand and gates nothing.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/live/rfm_smoke.py'


def _smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location('rfm_smoke', SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clean_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(('SDM_SMOKE_', 'DATABRICKS_', 'SDM_CORE_'))
    }


def test_an_argument_wins_over_its_variable_and_a_variable_is_read() -> None:
    smoke = _smoke()
    environ = {'SDM_SMOKE_ENDPOINT': 'from-env', 'SDM_SMOKE_PROFILE': 'p-env'}
    given = smoke.parse(['--endpoint', 'from-arg'])
    assert smoke.resolve(given, environ) == ('from-arg', 'p-env')
    assert smoke.resolve(smoke.parse([]), environ) == ('from-env', 'p-env')
    assert smoke.resolve(smoke.parse([]), {}) == (None, None)


def test_bad_usage_is_refused() -> None:
    smoke = _smoke()
    for argv in (['--show-body'], ['--timeout', '0'], ['--route', 'x']):
        with pytest.raises(SystemExit) as refused:
            smoke.parse(argv)
        assert refused.value.code == 2


def test_redaction_removes_secrets_and_anything_shaped_like_one() -> None:
    smoke = _smoke()
    redact = smoke.Redactor()
    redact.know('my-serving-endpoint', 'my-profile', None, '', 'ab')
    said = redact(
        'calling my-serving-endpoint (profile my-profile) at '
        'https://adb-1234567890123456.7.azuredatabricks.net/serving with '
        'Bearer abc.def and dapi0123456789abcdef0123456789abcdef as '
        'someone@example.com on dbc-a1b2c3d4-e5f6.cloud.databricks.com'
    )
    for secret in (
        'my-serving-endpoint',
        'my-profile',
        'adb-1234567890123456',
        'azuredatabricks',
        'abc.def',
        'dapi0123',
        'someone@',
        'example.com',
        'dbc-a1b2c3d4',
    ):
        assert secret not in said, secret
    assert said.count(smoke.REDACTED) >= 6
    # Text of no secret's shape passes; a value too short to be a secret is
    # not one.
    assert redact('rows to predict: 2 ab') == 'rows to predict: 2 ab'


def test_a_failure_is_reported_redacted_with_no_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke = _smoke()
    monkeypatch.setattr(smoke, '_quiet_the_world', lambda: None)

    def failing(route: str) -> Any:
        raise RuntimeError(
            'could not reach my-serving-endpoint at '
            'https://dbc-a1b2c3d4-e5f6.cloud.databricks.com with token '
            'dapi0123456789abcdef0123456789abcdef'
        )

    monkeypatch.setattr(smoke, 'build_request', failing)
    status = smoke.main(['--endpoint', 'my-serving-endpoint'])
    captured = capsys.readouterr()
    assert status == 1
    assert 'smoke failed: RuntimeError' in captured.err
    for secret in ('my-serving-endpoint', 'dbc-a1b2c3d4', 'dapi0123'):
        assert secret not in captured.err + captured.out
    assert 'Traceback' not in captured.err


def test_no_endpoint_is_bad_usage_and_nothing_is_sent() -> None:
    run = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env=_clean_env(),
        check=False,
    )
    assert run.returncode == 2, run.stderr
    assert 'SDM_SMOKE_ENDPOINT' in run.stderr


def test_the_result_check_names_what_is_wrong() -> None:
    import pandas as pd

    smoke = _smoke()
    good = pd.DataFrame(
        {'row_index': [0, 1], 'id': [0, 1], 'prediction': [1.0, 2.5]}
    )
    assert smoke.check(good, [0, 1]) == []
    assert smoke.check(good, [0, 1, 2])
    assert smoke.check(good.assign(prediction=[1.0, float('nan')]), [0, 1])
    assert smoke.check(good[['row_index', 'prediction']], [0, 1])


def test_the_dry_run_builds_the_core_s_request_with_no_connection() -> None:
    run = subprocess.run(
        [sys.executable, str(SCRIPT), '--dry-run'],
        capture_output=True,
        text=True,
        env=_clean_env(),
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'dry run, core route' in run.stdout
    assert 'context rows: 112, rows to predict: 2' in run.stdout
    assert 'smoke passed' in run.stdout


def test_an_argument_error_names_no_value() -> None:
    # argparse quotes what it refuses; the script's parser does not.
    for argv in (
        [
            '--dry-run',
            '--endpoint=SYNTHETIC_ENDPOINT',
            '--bad=SYNTHETIC_ENDPOINT',
        ],
        ['--route', 'SYNTHETIC_ENDPOINT'],
        ['--timeout', 'SYNTHETIC_ENDPOINT'],
        ['SYNTHETIC_ENDPOINT'],
    ):
        run = subprocess.run(
            [sys.executable, str(SCRIPT), *argv],
            capture_output=True,
            text=True,
            env=_clean_env(),
            check=False,
        )
        assert run.returncode == 2, argv
        assert 'SYNTHETIC' not in run.stdout + run.stderr, argv
        assert 'bad usage' in run.stderr


class _FakeSdk:
    r"""A stand-in ``databricks.sdk``, its client failing or configured as
    a test says; nothing is called."""

    def __init__(self, failure: Exception | None, host: str = '') -> None:
        failing = failure

        class WorkspaceClient:
            def __init__(self, profile: str | None = None) -> None:
                if failing is not None:
                    raise failing
                self.config = type(
                    'Config',
                    (),
                    {'host': host, 'username': 'SYNTH_USER', 'token': None},
                )()

        self.WorkspaceClient = WorkspaceClient


def _live_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sdk: _FakeSdk,
    environ: dict[str, str],
) -> tuple[int, str]:
    smoke = _smoke()
    monkeypatch.setattr(smoke, '_quiet_the_world', lambda: None)
    monkeypatch.setattr(smoke, 'build_request', lambda route: object())
    module = type(sys)('databricks.sdk')
    module.WorkspaceClient = sdk.WorkspaceClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'databricks.sdk', module)
    for name in [
        n for n in os.environ if n.startswith(('DATABRICKS_', 'SDM_SMOKE_'))
    ]:
        monkeypatch.delenv(name)
    monkeypatch.setenv('HOME', '/nonexistent-home')
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    status = smoke.main(['--endpoint', 'SYNTHETIC_ENDPOINT'])
    captured = capsys.readouterr()
    return status, captured.out + captured.err


@pytest.mark.parametrize(
    'environ',
    [
        {},
        {
            'DATABRICKS_USERNAME': 'SYNTH_USER',
            'DATABRICKS_CLIENT_ID': 'SYNTH_CID',
        },
    ],
)
def test_an_sdk_that_fails_to_configure_is_reported_without_its_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    environ: dict[str, str],
) -> None:
    failure = ValueError(
        'default auth: cannot configure: username=SYNTH_USER '
        'client_id=SYNTH_CID host=models.vanity-corp.test account SYNTH_ACCOUNT'
    )
    status, said = _live_main(monkeypatch, capsys, _FakeSdk(failure), environ)
    assert status == 1
    assert 'smoke failed: SmokeError' in said
    for secret in (
        'SYNTH_USER',
        'SYNTH_CID',
        'SYNTH_ACCOUNT',
        'vanity-corp',
        'SYNTHETIC_ENDPOINT',
    ):
        assert secret not in said, secret


def test_a_vanity_host_is_known_by_its_bare_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    smoke = _smoke()
    redact = smoke.Redactor()
    redact.know_host('https://models.vanity-corp.test:8443/path')
    redact.know_host('bare-host.vanity-corp.test')
    said = redact(
        'connecting to models.vanity-corp.test:443 and bare-host.vanity-corp.test'
    )
    assert 'vanity-corp' not in said
    # The SDK configured on a vanity host, then the call failing with text
    # naming the host and the user: only the class and code are shown.
    sdk = _FakeSdk(None, host='https://models.vanity-corp.test')
    status, said = _live_main(
        monkeypatch,
        capsys,
        sdk,
        {'DATABRICKS_HOST': 'https://models.vanity-corp.test'},
    )
    assert status == 1
    for secret in ('vanity-corp', 'SYNTH_USER', 'SYNTHETIC_ENDPOINT'):
        assert secret not in said, secret
    assert 'smoke failed:' in said


def test_an_interrupt_while_the_profile_is_read_is_reported_in_fixed_words(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke = _smoke()
    monkeypatch.setattr(smoke, '_quiet_the_world', lambda: None)
    monkeypatch.setenv('HOME', '/nonexistent-home')

    def interrupted(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt('SYNTHETIC_SECRET')

    monkeypatch.setattr(smoke.ConfigParser, 'read', interrupted)
    status = smoke.main(['--endpoint', 'SYNTHETIC_ENDPOINT'])
    captured = capsys.readouterr()
    said = captured.out + captured.err
    assert status == 1
    assert 'interrupted' in said
    assert 'SYNTHETIC' not in said and 'Traceback' not in said


def test_the_dry_run_reads_no_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke = _smoke()
    monkeypatch.setattr(smoke, '_quiet_the_world', lambda: None)
    monkeypatch.setattr(smoke, '_no_network', lambda: None)
    read: list[Any] = []
    monkeypatch.setattr(
        smoke.ConfigParser, 'read', lambda self, *a, **k: read.append(a)
    )

    def stop(route: str) -> Any:
        raise RuntimeError('stop here')

    monkeypatch.setattr(smoke, 'build_request', stop)
    assert smoke.main(['--dry-run']) == 1
    assert read == []
    capsys.readouterr()


def test_another_s_warning_is_shown_by_category_and_module_alone() -> None:
    script = (
        'import importlib.util, sys, warnings\n'
        f'spec = importlib.util.spec_from_file_location("s", {str(SCRIPT)!r})\n'
        'smoke = importlib.util.module_from_spec(spec)\n'
        'sys.modules["s"] = smoke\n'
        'spec.loader.exec_module(smoke)\n'
        'smoke._quiet_the_world()\n'
        'warnings.warn_explicit("token for SYNTH_UNKNOWN_USER at '
        'other-host.vanity-corp.test", UserWarning, '
        '"/x/site-packages/databricks/sdk/oauth.py", 1)\n'
        'warnings.warn_explicit("the context is small", UserWarning, '
        '"/x/site-packages/sdm_client/client.py", 1)\n'
    )
    run = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        env=_clean_env(),
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert 'warning UserWarning from oauth (text withheld)' in run.stderr
    assert 'warning UserWarning: the context is small' in run.stderr
    assert 'SYNTH_UNKNOWN_USER' not in run.stderr
    assert 'vanity-corp' not in run.stderr
