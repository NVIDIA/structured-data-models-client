# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.pytest-config (task 5): the engine's pytest configuration lets the
Snowflake live tests run under pyOpenSSL's ``X509.get_subject is
deprecated`` ``DeprecationWarning``, and still fails on any other
``DeprecationWarning``.

The engine's configuration files are copied beside two probe tests, so pytest
reads them as it does in ``packages/sdm-relational-engine``: one calls
pyOpenSSL's deprecated ``X509.get_subject``, as the Snowflake connector does,
and must pass; the other gives an unrelated ``DeprecationWarning``, and must
still fail.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _schema import REPO

ENGINE = REPO / 'packages' / 'structured-data-models-relational-engine'
CONFIGS = ('pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg')

_PYOPENSSL = """
def test_a_certificate_subject_is_read():
    from OpenSSL import crypto

    certificate = crypto.X509()
    certificate.get_subject()
"""

_OTHER = """
import warnings


def test_another_deprecation_still_fails():
    warnings.warn('this API is deprecated', DeprecationWarning)
"""


def _run(tmp: Path, name: str, body: str) -> subprocess.CompletedProcess[str]:
    for config in CONFIGS:
        if (ENGINE / config).is_file():
            shutil.copy(ENGINE / config, tmp / config)
    (tmp / name).write_text(body)
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', name],
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_pyopenssl_is_installed() -> None:
    pytest.importorskip('OpenSSL', reason='the phase venv has pyOpenSSL')
    from OpenSSL import crypto

    with pytest.warns(
        DeprecationWarning, match='X509.get_subject is deprecated'
    ):
        crypto.X509().get_subject()


def test_the_get_subject_deprecation_is_filtered(tmp_path: Path) -> None:
    proc = _run(tmp_path, 'test_probe_pyopenssl.py', _PYOPENSSL)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]


def test_other_deprecations_still_fail(tmp_path: Path) -> None:
    proc = _run(tmp_path, 'test_probe_other.py', _OTHER)
    assert proc.returncode == 1, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert 'this API is deprecated' in proc.stdout
