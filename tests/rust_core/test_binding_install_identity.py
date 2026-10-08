# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wheel setup refuses stale artifacts without splitting loaded identities."""

from __future__ import annotations

import importlib.machinery
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _rc


@pytest.fixture
def binding_install(tmp_path, monkeypatch):
    files = {
        'sdm_core/__init__.py': b'package = True\n',
        'sdm_core/answers.py': b'class ResultValues: pass\n',
        'sdm_core/_native.abi3.so': b'synthetic-native-artifact',
    }
    wheel = tmp_path / 'sdm_core-1.0.0-cp310-abi3-test.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    site = tmp_path / 'site-packages'
    for name, contents in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    for name in tuple(sys.modules):
        if name == 'sdm_core' or name.startswith('sdm_core.'):
            monkeypatch.delitem(sys.modules, name)
    loaded = {}
    for name, relative in (
        ('sdm_core', 'sdm_core/__init__.py'),
        ('sdm_core.answers', 'sdm_core/answers.py'),
        ('sdm_core._native', 'sdm_core/_native.abi3.so'),
    ):
        module = types.ModuleType(name)
        module.__file__ = str(site / relative)
        module.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, origin=module.__file__
        )
        monkeypatch.setitem(sys.modules, name, module)
        loaded[name] = module
    calls = []
    monkeypatch.setattr(_rc, 'site_packages', lambda: [site, site])
    monkeypatch.setattr(_rc, 'installer', lambda: ['installer-control'])
    monkeypatch.setattr(_rc, 'run', lambda command: calls.append(command))
    return wheel, site, loaded, calls


def test_equivalent_wheel_runs_installer_and_preserves_modules(binding_install):
    wheel, _, loaded, calls = binding_install
    _rc.install_wheel(wheel)
    assert calls == [['installer-control', '--reinstall', '--no-deps', wheel]]
    assert all(sys.modules[name] is module for name, module in loaded.items())


def test_first_install_does_not_require_prior_imports(
    binding_install, monkeypatch
):
    wheel, _, loaded, calls = binding_install
    for name in loaded:
        monkeypatch.delitem(sys.modules, name)
    _rc.install_wheel(wheel)
    assert len(calls) == 1
    assert not any(name in sys.modules for name in loaded)


@pytest.mark.parametrize('relative', ['answers.py', '_native.abi3.so'])
def test_changed_loaded_artifact_refuses_before_install(
    binding_install, relative
):
    wheel, site, loaded, calls = binding_install
    (site / 'sdm_core' / relative).write_bytes(b'non-equivalent artifact')
    with pytest.raises(AssertionError, match='fresh interpreter'):
        _rc.install_wheel(wheel)
    assert calls == []
    assert all(sys.modules[name] is module for name, module in loaded.items())


@pytest.mark.parametrize(
    'case',
    ['missing-file', 'unknown-origin', 'unknown-member', 'outside-install'],
)
def test_unknown_loaded_artifact_refuses_before_install(
    binding_install, case, tmp_path
):
    wheel, site, loaded, calls = binding_install
    module = loaded['sdm_core.answers']
    if case == 'missing-file':
        (site / 'sdm_core/answers.py').unlink()
    elif case == 'unknown-origin':
        module.__spec__.origin = None
    else:
        path = (
            site / 'sdm_core/unknown.py'
            if case == 'unknown-member'
            else tmp_path / 'answers.py'
        )
        path.write_bytes(b'class ResultValues: pass\n')
        module.__file__ = str(path)
        module.__spec__.origin = str(path)
    with pytest.raises(AssertionError, match='fresh interpreter'):
        _rc.install_wheel(wheel)
    assert calls == []


def test_installer_failure_is_not_replaced_by_cached_success(
    binding_install, monkeypatch
):
    wheel, _, loaded, calls = binding_install
    failure = AssertionError('installer deliberately failed')

    def refused(command):
        calls.append(command)
        raise failure

    monkeypatch.setattr(_rc, 'run', refused)
    with pytest.raises(AssertionError) as caught:
        _rc.install_wheel(wheel)
    assert caught.value is failure
    assert len(calls) == 1
    assert all(sys.modules[name] is module for name, module in loaded.items())


@pytest.mark.parametrize('defect', ['identity', 'installed-bytes'])
def test_installer_cannot_change_identity_or_install_wrong_bytes(
    binding_install, monkeypatch, defect
):
    wheel, site, _, calls = binding_install

    def altered(command):
        calls.append(command)
        if defect == 'identity':
            monkeypatch.setitem(
                sys.modules,
                'sdm_core.answers',
                types.ModuleType('sdm_core.answers'),
            )
        else:
            (site / 'sdm_core/answers.py').write_bytes(
                b'wrong installed artifact'
            )

    monkeypatch.setattr(_rc, 'run', altered)
    with pytest.raises(AssertionError, match='fresh interpreter'):
        _rc.install_wheel(wheel)
    assert len(calls) == 1
