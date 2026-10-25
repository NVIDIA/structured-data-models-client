# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fake warehouse fixture leaves subsequent driver imports intact."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('already_imported', [False, True])
def test_the_fake_warehouse_restores_the_parent_packages(
    already_imported: bool,
) -> None:
    fixture = Path(__file__).with_name('test_databricks_from_source.py')
    probe = r'''
import importlib
import importlib.util
import sys
import types
import pytest
import sdm_connectors.backends as backends

spec = importlib.util.spec_from_file_location('warehouse_fixture', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
name = 'sdm_connectors.backends.databricks'
original = types.ModuleType(name)
if sys.argv[2] == 'True':
    sys.modules[name] = original
    backends.databricks = original
else:
    sys.modules.pop(name, None)
    if hasattr(backends, 'databricks'):
        del backends.databricks
with pytest.MonkeyPatch.context() as patch:
    driver = module._driver.__wrapped__(patch)
    next(driver)
    fake_backend = importlib.import_module(name)
    assert fake_backend is not original
    assert not hasattr(fake_backend.databricks_sql, 'connect')
    try:
        next(driver)
    except StopIteration:
        pass
    else:
        raise AssertionError('fixture did not finish')
if sys.argv[2] == 'True':
    from sdm_connectors.backends import databricks
    assert databricks is original
    assert sys.modules[name] is original
else:
    assert name not in sys.modules
    assert not hasattr(backends, 'databricks')
'''
    result = subprocess.run(
        [sys.executable, '-I', '-c', probe, str(fixture), str(already_imported)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
