# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional test-module setup cannot publish an incomplete cached module."""

from __future__ import annotations

import sys

import pytest
from test_t136_forecast_capture import _module


@pytest.mark.parametrize('failure', ['skip', 'error'])
def test_repeated_optional_setup_preserves_its_actual_failure(
    tmp_path, monkeypatch, failure
):
    name = '_t44_optional_failed_' + failure
    monkeypatch.delitem(sys.modules, name, raising=False)
    path = tmp_path / 'optional.py'
    path.write_text(
        'import pytest\n'
        + (
            'pytest.skip("optional serving dependency absent")\n'
            if failure == 'skip'
            else 'raise RuntimeError("optional setup failed")\n'
        )
        + 'ready = True\n'
    )
    error = pytest.skip.Exception if failure == 'skip' else RuntimeError
    for _ in range(2):
        with pytest.raises(error):
            # The second consumer must see the same actual setup outcome,
            # rather than AttributeError on a half-initialized module.
            assert _module(name, path).ready is True
        assert name not in sys.modules


def test_completed_module_is_cached_without_reexecuting_setup(
    tmp_path, monkeypatch
):
    name = '_t44_optional_complete'
    monkeypatch.delitem(sys.modules, name, raising=False)
    path = tmp_path / 'complete.py'
    path.write_text('ready = True\n')
    module = _module(name, path)
    path.write_text('raise RuntimeError("must not execute cached module")\n')
    assert _module(name, path) is module
    assert module.ready is True


def test_failure_cleanup_does_not_remove_an_imports_replacement_owner(
    tmp_path, monkeypatch
):
    name = '_t44_optional_replaced'
    monkeypatch.delitem(sys.modules, name, raising=False)
    path = tmp_path / 'replacement.py'
    path.write_text(
        'import sys\nfrom types import ModuleType\n'
        'replacement = ModuleType(__name__)\nreplacement.ready = True\n'
        'sys.modules[__name__] = replacement\n'
        'raise RuntimeError("failed after publishing different owner")\n'
    )
    with pytest.raises(RuntimeError):
        _module(name, path)
    assert sys.modules[name].ready is True
