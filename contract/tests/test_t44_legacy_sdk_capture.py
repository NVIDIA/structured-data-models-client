# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current tooling can observe the checkpoint SDK's existing legacy port."""

from __future__ import annotations

import json

import pytest
from test_t136_forecast_capture import RUNNER

FUTURE_MODULES = {
    'sdm_client._rust.frame_native',
    'sdm_client._rust.native_sql',
    'sdm_client._rust.relational_predict',
}


def _checkpoint_modules(monkeypatch):
    original = RUNNER.importlib.import_module

    def existing(name, *args, **kwargs):
        if name in FUTURE_MODULES:
            raise ModuleNotFoundError(
                'authored checkpoint module absence', name=name
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(RUNNER.importlib, 'import_module', existing)


@pytest.mark.parametrize('name', sorted(FUTURE_MODULES))
def test_only_the_exact_missing_sdk_module_is_optional(monkeypatch, name):
    retained = ModuleNotFoundError(
        'missing internal dependency', name=name + '.dependency'
    )

    def missing(*args, **kwargs):
        raise retained

    monkeypatch.setattr(RUNNER.importlib, 'import_module', missing)
    with pytest.raises(ModuleNotFoundError) as caught:
        RUNNER._optional_sdk_module(name)
    assert caught.value is retained
    retained.name = name
    assert RUNNER._optional_sdk_module(name) is None


def test_other_import_failures_keep_the_actual_owner(monkeypatch):
    retained = ImportError('existing SDK module could not import')

    def failing(*args, **kwargs):
        raise retained

    monkeypatch.setattr(RUNNER.importlib, 'import_module', failing)
    with pytest.raises(ImportError) as caught:
        RUNNER._optional_sdk_module('sdm_client._rust.native_sql')
    assert caught.value is retained


def test_checkpoint_common_hooks_keep_actual_legacy_start_and_capabilities(
    monkeypatch,
):
    from sdm_client._rust import errors, run

    _checkpoint_modules(monkeypatch)
    original_native, original_presenter, original_carry = (
        run.native,
        errors.error_from_core,
        run._carried,
    )
    boundary = RUNNER._Boundary()
    try:
        assert run.native is not original_native
        assert errors.error_from_core is not original_presenter
        assert run._carried is not original_carry
        operation = {
            'op': 'predict',
            'model': 'kumo-relational',
            'request_id': 'legacy-request',
            'state': {'version': 1, 'query': 1, 'payload': 2},
        }
        raw = json.dumps({'operation': operation}).encode()
        payload = b'{"predict":{"instance_table":{"rows":[[1]]}}}'
        attachments = {
            1: b'"PREDICT users.age FOR EACH users.user_id"',
            2: payload,
        }
        boundary.capture.started(raw, attachments)
        assert boundary.capture.operations == ['predict']
        assert boundary.capture._entries == []
        assert boundary.capture._materialization_stack == []
        assert boundary.capture.core()['capture_capabilities'] == {
            'model_replies': True,
            'native_error_control': True,
        }
        assert raw == json.dumps({'operation': operation}).encode()
        assert attachments == {
            1: b'"PREDICT users.age FOR EACH users.user_id"',
            2: payload,
        }
        assert boundary.reset().operations == []
    finally:
        boundary.close()
    assert (run.native, errors.error_from_core, run._carried) == (
        original_native,
        original_presenter,
        original_carry,
    )


def test_checkpoint_hook_nesting_validates_all_owners_before_restoring(
    monkeypatch,
):
    from sdm_client._rust import run

    _checkpoint_modules(monkeypatch)
    original = run.native
    outer = RUNNER._Boundary()
    inner = RUNNER._Boundary()
    try:
        current = run.native
        with pytest.raises(RUNNER.RunnerFailure, match='LIFO'):
            outer.close()
        assert run.native is current
        inner.close()
        outer.close()
        assert run.native is original
    finally:
        inner.close()
        outer.close()


def test_a_present_sdk_module_with_an_incomplete_api_is_not_optional(
    monkeypatch,
):
    from types import SimpleNamespace

    original = RUNNER.importlib.import_module

    def incomplete(name, *args, **kwargs):
        if name == 'sdm_client._rust.relational_predict':
            return SimpleNamespace()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(RUNNER.importlib, 'import_module', incomplete)
    from sdm_client._rust import run

    previous = run.native
    with pytest.raises(AttributeError):
        RUNNER._Boundary()
    assert run.native is previous
