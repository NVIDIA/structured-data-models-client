# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed package ownership and actual native operation entry points."""

import contextlib
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

import pytest


@contextlib.contextmanager
def operations():
    from sdm_client import _native

    actual = _native.Run
    observed = []

    class Run:
        @staticmethod
        def start(control, attachments):
            operation = json.loads(control)['operation']
            observed.append(operation['op'])
            if operation.get('direct') is not None:
                observed.append(operation['op'] + ':direct')
            if 'action' in operation:
                observed.append(operation['op'] + ':' + operation['action'])
            return actual.start(control, attachments)

    with patch.object(_native, 'Run', Run):
        yield observed


def require_operations(observed, *required):
    assert set(required) <= set(observed), (required, observed)


def test_expected_operation_check_rejects_token_native_call():
    with pytest.raises(AssertionError):
        require_operations(['bind_direct'], 'decode_direct_response')


def test_installed_explanations_has_one_origin_and_no_retired_dependencies():
    import sdm_client
    import sdm_client.explanations as explanations

    root = Path(sdm_client.__file__).parent
    assert Path(explanations.__file__) == root / 'explanations.py'
    assert sdm_client.GradientExplanations is explanations.GradientExplanations
    assert not (root / 'explanations').exists()
    assert not hasattr(explanations, 'GradientDecoder')
    for module in root.rglob('*.py'):
        assert not (module.parent / module.stem / '__init__.py').exists(), (
            module
        )
    dependencies = (
        importlib.metadata.requires('structured-data-models-client') or []
    )
    assert not any(
        'structured-data-models-pql' in dep
        or 'structured-data-models-relational-engine' in dep
        for dep in dependencies
    )


def test_public_prediction_and_merge_require_rust_operations():
    from test_tabular_direct_core import Endpoint, client, frames

    from sdm_client import KumoTabular, PredictionResult, Request

    context, predict = frames()
    with operations() as observed, client(Endpoint()) as model_client:
        result = model_client.predict(
            Request(context, predict, target='target', task='classification'),
            model=KumoTabular(),
        )
        PredictionResult.merge([result, result])
    require_operations(
        observed, 'bind_direct', 'decode_direct_response', 'plan_result_merge'
    )
