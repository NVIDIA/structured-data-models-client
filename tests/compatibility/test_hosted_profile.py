# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required hosted execution cannot silently omit newly supported connectors."""

import pytest
from hosted_profile import profile

from contract import cells, manifest


def test_every_supported_cell_is_required_or_explicitly_optional():
    data = manifest()
    policy = profile(data)
    required = {row['connector'] for row in policy['required']}
    assert required == set(data['connectors']) - {'lakebase'}
    assert sum(row['cells'] for row in policy['required']) == sum(
        cell[1] != 'lakebase' for cell in cells(data)
    )
    assert policy['matrix_mode'] == 'materialization-only'
    assert policy['model_evidence'] == 'separate-pretrained-cpu-smoke'
    assert {row['service'] for row in policy['required']} == {
        'local',
        'emulated-s3',
        'real-warehouse',
    }


def test_new_connector_requires_an_explicit_execution_decision():
    data = manifest()
    data['connectors'].append('new-connector')
    with pytest.raises(ValueError, match='execution policy'):
        profile(data)
