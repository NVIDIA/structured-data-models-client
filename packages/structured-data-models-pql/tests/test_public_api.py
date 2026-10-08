# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The names this distribution promises to the rest of the repository.

Two packages import from here and neither may reach past this surface, so a
symbol disappearing is a breaking change to them rather than a local rename.
"""

from __future__ import annotations

import pytest

PUBLIC = (
    'PQLParser',
    'PredictiveQueryValidator',
    'ValidatedPredictiveQuery',
    'SchemaDefinition',
    'TableDefinition',
    'Dtype',
    'Stype',
    'Capabilities',
    'Profile',
    'RELATIONAL',
    'TABULAR',
)


@pytest.mark.parametrize('name', PUBLIC)
def test_the_package_exposes_its_public_surface(name):
    import predictive_query

    assert hasattr(predictive_query, name), f'missing {name}'
