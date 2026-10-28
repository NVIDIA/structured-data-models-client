# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary batch-size refusal precedes every native graph observation."""

import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from t45_frame_routes import FRAME_ROUTES
from test_t44_numeric_static_frames import materialize, query

from sdm_client import StructuredDataError
from sdm_client.materializers.relational import RelationalMaterializer


@pytest.mark.parametrize('rows', [0, -1, True, 1.5, '2'])
@pytest.mark.parametrize('statistic', [None, 'not-a-statistic'])
def test_rows_per_call_refuses_before_native_work_and_keeps_statistic_first(
    monkeypatch, rows, statistic
):
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core else '')
        asked = query()
        effects = []

        def native(*args, **kwargs):
            effects.append('native graph work')
            raise RuntimeError('native work reached before early option refusal')

        options = {'rows_per_call': rows}
        if statistic is not None:
            options['prediction_statistic'] = statistic
        with (
            warnings.catch_warnings(record=True) as notices,
            patch.object(pd.DataFrame, 'copy', native),
            patch.object(RelationalMaterializer, '_get_model', native),
        ):
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as caught:
                materialize(asked, **options)
        error = caught.value
        assert error.code == 'INVALID_REQUEST'
        assert error.__cause__ is None
        assert effects == []
        assert notices == []
        outcomes.append((type(error), error.code, str(error)))
    assert outcomes[1] == outcomes[0]
    assert ('prediction_statistic' if statistic else 'rows_per_call') in outcomes[1][2]
