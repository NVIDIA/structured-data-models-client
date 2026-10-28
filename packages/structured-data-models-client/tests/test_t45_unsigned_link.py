# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same public LINK journey with unsigned IDs on both engines."""

import pandas as pd
import pytest
import test_t45_frames as journey
from t38_forced_routes_record import bare
from t45_frame_routes import FRAME_ROUTES


@pytest.mark.parametrize('base', [0, 2**53 - 1, 2**53 + 1, 2**63 - 4])
@pytest.mark.parametrize(
    'shape', ['ordinary', 'null', 'empty-one-neighborhood']
)
def test_unsigned_link_success_keeps_full_payload_and_public_result(
    monkeypatch, base, shape
):
    original_configure = journey.configure

    def configure(kind, actual_shape):
        data, primary, times, links, query = original_configure(
            kind, actual_shape
        )
        for table in ('items', 'buys'):
            data[table]['item_id'] = pd.Series(
                [
                    None if pd.isna(value) else base + int(value)
                    for value in data[table]['item_id']
                ],
                dtype='UInt64',
            )
        return data, primary, times, links, query

    expected = next(
        case['expected']
        for case in journey.REFERENCE['cases']
        if case['kind'] == 'int64' and case['shape'] == shape
    )
    monkeypatch.setattr(journey, 'configure', configure)
    observed = []
    for core in (False, True):
        with monkeypatch.context() as route:
            route.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core else '')
            if core:

                def old_engine_was_called(*args, **kwargs):
                    raise AssertionError('core LINK called the Python engine')

                route.setattr(
                    journey.RelationalMaterializer,
                    '_get_model',
                    old_engine_was_called,
                )
            result = journey.observe('UInt64', shape, expected)
            assert result['outcome'] == 'success', result
            for warning in result['warnings']:
                warning['message'] = bare(warning['message'])
            observed.append(result)
    # Request IDs are replayed by the existing full public observer.
    # Every payload cell, notice, sent body and public result stays compared.
    assert observed[1] == observed[0]
