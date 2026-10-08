# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
from sdm_core.describe import numpy_text_prefix

np = pytest.importorskip('numpy')


MESSAGE = (
    'string or file could not be read to its end due to unmatched data; '
    'this will raise a ValueError in the future.'
)


def test_capability_is_measured_without_warning_leakage():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        capability = numpy_text_prefix()
    assert isinstance(capability, bool)
    assert not captured
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        try:
            values = np.fromstring('x', sep=',', dtype=int)
        except ValueError:
            assert capability is False
            assert not observed
        else:
            assert capability is True
            assert values.size == 0
            assert len(observed) == 1
            assert str(observed[0].message) == MESSAGE


@pytest.mark.parametrize(
    'defect', ['mixed', 'values', 'dtype', 'warning', 'message']
)
def test_unknown_capabilities_are_refused(monkeypatch, defect):
    def altered(text, sep, dtype):
        assert text == 'x' and sep == ','
        if defect == 'mixed' and dtype is int:
            raise ValueError('strict')
        warnings.warn(
            'different warning' if defect == 'message' else MESSAGE,
            UserWarning if defect == 'warning' else DeprecationWarning,
            stacklevel=2,
        )
        return np.array(
            [1] if defect == 'values' else [],
            dtype=bool if defect == 'dtype' else dtype,
        )

    monkeypatch.setattr(np, 'fromstring', altered)
    with pytest.raises(
        ValueError, match='unsupported NumPy text parser capability'
    ):
        numpy_text_prefix()
