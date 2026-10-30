# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native callback origin must survive the host protocol-error boundary."""

import numpy as np
import pandas as pd

from sdm_client._rust.codec import CodecError
from sdm_client._rust.tabular_numeric_owners import NumericOwners


def test_callback_codec_error_remains_a_native_value_error() -> None:
    values = np.empty(1, dtype=object)
    values[0] = float(1e100)
    owners = NumericOwners(pd.Index(values, dtype=object), 'fixed_weak_scalars')
    original = CodecError('raised by the user callback')
    events = []

    def callback(error: str, flags: int) -> None:
        events.append((error, flags))
        raise original

    previous = np.geterrcall()
    try:
        np.seterrcall(callback)
        with np.errstate(over='call'):
            loaded, _ = owners({'kind': 'load_probe_owner', 'position': 0})
            outcome, attached = owners(
                {
                    'kind': 'convert_owner',
                    'owner': loaded['owner'],
                    'destination': {'kind': 'binary_float', 'width': 4},
                    'overflow_possible': True,
                }
            )
        assert events == [('overflow', 2)]
        assert outcome['kind'] == 'raised'
        assert outcome['class'] == 'value_error'
        assert owners.exceptions[outcome['exception']] is original
        assert attached == {}
    finally:
        np.seterrcall(previous)
        owners.close()
