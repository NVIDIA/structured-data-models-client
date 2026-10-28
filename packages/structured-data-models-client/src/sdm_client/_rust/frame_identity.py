# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One fixed native conversion of a core-selected complete row identity.

Rust owns eligibility, ordered key definition, null admission and sanitation.
The actual native map/cast/add/alignment effects belong to this conversion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _render(value: object) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        raise ValueError('a key column cannot hold a null value')
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, float) and math.isnan(value):
        raise ValueError('a key column cannot hold a null value')
    return str(value)


def convert(
    frame: pd.DataFrame, definition: dict[str, Any], state: dict[str, Any]
) -> pd.Series:
    """Convert only this immutable identity at its reached assignment position."""
    from sdm_client._rust import codec
    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare

    if definition['codec'] != 'escaped_text_v1':
        raise codec.CodecError('unknown native identity encoding')
    rendered = []
    for column in definition['columns']:
        name = column['name']
        values = frame[name]
        missing = bool(values.isna().any())
        _adopt(
            _prepare(
                {
                    'stage': 'key_nulls',
                    'input': {
                        'definition': definition,
                        'column': name,
                        'missing': missing,
                    },
                },
                _Bindings(),
                state,
            ),
            'key_nulls',
        )
        rendered.append(
            values.map(
                lambda value: (
                    _render(value).replace('\\', '\\\\').replace('\x1f', '\\u')
                )
            ).astype(str)
        )
    joined = rendered[0]
    for part in rendered[1:]:
        joined = joined + '\x1f' + part
    # Use actual native assignment, including index alignment, on the same
    # retained shallow source owner. The complete aligned column is the fact.
    data = frame.copy(deep=False)
    data[definition['derived']] = joined
    return data[definition['derived']]
