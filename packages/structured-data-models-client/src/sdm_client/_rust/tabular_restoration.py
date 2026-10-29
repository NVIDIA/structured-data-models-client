# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached native dtype conversions over a closed complete response product.

The retained dtype owns its actual Python protocols, including the original
per-item order and errors. Rust owns source admission and final layout.
"""

from __future__ import annotations

from typing import Any

from sdm_client._rust.native_values import _encoded_native, _native
from sdm_client.wire.dtypes import as_declared


def restore(source: dict[str, Any], dtype: Any) -> dict[str, Any]:
    rows = []
    for row in source['rows']:
        prediction = as_declared(_native(row['prediction']), dtype)
        probabilities = None
        if row['probabilities'] is not None:
            probabilities = []
            for given in row['probabilities']:
                key = _native(given['value'])
                restored = as_declared(key, dtype)
                name = key if restored is key else str(restored)
                probabilities.append(
                    {
                        'ordinal': given['ordinal'],
                        'value': _encoded_native(restored),
                        'name': name,
                        'probability': given['probability'],
                    }
                )
        rows.append(
            {
                'ordinal': row['ordinal'],
                'prediction': _encoded_native(prediction),
                'probabilities': probabilities,
            }
        )
    return {'scope': source['scope'], 'rows': rows}
