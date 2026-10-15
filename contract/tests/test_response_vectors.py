# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prediction responses of ``contract/response-vectors.json`` against
the client's bindings (T1.11d): each body the vectors read is read by
``PredictionResponse.from_dict``, and each they refuse is refused, as the
relational adapter catches a refusal (``ResponseContractError``,
``TypeError`` or ``ValueError``, then ``INVALID_RESPONSE``). ``sdm-wire``'s
reader is held to the same file (``tests/response_vectors.rs``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sdm_api.generated.tfm_api import PredictionResponse, ResponseContractError

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = json.loads(
    (ROOT / 'contract/response-vectors.json').read_text(encoding='utf-8')
)
VECTORS = DOCUMENT['vectors']


def test_the_vector_file() -> None:
    assert DOCUMENT['format'] == 'sdm-response-vectors/1'
    names = [vector['name'] for vector in VECTORS]
    assert len(names) == len(set(names))
    # Not vacuous: both outcomes, and every field the reader refuses.
    assert {'response', 'id', 'model', 'predictions', 'metadata'} <= {
        vector.get('refused') for vector in VECTORS
    }
    assert any('refused' not in vector for vector in VECTORS)


@pytest.mark.parametrize('vector', VECTORS, ids=[v['name'] for v in VECTORS])
def test_the_bindings_read_each_vector_alike(vector: dict[str, Any]) -> None:
    body = json.loads(vector['body'])
    if 'refused' not in vector:
        PredictionResponse.from_dict(body)
        return
    with pytest.raises((ResponseContractError, TypeError, ValueError)):
        PredictionResponse.from_dict(body)
