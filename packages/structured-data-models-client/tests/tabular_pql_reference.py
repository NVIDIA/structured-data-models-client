# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent runtime identity and frozen-public receipt selection."""

import hashlib
import importlib
import json
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa


@lru_cache(maxsize=1)
def runtime_identity() -> dict[str, Any]:
    prefix = (
        'numpy.core' if np.__version__.split('.')[0] == '1' else 'numpy._core'
    )
    native = importlib.import_module(prefix + '._multiarray_umath')
    # These dispatch groups cover the actual audited argsort targets, not the
    # candidate profile registry. Full effective features remain in the receipt.
    features = {
        key: bool(value) for key, value in native.__cpu_features__.items()
    }
    identity = {
        'python': '.'.join(map(str, sys.version_info[:2])),
        'machine': platform.machine(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'pyarrow': pa.__version__,
        'native_sha256': hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest(),
        'dispatch_features': {
            key: features.get(key, False)
            for key in ('X86_V3', 'X86_V4', 'AVX512_SKX')
        },
    }
    promotion = getattr(np, '_get_promotion_state', None)
    if promotion is not None:
        identity['promotion_state'] = promotion()
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return {
        'fingerprint': fingerprint,
        'identity': identity,
        'python_full': sys.version,
        'effective_features': features,
    }


@lru_cache(maxsize=1)
def _receipt() -> dict[str, Any]:
    root = Path(__file__).parent / 'data/tabular_pql_reference'
    runtime = runtime_identity()
    path = root / (runtime['fingerprint'] + '.json')
    assert path.exists(), (
        f'No frozen PQL receipt for runtime {runtime["identity"]}'
    )
    receipt = json.loads(path.read_text())
    assert receipt['runtime']['identity'] == runtime['identity']
    assert receipt['runtime']['fingerprint'] == runtime['fingerprint']
    assert len(receipt['reference']['revision']) == 40
    return receipt


def reference_case(group: str, family: str, entrypoint: str) -> dict[str, Any]:
    return _receipt()['cases'][group][family][entrypoint]
