#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official Unicode data shared by the separate data generators.

This module performs no downloads and imports no product code.
Each generator owns its exact inputs, transformation and output provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

LICENSE_URL = 'https://www.unicode.org/license.txt'
LICENSE_SHA256 = (
    'e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96'
)


def sha256(path: Path) -> str:
    """Hash exact source or output bytes for the generator's provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(path: Path, digest: str) -> bytes:
    """Read an immutable input only when its declared digest matches."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'input digest mismatch: {path.name}')
    return data
