# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import sdm_relational_engine.relationallib  # noqa: F401
except Exception as e:
    import platform

    _msg = f"""RFM is not supported in your environment.

💻 Your Environment:
Python version: {platform.python_version()}
Operating system: {platform.system()}
CPU architecture: {platform.machine()}
glibc version: {platform.libc_ver()[1]}

✅ Prebuilt wheels are published for:
* CPython versions: 3.10, 3.11, 3.12, 3.13, 3.14
* Linux (x86_64 and arm64), glibc >=2.28
* macOS 11 and newer (arm64 and x86_64)
* Windows (x86_64)

No source distribution is published, so `pip install structured-data-models-relational-engine` resolves on those
platforms alone.

❌ Not supported:
* CPython below 3.10, or 3.15 and later; any PyPy
* Windows (arm64), musl-based Linux
* glibc versions: <2.28

Please create a feature request at 'https://github.com/NVIDIA/structured-data-models-client'."""

    raise RuntimeError(_msg) from e

from .table import LocalTable
from .graph_store import LocalGraphStore
from .sampler import LocalSampler

__all__ = [
    'LocalGraphStore',
    'LocalSampler',
    'LocalTable',
]
