# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

r"""The release, on its own so a backend can read it.

`__init__` imports `backends` before anything else, so a backend that needs
the version cannot import it from the package root without a cycle.
"""

__version__ = '1.0.0'
