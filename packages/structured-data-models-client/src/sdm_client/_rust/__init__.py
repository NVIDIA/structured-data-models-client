# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python value conversion, provider I/O and lifecycle adapters for Rust.

Importing this package does not load the native extension.
"""


class NativeBoundaryError(RuntimeError):
    """The native extension or a required host representation is unavailable.

    This is a runtime integration failure, distinct from a rejected user request.
    """
