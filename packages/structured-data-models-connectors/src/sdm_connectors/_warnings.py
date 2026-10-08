# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class ViewConversionWarning(UserWarning):
    r"""Parts of a metric or semantic view could not be represented.

    Filter this category to silence conversion warnings without silencing
    other client warnings.
    """
