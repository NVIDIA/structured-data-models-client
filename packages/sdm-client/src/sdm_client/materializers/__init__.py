# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Model-specific implementations behind query materialization."""

from sdm_client.materializers.relational import RelationalMaterializer
from sdm_client.materializers.tabular import TabularMaterializer

__all__ = ['RelationalMaterializer', 'TabularMaterializer']
