# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from predictive_query.schema.typing import Dtype


@dataclass
class SourceColumn:
    name: str
    dtype: Dtype | None
    is_primary_key: bool
    is_unique_key: bool
    is_nullable: bool


@dataclass
class SourceForeignKey:
    name: str
    dst_table: str
    primary_key: str
