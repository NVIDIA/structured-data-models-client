# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named inference options; defaults and task validation belong to Rust."""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable optional settings for relational inference.

    None means omitted. Rust applies defaults and validates task-specific values
    when materializing. Existing mappings remain available for explicit nulls.
    """

    num_estimators: int | None = None
    column_shuffle: bool | None = None
    category_shuffle: bool | None = None
    hop_shuffle: bool | None = None
    class_shuffle: bool | None = None
    target_transforms: Sequence[str | None] | None = None
    output_type: str | None = None

    def __post_init__(self) -> None:
        if self.target_transforms is not None:
            object.__setattr__(
                self, 'target_transforms', tuple(self.target_transforms)
            )

    def to_dict(self) -> dict[str, Any]:
        """Return fresh plain options, retaining only explicitly supplied fields."""
        result = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        }
        if self.target_transforms is not None:
            result['target_transforms'] = list(self.target_transforms)
        return result
