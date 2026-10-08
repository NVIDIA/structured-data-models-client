# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a model may be asked, as :class:`~sdm_client.ModelSpec` carries it.

The profiles are the core's. A spec carries one by name; what it allows is
read from the core when it is first asked, so importing this package asks
nothing of it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Capabilities:
    r"""The shape of request a profile allows, as the core says it."""

    model: str
    relationships: bool
    multiple_tables: bool
    link_prediction: bool
    forecasting: bool
    static: bool
    temporal: bool


@dataclass(frozen=True, eq=False)
class Profile:
    r"""A profile, by the name ``predictive_query`` knows it by, and the
    model id that stands for it in the core's registry."""

    name: str
    model_id: str
    _read: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def capabilities(self) -> Capabilities:
        if not self._read:
            from sdm_core import _native

            from sdm_client import _core

            found = json.loads(
                _core.call(_native.model_spec, json.dumps(self.model_id))
            )
            self._read.update(found)
        shape = self._read['shape']
        return Capabilities(model=self._read['model'], **shape)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Profile):
            return other.name == self.name
        # ``predictive_query``'s profile of the same name, where the old
        # implementation is loaded and asks; this never loads it.
        language = sys.modules.get('predictive_query')
        return language is not None and other is getattr(
            language, self.name, None
        )

    def __hash__(self) -> int:
        return hash(self.name)


#: The two profiles, by the names ``predictive_query`` gives them.
TABULAR = Profile('TABULAR', 'kumo-tabular')
RELATIONAL = Profile('RELATIONAL', 'kumo-relational')
