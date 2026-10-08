# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How a question was rewritten, held privately by its request.

A value the core decides and a request carries. ``sdm_client.decompose``,
the Python that decided it before the core did and that the old path still
runs, has a class of the same shape; a request made by that path, or
unpickled from one, carries that one instead. Both are read the same way:
``kind``, ``parts`` and ``top``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RANKING = 'ranking'
FORECAST = 'forecast'


@dataclass(frozen=True)
class Decomposition:
    r"""The questions one question was rewritten as, and what each is about.

    Carried on the request rather than re-derived from its text: a query
    naming a table ``rank_events`` is not a ranking, and a candidate read back
    out of the text it was written into is a string where a key was given.
    """

    kind: str
    parts: tuple[tuple[str, Any], ...]
    top: int | None = None
