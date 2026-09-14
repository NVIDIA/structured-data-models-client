# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Where a warning should say it came from."""

from __future__ import annotations

import sys

_PACKAGE = __name__.split('.')[0]


def caller_stacklevel() -> int:
    r"""How far up to point a warning so it lands on the caller's line.

    A fixed number is right for one route into a warning and blames this
    package on every other. Inference reaches the same setter through
    `Schema.from_data`, and `PredictiveQuery` reaches `Schema.from_data`
    through itself, so the same message is two, three or four frames from the
    line the reader wants to see.

    Counting every frame that belongs to this package answers for all of them,
    and keeps answering when a route gains a layer.
    """
    depth = 1
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get('__name__', '')
        if module != _PACKAGE and not module.startswith(f'{_PACKAGE}.'):
            break
        depth += 1
        frame = frame.f_back
    return depth
