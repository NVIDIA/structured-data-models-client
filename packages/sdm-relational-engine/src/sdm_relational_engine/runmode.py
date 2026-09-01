# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Types the client owns that the internalized ``sdm_relational_engine.api`` tree used to hold.

Both come from upstream ``kumo-api``'s ``model_plan.py``, which was dropped as
unreachable (see ``api/SOURCE.md``). They are the only two names in it this client
reaches, and they are used across the runtime, so they live here.
"""

from predictive_query.schema.common import StrEnum


class RunMode(StrEnum):
    r"""How much in-context data a prediction is given.

    The mode caps the number of in-context examples the client collects and also
    selects the per-hop neighbour fanout, trading accuracy against latency and
    payload size. It is sent as ``inference.run_mode``.

    The caps are 100 (``debug``), 1,000 (``fast``), 5,000 (``normal``) and
    10,000 (``best``); the exact counts live in ``sdm_relational_engine.rfm.rfm``. Temporal
    link prediction always uses the ``fast`` budget regardless of the mode.
    """

    FAST = 'fast'
    NORMAL = 'normal'
    BEST = 'best'
    DEBUG = 'debug'


class MissingType(StrEnum):
    r"""Sentinel for a table attribute the caller did not supply.

    Distinguishes "not given, so infer it" from an explicit ``None``, which
    means "this table has no such column".
    """

    VALUE = '???'
