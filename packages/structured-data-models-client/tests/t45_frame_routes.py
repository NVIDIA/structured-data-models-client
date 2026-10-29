# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force every frame task on the core while phase defaults remain separate."""

from sdm_client._routing import (
    CORE_ROUTED,
    FRAME_CONSTRUCTORS,
    FRAME_CONSUMERS,
    PREDICT,
)

FRAME_ROUTES = ','.join(
    sorted({*CORE_ROUTED, *FRAME_CONSTRUCTORS, *FRAME_CONSUMERS})
)
FRAME_PREDICT_ROUTES = ','.join(PREDICT['frames'])
