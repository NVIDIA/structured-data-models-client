# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force every frame task on the core while phase defaults remain separate."""

from sdm_client._routing import CORE_ROUTED

FRAME_ROUTES = ','.join(
    sorted(CORE_ROUTED)
    + [
        'schema.from-data/frames/frames',
        'schema.from-data/mapping/frames',
        *(
            f'{operation}/{kind}/frames'
            for operation, kinds in (
                (
                    'query.materialize',
                    (
                        'indices-binary',
                        'indices-multiclass',
                        'indices-regression',
                        'indices-link-prediction',
                        'indices-ranking',
                        'indices-forecast',
                    ),
                ),
                (
                    'client.predict',
                    (
                        'binary',
                        'multiclass',
                        'regression',
                        'link-prediction',
                        'ranking',
                        'forecast',
                    ),
                ),
            )
            for kind in kinds
        ),
    ]
)
