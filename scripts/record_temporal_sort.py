#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record native stable-sort controls without importing the candidate client."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    zone = ZoneInfo('America/New_York')
    other = ZoneInfo.no_cache('America/New_York')
    early = datetime(2020, 11, 1, 1, 30, tzinfo=zone)
    late = early.replace(fold=1)
    owners = [
        pd.Timestamp(early),
        pd.Timestamp(late),
        early,
        late,
        early.replace(tzinfo=other),
        late.replace(tzinfo=other),
    ]

    def micros(delta: timedelta | None) -> int:
        assert delta is not None
        return (
            delta.days * 86_400 + delta.seconds
        ) * 1_000_000 + delta.microseconds

    atoms = []
    for owner in owners:
        alternate = micros(
            datetime(
                2020, 11, 1, 1, 30, tzinfo=owner.tzinfo, fold=1 - owner.fold
            ).utcoffset()
        )
        identity = 1 if owner.tzinfo is zone else 2
        if type(owner) is pd.Timestamp:
            atoms.append(
                {
                    'kind': 'timestamp',
                    'ticks': int(owner.value),
                    'unit': 'nanosecond',
                    'year': 2020,
                    'month': 11,
                    'day': 1,
                    'micros': 5_400_000_000,
                    'nanosecond': 0,
                    'timezone': identity,
                    'alternate_offset_micros': alternate,
                }
            )
        else:
            atoms.append(
                {
                    'kind': 'zoned_date_time',
                    'ordinal': owner.toordinal(),
                    'micros': 5_400_000_000,
                    'timezone': identity,
                    'offset_micros': micros(owner.utcoffset()),
                    'alternate_offset_micros': alternate,
                }
            )
    cases = []
    for length in (17, 33, 65, 129, 320):
        for pattern in ('cyclic', 'shuffled'):
            codes = (
                np.arange(length) % len(owners)
                if pattern == 'cyclic'
                else np.random.RandomState(42).randint(0, len(owners), length)
            )
            values = np.empty(length, dtype=object)
            for i, code in enumerate(codes):
                values[i] = owners[code]
            cases.append(
                {
                    'name': f'{pattern}_{length}',
                    'codes': codes.tolist(),
                    'positions': np.lexsort((values,)).tolist(),
                }
            )
    args.output.write_text(
        json.dumps(
            {
                'generated_by': 'scripts/record_temporal_sort.py',
                'producer': {'numpy': np.__version__, 'pandas': pd.__version__},
                'atoms': atoms,
                'cases': cases,
            },
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
