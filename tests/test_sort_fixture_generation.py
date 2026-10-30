# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private kernel controls must match their independently recorded JSON."""

import subprocess
import sys
from pathlib import Path


def test_sort_fixture_constants_match_recorded_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / 'scripts/generate_sort_test_fixtures.py'),
            '--check',
        ],
        check=True,
    )
