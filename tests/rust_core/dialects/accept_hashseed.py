# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P6.hashseed: the capture takes no order from a set or a dict of hashed
keys. Run under two ``PYTHONHASHSEED`` values, it writes the same bytes, and
they are the committed files."""

from __future__ import annotations

import tempfile
from pathlib import Path

from _dialects import BACKENDS, CAPTURE_DIR, run_capture


def test_two_hash_seeds_write_the_same_capture() -> None:
    with tempfile.TemporaryDirectory(prefix='sdm-p6-hashseed-') as tmp:
        first, second = Path(tmp) / 'a', Path(tmp) / 'b'
        first.mkdir()
        second.mkdir()
        run_capture(first, hashseed='1')
        run_capture(second, hashseed='4242')
        names = [f'{backend}.json' for backend in BACKENDS]
        for out in (first, second):
            written = sorted(p.name for p in out.iterdir())
            assert written == sorted(names), f'the capture wrote {written}'
        differ = [
            name
            for name in names
            if (first / name).read_bytes() != (second / name).read_bytes()
        ]
        assert not differ, f'PYTHONHASHSEED changes {differ}'
        stale = [
            name
            for name in names
            if (first / name).read_bytes() != (CAPTURE_DIR / name).read_bytes()
        ]
        assert not stale, f'the committed capture differs in {stale}'
