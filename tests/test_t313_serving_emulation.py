# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A serving compatibility exception never opens another emulation module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))

import l0_scan  # noqa: E402


@pytest.mark.parametrize(
    ('entry', 'permitted'),
    [
        ('rust/crates/sdm-pql/src/float_repr.rs', True),
        ('rust/crates/sdm-request/src/serving_compat.rs', True),
        ('rust/crates/sdm-request/src/serving.rs', False),
        ('rust/crates/sdm-request/src/another_compat.rs', False),
        ('rust/crates/sdm-request/src/serving_compat/other.rs', False),
        ('rust/crates/sdm-core/src/serving_compat.rs', False),
    ],
)
def test_emulation_locations_remain_bounded(
    tmp_path: Path, entry: str, permitted: bool
) -> None:
    module = tmp_path / entry
    module.parent.mkdir(parents=True)
    module.write_text('pub fn python_value() {}\n')
    allowlist = tmp_path / l0_scan.ALLOWLIST
    allowlist.parent.mkdir(parents=True)
    allowlist.write_text(entry + '\n')

    entries, problems = l0_scan.read_allowlist(tmp_path, 'sdm-pql')

    if permitted:
        assert entries == [Path(entry)]
        assert problems == []
    else:
        assert entries == []
        assert len(problems) == 1
        assert 'is outside rust/crates/sdm-pql/src' in problems[0]
