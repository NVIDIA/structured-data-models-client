# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which of a crate's files ``rust_notices.py`` reproduces as its terms.

yaml-rust2 (T3.5) ships its two licences as ``MIT-LICENSE`` and
``Apache-LICENSE``, which the pattern missed, so the notices script stopped
with "ships no licence file" though the crate ships both.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    'rust_notices', ROOT / 'scripts/rust_core/rust_notices.py'
)
assert _SPEC is not None and _SPEC.loader is not None
rust_notices = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rust_notices)


@pytest.mark.parametrize(
    ('name', 'terms'),
    [
        ('MIT-LICENSE', True),
        ('Apache-LICENSE', True),
        ('LICENSE', True),
        ('LICENSE-MIT', True),
        ('LICENSE-APACHE', True),
        ('COPYING', True),
        ('NOTICE.txt', True),
        ('UNLICENSE', True),
        ('README.md', False),
        ('Cargo.toml', False),
        ('licensing.rs', False),
        ('GPL-LICENSE', False),
    ],
)
def test_a_crate_s_terms_files_are_found_by_name(
    name: str, terms: bool
) -> None:
    assert bool(rust_notices.TERMS.match(name)) is terms
