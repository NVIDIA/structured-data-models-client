# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which of a crate's files ``rust_notices.py`` reproduces as its terms, and
which IANA Time Zone Database release it attributes.

yaml-rust2 ships its two licences as ``MIT-LICENSE`` and
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


@pytest.mark.parametrize(
    ('version', 'release'),
    [
        ('Some(r"2026e")', '2026e'),
        ('Some(r"2027a")', None),
        ('None', None),
    ],
)
def test_the_embedded_tzdb_release_is_read_from_jiff_tzdb(
    tmp_path: Path, version: str, release: str | None
) -> None:
    (tmp_path / 'tzname.rs').write_text(
        f'pub(super) static VERSION: Option<&str> = {version};\n'
    )
    if release is None:
        with pytest.raises(SystemExit):
            rust_notices._tzdb_release(tmp_path)
    else:
        assert rust_notices._tzdb_release(tmp_path) == release
