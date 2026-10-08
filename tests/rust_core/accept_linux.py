# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P4.linux: cibuildwheel builds the manylinux_2_28 abi3 wheels, x86_64 and
aarch64, in Docker, and each passes the smoke test in its build container.

cibuildwheel runs from a copy of the repository with the binding's own
configuration, for both architectures. Its log must hold the smoke test's
success line, with this ``smoke.py``'s hash, from each architecture: that is
the configuration's ``test-command`` running this smoke test.
"""

from __future__ import annotations

import re

import pytest
from _rc import (
    SMOKE,
    linux_wheels,
    sha256,
    smoke_lines,
    wheel_by_arch,
    wheel_tags,
)

ARCHS = ('x86_64', 'aarch64')


@pytest.fixture(scope='module')
def built():
    return linux_wheels()


def test_one_abi3_manylinux_2_28_wheel_per_architecture(built):
    wheels, _ = built
    others = [
        w.name
        for w in wheels
        if not all(p.startswith('manylinux') for p in wheel_tags(w.name)[3])
    ]
    assert not others, f'built wheels beyond manylinux: {others}'
    by_arch = wheel_by_arch(wheels)
    assert set(by_arch) == set(ARCHS), f'built {[w.name for w in wheels]}'
    for arch, wheel in by_arch.items():
        _, py, abi, plats = wheel_tags(wheel.name)
        assert (py, abi) == ('cp310', 'abi3'), wheel.name
        for plat in plats:
            match = re.fullmatch(rf'manylinux_2_(\d+)_{arch}', plat) or (
                re.fullmatch(rf'manylinux2014_{arch}', plat)
            )
            assert match, f'{wheel.name}: {plat} is not a manylinux tag'
            if match.groups():
                assert int(match.group(1)) <= 28, (
                    f'{wheel.name} needs glibc newer than 2.28'
                )


@pytest.mark.parametrize('arch', ARCHS)
def test_the_smoke_test_passed_in_each_build_container(built, arch):
    _, log = built
    digest = sha256(SMOKE)
    passed = [
        line
        for line in smoke_lines(log)
        if line.get('machine') == arch and line.get('sha256') == digest
    ]
    assert passed, (
        f'no SDM_CORE_SMOKE_OK line from this smoke.py on {arch} in the '
        f'cibuildwheel log'
    )
