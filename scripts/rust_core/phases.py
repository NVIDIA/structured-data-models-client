#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Rust core migration's branches, named once (T2.0).

``rust-core`` is the epic branch. Each phase works on ``rust-core-phase-N``,
cut from it, with lanes ``rust-core-phase-N-lane-<x>``, topic branches
``rust-core-phase-N-<name>`` and archives ``rust-core-phase-N-archive-...``.
Every tool that needs the phase branch takes it from :func:`phase_branch`:
the tool's own flag (``--phase-branch``) when given, else
``SDM_PHASE_BRANCH`` (the variable the orchestration tools require), else
the phase of the branch the tool works on (a lane or topic branch names
its phase), else :data:`CURRENT`, the one setting a new phase changes. CI
names no phase: its patterns match every ``rust-core-phase-N``.

    phases.py [--branch NAME]

prints the phase branch for NAME (by default the branch checked out), as
:func:`phase_branch` resolves it, and exits 2 on a name that is not a
phase branch.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass

#: The epic branch, which every phase is cut from and merged back into.
EPIC = 'rust-core'
#: The phase in progress: the default wherever no flag, no
#: ``SDM_PHASE_BRANCH`` and no branch name says which.
CURRENT = 'rust-core-phase-2'
#: The variable that names the phase branch, shared with the orchestration
#: tools (``scripts/rust_core/orchestration/README.md``).
ENVIRONMENT = 'SDM_PHASE_BRANCH'
#: A phase branch: ``rust-core-phase-N``, N a number with no leading zero.
PHASE_BRANCH = re.compile(r'rust-core-phase-(0|[1-9][0-9]*)')
#: A branch under a phase branch, and the phase branch it is under.
UNDER = re.compile(rf'({PHASE_BRANCH.pattern})-.+')
#: A package: a task (``T1.11e``, ``T2.0a``) or a process package (``CI3``).
PACKAGE_ID = re.compile(r'T[0-9]+\.[0-9]+[a-z]?|CI[0-9]+')
#: A lane's name (decision E43, CI2).
LANE_NAME = re.compile(r'[a-z0-9]{1,16}')


class PhaseError(ValueError):
    r"""A name that is not a phase branch, named."""


@dataclass(frozen=True)
class Phase:
    r"""A phase branch and the names under it."""

    branch: str

    def __post_init__(self) -> None:
        if not PHASE_BRANCH.fullmatch(self.branch):
            raise PhaseError(
                f'{self.branch!r} is not a phase branch (rust-core-phase-N)'
            )

    @property
    def number(self) -> int:
        return int(self.branch.removeprefix('rust-core-phase-'))

    @property
    def lane_prefix(self) -> str:
        return self.branch + '-lane-'

    @property
    def archive_prefix(self) -> str:
        return self.branch + '-archive-'

    @property
    def topic(self) -> re.Pattern[str]:
        r"""A package's topic branch: a branch under the phase branch, a
        lane (``<phase>-lane-a``) or a task (``<phase>-t1.12``)."""
        return re.compile(
            re.escape(self.branch) + r'-[A-Za-z0-9](?:[A-Za-z0-9_-]|\.(?!\.))*'
        )

    def lane_of(self, branch: str) -> str | None:
        r"""The lane's name when ``branch`` is a lane of this phase."""
        if not branch.startswith(self.lane_prefix):
            return None
        name = branch.removeprefix(self.lane_prefix)
        return name if LANE_NAME.fullmatch(name) else None

    def holds(self, branch: str) -> bool:
        r"""``branch`` is the phase branch or a topic branch under it, never
        an archive branch."""
        return branch == self.branch or (
            self.topic.fullmatch(branch) is not None
            and not branch.startswith(self.archive_prefix)
        )


def phase_of(branch: str | None) -> str | None:
    r"""The phase branch ``branch`` is, or is under; None for any other."""
    if not branch:
        return None
    if PHASE_BRANCH.fullmatch(branch):
        return branch
    found = UNDER.fullmatch(branch)
    return found.group(1) if found else None


def phase_branch(
    given: str | None = None,
    *,
    branch: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    r"""The phase branch: ``given`` (a tool's flag), else
    ``SDM_PHASE_BRANCH``, else the phase ``branch`` is under, else
    :data:`CURRENT`. A name that is not a phase branch raises
    :class:`PhaseError`, so a typo never falls back to the default."""
    environ = os.environ if environ is None else environ
    for name in (given, environ.get(ENVIRONMENT) or None):
        if name is not None:
            return Phase(name).branch
    return phase_of(branch) or CURRENT


def checked_out() -> str | None:
    r"""The branch checked out in the working directory, or None."""
    done = subprocess.run(
        ['git', 'symbolic-ref', '--short', '-q', 'HEAD'],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--branch')
    args = parser.parse_args(argv)
    try:
        print(
            phase_branch(branch=args.branch if args.branch else checked_out())
        )
    except PhaseError as error:
        print(f'phases.py: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
