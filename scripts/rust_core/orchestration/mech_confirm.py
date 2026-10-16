#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mechanical post-move confirm (owner decision 28, 2026-10-14).

    mech_confirm.py --repo DIR --old-base A --old-tip X --new-base B --new-tip C \
        --review FILE [--review FILE ...] --out-dir DIR --name PKG

Confirms that C carries exactly the patches X carried: the commits in B..C pair one
to one, in order, with the commits in A..X, and each pair has the same subject and
the same changed lines (git show -U0, hunk headers dropped, so only moved line
numbers may differ). Each --review must be an APPROVE naming X. On success it
writes, for each review, a confirm file whose line 1 is APPROVE and line 2 is C's
full SHA, citing that review, and prints their paths. On any difference it writes
nothing, prints the first difference and exits 1: the reviewers look again.
"""

import argparse
import pathlib
import re
import subprocess
import sys


def git(repo, *a):
    return subprocess.run(
        ['git', '-C', repo, *a], check=True, capture_output=True, text=True
    ).stdout


def commits(repo, base, tip):
    return git(
        repo, 'rev-list', '--reverse', '--no-merges', f'{base}..{tip}'
    ).split()


def lines(repo, sha):
    out = git(repo, 'show', '-U0', '--format=%s%n%b', '--no-color', sha)
    return [
        ln
        for ln in out.splitlines()
        if not ln.startswith('@@')
        and not ln.startswith('index ')
        and not ln.startswith('Signed-off-by:')
    ]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for k in (
        'repo',
        'old-base',
        'old-tip',
        'new-base',
        'new-tip',
        'out-dir',
        'name',
    ):
        p.add_argument('--' + k, required=True)
    p.add_argument('--review', action='append', required=True)
    a = p.parse_args()
    r = a.repo
    x = git(r, 'rev-parse', a.old_tip).strip()
    c = git(r, 'rev-parse', a.new_tip).strip()
    for f in a.review:
        t = pathlib.Path(f).read_text().splitlines()
        if (
            not t
            or t[0] != 'APPROVE'
            or not any(
                x.startswith(w) and len(w) >= 7
                for w in re.findall(r'[0-9a-f]{7,40}', '\n'.join(t[:3]))
            )
        ):
            sys.exit(f'DIFFERENT: review {f} is not an APPROVE naming {x[:12]}')
    old, new = commits(r, a.old_base, x), commits(r, a.new_base, c)
    if len(old) != len(new):
        sys.exit(f'DIFFERENT: {len(old)} commits before, {len(new)} after')
    for o, n in zip(old, new, strict=True):
        if lines(r, o) != lines(r, n):
            sys.exit(f"DIFFERENT: {n[:12]} does not carry {o[:12]}'s change")
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(a.review, 1):
        dst = out / f'{a.name}-mech-confirm-{i}.md'
        dst.write_text(
            f'APPROVE\n{c}\n\nMechanical confirm (owner decision 28, 2026-10-14), not a new review.\n'
            f'{c} carries, commit for commit, the same changes as {x}, which was approved in {f}.\n'
            f'Old range {a.old_base}..{x} ({len(old)} commits); new range {a.new_base}..{c}.\n'
            f'Only the base and moved line numbers differ. CI still runs on {c} itself.\n'
        )
        print(dst)


if __name__ == '__main__':
    main()
