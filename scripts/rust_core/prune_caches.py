#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lists, and with ``--delete`` deletes, the repository's obviously stale
GitHub Actions caches (decision E43).

The repository has 10 GiB of caches. Under E43 only ``main``,
``rust-core``, ``rust-core-phase-1`` and the lane branches save, about
1.9 GB of Rust builds and 1.1 GB of pip downloads each, so the phase branch
and two lanes come to about 9 GB with nothing stale. GitHub evicts the
least recently used past the limit, which can be the phase branch's;
pruning first keeps it warm. A cache is obviously stale when:

- its branch is gone from the remote, or is a topic branch (not a lane)
  whose tip the phase branch already contains, so it has landed and no
  run on it will restore the cache again;
- or a newer cache of the same kind was saved on the same branch: a Rust
  build for an older ``Cargo.lock``, or pip's downloads of an earlier week,
  which a run restores only when no newer one matches.

Anything else, the current entries of every live branch and every cache of
an unlanded topic branch included, is kept. Without ``--delete`` it only
lists. ``--selftest`` runs its own checks, with no network.

    prune_caches.py [--delete] [--remote origin] [--selftest]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable

PHASE = 'rust-core-phase-1'
LONG_LIVED = ('main', 'rust-core', PHASE)
LANE = f'{PHASE}-lane-'
#: A cache's kind: the key without what changes from save to save (the
#: lockfile hash of a Rust build, the week and manifest hash of pip's).
KINDS = (
    re.compile(r'^(v0-rust-.+-[0-9a-f]{8})-[0-9a-f]{8}$'),
    re.compile(r'^(pip-.+?-[^-]+)-\d{4}-\d{2}-[0-9a-f]+$'),
)


def kind_of(key: str) -> str | None:
    for pattern in KINDS:
        match = pattern.match(key)
        if match:
            return match.group(1)
    return None


def stale(
    caches: list[dict[str, object]],
    branch_exists: Callable[[str], bool],
    landed: Callable[[str], bool],
) -> list[tuple[dict[str, object], str]]:
    """Each obviously stale cache, with why."""
    out: list[tuple[dict[str, object], str]] = []
    newest: dict[tuple[str, str], dict[str, object]] = {}
    for cache in caches:
        ref = str(cache['ref'])
        if not ref.startswith('refs/heads/'):
            continue
        branch = ref.removeprefix('refs/heads/')
        keep_branch = branch in LONG_LIVED or branch.startswith(LANE)
        if not branch_exists(branch):
            out.append((cache, f'{branch} is gone'))
            continue
        if (
            not keep_branch
            and branch.startswith(PHASE + '-')
            and landed(branch)
        ):
            out.append((cache, f'{branch} has landed in {PHASE}'))
            continue
        kind = kind_of(str(cache['key']))
        if kind is None:
            continue
        slot = (ref, kind)
        other = newest.get(slot)
        if other is None:
            newest[slot] = cache
            continue
        older, newer = sorted(
            (cache, other), key=lambda c: str(c.get('createdAt', ''))
        )
        newest[slot] = newer
        out.append((older, f'superseded on {branch} by {newer["key"]}'))
    return out


def run(*argv: str) -> str:
    return subprocess.run(
        list(argv), check=True, capture_output=True, text=True
    ).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--delete', action='store_true')
    parser.add_argument('--remote', default='origin')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    url = run('git', 'remote', 'get-url', args.remote).strip()
    match = re.search(r'github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$', url)
    if not match:
        print(f'prune: {args.remote} is not a GitHub remote: {url}')
        return 2
    repo = match.group(1)
    run('git', 'fetch', '--quiet', '--prune', args.remote)
    heads = set(
        run('git', 'for-each-ref', '--format=%(refname:lstrip=3)',
            f'refs/remotes/{args.remote}/').split()
    )  # fmt: skip
    phase = f'refs/remotes/{args.remote}/{PHASE}'

    def landed(branch: str) -> bool:
        tip = f'refs/remotes/{args.remote}/{branch}'
        return (
            subprocess.run(
                ['git', 'merge-base', '--is-ancestor', tip, phase],
                capture_output=True,
            ).returncode
            == 0
        )

    caches = json.loads(
        run('gh', 'cache', 'list', '--repo', repo, '--limit', '500', '--json',
            'id,key,ref,sizeInBytes,createdAt,lastAccessedAt')
    )  # fmt: skip
    found = stale(caches, lambda b: b in heads, landed)
    total = sum(int(c['sizeInBytes']) for c in caches)
    freed = sum(int(c['sizeInBytes']) for c, _ in found)
    for cache, why in found:
        size = int(cache['sizeInBytes']) / 1e6
        print(f'{cache["id"]} {size:8.1f} MB {cache["key"]} ({why})')
        if args.delete:
            run('gh', 'cache', 'delete', '--repo', repo, str(cache['id']))
    verb = 'deleted' if args.delete else 'would delete'
    print(
        f'prune: {len(found)} stale of {len(caches)} caches; {verb} '
        f'{freed / 2**30:.2f} of {total / 2**30:.2f} GiB'
    )
    return 0


def selftest() -> int:
    def cache(
        ref: str, key: str, created: str = '2026-10-09'
    ) -> dict[str, object]:
        return {'id': f'{ref}:{key}', 'ref': f'refs/heads/{ref}', 'key': key,
                'sizeInBytes': 1, 'createdAt': created}  # fmt: skip

    lane = f'{LANE}a'
    rust_old = cache(
        PHASE, 'v0-rust-rust-lint-Linux-x64-df5a5e57-11111111', '1'
    )
    rust_new = cache(
        PHASE, 'v0-rust-rust-lint-Linux-x64-df5a5e57-22222222', '2'
    )
    pip_old = cache(lane, 'pip-Linux-test-3.12-2026-40-' + 'a' * 64, '1')
    pip_new = cache(lane, 'pip-Linux-test-3.12-2026-41-' + 'a' * 64, '2')
    pip_other = cache(lane, 'pip-Linux-test-3.13-2026-40-' + 'a' * 64, '1')
    landed_topic = cache(f'{PHASE}-t1.11b', 'v0-rust-rust-test-x-df5a5e57-1')
    open_topic = cache(f'{PHASE}-t1.12', 'v0-rust-rust-test-x-df5a5e57-1')
    gone = cache(f'{PHASE}-t1.9', 'duckdb-extensions-v1.5.6-Linux-X64')
    ext = cache(PHASE, 'duckdb-extensions-v1.5.6-Linux-X64')
    landed_lane = cache(lane, 'duckdb-extensions-v1.5.6-Linux-X64')
    caches = [rust_old, rust_new, pip_old, pip_new, pip_other, landed_topic,
              open_topic, gone, ext, landed_lane]  # fmt: skip
    found = {
        str(c['id'])
        for c, _ in stale(
            caches,
            lambda b: not b.endswith('t1.9'),
            lambda b: b in (f'{PHASE}-t1.11b', lane),
        )
    }
    expected = {
        str(rust_old['id']),
        str(pip_old['id']),
        str(landed_topic['id']),
        str(gone['id']),
    }
    checks = [
        ('the superseded Rust build and pip week, and only those', found == expected),
        ('a lane is never pruned as landed', str(landed_lane['id']) not in found),
        ('an unlanded topic branch keeps its caches', str(open_topic['id']) not in found),
        ('another Python is another kind', str(pip_other['id']) not in found),
        ('kinds', kind_of(str(rust_new['key'])) == 'v0-rust-rust-lint-Linux-x64-df5a5e57'),
    ]  # fmt: skip
    for name, ok in checks:
        print(f'{"ok  " if ok else "FAIL"} {name}')
    if found != expected:
        print(f'     found {sorted(found)}')
    failed = [n for n, ok in checks if not ok]
    print(
        f'{len(checks) - len(failed)} of {len(checks)} self-tests as expected'
    )
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
