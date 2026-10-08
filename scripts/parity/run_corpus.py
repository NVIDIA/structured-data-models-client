#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Record and replay the parity corpus.

    run_corpus.py list [--json]
    run_corpus.py record [--corpus DIR] [--only NAME[,NAME...]] [--live]
    run_corpus.py replay [--corpus DIR] [--only NAME[,NAME...]]
    run_corpus.py list --kind sampling [--corpus DIR] [--json]
    run_corpus.py replay --kind sampling [--corpus DIR] [--only NAME[,NAME...]]
    run_corpus.py reanswer [--corpus DIR] [--only NAME[,NAME...]]

``record`` with no names records every offline scenario; ``--live`` records
every live one instead, and needs the credentials ``_parity/live.py`` lists.
PostgreSQL scenarios record against ``SDM_POSTGRES_TEST_DSN``, and two runs
recording into one database take turns. Each scenario is recorded all or
nothing, and stored only if its recording replays; one that cannot be
recorded is reported as ``cannot record <name>: <why>`` and the others still
are. It exits 0 only if every one was recorded.

``replay`` needs no network, database or credentials. It prints one line per
scenario, ``PASS <name>`` or ``FAIL <name> <reason>``, replays every one even
after a failure, and exits 0 only if all passed.

``list`` and ``replay`` accept ``--kind sampling`` for the existing sampling
boundary recordings, defaulting to ``tests/parity/sampling``. Replay uses
the retained native-only driver and requires an already installed binding;
it never builds a binding or records new fixture outputs.

``reanswer`` answers the model calls of stored transcripts again from
today's stand-in model, replaying their database steps: the live
recordings, by default, which cannot be recorded again without credentials.
It writes only what follows from the answers, and keeps a transcript whose
steps or payloads would change as it was.

The format is described in ``tests/parity/FORMAT.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _parity import TRANSCRIPTS
from _parity.corpus import (
    check,
    listing,
    one_line,
    reanswer,
    record_one,
    select,
    stored,
)
from _parity.runner import CannotRecord


def _names(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [name for name in text.split(',') if name]


def _list(args: argparse.Namespace) -> int:
    if args.kind == 'sampling':
        from _parity.sampling import listing as sampling_listing

        entries = sampling_listing(Path(args.corpus))
    else:
        entries = listing(Path(args.corpus))
    if args.json:
        print(json.dumps(entries, indent=1))
        return 0
    for entry in entries:
        print(f'{entry["status"]:<13} {entry["name"]}')
    return 0


def _record(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    try:
        scenarios = select(_names(args.only), live=args.live)
    except CannotRecord as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    failed = 0
    for scenario in scenarios:
        try:
            path = record_one(scenario, corpus)
        except CannotRecord as error:
            print(f'cannot record {scenario.name}: {error}', file=sys.stderr)
            failed += 1
            continue
        except Exception as error:  # noqa: BLE001 - reported; the rest go on
            # One scenario failing must not cost the others a credentialed
            # session. Nothing of it was stored (record_one is all or
            # nothing), and the run still exits non-zero.
            print(
                f'cannot record {scenario.name}: {type(error).__name__}: '
                f'{one_line(str(error))}',
                file=sys.stderr,
            )
            failed += 1
            continue
        print(f'recorded {scenario.name} -> {path}')
    return 2 if failed else 0


def _replay(args: argparse.Namespace) -> int:
    if args.kind == 'sampling':
        from _parity.sampling import replay as sampling_replay

        try:
            results = sampling_replay(Path(args.corpus), _names(args.only))
        except (ValueError, RuntimeError, AssertionError, OSError) as error:
            print(
                f'sampling replay failed: {one_line(str(error))}',
                file=sys.stderr,
            )
            return 1
        for name, reason in results:
            print(
                f'PASS {name}'
                if reason is None
                else f'FAIL {name} {one_line(reason)}',
                flush=True,
            )
        failed = sum(reason is not None for _name, reason in results)
        print(
            f'{len(results) - failed} passed, {failed} failed', file=sys.stderr
        )
        return 1 if failed else 0
    on_disk = stored(Path(args.corpus))
    names = _names(args.only) or sorted(on_disk)
    failed = 0
    for name in names:
        path = on_disk.get(name)
        reason = 'no transcript' if path is None else check(path)
        if reason is None:
            print(f'PASS {name}', flush=True)
        else:
            failed += 1
            print(f'FAIL {name} {one_line(reason)}', flush=True)
    print(f'{len(names) - failed} passed, {failed} failed', file=sys.stderr)
    return 1 if failed else 0


def _reanswer(args: argparse.Namespace) -> int:
    on_disk = stored(Path(args.corpus))
    names = _names(args.only) or sorted(
        name for name, path in on_disk.items() if _is_live(path)
    )
    failed = 0
    for name in names:
        path = on_disk.get(name)
        if path is None:
            print(f'cannot answer {name}: no transcript', file=sys.stderr)
            failed += 1
            continue
        try:
            reanswer(path)
        except CannotRecord as error:
            print(f'cannot answer {name}: {error}', file=sys.stderr)
            failed += 1
            continue
        print(f'answered {name} -> {path}')
    return 2 if failed else 0


def _is_live(path: Path) -> bool:
    from _parity.runner import LIVE

    return json.loads(path.read_text(encoding='utf-8'))['backend'] in LIVE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest='command', required=True)
    listed = commands.add_parser('list', help='every scenario and its status')
    listed.add_argument('--json', action='store_true')
    listed.add_argument('--corpus')
    listed.add_argument(
        '--kind', choices=('transcripts', 'sampling'), default='transcripts'
    )
    for name in ('record', 'replay', 'reanswer'):
        command = commands.add_parser(name)
        command.add_argument('--corpus')
        command.add_argument('--only', help='comma-separated scenario names')
        if name == 'replay':
            command.add_argument(
                '--kind',
                choices=('transcripts', 'sampling'),
                default='transcripts',
            )
        if name == 'record':
            command.add_argument(
                '--live', action='store_true', help='the live scenarios'
            )
    args = parser.parse_args(argv)
    if args.corpus is None:
        if getattr(args, 'kind', 'transcripts') == 'sampling':
            from _parity.sampling import RECORDINGS

            args.corpus = str(RECORDINGS)
        else:
            args.corpus = str(TRANSCRIPTS)
    return {
        'list': _list,
        'record': _record,
        'replay': _replay,
        'reanswer': _reanswer,
    }[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
