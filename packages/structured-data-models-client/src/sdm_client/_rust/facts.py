# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The host facts every run starts with.

What the core would otherwise read from the machine, read here and passed
in (contract ``HostFacts``):

- ``now``: the clock, a ``ts`` in microseconds, in UTC;
- ``local_zone``: the local zone, an IANA name, or a POSIX ``TZ`` string the
  host's parser takes, or none when this host cannot name one, by the
  core's rule (``sdm_host::local_zone``, which the Java host reads too);
- ``zone_roots``: the zone lookup roots, in the order Python's own zone
  library, :mod:`zoneinfo`, searches them (``zoneinfo.TZPATH``, then the
  ``tzdata`` package), each with the database version read there;
- ``platform``: the operating system, and whether its ``strftime`` pads a
  year below 1000 to four digits (``date-rules.md``);
- ``date_literal_range``: the instants this host's pandas can hold a PQL
  date literal as, when it cannot hold every year (pandas 2's nanosecond
  ``Timestamp``); left out when it can (pandas 3).
"""

from __future__ import annotations

import datetime
import importlib.util
import os
import re
import sys
import time
import zoneinfo
from pathlib import Path
from types import ModuleType
from typing import Any

from sdm_client._rust import codec

_VERSION = re.compile(r'#\s*version\s+(\S+)')


def _system_version(root: Path) -> str | None:
    r"""The version of the zone database in ``root``, if it says.

    macOS writes it to ``+VERSION``; Debian and most Linux distributions
    write it on the first line of ``tzdata.zi``.
    """
    try:
        found = (root / '+VERSION').read_text(encoding='ascii').strip()
        if found:
            return found
    except (OSError, UnicodeDecodeError):
        pass
    try:
        with (root / 'tzdata.zi').open(encoding='ascii') as zi:
            first = _VERSION.match(zi.readline())
        return first.group(1) if first else None
    except (OSError, UnicodeDecodeError):
        return None


def zone_roots() -> list[dict[str, Any]]:
    r"""The zone lookup roots, in :mod:`zoneinfo`'s order, with versions.

    ``zoneinfo.TZPATH`` first, the directories that exist, then the
    ``tzdata`` package's ``zoneinfo`` directory when it is installed, with
    its ``IANA_VERSION``.
    """
    roots = []
    for path in zoneinfo.TZPATH:
        root = Path(path)
        if root.is_dir():
            roots.append({'path': str(root), 'version': _system_version(root)})
    spec = importlib.util.find_spec('tzdata')
    if spec is not None and spec.submodule_search_locations:
        package = Path(next(iter(spec.submodule_search_locations)))
        root = package / 'zoneinfo'
        if root.is_dir():
            import tzdata  # type: ignore[import-not-found,unused-ignore]

            version = getattr(tzdata, 'IANA_VERSION', None)
            roots.append(
                {
                    'path': str(root),
                    'version': version if isinstance(version, str) else None,
                }
            )
    return roots


def local_zone(
    roots: list[dict[str, Any]], native: ModuleType
) -> dict[str, str] | None:
    r"""The local zone, as the platform names it.

    Read by the one rule every host reads it with, the core's
    (``native.local_zone``, ``sdm_host::local_zone``, T3.12), from ``TZ``
    and the zone lookup roots this host reads: ``TZ`` when set, an IANA
    name a root holds, else a POSIX ``TZ`` string a run's start reads, so a
    ``TZ`` the run could not read is passed as no zone and never refuses
    the run, else none; without ``TZ``, the zone ``/etc/localtime`` links
    to, named from its path under a ``zoneinfo`` directory. None when
    neither names one. ``native`` is ``sdm_client._native``.
    """
    found = native.local_zone(
        os.environ.get('TZ'), [str(root['path']) for root in roots]
    )
    if found is None:
        return None
    kind, name = found
    return {kind: name}


def platform() -> dict[str, Any]:
    r"""The operating system, and the platform's ``strftime`` year padding."""
    name = {'win32': 'windows', 'cygwin': 'windows'}.get(
        sys.platform, sys.platform
    )
    pads = datetime.date(1, 1, 1).strftime('%Y') == '0001'
    return {'os': name, 'strftime_pads_year': pads}


def date_literal_range() -> dict[str, int] | None:
    r"""The instants pandas can hold a PQL date literal as, if not every year.

    The reference reads a date literal with ``pandas.to_datetime``; pandas
    2 holds it as a nanosecond ``Timestamp``, so it refuses one outside
    ``Timestamp.min`` to ``Timestamp.max`` as a literal it cannot read,
    where pandas 3 reads every year. The core refuses such a literal alike
    when the host declares the range (contract ``HostFacts``), in
    nanoseconds from the epoch, inclusive.
    """
    import pandas

    try:
        pandas.to_datetime('0001-01-01')
    except pandas.errors.OutOfBoundsDatetime:
        return {
            'min_ns': int(pandas.Timestamp.min.value),
            'max_ns': int(pandas.Timestamp.max.value),
        }
    return None


def host_facts(native: ModuleType) -> dict[str, Any]:
    r"""The facts of a run starting now, as the contract's ``HostFacts``.

    ``native`` is ``sdm_client._native``, whose rule names the local zone
    (:func:`local_zone`).
    """
    roots = zone_roots()
    facts: dict[str, Any] = {
        'now': codec.timestamp(time.time_ns() // 1000, 'us', 'UTC'),
        'local_zone': local_zone(roots, native),
        'zone_roots': roots,
        'platform': platform(),
    }
    held = date_literal_range()
    if held is not None:
        facts['date_literal_range'] = held
    return facts
