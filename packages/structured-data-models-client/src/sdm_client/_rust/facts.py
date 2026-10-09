# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The host facts every run starts with.

What the core would otherwise read from the machine, read here and passed
in (contract ``HostFacts``):

- ``now``: the clock, a ``ts`` in microseconds, in UTC;
- ``local_zone``: the local zone, an IANA name, or a POSIX ``TZ`` string, or
  none when this host cannot name it;
- ``zone_roots``: the zone lookup roots, in the order Python's own zone
  library, :mod:`zoneinfo`, searches them (``zoneinfo.TZPATH``, then the
  ``tzdata`` package), each with the database version read there;
- ``platform``: the operating system, and whether its ``strftime`` pads a
  year below 1000 to four digits (``date-rules.md``).
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
from typing import Any

from sdm_client._rust import codec

#: A POSIX ``TZ`` string with a rule: a name, an offset and, optionally, a
#: daylight name, offset and rule. A string that looks like neither this nor
#: an IANA name is passed as no zone, so a run never refuses the
#: environment's ``TZ``.
_POSIX_TZ = re.compile(
    r'(?:[A-Za-z]{3,}|<[+-]?[A-Za-z0-9]{3,}>)[+-]?\d{1,2}(?::\d{2}){0,2}'
    r'(?:(?:[A-Za-z]{3,}|<[+-]?[A-Za-z0-9]{3,}>)'
    r'(?:[+-]?\d{1,2}(?::\d{2}){0,2})?'
    r'(?:,[^,\s]+,[^,\s]+)?)?\Z'
)
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


def _iana(name: str, roots: list[dict[str, Any]]) -> bool:
    if name.upper() == 'UTC':
        return True
    if not name or name.startswith('/') or '..' in name.split('/'):
        return False
    return any((Path(root['path']) / name).is_file() for root in roots)


def local_zone(roots: list[dict[str, Any]]) -> dict[str, str] | None:
    r"""The local zone, as the platform names it.

    ``TZ`` when set: an IANA name a root holds, else a POSIX ``TZ`` string,
    else none. Without ``TZ``, the zone ``/etc/localtime`` links to, named
    from its path under a ``zoneinfo`` directory. None when neither names
    one.
    """
    setting = os.environ.get('TZ')
    if setting is not None:
        name = setting[1:] if setting.startswith(':') else setting
        if _iana(name, roots):
            return {'iana': name}
        if _POSIX_TZ.match(setting):
            return {'posix': setting}
        return None
    try:
        target = Path('/etc/localtime').resolve(strict=True)
    except OSError:
        return None
    parts = target.parts
    if 'zoneinfo' not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index('zoneinfo')
    name = '/'.join(parts[index + 1 :])
    return {'iana': name} if name else None


def platform() -> dict[str, Any]:
    r"""The operating system, and the platform's ``strftime`` year padding."""
    name = {'win32': 'windows', 'cygwin': 'windows'}.get(
        sys.platform, sys.platform
    )
    pads = datetime.date(1, 1, 1).strftime('%Y') == '0001'
    return {'os': name, 'strftime_pads_year': pads}


def host_facts() -> dict[str, Any]:
    r"""The facts of a run starting now, as the contract's ``HostFacts``."""
    roots = zone_roots()
    return {
        'now': codec.timestamp(time.time_ns() // 1000, 'us', 'UTC'),
        'local_zone': local_zone(roots),
        'zone_roots': roots,
        'platform': platform(),
    }
