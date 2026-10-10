# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Our credentials, removed from an endpoint's refusal before it is shown.

Decision E47 (owner decision 2026-10-10) shows the model endpoint's own
error text on the core route, as the old route does. Its orchestrator
reading (a), owner to confirm, removes from that text every exact
credential of ours the endpoint echoes. This module does it in two steps,
each once and whole:

- :func:`credentials` collects what the transport sends: the URL's userinfo
  (its password always, its user name when it is 3 characters or more, so
  a short common name is not replaced everywhere), each URL query value,
  the API key, the user and password ``requests`` reads from ``.netrc`` for
  the URL, and a Databricks workspace's token, password and client secret.
  :func:`forms` turns them into every spelling an exact echo may take: each
  value raw, percent-encoded in upper and lower hex, and decoded; and the
  ``user:password`` pair of a ``Basic`` header in standard and URL-safe
  base64, each padded and unpadded.
- :func:`scrubbed` walks a value of any JSON shape without recursion (text,
  numbers, lists, objects, their keys included) and replaces each form
  with :data:`REDACTED`; a form shorter than 3 characters only as a whole
  token, so ordinary words keep their letters. Two keys that become one
  keep both values (the second's key gets a ``#2`` suffix, and so on).

Over-redaction fails closed: a value of the endpoint's that happens to
equal one of our credentials is replaced too. What an exact-value scrub
cannot catch is listed in E47: a credential the client's transport cut
(it truncates an endpoint's ``detail`` at 512 characters before this
runs), one the transport learns where this cannot read it (a Databricks
OAuth or CLI token), and a partial, split or otherwise transformed echo.
"""

from __future__ import annotations

import base64
import re
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlparse

#: What stands in for a credential of ours an endpoint echoes.
REDACTED = '<redacted>'

#: A form shorter than this is replaced only as a whole token.
SHORT = 3

#: Characters that make a token: a short form is replaced only where none
#: of them is beside it.
_TOKEN = r'A-Za-z0-9'


def credentials(transport: Any) -> list[str]:
    r"""Every credential of ours ``transport`` sends, as a value, each once.

    It reads only what the transport holds and the ``.netrc`` file
    ``requests`` reads for the URL, and opens no connection."""
    found: list[Any] = []
    url = getattr(transport, 'url', None)
    if isinstance(url, str):
        parsed = urlparse(url)
        user, password = parsed.username, parsed.password
        if '@' in parsed.netloc:
            # The userinfo as the URL spells it, percent escapes and all.
            found.append(parsed.netloc.rsplit('@', 1)[0])
        if password:
            found.append(password)
        if user and len(unquote(user)) >= SHORT:
            found.append(user)
        if user is not None or password is not None:
            found.append(f'{unquote(user or "")}:{unquote(password or "")}')
        for _, value in parse_qsl(parsed.query, keep_blank_values=False):
            found.append(value)
        found += _netrc(url)
    found.append(getattr(transport, 'api_key', None))
    config = getattr(getattr(transport, '_workspace', None), 'config', None)
    for name in ('token', 'password', 'client_secret'):
        found.append(getattr(config, name, None))
    seen: dict[str, None] = {}
    for value in found:
        if isinstance(value, str) and value:
            seen.setdefault(value, None)
    return list(seen)


def _netrc(url: str) -> list[str]:
    r"""The user and password ``requests`` would send for ``url`` from a
    ``.netrc`` file, and their pair; none when it reads none."""
    try:
        from requests.utils import get_netrc_auth

        found = get_netrc_auth(url)
    except Exception:  # noqa: BLE001 - no file, or one it cannot read
        return []
    if not found:
        return []
    user, password = found
    values = [password] if password else []
    if user and len(user) >= SHORT:
        values.append(user)
    values.append(f'{user}:{password}')
    return values


def _percent(value: str) -> list[str]:
    encoded = quote(value, safe='')
    lower = re.sub(r'%[0-9A-F]{2}', lambda m: m.group(0).lower(), encoded)
    return [encoded, lower]


def _basic(pair: str) -> list[str]:
    raw = pair.encode('utf-8')
    standard = base64.b64encode(raw).decode()
    urlsafe = base64.urlsafe_b64encode(raw).decode()
    latin = base64.b64encode(pair.encode('latin-1', 'replace')).decode()
    return [
        standard,
        standard.rstrip('='),
        urlsafe,
        urlsafe.rstrip('='),
        latin,
        latin.rstrip('='),
    ]


def forms(values: list[str]) -> list[str]:
    r"""Every spelling of ``values`` an exact echo may take, each once,
    longest first: each value raw, percent-encoded in upper and lower hex,
    and decoded; and for a ``user:password`` pair, the token of a ``Basic``
    header in standard and URL-safe base64, padded and unpadded."""
    found: dict[str, None] = {}
    for value in values:
        decoded = unquote(value)
        spellings = [value, decoded, *_percent(value), *_percent(decoded)]
        if ':' in value:
            spellings += _basic(value) + _basic(decoded)
        for spelling in spellings:
            if spelling:
                found.setdefault(spelling, None)
    return sorted(found, key=len, reverse=True)


def pattern(spellings: list[str]) -> re.Pattern[str] | None:
    r"""One pattern of every spelling, the longest first: a short one only
    as a whole token. ``None`` when there is none."""
    if not spellings:
        return None
    parts = [
        re.escape(form)
        if len(form) >= SHORT
        else rf'(?<![{_TOKEN}]){re.escape(form)}(?![{_TOKEN}])'
        for form in sorted(spellings, key=len, reverse=True)
    ]
    return re.compile('|'.join(parts))


def _text(value: str, found: re.Pattern[str] | None) -> str:
    return value if found is None else found.sub(REDACTED, value)


def _leaf(value: Any, found: re.Pattern[str] | None) -> Any:
    r"""A JSON leaf scrubbed: text as text; a number whose spelling holds a
    credential as :data:`REDACTED`; anything else as it is."""
    if isinstance(value, str):
        return _text(value, found)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        spelled = repr(value)
        if _text(spelled, found) != spelled:
            return REDACTED
    return value


def scrubbed(value: Any, found: re.Pattern[str] | None) -> Any:
    r"""``value``, of any JSON shape and depth, copied without recursion
    with every match of ``found`` replaced, in text, numbers and keys; two
    keys that become one keep both values, the later key suffixed."""
    if not isinstance(value, (dict, list, tuple)):
        return _leaf(value, found)
    root: Any = {} if isinstance(value, dict) else []
    stack: list[tuple[Any, Any]] = [(value, root)]
    while stack:
        source, target = stack.pop()
        pairs = (
            source.items() if isinstance(source, dict) else enumerate(source)
        )
        for key, item in pairs:
            if isinstance(item, (dict, list, tuple)):
                copied: Any = {} if isinstance(item, dict) else []
                stack.append((item, copied))
            else:
                copied = _leaf(item, found)
            if isinstance(target, dict):
                name = _leaf(key, found)
                name = name if isinstance(name, str) else str(name)
                unique, count = name, 1
                while unique in target:
                    count += 1
                    unique = f'{name}#{count}'
                target[unique] = copied
            else:
                target.append(copied)
    return root
