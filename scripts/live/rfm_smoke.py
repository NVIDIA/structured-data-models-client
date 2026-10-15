#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A live-model smoke of the Rust core's request (decision E27, T1.11d).

On demand and local only: it never runs in CI, gates nothing, and no
locked check reads it. It builds ``base.regression``'s request on the Rust
core (``Schema.from_source``, ``materialize`` and ``predict`` forced onto
the core over the ``shop`` base of ``contract/bases.yaml``, in DuckDB) and
sends it to a live model served by a Databricks Model Serving endpoint,
through the client's own ``SDMClient.for_databricks_serving``. It passes
when the model answers the core's request with one finite prediction for
each entity asked about, laid out as the client lays out a result.

    scripts/live/rfm_smoke.py [--endpoint NAME] [--profile NAME]
                              [--route core|old] [--timeout SECONDS]
                              [--dry-run [--show-body]]

The endpoint's name comes from ``--endpoint`` or ``SDM_SMOKE_ENDPOINT``,
and never from this repository; the workspace and its credentials only from
the local Databricks CLI profile (``--profile``, ``SDM_SMOKE_PROFILE``, or
the SDK's own ``DATABRICKS_CONFIG_PROFILE`` and environment). Nothing this
script writes names the endpoint, the workspace, a token or a user: every
line it prints and every warning it shows passes through :func:`redact`,
which knows, before any client exists, every value of the arguments, the
``DATABRICKS_`` and ``SDM_SMOKE_`` variables and the CLI profile, bare host
names included; an error is reported by its class, its code and its HTTP
status alone, never by its text, and with no traceback, a failure or an
interrupt while the CLI profile is read included; a log record of any
logger but the client's is shown by its name and level alone, and a warning
from any code but the client's by its category and module alone; and
argument errors name no value.

``--dry-run`` builds the same request on the core and shows what would be
sent, answered in process by the scripted endpoint's rule, with no network
access: it opens no socket and reads no Databricks configuration.
``--route old`` sends the old route's request instead, as a control; it
needs the relational engine installed.

Exit status: 0 when the smoke passes, 1 when it fails, 2 for bad usage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import socket
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from configparser import ConfigParser
from configparser import Error as ConfigError
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'contract/cases/s1/base.regression.json'
BASES = ROOT / 'contract/bases.yaml'
ENDPOINT_VARIABLE = 'SDM_SMOKE_ENDPOINT'
PROFILE_VARIABLE = 'SDM_SMOKE_PROFILE'

#: What is redacted whatever the secrets known: a URL, a Databricks personal
#: access token, a bearer token, an e-mail address and a workspace host.
_PATTERNS = (
    re.compile(r'[a-z][a-z0-9+.-]*://\S+', re.IGNORECASE),
    re.compile(r'\bdapi[0-9a-f]{16,}(-\d+)?\b', re.IGNORECASE),
    re.compile(r'\bbearer\s+\S+', re.IGNORECASE),
    re.compile(r'[\w.+-]+@[\w-]+(\.[\w-]+)+'),
    re.compile(
        r'\b[\w-]+(\.[\w-]+)*\.(cloud\.databricks\.com|azuredatabricks\.net|'
        r'gcp\.databricks\.com|databricks\.com)\b',
        re.IGNORECASE,
    ),
)
REDACTED = '<redacted>'


class Redactor:
    r"""Removes known secrets and anything shaped like one from text."""

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def know(self, *values: object) -> None:
        r"""Adds each value that is text of some length to the secrets."""
        for value in values:
            if isinstance(value, str) and len(value.strip()) >= 3:
                self._secrets.add(value.strip())

    def know_host(self, value: object) -> None:
        r"""Adds a URL or host, and its bare host name, to the secrets."""
        if not isinstance(value, str) or not value.strip():
            return
        self.know(value)
        for candidate in (value, f'https://{value}'):
            try:
                host = urlparse(candidate.strip()).hostname
            except ValueError:
                host = None
            if host:
                self.know(host)
                break

    def __call__(self, text: object) -> str:
        said = str(text)
        # Longest first, so a secret holding another is removed whole.
        for secret in sorted(self._secrets, key=len, reverse=True):
            said = said.replace(secret, REDACTED)
        for pattern in _PATTERNS:
            said = pattern.sub(REDACTED, said)
        return said


redact = Redactor()


def say(*parts: object, file: Any = None) -> None:
    r"""Prints, redacted."""
    print(redact(' '.join(str(p) for p in parts)), file=file or sys.stdout)


class _RedactingHandler(logging.Handler):
    r"""Shows a log record: the client's own text, redacted; any other
    logger's (the SDK's, the HTTP libraries') by its name and level only,
    since their text may hold a host, a user or an account this script was
    never told of."""

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith('sdm_'):
            say(
                f'log {record.levelname} {record.name} (text withheld)',
                file=sys.stderr,
            )
            return
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a record that cannot format
            message = '(a record that does not format)'
        say(f'log {record.levelname} {record.name}: {message}', file=sys.stderr)


def _quiet_the_world() -> None:
    r"""Every log record and warning shown, redacted, and none written by
    anything else: the SDK and HTTP libraries can log hosts and headers."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(_RedactingHandler())
    root.setLevel(logging.WARNING)
    for name in ('databricks', 'urllib3', 'requests'):
        logging.getLogger(name).setLevel(logging.WARNING)

    def show(
        message: Warning | str,
        category: type[Warning],
        filename: str = '',
        *_: object,
        **__: object,
    ) -> None:
        # The client's warnings, redacted; anyone else's (the SDK's, the
        # HTTP libraries') by category and module alone, since their text
        # may name what this script was never told of.
        module = Path(str(filename)).stem or '?'
        if _the_client_s(category):
            say(f'warning {category.__name__}: {message}', file=sys.stderr)
        else:
            say(
                f'warning {category.__name__} from {module} (text withheld)',
                file=sys.stderr,
            )

    warnings.showwarning = show  # type: ignore[assignment]


def _the_client_s(category: type[Warning]) -> bool:
    r"""Whether a warning is the client's own, by its category alone: a
    category defined in an ``sdm_client`` module (``SDMWarning`` and the
    client's subclasses of it, and the class the core's warnings are shown
    as); a subclass defined anywhere else is not, whatever it derives from.
    Its text and where it was raised say nothing: the client attributes its
    warnings to the caller's line, and anyone may write ``[CODE]``."""
    module = getattr(category, '__module__', '') or ''
    return module == 'sdm_client' or module.startswith('sdm_client.')


def _no_network() -> None:
    r"""Makes any socket connection fail, for ``--dry-run``."""

    def refuse(*_: object, **__: object) -> None:
        raise OSError('the dry run opens no connection')

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]


class _Parser(argparse.ArgumentParser):
    r"""An argument parser whose errors name no value: argparse quotes the
    arguments it refuses, and an argument may be an endpoint's name."""

    def error(self, message: str) -> NoReturn:  # noqa: ARG002 - never shown
        self.exit(2, 'bad usage: see --help\n')


def _usage(reason: str) -> NoReturn:
    r"""Exits for bad usage, with a fixed ``reason`` that names no value."""
    sys.stderr.write(f'bad usage: {reason}\n')
    raise SystemExit(2)


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _Parser(
        description='A live-model smoke of the Rust core request (E27).'
    )
    parser.add_argument(
        '--endpoint',
        help=f'the serving endpoint name; default ${ENDPOINT_VARIABLE}',
    )
    parser.add_argument(
        '--profile',
        help=f'the Databricks CLI profile; default ${PROFILE_VARIABLE}, '
        "then the SDK's own configuration",
    )
    parser.add_argument('--route', choices=('core', 'old'), default='core')
    parser.add_argument(
        '--timeout',
        type=float,
        default=600.0,
        help='seconds one attempt may take (a scaled-to-zero endpoint '
        'starts in minutes)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='build the request and show what would be sent, offline',
    )
    parser.add_argument(
        '--show-body',
        action='store_true',
        help='with --dry-run, print the whole request body',
    )
    arguments = parser.parse_args(argv)
    if arguments.show_body and not arguments.dry_run:
        _usage('--show-body goes with --dry-run')
    if arguments.timeout <= 0:
        _usage('--timeout must be positive')
    return arguments


def resolve(
    arguments: argparse.Namespace, environ: dict[str, str] | None = None
) -> tuple[str | None, str | None]:
    r"""The endpoint and the profile: an argument, else its variable."""
    environ = dict(os.environ if environ is None else environ)
    endpoint = arguments.endpoint or environ.get(ENDPOINT_VARIABLE) or None
    profile = arguments.profile or environ.get(PROFILE_VARIABLE) or None
    redact.know(endpoint, profile)
    know_environment(environ)
    return endpoint, profile


def know_environment(environ: dict[str, str]) -> None:
    r"""Adds to the secrets each ``DATABRICKS_`` and ``SDM_SMOKE_``
    variable's value, a host's bare name included; it reads no file."""
    for name, value in environ.items():
        if name.startswith(('DATABRICKS_', 'SDM_SMOKE_')):
            redact.know_host(value)


def know_profile(environ: dict[str, str], profile: str | None) -> None:
    r"""Adds to the secrets, before any client exists, each value of the
    profile's section of the CLI's configuration file
    (``DATABRICKS_CONFIG_FILE``, or ``~/.databrickscfg``), a host's bare
    name included. It reads that file, so :func:`main` calls it only for a
    live smoke, inside the boundary that reports a failure or an interrupt
    in fixed words."""
    path = Path(
        environ.get('DATABRICKS_CONFIG_FILE')
        or Path(environ.get('HOME', '~')).expanduser() / '.databrickscfg'
    )
    parser = ConfigParser(interpolation=None)
    try:
        parser.read(path, encoding='utf-8')
    except (OSError, ConfigError, UnicodeDecodeError):
        return
    section = (
        profile
        or environ.get('DATABRICKS_CONFIG_PROFILE')
        or parser.default_section
    )
    if section != parser.default_section and not parser.has_section(section):
        return
    for value in parser[section].values():
        redact.know_host(value)


def _shop() -> list[str]:
    r"""The ``shop`` base's statements, read without a YAML library."""
    lines = BASES.read_text(encoding='utf-8').splitlines()
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def build_request(route: str) -> Any:
    r"""``base.regression``'s request, built on ``route``."""
    import duckdb
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema, _routing
    from sdm_connectors import DuckDBSource

    case = json.loads(CASE.read_text(encoding='utf-8'))
    if route == 'core':
        os.environ[_routing.ENVIRONMENT] = ','.join(case['routes'])
    else:
        os.environ.pop(_routing.ENVIRONMENT, None)
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    given = case['input']
    schema = Schema.from_source(DuckDBSource(connection))
    materialize = given['materialize']
    return PredictiveQuery(given['pql'], schema).materialize(
        indices=list(materialize['indices']),
        anchor_time=pd.Timestamp(materialize['anchor_time']),
    )


class _Recorded:
    r"""A transport answering in process with the scripted endpoint's rule,
    recording each body it is handed."""

    serves_sessions = False
    url = 'in-process'

    def __init__(self) -> None:
        spec = importlib.util.spec_from_file_location(
            'scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._answer: Callable[..., tuple[int, Any]] = module.answer
        self.sent: list[bytes] = []

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode()
        self.sent.append(body)
        status, answer = self._answer('POST', '/v1/predictions', body)
        if status != 200:
            raise RuntimeError(f'the scripted rule refused the body ({status})')
        return answer

    def close(self) -> None:
        pass


def _summary(body: bytes) -> list[str]:
    sent = json.loads(body)
    rows = {
        half: len(sent.get(half, {}).get('instance_table', {}).get('rows', []))
        for half in ('context', 'predict')
    }
    config = sent.get('inference', {}).get('inference_config', {})
    return [
        f'model: {sent.get("model")}',
        f'task: {sent.get("task", {}).get("kind")}',
        f'context rows: {rows["context"]}, rows to predict: {rows["predict"]}',
        f'output: {json.dumps(sent.get("output"))}',
        f'inference options: {sorted(config)}',
        f'metadata: {sorted(sent.get("metadata", {}))} '
        '(request_id drawn per call)',
        f'body: {len(body)} bytes, sha256 {hashlib.sha256(body).hexdigest()}',
    ]


def check(frame: Any, indices: Iterable[Any]) -> list[str]:
    r"""What is wrong with a result: a column, a row or a prediction."""
    problems = []
    columns = list(frame.columns)
    if columns[:3] != ['row_index', 'id', 'prediction']:
        problems.append(f'the result has the columns {columns}')
        return problems
    asked = list(indices)
    if sorted(frame['id'].tolist()) != sorted(asked):
        problems.append(
            f'the result answers {len(frame)} rows for {len(asked)} entities'
        )
    for value in frame['prediction'].tolist():
        if not isinstance(value, float) or not math.isfinite(value):
            problems.append('a prediction is not a finite number')
            break
    return problems


class SmokeError(Exception):
    r"""A failure this script describes in its own fixed words."""


def _client(endpoint: str, profile: str | None, timeout: float) -> Any:
    r"""A client of the serving endpoint, through the local Databricks
    configuration. The SDK's own text on a failure is never shown: it may
    name a user, a client id or a host this script was never told of."""
    from databricks.sdk import WorkspaceClient
    from sdm_client import SDMClient

    try:
        workspace = (
            WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        )
    except Exception:  # noqa: BLE001 - its text is not shown
        raise SmokeError(
            'the Databricks client could not be configured from the local '
            'profile or environment'
        ) from None
    config = workspace.config
    for name in (
        'host',
        'token',
        'username',
        'password',
        'client_id',
        'client_secret',
        'account_id',
        'profile',
        'azure_client_id',
        'azure_tenant_id',
    ):
        redact.know_host(getattr(config, name, None))
    return SDMClient.for_databricks_serving(
        endpoint, workspace_client=workspace, timeout=timeout
    )


def _diagnosis(failed: BaseException) -> str:
    r"""What a failure is, from an allowlist: the exception's class, the
    client's code and HTTP status when it carries them, and this script's
    own words; never the exception's text, which an SDK or an endpoint
    wrote."""
    said = type(failed).__name__
    code = getattr(failed, 'code', None)
    if isinstance(code, str) and re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}', code):
        said += f' [{code}]'
    status = getattr(failed, 'status_code', None)
    if isinstance(status, int) and not isinstance(status, bool):
        said += f' (HTTP {status})'
    if isinstance(failed, SmokeError):
        said += f': {failed}'
    return said


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse(argv)
    endpoint, profile = resolve(arguments)
    _quiet_the_world()
    if not arguments.dry_run and endpoint is None:
        say(
            f'no endpoint: pass --endpoint or set {ENDPOINT_VARIABLE}',
            file=sys.stderr,
        )
        return 2
    try:
        if arguments.dry_run:
            _no_network()
        else:
            # The CLI profile is read for a live smoke only, inside this
            # boundary: a failure or an interrupt while reading it is
            # reported in fixed words.
            know_profile(dict(os.environ), profile)
        from sdm_client import KumoRelational, SDMClient

        case = json.loads(CASE.read_text(encoding='utf-8'))
        indices = case['input']['materialize']['indices']
        request = build_request(arguments.route)
        if arguments.dry_run:
            transport = _Recorded()
            client = SDMClient.__new__(SDMClient)
            client._start(transport)
            result = client.predict(request, model=KumoRelational())
            [body] = transport.sent
            say(f'dry run, {arguments.route} route: what would be sent')
            for line in _summary(body):
                say(f'  {line}')
            if arguments.show_body:
                say(json.dumps(json.loads(body), indent=1))
        else:
            assert endpoint is not None
            say(
                f'live smoke, {arguments.route} route: sending to the endpoint '
                f'from --endpoint or ${ENDPOINT_VARIABLE}'
            )
            with _client(endpoint, profile, arguments.timeout) as client:
                result = client.predict(request, model=KumoRelational())
        frame = result.to_pandas()
        say(frame.to_string(index=False))
        problems = check(frame, indices)
    except BaseException as failed:  # noqa: BLE001 - reported, redacted
        if isinstance(failed, KeyboardInterrupt):
            say('interrupted', file=sys.stderr)
            return 1
        say(f'smoke failed: {_diagnosis(failed)}', file=sys.stderr)
        return 1
    for problem in problems:
        say(f'smoke failed: {problem}', file=sys.stderr)
    if problems:
        return 1
    say('smoke passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
