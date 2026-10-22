# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The forwarding provider's client to the stand-in (Phase 3 design,
choice 2; T3.3): python-core's warehouse legs send each fetch step the core
asks to ``sdm-standin`` (``sdm-testkit``'s stand-in, served over a pipe)
and hand its answer back to the core.

    with StandIn(capture, binary=path) as stand_in:
        answer = stand_in.fetch(step, attachments)

``capture`` is one case's capture on one leg, as
``contract/runners/captures.py`` wrote it; ``step`` a fetch step as the
core sends it (``{"id", "request", "plan"}``, the plan the leg's own
dialect rendered), as JSON data; ``attachments`` the key sets it binds,
``pyarrow.Table`` by attachment id. The :class:`Answer` holds the rows
(``table``), the provider's error (``error``), or the reason the step is
judged live only (``live_only``: ``B1``, ``B2``), and the codes it was
flagged with (``flagged``: ``B4``). What the stand-in refuses by name (a
recorded table or draw missing, a draw asked with another seed or
fraction text) raises :class:`StandInFailure`: a runner failure, never an
answer.

The pipe: each message a 4-byte big-endian length and that many bytes;
the stand-in's ``ready`` frame first, then per step a JSON request frame
and one Arrow IPC stream frame per attachment, answered by a JSON frame
and, for rows, one Arrow IPC stream frame (``sdm-standin``'s docs).

The binary is ``binary``, else ``SDM_STANDIN``, else ``sdm-standin`` in
``CARGO_TARGET_DIR`` (or ``rust/target``)'s debug and release builds.
"""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
#: The legs the stand-in serves; Lakebase's local engine is the local
#: Postgres, read by the Python host's own provider.
LEGS = ('databricks-sql', 'databricks-spark', 'snowflake')
#: The reason codes a live-only answer carries
#: (``contract/compare-policy.yaml``'s list, T3.2).
CODES = ('B1', 'B2', 'B4')
#: How long one answer may take before the stand-in is taken as hung.
TIMEOUT = 600


class StandInFailure(Exception):
    r"""What the stand-in refused by name, or a stand-in that could not
    start or answer: a runner failure."""


@dataclass
class Answer:
    r"""The stand-in's answer to one fetch step: exactly one of ``table``,
    ``error`` and ``live_only``."""

    table: Any = None
    error: dict[str, Any] | None = None
    live_only: str | None = None
    flagged: list[str] = field(default_factory=list)
    #: How it was answered: ``replay``, ``draw``, ``local`` or
    #: ``live_only`` (diagnostic).
    by: str | None = None


def binary_path() -> Path:
    r"""The ``sdm-standin`` binary: ``SDM_STANDIN``, else the newest build
    under the Cargo target directory."""
    named = os.environ.get('SDM_STANDIN')
    if named:
        return Path(named)
    target = Path(os.environ.get('CARGO_TARGET_DIR') or ROOT / 'rust/target')
    built = [
        target / profile / 'sdm-standin'
        for profile in ('debug', 'release')
        if (target / profile / 'sdm-standin').is_file()
    ]
    if not built:
        raise StandInFailure(
            'no sdm-standin: build it (cargo build -p sdm-testkit --bin '
            'sdm-standin) or name it in SDM_STANDIN'
        )
    return max(built, key=lambda path: path.stat().st_mtime)


def _ipc(table: Any) -> bytes:
    import pyarrow as pa

    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _table(data: bytes) -> Any:
    import pyarrow as pa

    return pa.ipc.open_stream(io.BytesIO(data)).read_all()


class StandIn:
    r"""A stand-in serving one capture, through ``sdm-standin``."""

    def __init__(
        self, capture: Path | str, *, binary: Path | str | None = None
    ):
        self.capture = Path(capture)
        self.binary = Path(binary) if binary is not None else None
        self._process: subprocess.Popen[bytes] | None = None
        self.case: str | None = None
        self.leg: str | None = None

    # The pipe.

    def _write(self, data: bytes) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(struct.pack('>I', len(data)) + data)

    def _read(self) -> bytes:
        assert self._process is not None and self._process.stdout is not None
        head = self._process.stdout.read(4)
        if len(head) < 4:
            raise StandInFailure(self._ended())
        (length,) = struct.unpack('>I', head)
        data = self._process.stdout.read(length)
        if len(data) < length:
            raise StandInFailure(self._ended())
        return data

    def _ended(self) -> str:
        process = self._process
        error = b''
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            if process.stderr is not None:
                error = process.stderr.read() or b''
        said = error.decode('utf-8', 'replace').strip()[-2000:]
        return 'sdm-standin ended' + (f': {said}' if said else '')

    # The context.

    def __enter__(self) -> StandIn:
        binary = self.binary or binary_path()
        self._process = subprocess.Popen(
            [str(binary), '--capture', str(self.capture)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ready = json.loads(self._read())
        if not ready.get('ready'):
            message = ready.get('message', 'no reason given')
            self.close()
            raise StandInFailure(
                f'sdm-standin cannot serve {self.capture}: {message}'
            )
        self.case, self.leg = ready.get('case'), ready.get('leg')
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        for stream in (process.stdin,):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    # A step.

    def fetch(
        self, step: dict[str, Any], attachments: dict[int, Any] | None = None
    ) -> Answer:
        r"""The stand-in's answer to ``step``, its key sets ``attachments``;
        :class:`StandInFailure` for what it refuses by name."""
        if self._process is None:
            raise StandInFailure('the stand-in is not started')
        attachments = attachments or {}
        ids = sorted(attachments)
        self._write(
            json.dumps({'step': step, 'attachments': ids}).encode('utf-8')
        )
        for id_ in ids:
            self._write(_ipc(attachments[id_]))
        assert self._process.stdin is not None
        self._process.stdin.flush()
        head = json.loads(self._read())
        answer = head.get('answer')
        flagged = list(head.get('flagged') or [])
        by = head.get('by')
        if answer == 'data':
            return Answer(table=_table(self._read()), flagged=flagged, by=by)
        if answer == 'error':
            return Answer(error=head.get('error'), flagged=flagged, by=by)
        if answer == 'live_only':
            code = head.get('code')
            if code not in CODES:
                raise StandInFailure(
                    f'a live-only answer with no reason: {head}'
                )
            return Answer(live_only=code, flagged=flagged, by=by)
        if answer == 'refused':
            raise StandInFailure(str(head.get('message')))
        raise StandInFailure(f'an answer the client does not read: {head}')


# --- the forwarding provider ---------------------------------------------------------


class LiveOnly(Exception):
    r"""A step the stand-in cannot judge offline: the run stops there, and
    its verdict (:meth:`Session.verdict`) carries the reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(f'judged live only ({code})')
        self.code = code


class WarehouseError(Exception):
    r"""The warehouse's error to a replayed statement, as the provider's
    error: the core reads it as the source's failure."""

    def __init__(self, error: dict[str, Any] | None) -> None:
        super().__init__('the warehouse refused the statement')
        self.error = error


def _arrow_keys(keys: Any) -> dict[int, Any]:
    r"""The key sets the core handed over (objects exporting an Arrow
    stream, or tables), as pyarrow tables by attachment id."""
    import pyarrow as pa

    found = {}
    for id_, stream in (keys or {}).items():
        if isinstance(stream, pa.Table):
            found[int(id_)] = stream
        else:
            found[int(id_)] = pa.RecordBatchReader._import_from_c_capsule(
                stream.__arrow_c_stream__()
            ).read_all()
    return found


class Session:
    r"""One case's warehouse leg served by the stand-in for python-core
    (``open``): ``provider(step, keys)``, the forwarding provider
    ``run_python.py`` plugs in as the leg's provider; ``names``, each
    placeholder's name offline (the placeholder itself, as the capture
    holds it); ``verdict()``, the stand-in's verdict for the observation's
    ``standin`` field; ``close()``."""

    def __init__(self, stand_in: StandIn, names: dict[str, str]) -> None:
        self._stand_in = stand_in
        self.names = names
        self._live_only: str | None = None
        self._flagged: list[str] = []

    def provider(self, step: dict[str, Any], keys: Any = None) -> Any:
        answer = self._stand_in.fetch(step, _arrow_keys(keys))
        for code in answer.flagged:
            if code not in self._flagged:
                self._flagged.append(code)
        if answer.live_only is not None:
            self._live_only = self._live_only or answer.live_only
            raise LiveOnly(answer.live_only)
        if answer.error is not None:
            raise WarehouseError(answer.error)
        return answer.table

    def verdict(self) -> dict[str, Any]:
        return {'live_only': self._live_only, 'flagged': list(self._flagged)}

    def close(self) -> None:
        self._stand_in.close()


#: Each placeholder a capture holds, read offline as itself.
PLACEHOLDER_NAMES = (
    'catalog',
    'schema',
    'host',
    'user',
    'adventureworks_metric_view',
    'shop_metric_view',
    'shop_semantic_view',
)


class LakebaseSession:
    r"""Lakebase's leg offline (design, choice 2): its local engine is the
    local Postgres CI runs, so the core's own Postgres SQL runs there. The
    case's tables are loaded by the contract loader into a database of the
    session's own, under a schema named as the capture names Lakebase's
    test schema (``{schema}``), exactly as C-1 loaded them on Lakebase;
    ``provider`` is the Python host's Postgres provider over it."""

    def __init__(self, case: dict[str, Any], bases: Any) -> None:
        import importlib.util
        import uuid

        import psycopg
        from psycopg.conninfo import make_conninfo

        dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
        if not dsn:
            raise StandInFailure(
                f'{case["id"]}@lakebase runs on the local Postgres: set '
                'SDM_POSTGRES_TEST_DSN'
            )
        here = Path(__file__).resolve().parent

        def sibling(name: str) -> Any:
            spec = importlib.util.spec_from_file_location(
                f'standin_client_{name}', here / f'{name}.py'
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        runner, loader = sibling('run_python'), sibling('databases')
        self.names = {name: '{' + name + '}' for name in PLACEHOLDER_NAMES}
        self._dsn = dsn
        self._name = f'sdm_lakebase_{uuid.uuid4().hex[:16]}'
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'CREATE DATABASE {self._name}')
        try:
            scoped = make_conninfo(dsn, dbname=self._name)
            self._connection = psycopg.connect(scoped)
            schema = self.names['schema']
            self._connection.execute(f'CREATE SCHEMA {loader.quote(schema)}')
            self._connection.commit()
            duck = runner._connection(case, bases or runner.load_bases())
            try:
                loader.load(
                    duck, 'lakebase', self._connection, namespace=(schema,)
                )
            finally:
                duck.close()
            from sdm_client._rust.providers.postgres import PostgresProvider

            self.provider = PostgresProvider(self._connection)
        except BaseException:
            self.close()
            raise

    def verdict(self) -> dict[str, Any]:
        return {'live_only': None, 'flagged': []}

    def close(self) -> None:
        import psycopg

        connection = getattr(self, '_connection', None)
        if connection is not None:
            connection.close()
            self._connection = None
        with psycopg.connect(self._dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS {self._name} WITH (FORCE)')


def open(  # noqa: A001 - the name run_python.py calls
    leg: str,
    case: dict[str, Any],
    bases: Any,
    captures: Path | str,
    *,
    binary: Path | str | None = None,
) -> Session:
    r"""The stand-in serving ``case``'s capture on ``leg``
    (``captures/<leg>/<case id>/``), or, on ``lakebase``, the local
    Postgres loaded with the case's data (:class:`LakebaseSession`);
    :class:`StandInFailure` when the case has no capture on the leg
    (pending live)."""
    if leg == 'lakebase':
        return LakebaseSession(case, bases)  # type: ignore[return-value]
    if leg not in LEGS:
        raise StandInFailure(
            f'{leg} is not served by the stand-in (it serves {", ".join(LEGS)})'
        )
    directory = Path(captures) / leg / case['id']
    if not (directory / 'capture.json').is_file():
        raise StandInFailure(
            f'{case["id"]}@{leg}: no capture (pending live until E-0 records it)'
        )
    stand_in = StandIn(directory, binary=binary).__enter__()
    names = {name: '{' + name + '}' for name in PLACEHOLDER_NAMES}
    return Session(stand_in, names)
