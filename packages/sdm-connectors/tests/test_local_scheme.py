# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A path read as local must stay local."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from sdm_connectors import read
from sdm_connectors.sql import ConnectorError

# These bind a local socket, which a sandboxed or locked-down runner may
# refuse. Marked so such a runner can deselect them rather than report a
# failure it cannot act on.
pytestmark = pytest.mark.socket


_CSV = b'a,b\n1,2\n'


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:
        self.server.received.append(self.path)
        self.send_response(200)
        self.send_header('Content-Type', 'text/csv')
        self.send_header('Content-Length', str(len(_CSV)))
        self.end_headers()
        self.wfile.write(_CSV)

    def log_message(self, *args: object) -> None:
        pass


@contextmanager
def _serve() -> Iterator[tuple[ThreadingHTTPServer, str]]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


def test_read_local_does_not_fetch_http_urls():
    with _serve() as (server, url):
        with pytest.raises(ConnectorError) as excinfo:
            read(f'{url}/internal-metadata.csv')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert server.received == []


@pytest.mark.parametrize(
    'path, says',
    [
        ('https://example.invalid/table.csv', 'local'),
        ('gs://bucket/key.csv', 'local'),
        ('ftp://example.invalid/table.csv', 'local'),
        # Refused as S3 in the wrong case, not as a scheme read cannot take.
        ('S3://bucket/key.csv', 'lowercase, s3://'),
    ],
)
def test_read_local_rejects_remote_uris(path, says):
    with pytest.raises(ConnectorError) as excinfo:
        read(path)
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert says in str(excinfo.value)


@pytest.mark.parametrize(
    'path',
    [
        'file://example.invalid/table.csv',
        'file://10.255.255.1/share/table.csv',
    ],
)
def test_read_local_rejects_file_uris_naming_a_host(path):
    r"""``urllib`` drops the authority and reads the path locally, so
    ``file://example.invalid/table.csv`` silently becomes ``/table.csv``.
    That is not remote access, but the URI does not mean what it says.
    """
    with pytest.raises(ConnectorError) as excinfo:
        read(path)
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_local_still_reads_plain_and_file_uri_paths(tmp_path):
    path = tmp_path / 'table.csv'
    pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}).to_csv(path, index=False)
    assert read(str(path)).shape == (2, 2)
    assert read(path.as_uri()).shape == (2, 2)


def test_read_local_keeps_windows_style_drive_letters(tmp_path):
    r"""``urlparse('C:/x.csv').scheme`` is ``'c'``; a drive letter is a path,
    not a URI scheme, so it must reach the reader rather than be refused.
    """
    with pytest.raises(ConnectorError) as excinfo:
        read('C:/does-not-exist.csv')
    assert excinfo.value.code == 'NOT_FOUND'
