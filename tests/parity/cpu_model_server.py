# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loopback-only pretrained CPU server for cross-language model smoke tests."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def serve(url_file):
    from sdm_client.in_process.backend import InProcessBackend

    backend = InProcessBackend(device='cpu')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value, status=200):
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            try:
                payload = json.loads(
                    self.rfile.read(int(self.headers['Content-Length']))
                )
                parts = self.path.strip('/').split('/')
                if parts == ['v1', 'predictions']:
                    result = backend.predict(payload)
                elif parts == ['v1', 'sessions']:
                    result = backend.create_session(payload)
                elif (
                    len(parts) == 4
                    and parts[:2] == ['v1', 'sessions']
                    and parts[3] == 'predictions'
                ):
                    result = backend.session_predict(parts[2], payload)
                else:
                    self.reply({'error': 'unknown smoke-test route'}, 404)
                    return
                self.reply(result)
            except Exception:
                # Do not expose request rows or dependency diagnostics over HTTP.
                self.reply({'error': 'CPU model smoke request failed'}, 500)

        def do_DELETE(self):
            parts = self.path.strip('/').split('/')
            if len(parts) != 3 or parts[:2] != ['v1', 'sessions']:
                self.reply({'error': 'unknown smoke-test route'}, 404)
                return
            backend.delete_session(parts[2])
            self.reply({})

    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        url_file.write_text(f'http://127.0.0.1:{server.server_port}')
        try:
            server.serve_forever()
        finally:
            backend.close()
            url_file.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url-file', required=True, type=Path)
    serve(parser.parse_args().url_file)
