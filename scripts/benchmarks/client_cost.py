#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alternate verified client installations against one fixed HTTP fixture."""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent


def runtime_environment():
    env = dict(os.environ)
    for variable in (
        'SDM_PARITY_TRACE',
        'JAVA_TOOL_OPTIONS',
        'JDK_JAVA_OPTIONS',
        '_JAVA_OPTIONS',
        'PYTHONPATH',
        'PYTHONHOME',
    ):
        env.pop(variable, None)
    return env


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_revision(repo, expected):
    def git(*args):
        return subprocess.check_output(
            ['git', '-C', str(repo), *args], text=True
        ).strip()

    actual = git('rev-parse', 'HEAD')
    if actual != expected or git('status', '--porcelain'):
        raise ValueError(
            'benchmark source must be clean and match the full pinned revision'
        )
    return actual


def answer(payload, model, gradients):
    """Independent constant regression/gradient fixture, also used by Java."""
    table = payload['predict']['instance_table']
    predictions = []
    for index, row in enumerate(table['rows']):
        item = {'row_index': index, 'prediction': 1.0}
        if model == 'kumo-relational':
            item['id'] = str(row[table['columns'].index('instance_id')])
        if gradients:

            def block(positions, columns):
                return {
                    'row_indices': positions,
                    'columns': columns,
                    'gradients': [[[0.25] * len(columns) for _ in positions]],
                }

            columns = [
                name for name in table['columns'] if name.startswith('x')
            ]
            related = {}
            if model == 'kumo-relational':
                # The entity's numerical features live in its related users table.
                # Positions are local to the actual outgoing table, scoped by owner.
                users = payload['predict']['related_tables']['users']
                owner_column = users['columns'].index('instance_id')
                owner = row[table['columns'].index('instance_id')]
                positions = [
                    at
                    for at, values in enumerate(users['rows'])
                    if values[owner_column] == owner
                ]
                feature_columns = [
                    name for name in users['columns'] if name.startswith('x')
                ]
                if not positions or not feature_columns:
                    raise ValueError(
                        'relational gradient fixture needs owned users rows and numerical x features'
                    )
                related['users'] = block(positions, feature_columns)
                columns = []
            elif not columns:
                raise ValueError(
                    'fixed gradient fixture requires numerical feature columns'
                )
            item['explanation'] = {
                'format': 'sdm_gradient_v1',
                'details': {
                    'input_space': 'preprocessed_numerical',
                    'outputs': [{'kind': 'prediction', 'statistic': 'mean'}],
                    'instance_table': block([index], columns),
                    'related_tables': related,
                },
            }
        predictions.append(item)
    return {
        'id': 'benchmark',
        'model': model,
        'metadata': {},
        'predictions': predictions,
    }


def request_rows(payload, kind):
    """Counts only: retain no full request data or captured Arrow attachments."""
    counts = {'kind': kind}
    for half in ('context', 'predict'):
        part = payload.get(half, {})
        counts[half] = {
            'present': half in payload,
            'instance_rows': len(
                part.get('instance_table', {}).get('rows', [])
            ),
            'related_rows': {
                name: len(table.get('rows', []))
                for name, table in part.get('related_tables', {}).items()
            },
        }
    return counts


class Fixture(ThreadingHTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        # Separate header/body writes must not add a fixture-induced delayed ACK.
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection, address

    def __init__(self, gradients):
        super().__init__(('127.0.0.1', 0), Handler)
        self.gradients = gradients
        self.sessions = {}
        self.row_counts = []
        self.counts = {
            'predict': 0,
            'fit': 0,
            'close': 0,
            'request_bytes': 0,
            'response_bytes': 0,
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def respond(self, status, value=None):
        content = (
            json.dumps(value, separators=(',', ':')).encode()
            if value is not None
            else b''
        )
        self.server.counts['response_bytes'] += len(content)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        self.server.counts['request_bytes'] += len(raw)
        payload = json.loads(raw)
        self.server.row_counts.append(
            request_rows(
                payload, 'fit' if self.path == '/v1/sessions' else 'predict'
            )
        )
        if self.path == '/v1/sessions':
            self.server.counts['fit'] += 1
            sid = str(self.server.counts['fit'])
            self.server.sessions[sid] = payload.get('model', 'kumo-tabular')
            self.respond(200, {'session_id': sid})
        else:
            self.server.counts['predict'] += 1
            sid = (
                self.path.split('/')[3]
                if self.path.startswith('/v1/sessions/')
                else None
            )
            model = payload.get(
                'model', self.server.sessions.get(sid, 'kumo-tabular')
            )
            self.respond(200, answer(payload, model, self.server.gradients))

    def do_DELETE(self):
        self.server.counts['close'] += 1
        raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.server.counts['request_bytes'] += len(raw)
        self.server.sessions.pop(self.path.rsplit('/', 1)[-1], None)
        self.respond(204)


def rss_summary(samples):
    """Trapezoidal mean over observed interval, never an allocation counter."""
    if len(samples) < 2 or samples[-1][0] <= samples[0][0]:
        raise ValueError('at least two distinct RSS sample times are required')
    if any(
        b[0] <= a[0] for a, b in zip(samples, samples[1:], strict=False)
    ) or any(rss < 0 for _, rss in samples):
        raise ValueError(
            'RSS samples need monotonic times and nonnegative values'
        )
    seconds = samples[-1][0] - samples[0][0]
    area = sum(
        (b[0] - a[0]) * (a[1] + b[1]) / 2
        for a, b in zip(samples, samples[1:], strict=False)
    )
    return {
        'sampled_peak_bytes': max(rss for _, rss in samples),
        'time_weighted_mean_bytes': area / seconds,
        'observed_seconds': seconds,
        'sample_count': len(samples),
    }


def launch(command, env, interval, timeout, failure_log):
    import psutil

    start = time.perf_counter()
    start_epoch = time.time()
    samples, descendants = [], set()
    # Temporary files prevent child stdout/stderr pipe backpressure from affecting timings.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command, env=env, stdout=stdout, stderr=stderr
        )
        observed = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                now = time.perf_counter()
                if now - start > timeout:
                    raise TimeoutError('benchmark child timed out')
                try:
                    samples.append((now - start, observed.memory_info().rss))
                    descendants.update(
                        child.pid for child in observed.children(recursive=True)
                    )
                except psutil.NoSuchProcess:
                    break
                time.sleep(interval)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        elapsed = time.perf_counter() - start
        stdout.seek(0)
        stderr.seek(0)
        if code:
            # Endpoint errors can contain credentials; retain stderr privately, never echo it.
            with os.fdopen(
                os.open(
                    failure_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
                ),
                'wb',
            ) as log:
                log.write(stderr.read())
            raise RuntimeError(
                f'benchmark child exited {code}; private diagnostics: {failure_log}'
            )
        lines = stdout.read().decode().splitlines()
        result = json.loads(lines[-1])
    result['process_wall_seconds'] = elapsed
    result['startup_to_harness_entry_seconds'] = (
        result.pop('entry_epoch_seconds') - start_epoch
    )
    result['rss'] = {
        **rss_summary(samples),
        'interval_seconds': interval,
        'scope': 'benchmark process only, server and descendants excluded',
        'observed_descendant_pids': sorted(descendants),
        'samples_seconds_bytes': samples,
        'coverage': 'first through last parent sample, excludes unsampled startup/exit tails',
    }
    return result


def verify_wheel(wheel, identity, revision, packages):
    import zipfile

    if (
        not identity.get('tree_clean')
        or identity.get('commit') != revision
        or identity.get('sha256') != digest(wheel)
    ):
        raise ValueError('wheel sidecar differs from clean pinned artifact')
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            name, _, relative = member.partition('/')
            if name in packages and relative and not member.endswith('/'):
                installed = Path(packages[name]) / relative
                actual = hashlib.sha256(archive.read(member)).hexdigest()
                if actual != digest(installed):
                    raise ValueError(
                        'installed package bytes differ from wheel'
                    )
                recorded_native = identity.get('native_files', {}).get(member)
                if recorded_native and recorded_native != actual:
                    raise ValueError(
                        'legacy native receipt differs from artifact'
                    )


def verify_variant(variant):
    clean_revision(variant['repo'], variant['revision'])
    verified = {'revision': variant['revision'], 'label': variant['label']}
    if variant['language'] == 'python':
        probe = json.loads(
            subprocess.check_output(
                [variant['python'], str(HERE / 'python_case.py'), '--probe'],
                text=True,
                env=runtime_environment(),
            )
        )
        verified['installed'] = probe
        expected = variant.get('dependency_sha256')
        if expected and expected != probe['dependency_sha256']:
            raise ValueError(
                'installed dependency versions differ from expected digest'
            )
        # Verify installed Python source bytes against the clean source checkout,
        # including legacy engine/PQL modules. Non-editable installs are valid too.
        for name, package in probe['packages'].items():
            if name == 'sdm_client' and variant.get('wheel'):
                continue
            matches = list(Path(variant['repo']).glob(f'packages/*/src/{name}'))
            if len(matches) != 1:
                raise ValueError(f'cannot locate {name} in verified checkout')
            installed = Path(package)
            source_files = {
                path.relative_to(matches[0])
                for path in matches[0].rglob('*.py')
            }
            installed_files = {
                path.relative_to(installed) for path in installed.rglob('*.py')
            }
            if source_files != installed_files or any(
                digest(installed / path) != digest(matches[0] / path)
                for path in source_files
            ):
                raise ValueError(
                    f'{name} installed source bytes differ from pinned checkout'
                )
        if variant.get('wheel'):
            wheel = Path(variant['wheel'])
            identity = json.loads(Path(str(wheel) + '.json').read_text())
            verify_wheel(
                wheel, identity, variant['revision'], probe['packages']
            )
            if identity['native_sha256'] != probe['native_sha256']:
                raise ValueError('installed native bytes differ from wheel')
            verified['artifact'] = identity
        elif probe['native_sha256']:
            raise ValueError(
                'native Python variants require a wheel and its build sidecar'
            )
        if 'sdm_relational_engine' in probe['packages']:
            receipt_path = Path(variant['engine_provenance'])
            identity = json.loads(receipt_path.read_text())
            verify_wheel(
                receipt_path.parent / identity['wheel'],
                identity,
                variant['revision'],
                probe['packages'],
            )
            verified['legacy_engine_artifact'] = identity
        command = [variant['python'], str(HERE / 'python_case.py')]
    elif variant['language'] == 'java':
        receipt = json.loads(Path(variant['receipt']).read_text())
        jar = str(Path(variant['jar']).resolve())
        if receipt.get('commit') != variant['revision'] or receipt.get(
            'package_hash'
        ) != digest(jar):
            raise ValueError('Java artifact identity mismatch')
        import zipfile

        with zipfile.ZipFile(jar) as archive:
            os_name = {
                'Linux': 'linux',
                'Darwin': 'macos',
                'Windows': 'windows',
            }[platform.system()]
            arch = {'AMD64': 'x86_64', 'arm64': 'aarch64'}.get(
                platform.machine(), platform.machine()
            )
            filename = {
                'linux': 'libsdm_jni.so',
                'macos': 'libsdm_jni.dylib',
                'windows': 'sdm_jni.dll',
            }[os_name]
            native = f'com/nvidia/sdm/native/{os_name}-{arch}/{filename}'
            if hashlib.sha256(archive.read(native)).hexdigest() != receipt.get(
                'native_hash'
            ):
                raise ValueError('packaged JNI does not match Java provenance')
        verified['artifact'] = receipt
        jars = [Path(jar), *[Path(path) for path in variant['dependencies']]]
        verified['artifacts'] = {str(path): digest(path) for path in jars}
        actual_dependencies = hashlib.sha256(
            json.dumps(verified['artifacts'], sort_keys=True).encode()
        ).hexdigest()
        verified['dependency_sha256'] = actual_dependencies
        if (
            variant.get('dependency_sha256')
            and variant['dependency_sha256'] != actual_dependencies
        ):
            raise ValueError(
                'actual dependency JAR hashes differ from expected digest'
            )
        verified['jvm'] = subprocess.run(
            [variant['java'], '-version'],
            env=runtime_environment(),
            capture_output=True,
            text=True,
            check=True,
        ).stderr
        Path(variant['harness_dir']).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(Path(variant['java']).with_name('javac')),
                '-cp',
                os.pathsep.join(str(path) for path in jars),
                '-d',
                variant['harness_dir'],
                str(HERE / 'ClientCost.java'),
            ],
            env=runtime_environment(),
            check=True,
        )
        verified['harness_sha256'] = digest(
            Path(variant['harness_dir']) / 'ClientCost.class'
        )
        command = [
            variant['java'],
            '--add-opens=java.base/java.nio=ALL-UNNAMED',
            '--enable-native-access=ALL-UNNAMED',
            '-cp',
            os.pathsep.join([variant['harness_dir'], *(str(p) for p in jars)]),
            'ClientCost',
        ]
    else:
        raise ValueError('language must be python or java')
    return verified, command


def run(args):
    config = json.loads(args.config.read_text())
    variants = config['variants']
    if len({v['label'] for v in variants}) != len(variants):
        raise ValueError('variant labels must be unique')
    verified = [verify_variant(v) for v in variants]
    if args.real_config:
        private = json.loads(args.real_config.read_text())
        endpoint = private['url']
        endpoint_identity = private['model_identity']
        server = None
    else:
        server = Fixture(args.gradients)
        endpoint = f'http://127.0.0.1:{server.server_port}'
        endpoint_identity = 'independent constant-regression fixture v1'
    results = []
    with contextlib.ExitStack() as stack:
        if server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
        for repetition in range(args.processes):
            order = list(range(len(variants)))
            if repetition % 2:
                order.reverse()
            for index in order:
                identity, command = verified[index]
                before = dict(server.counts) if server else None
                before_rows = len(server.row_counts) if server else 0
                env = runtime_environment()
                env['SDM_BENCH_URL'] = endpoint
                env['SDM_BENCH_FIXED'] = '1' if server else '0'
                result = launch(
                    [
                        *command,
                        '--workload',
                        args.workload,
                        '--rows',
                        str(args.rows),
                        '--scores',
                        str(args.scores),
                        '--width',
                        str(args.width),
                        '--degree',
                        str(args.degree),
                        '--warmups',
                        str(args.warmups),
                        '--loops',
                        str(args.loops),
                        '--gradients',
                        str(int(args.gradients)),
                    ],
                    env,
                    args.interval,
                    args.timeout,
                    args.out.with_suffix(f'.{index}.{repetition}.stderr.log'),
                )
                if (
                    variants[index]['language'] == 'java'
                    and Path(
                        unquote(urlparse(result.get('runtime_jar', '')).path)
                    ).resolve()
                    != Path(variants[index]['jar']).resolve()
                ):
                    raise ValueError('JVM loaded a different client artifact')
                clean_revision(
                    variants[index]['repo'], variants[index]['revision']
                )
                result.update(
                    label=variants[index]['label'],
                    repetition=repetition,
                    identity=identity,
                )
                result['http_counts'] = (
                    {
                        key: value - before[key]
                        for key, value in server.counts.items()
                    }
                    if server
                    else None
                )
                result['http_request_rows'] = (
                    server.row_counts[before_rows:] if server else None
                )
                results.append(result)
                args.out.write_text(
                    json.dumps(
                        {
                            'endpoint_identity': endpoint_identity,
                            'lane': 'real-model'
                            if args.real_config
                            else 'fixed-http',
                            'harness_sha256': {
                                p.name: digest(p)
                                for p in HERE.glob('*')
                                if p.suffix in {'.py', '.java'}
                            },
                            'results': results,
                        },
                        indent=2,
                    )
                    + '\n'
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    bench = parser
    bench.add_argument('--config', required=True, type=Path)
    bench.add_argument('--out', required=True, type=Path)
    bench.add_argument(
        '--real-config',
        type=Path,
        help='Private JSON: url and pinned model_identity',
    )
    bench.add_argument(
        '--workload',
        choices=['tabular-session', 'relational-memory', 'relational-sqlite'],
        default='tabular-session',
    )
    for name, default in [
        ('rows', 1000),
        ('scores', 10),
        ('width', 4),
        ('degree', 4),
        ('warmups', 3),
        ('loops', 8),
        ('processes', 4),
    ]:
        bench.add_argument('--' + name, type=int, default=default)
    bench.add_argument('--gradients', action='store_true')
    bench.add_argument('--interval', type=float, default=0.01)
    bench.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if (
        min(
            args.rows,
            args.scores,
            args.width,
            args.degree,
            args.loops,
            args.processes,
            args.interval,
            args.timeout,
        )
        <= 0
        or args.warmups < 1
        or args.scores >= args.rows
    ):
        parser.error(
            'positive dimensions, scores < rows, and at least one warmup required'
        )
    run(args)


if __name__ == '__main__':
    main()
