# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded public capture chunks using the unchanged reviewed capture worker."""

from __future__ import annotations

import copy
import email
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
CAPTURE = HERE.parent / 'p97-real-adapter-staging'
PRODUCT = '69c24b6af6303f149369a74aaf24291e4ed007c3'
NATIVE = '1deeee0ef3b4ce20622985b32fc1822b66a8b7bff4fc4905a561ae1a91a82900'
MAX_CASES = 32
MAX_INPUT = 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024
DEADLINE = 120
BINDINGS = {
    'p97-real-adapter-staging/capture.py': 'c8c4d446cb035a635df23fde8b75a8e8fa2389ad477b8d164c5d8ea48128483c',
    'p97-real-adapter-staging/common.py': 'd1f5e0e71bfcf4f8b970705d9df32e92169adfff7509efdd40e80156b2a353e8',
    'p97-execution-coupling/coupling.py': 'db92950ffccd7a36c7f09652e63f08d3eb62f2ed4c641d49b5f80c020facae8f',
    'p97-acceptance-staging/accept_contract.py': 'bd02b6f5f00e5f737230d68df14d5af5dac87c027be49761224509e000e7d6f3',
}


def verify_capture_sources() -> None:
    for relative, expected in BINDINGS.items():
        if (
            hashlib.sha256((HERE.parent / relative).read_bytes()).hexdigest()
            != expected
        ):
            raise RuntimeError('batch.capture_binding')


verify_capture_sources()
sys.path.insert(0, str(CAPTURE))
from capture import (  # noqa: E402
    complete_public_capture_v1,
    validate_materialized_rows,
)
from common import (  # noqa: E402
    Rejected,
    outcome,
    require,
    require_path,
    strict_json,
)

__all__ = ['Rejected', 'outcome']
validate_observation = sys.modules['accept_contract'].observation


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(',', ':'), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_cases(cases: list) -> None:
    require(
        isinstance(cases, list) and 1 <= len(cases) <= MAX_CASES,
        'batch.chunk_size',
    )
    identities = []
    for case in cases:
        require(
            isinstance(case, dict) and set(case) == {'id', 'recipe'},
            'batch.case_fields',
        )
        require(
            isinstance(case['id'], str) and 0 < len(case['id']) <= 256,
            'batch.case_id',
        )
        require(isinstance(case['recipe'], dict), 'batch.recipe')
        identities.append(case['id'])
    require(len(set(identities)) == len(identities), 'batch.duplicate_id')
    require(len(json.dumps(cases).encode()) <= MAX_INPUT, 'batch.input_bound')


def validate_results(
    cases: list, results: object, selector: str, major: str
) -> dict:
    validate_cases(cases)
    require(
        isinstance(results, list) and len(results) == len(cases),
        'batch.result_count',
    )
    observed = {}
    for case, result in zip(cases, results, strict=True):
        require(
            isinstance(result, dict)
            and set(result) == {'id', 'input_sha256', 'capture'},
            'batch.result_fields',
        )
        require(result['id'] == case['id'], 'batch.result_id')
        require(
            result['input_sha256'] == fingerprint(case['recipe']),
            'batch.result_input',
        )
        record = result['capture']
        require(isinstance(record, dict), 'batch.capture_type')
        require(
            set(record)
            == {
                'observation',
                'public_calls',
                'legacy_calls',
                'native_calls',
                'selector',
                'pandas_major',
                'materialized_rows',
                'native_sha256',
            },
            'batch.capture_fields',
        )
        require(
            record['selector'] == selector and record['pandas_major'] == major,
            'batch.lane',
        )
        require(record['native_sha256'] == NATIVE, 'batch.native')
        require(
            all(
                type(record[key]) is int and record[key] >= 0
                for key in ('public_calls', 'legacy_calls', 'native_calls')
            ),
            'batch.count_types',
        )
        require_path(record, selector)
        validate_observation(record['observation'])
        validate_materialized_rows(record)
        observed[case['id']] = record
    require(set(observed) == {case['id'] for case in cases}, 'batch.case_map')
    return observed


def bounded(
    command: list[str],
    raw: bytes,
    environment: dict,
    cwd: Path,
    *,
    deadline: float = DEADLINE,
    output_bound: int = MAX_OUTPUT,
) -> tuple[int, bytes, float]:
    started = time.monotonic()
    require(len(raw) <= MAX_INPUT, 'batch.input_bound')
    with tempfile.TemporaryFile(dir=environment.get('TMPDIR', cwd)) as request:
        request.write(raw)
        request.seek(0)
        with subprocess.Popen(
            command,
            stdin=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            cwd=cwd,
            start_new_session=True,
        ) as child:
            output = bytearray()
            count = 0

            def terminated(_signum, _frame):
                raise RuntimeError('batch.terminated')

            previous = None
            try:
                previous = signal.signal(signal.SIGTERM, terminated)
                with selectors.DefaultSelector() as selected:
                    for stream in (child.stdout, child.stderr):
                        selected.register(stream, selectors.EVENT_READ)
                    while selected.get_map():
                        remaining = deadline - (time.monotonic() - started)
                        require(remaining > 0, 'batch.deadline')
                        for key, _event in selected.select(min(remaining, 0.1)):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                selected.unregister(key.fileobj)
                                continue
                            count += len(chunk)
                            require(count <= output_bound, 'batch.output_bound')
                            if key.fileobj is child.stdout:
                                output.extend(chunk)
                    remaining = deadline - (time.monotonic() - started)
                    require(remaining > 0, 'batch.deadline')
                    require(
                        child.wait(timeout=remaining) == 0,
                        'batch.worker_failed',
                    )
                return child.pid, bytes(output), time.monotonic() - started
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                if previous is not None:
                    signal.signal(signal.SIGTERM, previous)


def environment(
    repo: Path, runtime: Path, scratch: Path, selector: str, major: str
) -> dict:
    paths = [
        repo / 'packages' / name / 'src'
        for name in (
            'structured-data-models-api',
            'structured-data-models-pql',
            'structured-data-models-client',
            'structured-data-models-connectors',
            'structured-data-models-relational-engine',
        )
    ]
    if major == '2':
        paths.insert(0, HERE.parent / 'p9-product-fixes/legacy-pandas2')
    for name in ('home', 'tmp', 'cache'):
        (scratch / name).mkdir()
    return {
        'PATH': f'{runtime}/bin:/usr/bin:/bin',
        'HOME': str(scratch / 'home'),
        'TMPDIR': str(scratch / 'tmp'),
        'XDG_CACHE_HOME': str(scratch / 'cache'),
        'LANG': 'en_US.UTF-8',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONNOUSERSITE': '1',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'SDM_CLIENT_IMPLEMENTATION': selector,
        'PYTHONPATH': ':'.join(map(str, paths)),
    }


def lane_metadata(repo: Path, major: str) -> tuple[Path, dict]:
    root = HERE.parent / 'p9-product-fixes'
    provenance = strict_json((root / 'PROVENANCE.json').read_bytes())
    name = 'runtime' if major == '3' else 'runtime-pandas2'
    installed = provenance['dependencies']['installed_metadata'][name]
    python_version = '.'.join(installed['python'].split('.')[:2])
    version = installed['packages']['pandas']
    require(version.split('.')[0] == major, 'batch.parent_major')
    runtime = root / name
    site = runtime / 'lib' / f'python{python_version}' / 'site-packages'
    metadata = email.message_from_bytes(
        (site / f'pandas-{version}.dist-info/METADATA').read_bytes()
    )
    require(metadata['Version'] == version, 'batch.pandas_metadata')
    helper = (site / 'sdm_core/answers.py').read_bytes()
    require(
        helper
        == (
            repo / 'rust/bindings/python/python/sdm_core/answers.py'
        ).read_bytes(),
        'batch.parent_helper',
    )
    return runtime, {
        'python_version': installed['python'],
        'pandas_version': version,
        'installed_helper_sha256': hashlib.sha256(helper).hexdigest(),
        'native_sha256': NATIVE,
    }


def run_chunk(
    repo: Path, cases: list, major: str, *, fresh: bool = False
) -> dict:
    validate_cases(cases)
    require(major in {'2', '3'}, 'batch.major')
    require(not fresh or len(cases) == 1, 'batch.fresh_size')
    require(
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo)
        .decode()
        .strip()
        == PRODUCT,
        'batch.product_commit',
    )
    require(
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=repo).returncode
        == 0,
        'batch.product_dirty',
    )
    runtime, metadata = lane_metadata(repo, major)
    lanes = {}
    for selector in ('python', 'core'):
        verify_capture_sources()
        with tempfile.TemporaryDirectory(
            prefix='chunk-', dir=HERE
        ) as temporary:
            scratch = Path(temporary)
            env = environment(repo, runtime, scratch, selector, major)
            command = [str(runtime / 'bin/python'), '-B']
            if fresh:
                command += [str(CAPTURE / 'capture.py'), str(repo)]
                request = {'recipe': cases[0]['recipe']}
            else:
                command += [
                    str(Path(__file__).resolve()),
                    '--worker',
                    str(repo),
                ]
                request = {'cases': cases}
            pid, raw, elapsed = bounded(
                command, json.dumps(request).encode(), env, repo
            )
            results = strict_json(raw)
            if fresh:
                results = [
                    {
                        'id': cases[0]['id'],
                        'input_sha256': fingerprint(cases[0]['recipe']),
                        'capture': results,
                    }
                ]
            observed = validate_results(cases, results, selector, major)
            verify_capture_sources()
            lanes[selector] = {
                'process_id': pid,
                'elapsed_seconds': elapsed,
                'source_commit': PRODUCT,
                'pandas_major': major,
                'installed_metadata': metadata,
                'selector': selector,
                'input_sha256': {
                    case['id']: fingerprint(case['recipe']) for case in cases
                },
                'cases': observed,
                'output_bytes': len(raw),
            }
        require(not scratch.exists(), 'batch.scratch_cleanup')
    require(
        lanes['python']['process_id'] != lanes['core']['process_id'],
        'batch.process_separation',
    )
    require(
        subprocess.run(['git', 'diff', '--quiet', PRODUCT], cwd=repo).returncode
        == 0,
        'batch.product_changed',
    )
    return lanes


def worker(repo: Path, cases: list) -> list:
    validate_cases(cases)
    selector = os.environ['SDM_CLIENT_IMPLEMENTATION']
    if selector == 'python':
        from sdm_relational_engine.rfm.base import sampler
    else:
        from sdm_client import _request_core as sampler
    import sdm_core.answers as answers

    require(
        Path(answers.__file__).read_bytes()
        == (
            repo / 'rust/bindings/python/python/sdm_core/answers.py'
        ).read_bytes(),
        'batch.installed_helper',
    )
    results = []
    for case in cases:
        sampler._coverage_warned = False
        previous_tempdir = tempfile.tempdir
        previous_tmp = os.environ.get('TMPDIR')
        with tempfile.TemporaryDirectory(prefix='case-') as temporary:
            try:
                tempfile.tempdir = temporary
                os.environ['TMPDIR'] = temporary
                record = complete_public_capture_v1(
                    repo, copy.deepcopy(case['recipe'])
                )
                results.append(
                    {
                        'id': case['id'],
                        'input_sha256': fingerprint(case['recipe']),
                        'capture': record,
                    }
                )
            finally:
                tempfile.tempdir = previous_tempdir
                if previous_tmp is None:
                    os.environ.pop('TMPDIR', None)
                else:
                    os.environ['TMPDIR'] = previous_tmp
        require(
            not Path(temporary).exists() and sys.getprofile() is None,
            'batch.case_cleanup',
        )
    return results


if __name__ == '__main__':
    require(
        len(sys.argv) == 3 and sys.argv[1] == '--worker',
        'batch.worker_arguments',
    )
    supplied = sys.stdin.buffer.read(MAX_INPUT + 1)
    require(len(supplied) <= MAX_INPUT, 'batch.input_bound')
    document = strict_json(supplied)
    require(set(document) == {'cases'}, 'batch.request_fields')
    print(
        json.dumps(
            worker(Path(sys.argv[2]), document['cases']), allow_nan=False
        )
    )
