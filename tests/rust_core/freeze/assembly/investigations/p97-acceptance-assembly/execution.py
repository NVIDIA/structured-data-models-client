# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent-owned executions, provenance and exact ordered recipe/result joins."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from candidate import context, lane
from inline_old import validate as validate_inline_old
from recipes import open_recipe
from support import (
    HERE,
    INVESTIGATIONS,
    contract,
    json_bytes,
    load,
    require,
    same_json,
    unique_json,
)

batch = load(
    '_p97_batch_reuse', INVESTIGATIONS / 'p97-batch-capture-staging/batch.py'
)


def source(root: Path, expected: str):
    require(
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root)
        .decode()
        .strip()
        == expected,
        'execution.source',
    )
    require(
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root).returncode
        == 0,
        'execution.dirty',
    )


def capture_chunk(
    root: Path,
    cases: list,
    major: str,
    binding: dict,
    templates: dict | None = None,
    reference_scopes: dict | None = None,
    *,
    provider_path: Path | None = None,
    provider_sha256: str | None = None,
) -> dict:
    require(
        (provider_path is None) == (provider_sha256 is None),
        'provider.arguments',
    )
    require(
        major in {'2', '3'} and 1 <= len(cases) <= batch.MAX_CASES,
        'execution.chunk',
    )
    identities = []
    for case in cases:
        require(
            set(case) == {'id', 'recipe'} and type(case['id']) is str,
            'execution.case_fields',
        )
        open_recipe(case['recipe'])
        identities.append(case['id'])
    require(len(set(identities)) == len(identities), 'execution.duplicate_case')
    candidate = binding['candidate']
    source(root, candidate)
    context(root, binding)
    runtime, metadata = lane(root, major, binding)
    lanes = {}
    for selector in ('python', 'core'):
        with tempfile.TemporaryDirectory(
            prefix='capture-', dir=HERE
        ) as temporary:
            scratch = Path(temporary)
            environment = batch.environment(
                root, runtime, scratch, selector, major
            )
            if major == '2':
                copied = (
                    Path(binding['lanes'][major]['legacy_copy_root'])
                    .expanduser()
                    .resolve()
                )
                environment['PYTHONPATH'] = ':'.join(
                    [
                        str(copied.parent),
                        *environment['PYTHONPATH'].split(':')[1:],
                    ]
                )
                environment['P97_LEGACY_COPY_ROOT'] = str(copied)
            pid, output, elapsed = batch.bounded(
                [
                    str(runtime / 'bin/python'),
                    '-B',
                    str(HERE / 'capture_worker.py'),
                    str(root),
                    *(
                        []
                        if provider_path is None
                        else [str(provider_path.resolve()), provider_sha256]
                    ),
                ],
                json_bytes(
                    {
                        'cases': cases,
                        'templates': templates or {},
                        'reference_scopes': reference_scopes or {},
                    }
                ),
                environment,
                root,
            )
            records = unique_json(output)
            require(
                type(records) is list and len(records) == len(cases),
                'execution.result_count',
            )
            for case, record in zip(cases, records, strict=True):
                require(
                    set(record) == {'id', 'recipe', 'capture'}
                    and record['id'] == case['id']
                    and same_json(record['recipe'], case['recipe']),
                    'execution.case_join',
                )
                captured = record['capture']
                require(
                    captured['selector'] == selector
                    and captured['pandas_major'] == major
                    and captured['native_sha256'] == metadata['native_sha256'],
                    'execution.lane',
                )
                contract.observation(captured['observation'])
                validate_inline_old(root, case['recipe'], captured)
                if (
                    open_recipe(case['recipe'])['operation']
                    == 'fixture.reference'
                ):
                    from references import validate

                    validate(
                        root, case['recipe'], captured, reference_scopes or {}
                    )
                batch.validate_materialized_rows(captured)
                require(
                    all(
                        type(captured[key]) is int and captured[key] >= 0
                        for key in (
                            'public_calls',
                            'native_calls',
                            'legacy_calls',
                            'copied_legacy_calls',
                        )
                    ),
                    'execution.count_types',
                )
            lanes[selector] = {
                'process_id': pid,
                'elapsed_seconds': elapsed,
                'source_commit': candidate,
                'metadata': metadata,
                'cases': {record['id']: record for record in records},
            }
        require(not scratch.exists(), 'execution.cleanup')
    require(
        lanes['python']['process_id'] != lanes['core']['process_id'],
        'execution.same_pid',
    )
    source(root, candidate)
    context(root, binding)
    lane(root, major, binding)
    return lanes
