# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance-owned LLVM branch-arm measurement of immutable replay partitions."""

from __future__ import annotations

import posixpath
import tempfile
from pathlib import Path

from execution import batch
from native_binding import verify_no_python
from support import (
    checked_path,
    json_bytes,
    require,
    same_json,
    sha,
    unique_json,
)

FLAGS = ['-C', 'instrument-coverage', '-Z', 'coverage-options=branch']
CLASSES = ('contract', 'scaffold')


def command(
    arguments, *, cwd, environment, payload=None, deadline=1200, receipts=None
):
    arguments = [str(argument) for argument in arguments]
    require(
        payload is None or len(payload) <= 512 * 1024 * 1024,
        'scaffold.snapshot_bound',
    )
    with tempfile.NamedTemporaryFile(dir=environment['TMPDIR']) as request:
        request.write(payload or b'')
        request.flush()
        process_id, output, elapsed = batch.bounded(
            [
                '/bin/sh',
                '-c',
                'exec "$@" < "$P97_SCAFFOLD_INPUT"',
                'p97-scaffold',
                *arguments,
            ],
            b'',
            {**environment, 'P97_SCAFFOLD_INPUT': request.name},
            cwd,
            deadline=deadline,
            output_bound=128 * 1024 * 1024,
        )
    if receipts is not None:
        receipts.append(
            {
                'process_id': process_id,
                'elapsed_seconds': elapsed,
                'command': [
                    argument.replace(str(cwd), '<source-root>').replace(
                        str(Path.home()), '~'
                    )
                    for argument in arguments
                ],
                'stdin_sha256': sha(payload or b''),
                'stdout_sha256': sha(output),
            }
        )
    return output


def environment(scratch: Path) -> dict:
    for name in ('home', 'tmp', 'cache'):
        (scratch / name).mkdir()
    return {
        'HOME': str(scratch / 'home'),
        'TMPDIR': str(scratch / 'tmp'),
        'XDG_CACHE_HOME': str(scratch / 'cache'),
        'PATH': '/usr/bin:/bin',
        'LANG': 'en_US.UTF-8',
    }


def partition(manifest: list, contents: dict, routes: dict) -> dict:
    require(len(manifest) == len(contents), 'scaffold.manifest_files')
    owners, names = {}, set()
    selected = {kind: [] for kind in CLASSES}
    for record in manifest:
        name, kind = record['path'], record['kind']
        require(
            name not in names and name in contents, 'scaffold.manifest_files'
        )
        names.add(name)
        require(kind in CLASSES, 'scaffold.classification')
        require(
            sha(contents[name]) == record['sha256'], 'scaffold.fixture_digest'
        )
        for identity in record['semantic_cases']:
            require(identity not in owners, 'scaffold.ambiguous_case_owner')
            owners[identity] = kind
            selected[kind].append(identity)
    require(set(owners) == set(routes), 'scaffold.case_partition')
    return {kind: sorted(cases) for kind, cases in selected.items()}


def build(root: Path, scratch: Path, toolchain: dict, base_environment: dict):
    compiler_flags = [
        *FLAGS,
        '--remap-path-prefix',
        f'{Path.home()}=/p97-home',
        '--remap-path-prefix',
        f'{root}=/p97-source',
    ]
    build_environment = {
        **base_environment,
        'RUSTC_BOOTSTRAP': '1',
        'RUSTC': str(toolchain['rustc']),
        'CARGO_HOME': str(toolchain['cargo_home']),
        'CARGO_ENCODED_RUSTFLAGS': '\x1f'.join(compiler_flags),
        'CARGO_BUILD_JOBS': '2',
    }
    arguments = [
        toolchain['cargo'],
        'build',
        '--offline',
        '--locked',
        '--manifest-path',
        root / 'rust/Cargo.toml',
        '--bin',
        'p97-replay',
        '--target-dir',
        scratch / 'target',
    ]
    executions = []
    command(
        arguments, cwd=root, environment=build_environment, receipts=executions
    )
    binary = scratch / 'target/debug/p97-replay'
    require(binary.is_file(), 'scaffold.instrumented_target_absent')
    return binary, {
        'artifact_sha256': sha(binary.read_bytes()),
        'execution': executions[0],
        'coverage_flags': FLAGS,
        'remapped_source_root': '/p97-source',
        'remapped_user_home': '/p97-home',
        'unstable_flags_enabled_in_build_child_only': True,
        'cargo_arguments': [
            'build',
            '--offline',
            '--locked',
            '--manifest-path',
            'rust/Cargo.toml',
            '--bin',
            'p97-replay',
            '--target-dir',
            '<owned-measurement>/target',
        ],
        'tools': {
            name: sha(Path(toolchain[name]).read_bytes())
            for name in ('cargo', 'rustc', 'llvm_cov', 'llvm_profdata')
        },
    }


def branches(export: dict, source_bindings: dict) -> set[str]:
    require(
        export.get('type') == 'llvm.coverage.json.export'
        and export.get('version', '').startswith('3.')
        and type(export.get('data')) is list,
        'scaffold.coverage_format',
    )
    reached = set()
    for unit in export['data']:
        for function in unit['functions']:
            for record in function['branches']:
                require(
                    len(record) == 9
                    and all(
                        type(value) is int and value >= 0 for value in record
                    )
                    and record[8] == 4,
                    'scaffold.branch_record',
                )
                require(
                    record[6] < len(function['filenames']),
                    'scaffold.branch_file',
                )
                filename = posixpath.normpath(function['filenames'][record[6]])
                prefix = '/p97-source/'
                if not filename.startswith(prefix):
                    continue
                relative = filename[len(prefix) :]
                if not (
                    relative.startswith('rust/crates/')
                    and '/src/' in relative
                    and relative.endswith('.rs')
                ):
                    continue
                require(
                    relative in source_bindings,
                    'scaffold.unreviewed_branch_source',
                )
                location = ':'.join(str(value) for value in record[:4])
                for arm, count in zip(
                    ('true', 'false'), record[4:6], strict=False
                ):
                    if count:
                        reached.add(
                            f'{relative}@{source_bindings[relative]}:{location}:{arm}'
                        )
    return reached


def run_class(
    root: Path,
    scratch: Path,
    kind: str,
    binary: Path,
    contents: dict,
    routes: dict,
    cases: list,
    native_sha256: str,
    source_bindings: dict,
    toolchain: dict,
    base_environment: dict,
    expected: dict,
) -> tuple[set, dict]:
    class_root = scratch / kind
    class_root.mkdir()
    request = json_bytes(
        {
            'files': {name: value.hex() for name, value in contents.items()},
            'cases': cases,
            'routes': {identity: routes[identity] for identity in cases},
            'native_sha256': native_sha256,
        }
    )
    native = {
        identity
        for identity in cases
        if routes[identity]['kind'] == 'native_semantic'
    }
    if not native:
        return set(), {
            'kind': kind,
            'case_ids': cases,
            'execution': 'no_native_semantic_cases',
            'complete_files': {
                name: sha(raw) for name, raw in contents.items()
            },
            'artifacts': {},
        }
    executions = []
    output = command(
        [binary, '--format', 'p97-snapshot-v1'],
        cwd=root,
        environment={
            **base_environment,
            'LLVM_PROFILE_FILE': str(class_root / 'run-%p-%m.profraw'),
        },
        payload=request,
        receipts=executions,
    )
    result = unique_json(output)
    host = set(cases) - native
    require(
        set(result['observations']) == native, 'scaffold.executed_native_cases'
    )
    require(
        same_json(
            result['observations'],
            {identity: expected[identity] for identity in native},
        ),
        'scaffold.wrong_observation',
    )
    require(
        set(result['host_dispositions']) == host, 'scaffold.executed_host_cases'
    )
    require(
        len(result['files_verified']) == len(contents)
        and set(result['files_verified']) == set(contents),
        'scaffold.verified_files',
    )
    profiles = sorted(class_root.glob('*.profraw'))
    require(bool(profiles), 'scaffold.no_fresh_profile')
    profile = class_root / 'merged.profdata'
    command(
        [
            toolchain['llvm_profdata'],
            'merge',
            '-sparse',
            *profiles,
            '-o',
            profile,
        ],
        cwd=root,
        environment=base_environment,
        receipts=executions,
    )
    exported = command(
        [toolchain['llvm_cov'], 'export', binary, f'-instr-profile={profile}'],
        cwd=root,
        environment=base_environment,
        receipts=executions,
    )
    require(
        str(Path.home()).encode() not in exported,
        'scaffold.unmapped_source_path',
    )
    document = unique_json(exported)
    reached = branches(document, source_bindings)
    export_path = class_root / 'branches.coverage.json'
    export_path.write_bytes(exported)
    artifacts = {
        str(path.relative_to(scratch)): sha(path.read_bytes())
        for path in [*profiles, profile, export_path]
    }
    return reached, {
        'kind': kind,
        'case_ids': cases,
        'snapshot_sha256': sha(request),
        'complete_files': {name: sha(raw) for name, raw in contents.items()},
        'output_sha256': sha(output),
        'commands': executions,
        'result': result,
        'artifacts': artifacts,
    }


def measure(
    root: Path,
    manifest: list,
    contents: dict,
    candidate_binding: dict,
    scratch: Path,
    *,
    routes: dict,
    binary: Path,
    no_python_binding: dict,
    review: dict,
    toolchain: dict,
    expected: dict,
) -> dict:
    root, scratch = root.resolve(), scratch.resolve()
    require(not scratch.exists(), 'scaffold.scratch_not_fresh')
    verified = verify_no_python(root, binary, no_python_binding, review)
    sources = verified['source_bindings']
    for relative in ('rust/Cargo.toml', 'rust/Cargo.lock'):
        require(relative in sources, 'scaffold.build_source_closure')
    selected = partition(manifest, contents, routes)
    require(set(expected) == set(routes), 'scaffold.expected_case_map')
    before = {name: sha(raw) for name, raw in contents.items()}
    scratch.mkdir(parents=True)
    base_environment = environment(scratch)
    head = command(
        ['git', 'rev-parse', 'HEAD'], cwd=root, environment=base_environment
    )
    require(
        head.decode().strip() == candidate_binding['candidate'],
        'scaffold.candidate',
    )
    instrumented, build_receipt = build(
        root, scratch, toolchain, base_environment
    )
    sets, executions = {}, {}
    for kind in CLASSES:
        sets[kind], executions[kind] = run_class(
            root,
            scratch,
            kind,
            instrumented,
            contents,
            routes,
            selected[kind],
            candidate_binding['native_build']['native_sha256'],
            sources,
            toolchain,
            base_environment,
            expected,
        )
    require(
        sha(instrumented.read_bytes()) == build_receipt['artifact_sha256'],
        'scaffold.instrumented_artifact_mutated',
    )
    verify_no_python(root, binary, no_python_binding, review)
    for name, digest in sources.items():
        require(
            sha(checked_path(root, name).read_bytes()) == digest,
            'scaffold.source_mutated',
        )
    require(
        {name: sha(raw) for name, raw in contents.items()} == before,
        'scaffold.snapshot_mutated',
    )
    final_head = command(
        ['git', 'rev-parse', 'HEAD'], cwd=root, environment=base_environment
    )
    require(final_head == head, 'scaffold.candidate_mutated')
    artifacts = {}
    for execution in executions.values():
        for name, digest in execution['artifacts'].items():
            require(
                sha(checked_path(scratch, name).read_bytes()) == digest,
                'scaffold.artifact_mutated',
            )
            artifacts[name] = digest
    return {
        'sha': candidate_binding['candidate'],
        'measurement': 'instrumented-core-branches',
        'measurement_artifacts': artifacts,
        'contract_branches': sorted(sets['contract']),
        'scaffold_branches': sorted(sets['scaffold']),
        'scaffold_only_branches': sorted(sets['scaffold'] - sets['contract']),
        'source_bindings': sources,
        'ordinary_artifact_sha256': verified['artifact_sha256'],
        'instrumented_build': build_receipt,
        'manifest_sha256': sha(json_bytes(manifest)),
        'executions': executions,
    }
