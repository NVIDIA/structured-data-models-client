# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance-owned reviewed native artifact binding, not a descendant census."""

from __future__ import annotations

from pathlib import Path

from support import checked_path, require, sha

NATIVE_MAGIC = (
    b'\x7fELF',
    b'\xcf\xfa\xed\xfe',
    b'\xfe\xed\xfa\xcf',
    b'\xca\xfe\xba\xbe',
    b'\xca\xfe\xba\xbf',
)
FORBIDDEN_DEPENDENCIES = {
    'pyo3',
    'pyo3-ffi',
    'cpython',
    'rust-cpython',
    'python3-sys',
}


def verify_no_python(
    root: Path, binary: Path, binding: dict, review: dict
) -> dict:
    require(
        binding == review['approved_no_python_binding'],
        'native.unreviewed_binding',
    )
    require(
        binding['basis'] == 'reviewed-source-dependency-link-closure'
        and review['source_derived_not_descendant_census'] is True,
        'native.attribution',
    )
    raw = binary.read_bytes()
    require(
        raw.startswith(NATIVE_MAGIC) and sha(raw) == binding['artifact_sha256'],
        'native.artifact',
    )
    require(binary.name == 'p97-replay', 'native.command')
    require(
        binding['command_arguments'] == ['--format', 'p97-snapshot-v1'],
        'native.arguments',
    )
    require(
        type(binding['sources']) is dict
        and bool(binding['sources'])
        and 'rust/Cargo.lock' in binding['sources'],
        'native.source_closure',
    )
    for name, expected in binding['sources'].items():
        require(
            sha(checked_path(root, name).read_bytes()) == expected,
            'native.stale_source',
        )
    dependencies = binding['dependency_packages']
    require(
        type(dependencies) is list
        and bool(dependencies)
        and not FORBIDDEN_DEPENDENCIES.intersection(dependencies),
        'native.python_dependency',
    )
    require(
        binding['linked_python_libraries'] == []
        and binding['python_embedding_symbols'] == []
        and binding['replay_process_spawn_sites'] == [],
        'native.python_or_spawn_surface',
    )
    require(
        review['reviewed_artifact_sha256'] == binding['artifact_sha256']
        and review['reviewed_source_bindings'] == binding['sources'],
        'native.review_identity',
    )
    return {
        'basis': 'source-derived, not runtime descendant census',
        'artifact_sha256': binding['artifact_sha256'],
        'source_bindings': binding['sources'],
        'dependency_packages': dependencies,
        'command_arguments': binding['command_arguments'],
    }
