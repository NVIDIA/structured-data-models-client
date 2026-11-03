# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only pinned dataset I/O; schema and sampling policy stay in Rust."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from sdm_client.errors import StructuredDataError

_HUB = 'https://huggingface.co/datasets'


def _catalog() -> tuple[dict[str, Any], dict[str, Any]]:
    root = files('sdm_client').joinpath('assets/relbench')
    return (
        json.loads(root.joinpath('registry.json').read_text()),
        json.loads(root.joinpath('overrides.json').read_text()),
    )


def _relative(repository: str, revision: str, dataset: str, table: str) -> Path:
    parts = [
        *repository.split('/'),
        revision,
        dataset,
        'db',
        table + '.parquet',
    ]
    if len(repository.split('/')) != 2 or any(
        not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', part) for part in parts
    ):
        raise StructuredDataError(
            'unsafe RelBench manifest path', code='INVALID_REQUEST'
        )
    return Path(*parts)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(path: Path, url: str, expected: str, offline: bool) -> None:
    if path.is_file() and _digest(path) == expected:
        return
    if offline:
        raise OSError(
            'offline cache entry is missing or has an invalid SHA-256'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix='.' + path.name, delete=False
        ) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            with urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise OSError(
                    'download SHA-256 does not match the pinned manifest'
                )
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load(
    cls: Any, name: str, *, cache_dir: str | Path | None, offline: bool
) -> Any:
    import pyarrow.parquet as parquet

    from sdm_client.schema import Relationship

    registry, overrides = _catalog()
    if not isinstance(name, str) or not isinstance(offline, bool):
        raise StructuredDataError(
            'RelBench name must be a string and offline a bool',
            code='INVALID_REQUEST',
        )
    dataset = name.lower()
    if dataset not in registry['datasets']:
        dataset = 'rel-' + dataset
    if dataset not in registry['datasets']:
        raise StructuredDataError(
            f'unknown RelBench dataset {name!r}; available: {", ".join(registry["datasets"])}',
            code='INVALID_REQUEST',
        )
    try:
        cache = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else Path.home() / '.cache/sdm/relbench'
        )
    except TypeError as error:
        raise StructuredDataError(
            'cache_dir must be a filesystem path', code='INVALID_REQUEST'
        ) from error
    entry = registry['datasets'][dataset]
    repository = entry['repository']
    revision = registry['revisions'][repository]
    frames = {}
    for table, spec in entry['tables'].items():
        relative = _relative(repository, revision, dataset, table)
        path = cache / relative
        try:
            _fetch(
                path,
                f'{_HUB}/{repository}/resolve/{revision}/{dataset}/db/{table}.parquet',
                spec['sha256'],
                offline,
            )
            excluded = (
                overrides.get(dataset, {}).get(table, {}).get('exclude', [])
            )
            columns = [
                column
                for column in parquet.read_schema(path).names
                if column not in excluded
            ]
            frames[table] = parquet.read_table(
                path, columns=columns
            ).to_pandas()
        except (OSError, ValueError) as error:
            raise StructuredDataError(
                f'could not load RelBench {dataset}.{table}: {error}',
                code='PROVIDER_ERROR',
                details={'dataset': dataset, 'table': table},
            ) from error
    links = [
        Relationship(
            table, column, target, entry['tables'][target]['primary_key']
        )
        for table, spec in entry['tables'].items()
        for column, target in spec['foreign_keys'].items()
    ]
    schema = cls.from_data(
        frames,
        primary_keys={
            table: spec['primary_key']
            for table, spec in entry['tables'].items()
            if spec['primary_key'] is not None
        },
        time_columns={
            table: spec['time_column']
            for table, spec in entry['tables'].items()
        },
        relationships=[],
    )
    for table, spec in entry['tables'].items():
        if spec['primary_key'] is None:
            schema[table].primary_key = None
    schema.relationships = links
    for table, declarations in overrides.get(dataset, {}).items():
        for column, stype in declarations.get('stypes', {}).items():
            schema[table][column].stype = stype
    return schema
