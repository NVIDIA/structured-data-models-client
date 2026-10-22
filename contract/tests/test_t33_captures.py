# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.3's acceptance tests for the capture writer and its converter (Phase 3
design, choice 2, "What C-1 records"; T3.3's brief, exits 1 and 4), written
by the package's test author before the implementation.

The interface tested, ``contract/runners/captures.py``, named here because
nothing names it before the implementation:

- ``FORMAT``: the format version written, ``sdm-capture/1`` until it moves;
- ``PLACEHOLDERS``: each environment variable whose value is a name the
  writer puts a placeholder in place of (``{catalog}``, ``{schema}``,
  ``{host}``, ``{user}``), by variable;
- ``PRIVATE``: every environment variable whose value must never be
  written, the placeholders' included;
- ``write(path, capture)``: the capture (``t33_fixtures.py`` gives its
  shape) written at ``path``, a file or a directory, with the placeholders
  put in before anything is written; a private value still in it fails the
  write with ``PrivateName``, naming the variable and never its value, and
  nothing is left at ``path``;
- ``read(path)``: the capture at ``path``, of any version the converter
  reads, in the current version's form, ``format`` included.

The converter's version-1 fixture, ``contract/tests/t33-capture-v1``, is
recorded once by ``contract/tests/t33_capture_v1_record.py`` from the
writer, and never again: a later version reads it through the converter.

Each test listed in ``contract/tests/t33-pending.json``'s ``pending`` (as
``py:<function name>``) failed when written and is held to failing
(``t33_gate.py``) until the implementer takes it out.
"""

from __future__ import annotations

import importlib.util
import secrets
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = HERE / 't33-capture-v1'
#: The Snowflake variables of the owner's env file (KO3-24), which no
#: capture may hold.
KO3_24 = ('SNOWFLAKE_URL', 'SNOWFLAKE_USERNAME', 'SNOWFLAKE_ACCESS_TOKEN')


def _load(name: str, path: Path) -> ModuleType:
    key = f't33_captures_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[key]
        raise
    return module


fixtures = _load('fixtures', HERE / 't33_fixtures.py')


def _captures() -> ModuleType:
    return _load('captures', ROOT / 'contract/runners/captures.py')


def _gate() -> ModuleType:
    key = 't33_gate'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            key, Path(__file__).with_name('t33_gate.py')
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[key] = module
    return sys.modules[key]


#: Runs a test through the gate: it must pass, or fail while pending.
pending = _gate().gated


def _bytes(path: Path) -> bytes:
    r"""Everything written at ``path``, a file or every file under it."""
    if path.is_file():
        return path.read_bytes()
    return b''.join(
        found.read_bytes()
        for found in sorted(path.rglob('*'))
        if found.is_file()
    )


def _token() -> str:
    return 'zq' + secrets.token_hex(8)


def _clear(monkeypatch: pytest.MonkeyPatch, captures: ModuleType) -> None:
    for name in captures.PRIVATE:
        monkeypatch.delenv(name, raising=False)


# --- exit 1: a private name in a capture fails the write ---------------------


@pending
def test_t33_the_private_variables_cover_every_warehouse() -> None:
    r"""Brief, "Threat model": the owner's Snowflake env file (KO3-24) and the
    Databricks names are private; catalogs, schemas, hosts and users each
    have a placeholder, and every placeholder's variable is private."""
    captures = _captures()
    assert set(KO3_24) <= set(captures.PRIVATE)
    assert any(name.startswith('DATABRICKS_') for name in captures.PRIVATE)
    assert {'{catalog}', '{schema}', '{host}', '{user}'} <= set(
        captures.PLACEHOLDERS.values()
    )
    assert set(captures.PLACEHOLDERS) <= set(captures.PRIVATE)


@pending
def test_t33_a_private_name_fails_the_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    r"""Exit 1: a private name from the environment in a capture fails the
    write, naming the variable and never its value, and leaves nothing
    written. Each private variable is seeded in turn, its value inside a
    result row in upper case, as Snowflake folds a name it returns; a
    placeholder's variable may instead be replaced there, so long as no
    spelling of the value is written."""
    captures = _captures()
    refused = 0
    for name in sorted(captures.PRIVATE):
        _clear(monkeypatch, captures)
        value = _token()
        monkeypatch.setenv(name, value)
        import pyarrow as pa

        found = fixtures.capture()
        found['statements'] = [
            *found['statements'],
            {
                'text': 'SELECT current_user() AS who',
                'parameters': [],
                'result': pa.table({'who': [f'owner={value.upper()};']}),
            },
        ]
        target = tmp_path / name / 'capture'
        target.parent.mkdir()
        try:
            captures.write(target, found)
        except captures.PrivateName as error:
            assert name in str(error), error
            assert value.lower() not in str(error).lower(), name
            assert list(target.parent.iterdir()) == [], name
            refused += 1
            continue
        # Written: only a placeholder's variable, and no spelling of it.
        assert name in captures.PLACEHOLDERS, f'{name} was written'
        assert value.lower().encode() not in _bytes(target).lower(), name
    # Not vacuous: a credential has no placeholder, so it is refused.
    assert refused >= len(set(KO3_24) - set(captures.PLACEHOLDERS)) > 0


@pending
def test_t33_placeholders_are_put_in_before_anything_is_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    r"""Choice 2: names of catalogs, schemas, hosts and users are replaced by
    placeholders before anything is written: a statement naming the
    catalog and schema reads back with ``{catalog}`` and ``{schema}``, and
    the file holds no seeded value; the capture names its format, its
    recording and reference commits, its versions and its input's hash."""
    captures = _captures()
    _clear(monkeypatch, captures)
    seeded = {}
    for name, placeholder in sorted(captures.PLACEHOLDERS.items()):
        if placeholder not in seeded.values():
            seeded[name] = placeholder
            monkeypatch.setenv(name, _token())
    names = {placeholder: name for name, placeholder in seeded.items()}
    import os

    catalog = os.environ[names['{catalog}']]
    schema = os.environ[names['{schema}']]
    found = fixtures.capture()
    found['statements'] = [
        {
            **statement,
            'text': statement['text']
            .replace('{catalog}', catalog)
            .replace('{schema}', schema),
        }
        for statement in found['statements']
    ]
    target = tmp_path / 'capture'
    captures.write(target, found)
    written = _bytes(target)
    for name in seeded:
        assert os.environ[name].encode() not in written, name
    back = captures.read(target)
    assert [s['text'] for s in back['statements']] == [fixtures.REPLAYED_TEXT]
    assert back['format'] == captures.FORMAT == 'sdm-capture/1'
    for key in (
        'case',
        'leg',
        'recording_commit',
        'reference_commit',
        'versions',
        'input_hash',
        'session',
    ):
        assert back[key] == found[key], key


# --- exit 4: the converter reads a version-1 capture -------------------------


def _same(back: dict[str, Any], expected: dict[str, Any]) -> None:
    r"""Every part ``expected`` holds, held by ``back`` unchanged: tables as
    Arrow, the rest as data."""
    for key, value in expected.items():
        if key == 'tables':
            assert sorted(back['tables']) == sorted(value)
            for name, data in value.items():
                assert back['tables'][name].equals(data), name
        elif key in ('draws', 'statements'):
            assert len(back[key]) == len(value), key
            for got, want in zip(back[key], value, strict=True):
                for part, held in want.items():
                    if hasattr(held, 'equals'):
                        assert got[part].equals(held), (key, part)
                    else:
                        assert got[part] == held, (key, part)
        else:
            assert back[key] == value, key


@pending
def test_t33_the_converter_reads_a_version_1_capture() -> None:
    r"""Exit 4: the converter reads the committed version-1 fixture, whatever
    the current version is, into the content it was recorded from."""
    assert FIXTURE.exists(), (
        'record it once: python contract/tests/t33_capture_v1_record.py'
    )
    assert b'sdm-capture/1' in _bytes(FIXTURE)
    recorder = _load('recorder', HERE / 't33_capture_v1_record.py')
    captures = _captures()
    back = captures.read(FIXTURE)
    assert back['format'] == captures.FORMAT
    _same(back, recorder.content())


@pending
def test_t33_a_capture_of_an_unknown_version_is_refused_by_name(
    tmp_path: Path,
) -> None:
    r"""Brief, "Input axes" (T3.2's too): a capture of a version the
    converter does not know is refused, naming the version."""
    assert FIXTURE.exists()
    target = tmp_path / FIXTURE.name
    sources = (
        [found for found in FIXTURE.rglob('*') if found.is_file()]
        if FIXTURE.is_dir()
        else [FIXTURE]
    )
    changed = 0
    for source in sources:
        found = (
            target / source.relative_to(FIXTURE) if FIXTURE.is_dir() else target
        )
        found.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if b'sdm-capture/1' in data:
            data = data.replace(b'sdm-capture/1', b'sdm-capture/999')
            changed += 1
        found.write_bytes(data)
    assert changed
    with pytest.raises(Exception, match='sdm-capture/999'):
        _captures().read(target)
