# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T3.12: one copy of the host's rules both hosts share, in the core.

- The local zone a run's facts name is read by the core's rule
  (``sdm_host::local_zone``), which the Java host reads too: the Python
  host reads ``TZ`` and its lookup roots and asks the native module, and
  holds no copy of the rule (T3.11 review, P2 2).
- The model capabilities the core exports (``_native.capabilities()``),
  which the Java host's ``SdmClient.capabilities`` reads, are what the
  Python client describes each model as (``sdm_client.base.describe``).
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from sdm_client import _native
from sdm_client._rust import facts
from sdm_client.base import describe
from sdm_client.models import spec_for

FACTS = Path(facts.__file__)


def test_the_local_zone_is_the_cores_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'Europe').mkdir()
    (tmp_path / 'Europe' / 'Berlin').write_bytes(b'TZif')
    roots = [{'path': str(tmp_path), 'version': None}]
    asked: list[Any] = []

    class Native:
        @staticmethod
        def local_zone(setting: Any, paths: Any) -> Any:
            asked.append((setting, paths))
            return _native.local_zone(setting, paths)

    for setting, expected in [
        ('Europe/Berlin', {'iana': 'Europe/Berlin'}),
        (':Europe/Berlin', {'iana': 'Europe/Berlin'}),
        ('utc', {'iana': 'utc'}),
        ('EST5EDT,M3.2.0,M11.1.0', {'posix': 'EST5EDT,M3.2.0,M11.1.0'}),
        ('Mars/Olympus', None),
        ('../Europe/Berlin', None),
    ]:
        monkeypatch.setenv('TZ', setting)
        assert facts.local_zone(roots, Native) == expected, setting
        assert asked[-1] == (setting, [str(tmp_path)])
    monkeypatch.delenv('TZ')
    found = facts.local_zone(roots, Native)
    assert asked[-1] == (None, [str(tmp_path)])
    assert found is None or set(found) == {'iana'}


def test_the_host_holds_no_copy_of_the_zone_rule() -> None:
    # The rule's own words are the core's alone: the host reads TZ and its
    # roots, and never /etc/localtime, a '..' part, a leading colon or a
    # POSIX check (docstrings aside).
    tree = ast.parse(FACTS.read_text(encoding='utf-8'))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    texts = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }
    for word in ('/etc/localtime', '..', ':'):
        assert word not in texts, word
    names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert 'is_posix_zone' not in names
    assert 'resolve' not in names


def test_the_cores_capabilities_are_the_clients() -> None:
    read = json.loads(_native.capabilities())
    told = {model['model']: model for model in read['models']}
    from sdm_client.models import MODELS

    known = set(MODELS)
    assert known, 'the registry names no model'
    assert set(told) == known
    for name, model in told.items():
        described = dataclasses.asdict(describe(spec_for(name)))
        assert set(model) == set(described), name
        for field, value in described.items():
            given = model[field]
            assert (
                tuple(given) if isinstance(value, tuple) else given
            ) == value, (name, field)


def test_the_core_route_describes_each_model_from_the_core_s_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Claude T3.12 r1 P2-1: the capability table is the core's alone; the
    # client's description on its core route is read from it, so a change
    # to the core's table is what the client describes.
    from sdm_client import SDMClient

    read = json.loads(_native.capabilities())
    changed = {
        'models': [
            {**model, 'tasks': ['only-in-the-core']} for model in read['models']
        ]
    }
    monkeypatch.setattr(_native, 'capabilities', lambda: json.dumps(changed))
    monkeypatch.setenv(
        'SDM_CORE_ROUTES', 'client.capabilities/capabilities/any'
    )
    client = SDMClient(url='http://localhost:8000')
    for model in read['models']:
        described = client.capabilities(model['model'])
        assert described.tasks == ('only-in-the-core',), model['model']
        assert described.outputs == tuple(model['outputs'])
        assert described.source == 'sdk'
    # A name the core's table holds no model of is refused as before.
    with pytest.raises(Exception) as refused:
        client.capabilities('no-such-model')
    assert type(refused.value).__name__ != 'KeyError'
