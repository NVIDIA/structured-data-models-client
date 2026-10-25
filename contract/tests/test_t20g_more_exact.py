# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0g: a register entry the owner approved lists, under ``more_exact``,
the fields where the core is more exact than the reference; ``compare.py``
leaves out only those, for that case and runner, and compares every other
field.

The reference's own observation of ``base.regression``, observed as
``python-core`` but for a payload number and the result's digest, stands
for the core's nearest double and exact text id. Fix 1 (review round 1):
the entry counts only with an owner-approved decision row of the
harness's own, and a path is excused only where both sides hold one value.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'contract/cases/s1/base.regression.json'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')
compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')

#: The payload's first context row's target, as compare prints its path.
TARGET = 'public.payloads/0/2/context/instance_table/rows/0/3'
DIGEST = 'public.result/digest'


def _differences(
    tmp_path: Path,
    entry: dict[str, Any],
    *,
    other_field: bool = False,
    target_as: Any = None,
    reference_text: str | None = '1' * 64,
    observed_text: str | None = '1' * 64,
) -> set[str | None]:
    r"""The rules that fail, the observation differing from the reference
    in ``TARGET`` and ``DIGEST`` (and in ``metadata.source_query`` with
    ``other_field``), under a register holding ``entry``."""
    cases = tmp_path / 'cases'
    cases.mkdir(exist_ok=True)
    (cases / CASE.name).write_bytes(CASE.read_bytes())
    reference = next(
        json.loads(line)
        for line in REFERENCE.read_text().splitlines()
        if json.loads(line)['case'] == 'base.regression'
    )
    reference['identity']['inputs']['case'] = run_python.digest(
        CASE.read_bytes()
    )
    # These are synthetic observations of matching result rows. Keep
    # independent text evidence when the typed digest is excused.
    reference['public']['result_text'] = reference_text
    observation = copy.deepcopy(reference)
    observation['public']['result_text'] = observed_text
    observation['identity']['runner'] = 'python-core'
    observation['identity']['reference'] = None
    parsed = observation['public']['payloads'][0]['parsed']
    target = parsed['context']['instance_table']['rows'][0][3]
    target['number'] += '1'
    if target_as is not None:
        parsed['context']['instance_table']['rows'][0][3] = target_as
    observation['public']['result']['digest'] = '0' * 64
    if other_field:
        parsed['metadata']['source_query'] = 'PREDICT x'
    files = []
    for name, item in (('reference', reference), ('core', observation)):
        path = tmp_path / f'{name}.jsonl'
        path.write_text(json.dumps(item) + '\n', encoding='utf-8')
        files.append(path)
    register = tmp_path / 'register.yaml'
    register.write_text(yaml.safe_dump({'entries': [entry]}), encoding='utf-8')
    report = compare.compare(
        [cases], files[0], files[1:], improvements=register
    )
    return {rule for rule, _ in report.failures}


def _approved_entry() -> dict[str, Any]:
    return {
        'id': 'P2E-1',
        'status': 'approved',
        'approved_by': 'owner',
        # A row of this harness's decisions the owner approved.
        'decision_ref': {
            'file': 'docs/design/rust-core/phase-2/decisions.md',
            'row': 'KO2-35',
        },
        'more_exact': [
            {
                'case': 'base.regression',
                'runners': ['python-core'],
                'fields': [TARGET, DIGEST],
            }
        ],
    }


def test_an_approved_more_exact_entry_excuses_only_its_fields(
    tmp_path: Path,
) -> None:
    entry = _approved_entry()
    # Not vacuous: with the entry proposed, both differences fail.
    proposed = {**entry, 'status': 'proposed', 'approved_by': None}
    assert {'payload', 'result'} <= _differences(tmp_path, proposed)
    # Approved by the owner: the named fields are left out.
    excused = _differences(tmp_path, entry)
    assert not {'payload', 'result'} & excused, excused
    # A difference in a field the entry does not name still fails.
    assert 'payload' in _differences(tmp_path, entry, other_field=True)
    # Approved by the owner only in the candidate's own register, with no
    # owner-approved decision row behind it: nothing is excused.
    unbacked = {**entry, 'decision_ref': None}
    assert {'payload', 'result'} <= _differences(tmp_path, unbacked)
    # A path naming a whole payload, not one value, excuses nothing.
    whole = copy.deepcopy(entry)
    whole['more_exact'][0]['fields'] = ['public.payloads/*', DIGEST]
    assert 'payload' in _differences(tmp_path, whole)
    # The path compare prints for a changed number, its exact value (fix 3).
    printed = copy.deepcopy(entry)
    printed['more_exact'][0]['fields'] = [f'{TARGET}/2', DIGEST]
    assert 'payload' not in _differences(tmp_path, printed)
    # A row that is no ID excuses nothing, and does not crash (fix 3).
    listed = {
        **entry,
        'decision_ref': {**entry['decision_ref'], 'row': ['KO2-35']},
    }
    assert {'payload', 'result'} <= _differences(tmp_path, listed)
    # A list shaped like a number, but not one, at the named path: not one
    # value, so it still differs (fix 2).
    fake = ['number', {'unexpected': True}, ['wrong'], None]
    assert 'payload' in _differences(tmp_path, entry, target_as=fake)


@pytest.mark.parametrize(
    ('reference_text', 'observed_text'),
    [(None, None), ('a' * 64, None), (None, 'a' * 64), ('a' * 64, 'b' * 64)],
)
def test_excusing_the_typed_digest_still_requires_matching_text(
    tmp_path: Path, reference_text: str | None, observed_text: str | None
) -> None:
    assert 'result' in _differences(
        tmp_path,
        _approved_entry(),
        reference_text=reference_text,
        observed_text=observed_text,
    )
