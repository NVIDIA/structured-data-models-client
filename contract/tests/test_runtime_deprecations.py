# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``warnings`` rule's narrowing in a cell other than the reference's
(T1.29, decision E81, register HOST-5), seeded fail-closed.

In ``floor-pandas2.2`` the old route's sampler converts each fetch with
pyarrow 14's ``to_pandas()``, and pandas 2.2.0 raises a BlockManager
``DeprecationWarning`` for each; the core route converts none. With HOST-5
approved by the owner, ``compare.py`` leaves out of python-core against
python-old a ``DeprecationWarning`` python-old alone shows whose recorded
origin is pandas or pyarrow. Each claim and what holds it:

- The narrowing applies only while HOST-5 is approved by the owner in the
  register the comparator reads: proposed, approved by a reviewer, missing,
  unreadable or nested past the reader, it says no.
- Still failing with HOST-5 approved: a warning python-core shows that
  python-old does not, a ``DeprecationWarning`` of another origin
  (``sdm_client``'s own, the runner's, none), another class from pandas,
  a class out of order; and a record of origins missing, empty, short,
  retyped or naming another class than the warning list.
- The rule is wired into ``_host_fields`` only against python-old: against
  the reference the same difference fails whatever the register says.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _compare() -> Any:
    if 'compare' in sys.modules:
        return sys.modules['compare']
    spec = importlib.util.spec_from_file_location(
        'compare', ROOT / 'scripts/rust_core/compare.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['compare'] = module
    spec.loader.exec_module(module)
    return module


compare = _compare()

DEPRECATED = ('DeprecationWarning', 'pandas')
USER = ('UserWarning', '__main__')


def host(*warnings: tuple[str, str | None]) -> dict[str, Any]:
    return {
        'warnings': [name for name, _ in warnings],
        'origins': [
            {'class': name, 'module': module} for name, module in warnings
        ],
    }


#: base.regression in floor-pandas2.2 as measured at T1.29: python-old's
#: five BlockManager warnings around the client's UserWarning, which the
#: engine raises there and the client on the core route.
OLD = host(
    DEPRECATED,
    DEPRECATED,
    ('UserWarning', 'sdm_relational_engine'),
    DEPRECATED,
    DEPRECATED,
    ('DeprecationWarning', 'pyarrow'),
)
CORE = host(USER)


def test_the_measured_difference_is_narrowed_only_when_approved() -> None:
    assert compare._runtime_only(CORE, OLD, True)
    assert not compare._runtime_only(CORE, OLD, False)
    # Equal records narrow nothing because nothing differs; the rule is
    # only consulted on a difference.
    assert compare._runtime_only(OLD, OLD, True)


@pytest.mark.parametrize(
    'old, core',
    [
        # A warning python-core shows that python-old does not.
        (host(USER), host(USER, DEPRECATED)),
        (host(USER), host(DEPRECATED, USER)),
        # A DeprecationWarning of another origin.
        (host(USER, ('DeprecationWarning', 'sdm_client')), host(USER)),
        (host(USER, ('DeprecationWarning', '__main__')), host(USER)),
        (host(USER, ('DeprecationWarning', None)), host(USER)),
        (host(USER, ('DeprecationWarning', 'numpy')), host(USER)),
        # Another class from the runtime's libraries.
        (host(USER, ('FutureWarning', 'pandas')), host(USER)),
        (host(USER, ('UserWarning', 'pyarrow')), host(USER)),
        (host(USER, ('PendingDeprecationWarning', 'pandas')), host(USER)),
        # The client's own warnings out of order or missing.
        (
            host(USER, ('SDMWarning', '__main__')),
            host(('SDMWarning', '__main__'), USER),
        ),
        (host(USER, DEPRECATED), host()),
    ],
)
def test_every_other_difference_still_fails(
    old: dict[str, Any], core: dict[str, Any]
) -> None:
    assert not compare._runtime_only(core, old, True)


def _broken(which: str, how: str) -> tuple[dict[str, Any], dict[str, Any]]:
    core, old = copy.deepcopy(CORE), copy.deepcopy(OLD)
    target = core if which == 'core' else old
    if how == 'missing':
        del target['origins']
    elif how == 'empty':
        target['origins'] = []
    elif how == 'short':
        target['origins'] = target['origins'][:-1]
    elif how == 'retyped':
        target['origins'] = {'class': 'UserWarning', 'module': None}
    elif how == 'item retyped':
        target['origins'][0] = ['UserWarning', None]
    elif how == 'class mismatched':
        target['origins'][0]['class'] = 'SyntaxWarning'
    elif how == 'module retyped':
        target['origins'][0]['module'] = 7
    elif how == 'extra key':
        target['origins'][0]['line'] = 1
    elif how == 'no host':
        return (None, old) if which == 'core' else (core, None)  # type: ignore[return-value]
    return core, old


@pytest.mark.parametrize('which', ['core', 'old'])
@pytest.mark.parametrize(
    'how',
    [
        'missing',
        'empty',
        'short',
        'retyped',
        'item retyped',
        'class mismatched',
        'module retyped',
        'extra key',
        'no host',
    ],
)
def test_a_missing_or_retyped_record_narrows_nothing(
    which: str, how: str
) -> None:
    core, old = _broken(which, how)
    assert not compare._runtime_only(core, old, True)


def _register(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'register.yaml'
    path.write_text(text, encoding='utf-8')
    return path


ENTRY = 'entries:\n- id: HOST-5\n  status: {status}\n  approved_by: {by}\n'


def test_only_the_owner_s_approval_activates_it(tmp_path: Path) -> None:
    approved = _register(tmp_path, ENTRY.format(status='approved', by='owner'))
    assert compare.owner_approved('HOST-5', approved)
    assert not compare.owner_approved('HOST-4', approved)
    for status, by in [
        ('proposed', 'null'),
        ('approved', 'reviewer'),
        ('approved', 'null'),
        ('rejected', 'owner'),
        ('proposed', 'owner'),
    ]:
        path = _register(tmp_path, ENTRY.format(status=status, by=by))
        assert not compare.owner_approved('HOST-5', path), (status, by)
    for text in ['', 'entries: {}\n', '- id: HOST-5\n', ': : :\n', '[' * 5000]:
        path = _register(tmp_path, text)
        assert not compare.owner_approved('HOST-5', path), text[:20]
    assert not compare.owner_approved('HOST-5', tmp_path / 'absent.yaml')
    (tmp_path / 'not-utf8.yaml').write_bytes(b'\xff\xfe\x00')
    assert not compare.owner_approved('HOST-5', tmp_path / 'not-utf8.yaml')


def test_the_committed_register_holds_the_owner_s_approval() -> None:
    # HOST-5 was proposed when T1.29's part 1 landed; the owner approved it
    # on 2026-10-13 (relayed, recorded at T1.32d), so the narrowing is
    # active once that approval is on the phase head, where T1.29's part 2
    # admits the baseline cases.
    assert compare.owner_approved('HOST-5')


def _observation(warnings: dict[str, Any]) -> dict[str, Any]:
    return {
        'host': warnings,
        'public': {'schema': None, 'result': None, 'payloads': []},
    }


@pytest.mark.parametrize('narrowed', [True, False])
def test_the_rule_narrows_python_old_s_comparison_alone(narrowed: bool) -> None:
    case = {'id': 'base.regression'}
    rules = {
        'schema': 'gate',
        'warnings': 'gate',
        'result': 'gate',
        'payload': 'gate',
    }
    for against, fails in [
        ('python-old', not narrowed),
        ('python-ref', True),
    ]:
        report = compare.Report(rules)
        report.narrowed = narrowed
        compare._host_fields(
            report,
            case,
            'python-core',
            _observation(CORE),
            _observation(OLD),
            against,
        )
        assert bool(report.failures) is fails, against
        if not fails:
            assert any('HOST-5' in note for note in report.notes)


def _runner() -> Any:
    if 'run_python' in sys.modules:
        return sys.modules['run_python']
    spec = importlib.util.spec_from_file_location(
        'run_python', ROOT / 'contract/runners/run_python.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['run_python'] = module
    spec.loader.exec_module(module)
    return module


def test_the_runner_records_the_module_whose_files_raised_it() -> None:
    import pandas.core.frame
    import pyarrow
    import sdm_client.schema

    run_python = _runner()
    assert run_python._origin(pandas.core.frame.__file__) == 'pandas'
    assert run_python._origin(pyarrow.__file__) == 'pyarrow'
    assert run_python._origin(sdm_client.schema.__file__) == 'sdm_client'
    # A file no loaded module holds, and a prefix that is not a package's
    # directory, name none.
    assert run_python._origin('/nowhere/frame.py') is None
    assert (
        run_python._origin(
            str(Path(pandas.core.frame.__file__).parents[2]) + 'x'
        )
        is None
    )
