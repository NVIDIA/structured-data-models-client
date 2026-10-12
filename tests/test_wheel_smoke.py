# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/rust_core/wheel_smoke.py``'s own rules (T1.17): its verdict on
a test report and on what resolved fails closed, its lanes are decision
E16's, and CI runs both of them.

The smoke itself runs in CI's ``wheel-smoke`` job, against the wheels."""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys
import zipfile
from typing import Any

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'scripts/rust_core/wheel_smoke.py'
CI = ROOT / '.github/workflows/ci.yml'

_spec = importlib.util.spec_from_file_location('wheel_smoke', SCRIPT)
assert _spec and _spec.loader
smoke = importlib.util.module_from_spec(_spec)
sys.modules['wheel_smoke'] = smoke
_spec.loader.exec_module(smoke)

FILES = ('test_a.py', 'test_b.py')


def _report(tmp_path: pathlib.Path, cases: str) -> pathlib.Path:
    path = tmp_path / 'junit.xml'
    path.write_text(f'<testsuites><testsuite>{cases}</testsuite></testsuites>')
    return path


def _case(module: str, name: str, inside: str = '') -> str:
    classname = f'rust.bindings.sdm-py.tests.{module}'
    return (
        f'<testcase classname="{classname}" name="{name}">{inside}</testcase>'
    )


PASSING = _case('test_a', 'one') + _case('test_b', 'two')


def test_a_report_with_every_file_passing_is_counted(
    tmp_path: pathlib.Path,
) -> None:
    counts = smoke.junit_verdict(_report(tmp_path, PASSING), FILES)
    assert counts['test_a.py']['passed'] == 1
    assert counts['test_b.py']['passed'] == 1


@pytest.mark.parametrize(
    ('cases', 'fault'),
    [
        ('', 'no test at all'),
        (_case('test_a', 'one'), 'a file that ran nothing'),
        (PASSING + _case('test_b', 'x', '<failure/>'), 'a failure'),
        (PASSING + _case('test_b', 'x', '<error/>'), 'an error'),
        (PASSING + _case('test_b', 'x', '<skipped/>'), 'a skip'),
        (PASSING + _case('test_c', 'x'), 'a file not asked for'),
    ],
)
def test_a_report_is_refused_on(
    tmp_path: pathlib.Path, cases: str, fault: str
) -> None:
    with pytest.raises(smoke.SmokeFailed):
        smoke.junit_verdict(_report(tmp_path, cases), FILES)


def test_a_missing_or_unreadable_report_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(smoke.SmokeFailed):
        smoke.junit_verdict(tmp_path / 'absent.xml', FILES)
    unreadable = tmp_path / 'junit.xml'
    unreadable.write_text('<testsuites>')
    with pytest.raises(smoke.SmokeFailed):
        smoke.junit_verdict(unreadable, FILES)


def test_what_resolved_is_held_to_the_lane() -> None:
    floor, newest = smoke.LANES['floor'], smoke.LANES['newest']
    smoke.check_resolved(
        floor, {'pyarrow': '14.0.1', 'pandas': '2.2.0', 'numpy': '1.26.4'}
    )
    smoke.check_resolved(newest, {'pandas': '3.0.6', 'numpy': '2.5.4'})
    for lane, versions in [
        (floor, {'pyarrow': '14.0.2', 'pandas': '2.2.0', 'numpy': '1.26.4'}),
        (floor, {'pandas': '2.2.0', 'numpy': '1.26.4'}),
        (floor, {}),
        (newest, {'pandas': '2.3.3', 'numpy': '2.5.4'}),
        (newest, {'pandas': '3.0.6', 'numpy': '1.26.4'}),
        (newest, {'pandas': '30.0', 'numpy': '2.5.4'}),
        (newest, {'numpy': '2.5.4'}),
    ]:
        with pytest.raises(smoke.SmokeFailed):
            smoke.check_resolved(lane, versions)


def test_the_lanes_are_e16_s_floor_and_newest() -> None:
    floor, newest = smoke.LANES['floor'], smoke.LANES['newest']
    assert set(smoke.LANES) == {'floor', 'newest'}
    assert floor.python == '3.10' and newest.python == '3.14'
    for lane in (floor, newest):
        image, digest = lane.image.split('@')
        assert f'/python:{lane.python}.' in image
        assert digest.startswith('sha256:') and len(digest) == 71
    # The floor is the floor-pandas2.2 cell's and the binding job's.
    env = yaml.safe_load(CI.read_text())['env']
    jobs = yaml.safe_load(CI.read_text())['jobs']
    cell = jobs['contract-python-floor-pandas2-2']['env']['CELL_PINS'].split()
    assert sorted(floor.pins) == sorted(cell)
    assert set(floor.pins) <= set(env['BINDING_FLOOR'].split())
    assert sorted(newest.pins) == ['numpy>=2', 'pandas>=3', 'pyarrow']


def test_ci_runs_both_lanes_on_the_built_wheels() -> None:
    job = yaml.safe_load(CI.read_text())['jobs']['wheel-smoke']
    assert job['strategy']['matrix']['lane'] == ['floor', 'newest']
    assert set(job['needs']) == {'native-wheel', 'native-test-ops'}
    text = '\n'.join(str(step.get('run', '')) for step in job['steps'])
    assert 'scripts/rust_core/wheel_smoke.py' in text
    assert '--container' in text
    downloads = {
        step['with']['name']
        for step in job['steps']
        if str(step.get('uses', '')).startswith('actions/download-artifact')
    }
    assert downloads == {'native-wheel', 'native-wheel-test-ops'}


def test_third_party_leaves_out_this_repository_s_distributions() -> None:
    freeze = '\n'.join(
        [
            'numpy==1.26.4',
            'structured-data-models-client @ file:///w/x.whl',
            'structured-data-models-api @ file:///w/a.whl',
            '-e git+https://example/x#egg=y',
            'pip==25.0',
            '',
        ]
    )
    assert smoke.third_party(freeze) == ['numpy==1.26.4', 'pip==25.0']


@pytest.mark.parametrize(
    ('name', 'platform'),
    [
        ('c-1.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
         'linux/amd64'),
        ('c-1.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl',
         'linux/arm64'),
    ],
)  # fmt: skip
def test_a_linux_wheel_names_its_container_platform(
    name: str, platform: str
) -> None:
    assert smoke._docker_platform(pathlib.Path(name)) == platform


def test_a_macos_wheel_is_refused_a_container() -> None:
    with pytest.raises(smoke.SmokeFailed):
        smoke._docker_platform(
            pathlib.Path('c-1.0-cp310-abi3-macosx_11_0_arm64.whl')
        )


def _wheel(path: pathlib.Path, names: list[str]) -> pathlib.Path:
    with zipfile.ZipFile(path, 'w') as wheel:
        for name in names:
            wheel.writestr(name, b'x')
    return path


def test_the_control_wheel_lacks_only_the_native_library(
    tmp_path: pathlib.Path,
) -> None:
    names = [
        'sdm_client/__init__.py',
        'sdm_client/_native.pyi',
        'sdm_client/_native.abi3.so',
        'structured_data_models_client-1.0.0.dist-info/RECORD',
    ]
    wheel = _wheel(tmp_path / 'c-1.0-cp310-abi3-any.whl', names)
    doctored = smoke.without_native(wheel, tmp_path / 'control')
    with zipfile.ZipFile(doctored) as copy:
        assert sorted(copy.namelist()) == sorted(
            n for n in names if not n.endswith('.so')
        )


def test_a_wheel_with_no_native_library_is_no_control(
    tmp_path: pathlib.Path,
) -> None:
    wheel = _wheel(
        tmp_path / 'c-1.0-cp310-abi3-any.whl', ['sdm_client/__init__.py']
    )
    with pytest.raises(smoke.SmokeFailed):
        smoke.without_native(wheel, tmp_path / 'control')


COUNTS = {'test_binding.py': {'passed': 12}}
ITEMS = [
    f'rust/bindings/sdm-py/tests/test_binding.py::{name}'
    for name in smoke.EXPECTED['test_binding.py']
]


def _origins(home: pathlib.Path) -> dict[str, str]:
    site = home / 'lib/python3.10/site-packages/sdm_client'
    return {
        'sdm_client': str(site / '__init__.py'),
        'sdm_client._native': str(site / '_native.abi3.so'),
    }


def _collection(tmp_path: pathlib.Path, record: Any) -> pathlib.Path:
    path = tmp_path / 'collection.json'
    path.write_text(record if isinstance(record, str) else json.dumps(record))
    return path


def _record(home: pathlib.Path, **changes: Any) -> dict[str, Any]:
    return {'items': ITEMS, 'deselected': [], 'origins': _origins(home)} | (
        changes
    )


def test_a_collection_of_every_expected_test_passed_is_taken(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / 'env'
    smoke.collection_verdict(_collection(tmp_path, _record(home)), COUNTS, home)


def _faults(home: pathlib.Path) -> list[tuple[Any, str]]:
    checkout = smoke.REPO / 'packages/structured-data-models-client/src'
    return [
        (_record(home, items=ITEMS[1:], deselected=ITEMS[:1]), 'deselected'),
        (
            _record(home, items=[*ITEMS[1:], ITEMS[1] + '[again]']),
            'an expected test dropped silently, the count kept',
        ),
        (_record(home, items=ITEMS[1:]), 'fewer collected than passed'),
        (_record(home, items=[*ITEMS, ITEMS[0] + 'x']), 'more than passed'),
        (_record(home, items=[]), 'nothing collected'),
        (_record(home, items='x'), 'items retyped'),
        ({'items': ITEMS, 'origins': _origins(home)}, 'no deselections'),
        ({'items': ITEMS, 'deselected': []}, 'no origins'),
        (_record(home, origins=[]), 'origins retyped'),
        (_record(home, origins={}), 'no module recorded'),
        (
            _record(
                home,
                origins=_origins(home)
                | {'sdm_client._native': str(checkout / 'sdm_client/x.so')},
            ),
            'the native module from the checkout',
        ),
        (
            _record(
                home,
                origins=_origins(home)
                | {'sdm_client': str(checkout / 'sdm_client/__init__.py')},
            ),
            'the client from the checkout',
        ),
        (
            _record(home, origins=_origins(home.parent / 'elsewhere')),
            'another environment',
        ),
        ([ITEMS], 'not a record'),
        ('', 'empty'),
        ('[' * 100_000, 'nested past any reader'),
    ]


@pytest.mark.parametrize('case', range(16))
def test_a_collection_is_refused_on(tmp_path: pathlib.Path, case: int) -> None:
    home = tmp_path / 'env'
    faults = _faults(home)
    assert len(faults) == 16
    record, fault = faults[case]
    with pytest.raises(smoke.SmokeFailed):
        smoke.collection_verdict(_collection(tmp_path, record), COUNTS, home)


def test_a_missing_collection_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(smoke.SmokeFailed):
        smoke.collection_verdict(tmp_path / 'absent.json', COUNTS, tmp_path)


def test_a_file_with_no_record_of_its_tests_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / 'env'
    items = ['t/test_new.py::test_one']
    record = _record(home, items=items)
    with pytest.raises(smoke.SmokeFailed):
        smoke.collection_verdict(
            _collection(tmp_path, record),
            {'test_new.py': {'passed': 1}},
            home,
        )


def _parameters(node: ast.AST) -> int | None:
    """How many parameter sets a test's ``parametrize`` decorators give, or
    None when one is not written as a literal list or tuple (0: none)."""
    count = 0
    for decorator in getattr(node, 'decorator_list', []):
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == 'parametrize'
        ):
            continue
        values = (
            decorator.args[1]
            if len(decorator.args) > 1
            else next(
                (k.value for k in decorator.keywords if k.arg == 'argvalues'),
                None,
            )
        )
        if not isinstance(values, (ast.List, ast.Tuple)):
            return None
        count = (count or 1) * len(values.elts)
    return count


def _defined_tests(source: str) -> dict[str, int | None]:
    """A test file's tests as pytest names them after the file, each with its
    parameter count: module-level functions, async ones included, and the
    methods of ``Test`` classes, as ``Class::method``."""
    found: dict[str, int | None] = {}
    test_function = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.parse(source).body:
        if isinstance(node, test_function) and node.name.startswith('test_'):
            found[node.name] = _parameters(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            for item in node.body:
                if isinstance(item, test_function) and item.name.startswith(
                    'test_'
                ):
                    found[f'{node.name}::{item.name}'] = _parameters(item)
    return found


def test_the_expected_tests_are_the_files_own() -> None:
    """Every test file of the binding is recorded, with each test it defines,
    async tests and test classes' methods included; a parametrized one under
    each of its parameters, as many as its literal parameter list holds, and
    a test with no parameters under its bare name."""
    tests = smoke.REPO / smoke.TESTS
    assert sorted(smoke.EXPECTED) == sorted(
        p.name for p in tests.glob('test_*.py')
    )
    for name, recorded in smoke.EXPECTED.items():
        defined = _defined_tests((tests / name).read_text())
        assert {test.split('[', 1)[0] for test in recorded} == set(defined), (
            name
        )
        assert len(set(recorded)) == len(recorded), name
        for test, count in defined.items():
            bare = [x for x in recorded if x == test]
            with_parameters = [x for x in recorded if x.startswith(test + '[')]
            if count == 0:
                assert bare and not with_parameters, (name, test)
            elif count is None:
                assert with_parameters and not bare, (name, test)
            else:
                assert len(with_parameters) == count and not bare, (name, test)


def test_the_expected_tests_sync_sees_classes_async_tests_and_parameters() -> (
    None
):
    """The sync check's reading, on a file holding each shape it must see."""
    source = (
        'import pytest\n'
        'def test_plain(): pass\n'
        'async def test_async(): pass\n'
        "@pytest.mark.parametrize('x', [1, 2, 3])\n"
        'def test_three(x): pass\n'
        "@pytest.mark.parametrize('x', VALUES)\n"
        'def test_unknown(x): pass\n'
        'class TestGroup:\n'
        '    def test_member(self): pass\n'
        '    async def test_async_member(self): pass\n'
        '    def helper(self): pass\n'
        'class Helper:\n'
        '    def test_not_collected(self): pass\n'
        'def helper_test(): pass\n'
    )
    assert _defined_tests(source) == {
        'test_plain': 0,
        'test_async': 0,
        'test_three': 3,
        'test_unknown': None,
        'TestGroup::test_member': 0,
        'TestGroup::test_async_member': 0,
    }


def test_the_tests_run_with_nothing_of_the_caller_s_pytest(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> None:
        seen.update(command=command, **kwargs)

    monkeypatch.setattr(smoke, '_run', run)
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k "not sharing"')
    monkeypatch.setenv('PYTEST_PLUGINS', 'other')
    smoke._pytest(
        'python', tmp_path, ('test_binding.py',), 'production',
        tmp_path / 'r.xml', tmp_path / 'c.json',
    )  # fmt: skip
    env, command = seen['env'], seen['command']
    assert 'PYTEST_ADDOPTS' not in env and 'PYTEST_PLUGINS' not in env
    assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
    assert env['SDM_SMOKE_COLLECTION'] == str(tmp_path / 'c.json')
    at = command.index('-c', 5)
    assert command[at + 1] == str(tmp_path / 'guard' / 'pytest.ini')
    assert (tmp_path / 'guard' / 'pytest.ini').read_text() == '[pytest]\n'
    assert command[command.index('-p') + 1] == 'smoke_guard'
    assert (tmp_path / 'guard' / 'smoke_guard.py').read_text() == smoke._GUARD
