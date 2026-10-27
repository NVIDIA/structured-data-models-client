# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/rust_core/wheel_smoke.py``'s own rules (T1.17): its verdict on
a test report and on what resolved fails closed, its lanes are decision
E16's, and CI runs both of them.

Since T1.20, the tree it takes the candidate's contract inputs from
(``--inputs``), judged by its own checkout's tests, and the wheels'
provenance (``--commit``).

The smoke itself runs in CI's ``wheel-smoke`` job, against the wheels."""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import types
import xml.etree.ElementTree as ElementTree
import zipfile
from typing import Any

import pytest
import yaml

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

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
        floor, {'pyarrow': '14.0.1', 'pandas': '2.2.2', 'numpy': '1.23.5'}
    )
    smoke.check_resolved(newest, {'pandas': '3.0.6', 'numpy': '2.5.4'})
    for lane, versions in [
        (floor, {'pyarrow': '14.0.2', 'pandas': '2.2.2', 'numpy': '1.23.5'}),
        (floor, {'pandas': '2.2.2', 'numpy': '1.23.5'}),
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
    assert set(smoke.LANES) == {
        'floor',
        'newest',
        'aarch64-floor',
        'aarch64-newest',
    }
    # The aarch64 lanes are the same two on an aarch64 machine (KO2-40).
    for lane in (floor, newest):
        assert lane.machine is None
    assert smoke.LANES['aarch64-floor'] == dataclasses.replace(
        floor, machine='aarch64'
    )
    assert smoke.LANES['aarch64-newest'] == dataclasses.replace(
        newest, machine='aarch64'
    )
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


def test_production_smoke_includes_the_frame_ownership_tests() -> None:
    # These tests use shipped operations: the ownership boundary must hold
    # on the production wheel as well as the test-ops wheel.
    files = {
        'test_t42_frames.py',
        'test_t42_string_ownership.py',
        'test_t42_temporal_ownership.py',
        'test_t44_numeric_checkpoint_ownership.py',
        'test_t46_direct_ownership.py',
    }
    for build in ('production', 'test-ops'):
        assert files <= set(smoke.FILES[build])
    assert len(smoke.EXPECTED['test_t42_temporal_ownership.py']) == 16
    assert len(smoke.EXPECTED['test_t42_string_ownership.py']) == 4


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


# --- T1.20: the candidate's contract inputs, the checkpoint's judges ----------

BINDING_TESTS = 'rust/bindings/sdm-py/tests'
#: The contract inputs ``suite()`` takes from the inputs tree, at their
#: repository paths: the contract's sources and the native module's stub.
#: The message builder is not one: it decodes what the tests judge.
INPUTS = (
    'rust/crates/sdm-contract/src/lib.rs',
    'rust/crates/sdm-contract/src/error.rs',
    'packages/structured-data-models-client/src/sdm_client/_native.pyi',
)
#: The binding's test files ``suite()`` takes from its own checkout as they
#: are: every file of the tests directory but ``messages.py``, which it
#: takes from its own checkout too, its ``CONTRACT`` line set to the
#: inputs' version.
JUDGES = (
    'test_binding.py',
    'test_arrow_lifetimes.py',
    'test_test_ops.py',
    'conftest.py',
    'getenv_shim.c',
)
#: The one line of ``messages.py`` the suite sets.
CONTRACT_LINE = re.compile(r'^CONTRACT = (?:\d+|_crate_version\(\))$', re.M)


def _builder(label: str, version: int) -> str:
    """A tree's ``messages.py``: its version line among lines of its own."""
    return (
        f'# the builder of {label}\n'
        'import json\n'
        f'CONTRACT = {version}\n'
        'def read(control):\n'
        '    return json.loads(control)\n'
    )


def _at(text: str, version: int) -> str:
    """`text`, a ``messages.py``, with its one ``CONTRACT`` line set to
    `version` and nothing else changed."""
    assert len(CONTRACT_LINE.findall(text)) == 1
    return CONTRACT_LINE.sub(f'CONTRACT = {version}', text)


def _tree(root: pathlib.Path, version: int) -> pathlib.Path:
    """A fake checkout at contract version `version`: the contract inputs,
    the binding's test files and message builder (each naming the tree it
    is of), and the two sibling distributions."""
    label = root.name
    files = {
        f'{BINDING_TESTS}/messages.py': _builder(label, version),
        'rust/crates/sdm-contract/src/lib.rs': (
            f'//! bumps [`CONTRACT_VERSION`] on any change\n'
            f'pub const CONTRACT_VERSION: u32 = {version};\n'
        ),
        'rust/crates/sdm-contract/src/error.rs': (
            f'// the error codes of {label}, at {version}\n'
        ),
        'packages/structured-data-models-client/src/sdm_client/_native.pyi': (
            f'# the stub of {label}, at {version}\n'
        ),
        f'{BINDING_TESTS}/conftest.py': f'# the conftest of {label}\n',
        f'{BINDING_TESTS}/getenv_shim.c': f'/* the shim of {label} */\n',
        'packages/structured-data-models-api/pyproject.toml': '[project]\n',
        'packages/structured-data-models-pql/pyproject.toml': '[project]\n',
    }
    for name in (
        'test_binding.py',
        'test_arrow_lifetimes.py',
        'test_test_ops.py',
    ):
        files[f'{BINDING_TESTS}/{name}'] = (
            f'# {name} of {label}\ndef test_one() -> None:\n    pass\n'
        )  # fmt: skip
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    return root


def _scratch(tmp_path: pathlib.Path) -> pathlib.Path:
    into = tmp_path / 'scratch'
    into.mkdir()
    return into


#: A message builder's ``read()`` that turns a numeric ``accepted`` into a
#: bool: what a candidate could write to pass an assertion it fails.
NORMALIZING = """
plain_read = read


def read(control):
    value = plain_read(control)
    if 'result' in value and 'accepted' in value['result']:
        value['result']['accepted'] = bool(value['result']['accepted'])
    return value
"""


def test_the_suite_is_the_inputs_contract_and_its_own_tests(
    tmp_path: pathlib.Path,
) -> None:
    """The inputs' three contract files at their repository paths, the own
    tree's test files, conftest, shim and message builder beside them, and
    nothing of the inputs' tests: an extra test in the inputs'
    ``test_binding.py`` and a test file only the inputs hold stay out."""
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    binding = inputs / BINDING_TESTS / 'test_binding.py'
    binding.write_text(
        binding.read_text() + 'def test_extra() -> None:\n    pass\n'
    )
    (inputs / BINDING_TESTS / 'test_extra.py').write_text(
        'def test_extra() -> None:\n    pass\n'
    )
    into = _scratch(tmp_path)
    tests = smoke.suite(own, inputs, into)
    assert tests.is_dir()
    assert tests.resolve().is_relative_to(into.resolve())
    root = tests.parents[3]
    assert tests == root / BINDING_TESTS
    assert tuple(smoke.INPUTS) == INPUTS
    for path in INPUTS:
        assert (root / path).read_text() == (inputs / path).read_text(), path
    for name in JUDGES:
        assert (tests / name).read_text() == (
            own / BINDING_TESTS / name
        ).read_text(), name
    assert (tests / 'messages.py').read_text() == _builder('own', 20)
    assert {p.name for p in tests.iterdir() if p.is_file()} == {
        *JUDGES,
        'messages.py',
    }


def test_the_own_tree_s_messages_py_is_kept_at_the_inputs_version(
    tmp_path: pathlib.Path,
) -> None:
    """The suite's message builder is the own tree's, byte for byte but for
    its ``CONTRACT`` line, which says the inputs' ``CONTRACT_VERSION``; the
    own tree is left as it was."""
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    (inputs / BINDING_TESTS / 'messages.py').write_text(
        "CONTRACT = 20\n# the inputs' builder\n"
    )
    tests = smoke.suite(own, inputs, _scratch(tmp_path))
    staged = (tests / 'messages.py').read_bytes()
    assert staged == _builder('own', 20).encode()
    assert b"the inputs' builder" not in staged
    assert (own / BINDING_TESTS / 'messages.py').read_text() == _builder(
        'own', 19
    )


@pytest.mark.parametrize('missing', INPUTS)
def test_an_input_missing_from_the_inputs_is_refused_by_name(
    tmp_path: pathlib.Path, missing: str
) -> None:
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    (inputs / missing).unlink()
    with pytest.raises(smoke.SmokeFailed) as refused:
        smoke.suite(own, inputs, _scratch(tmp_path))
    assert pathlib.PurePosixPath(missing).name in str(refused.value)


def test_a_suite_at_n_judges_inputs_at_n_plus_1(tmp_path: pathlib.Path) -> None:
    """The control: the own tree at contract version N, the inputs at N+1.
    Where the binding's test reads the repository (``parents[4]`` of its
    own file), the suite says N+1; the test is the own tree's, and so is
    its message builder, at ``CONTRACT = N+1``."""
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    tests = smoke.suite(own, inputs, _scratch(tmp_path))
    test = tests / 'test_binding.py'
    repo = test.resolve().parents[4]
    assert repo not in (own.resolve(), inputs.resolve())
    lib = (repo / 'rust/crates/sdm-contract/src/lib.rs').read_text()
    assert 'pub const CONTRACT_VERSION: u32 = 20;' in lib
    builder = (tests / 'messages.py').read_text()
    assert builder == _builder('own', 20)
    assert CONTRACT_LINE.findall(builder) == ['CONTRACT = 20']
    assert '# the builder of own' in builder
    assert (
        test.read_text()
        == (own / BINDING_TESTS / 'test_binding.py').read_text()
    )
    assert '# test_binding.py of own' in test.read_text()


def test_the_checkout_s_own_tests_judge_another_tree_s_inputs(
    tmp_path: pathlib.Path,
) -> None:
    """The smoke's own checkout as the judge: its binding tests byte for
    byte, and its message builder but for the ``CONTRACT`` line, beside a
    fake tree's contract inputs."""
    inputs = _tree(tmp_path / 'inputs', 20)
    tests = smoke.suite(smoke.REPO, inputs, _scratch(tmp_path))
    own = smoke.REPO / BINDING_TESTS
    judges = [
        p.name
        for p in own.iterdir()
        if p.is_file() and p.name != 'messages.py'
        and (p.name.startswith('test_') or p.suffix == '.c'
             or p.name == 'conftest.py')
    ]  # fmt: skip
    assert set(JUDGES) <= set(judges)
    for name in judges:
        assert (tests / name).read_bytes() == (own / name).read_bytes(), name
    assert (tests / 'messages.py').read_text() == _at(
        (own / 'messages.py').read_text(), 20
    )
    root = tests.parents[3]
    for path in INPUTS:
        assert (root / path).read_text() == (inputs / path).read_text(), path


def test_a_crate_reading_builder_is_staged_without_executing_it(
    tmp_path: pathlib.Path,
) -> None:
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    builder = (
        'def _crate_version():\n'
        '    raise AssertionError("the smoke must not execute its builder")\n'
        'CONTRACT = _crate_version()\n'
    )
    (own / BINDING_TESTS / 'messages.py').write_text(builder)
    tests = smoke.suite(own, inputs, _scratch(tmp_path))
    assert (tests / 'messages.py').read_text() == builder.replace(
        'CONTRACT = _crate_version()', 'CONTRACT = 20'
    )


def test_a_normalizing_messages_py_in_the_inputs_changes_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """Codex's seed: the inputs' message builder reads a numeric
    ``accepted`` as a bool, and says another version than the inputs'
    contract. The suite's builder is still the own tree's, but for its
    ``CONTRACT`` line, set from the inputs' ``lib.rs``; and with no builder
    in the inputs at all, the suite is the same."""
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    candidate = inputs / BINDING_TESTS / 'messages.py'
    candidate.write_text(_builder('inputs', 21) + NORMALIZING)
    into = tmp_path / 'first'
    into.mkdir()
    tests = smoke.suite(own, inputs, into)
    staged = (tests / 'messages.py').read_text()
    assert staged == _builder('own', 20)
    assert 'bool(' not in staged and 'plain_read' not in staged
    candidate.unlink()
    again = tmp_path / 'second'
    again.mkdir()
    tests = smoke.suite(own, inputs, again)
    assert (tests / 'messages.py').read_text() == _builder('own', 20)


def _load(
    monkeypatch: pytest.MonkeyPatch, name: str, path: pathlib.Path
) -> Any:
    """The module at `path`, imported as `name`; `sys.modules` restored by
    `monkeypatch`."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _fake_native(
    monkeypatch: pytest.MonkeyPatch, numeric: bool, seen: list[int]
) -> None:
    """``sdm_client._native`` as a stand-in whose ``Run`` validates a query
    as the binding does, answering ``accepted`` as 0/1 when `numeric`, as
    a bool otherwise; each start's contract version recorded in `seen`."""

    class Run:
        accepted: object

        @classmethod
        def start(cls, control: bytes, attachments: dict[int, bytes]) -> Run:
            seen.append(json.loads(control)['contract'])
            run = cls()
            good = json.loads(attachments[1]) == (
                'PREDICT users.age FOR EACH users.id'
            )
            run.accepted = int(good) if numeric else good
            return run

        def next(self) -> tuple[bytes, dict[int, bytes]]:
            step = {
                'message': 'step',
                'step': 'done',
                'result': {'op': 'validate_query', 'accepted': self.accepted},
            }
            return json.dumps(step).encode(), {}

    native = types.ModuleType('sdm_client._native')
    native.Run = Run  # type: ignore[attr-defined]
    client = types.ModuleType('sdm_client')
    client._native = native  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'sdm_client', client)
    monkeypatch.setitem(sys.modules, 'sdm_client._native', native)


def _normalizing_inputs(root: pathlib.Path, version: int) -> pathlib.Path:
    """A copy of this checkout's contract inputs at `root`, its
    ``CONTRACT_VERSION`` set to `version`, beside a message builder that
    normalizes ``accepted``."""
    for path in INPUTS:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes((smoke.REPO / path).read_bytes())
    lib = root / 'rust/crates/sdm-contract/src/lib.rs'
    text, count = re.subn(
        r'pub const CONTRACT_VERSION: u32 = \d+;',
        f'pub const CONTRACT_VERSION: u32 = {version};',
        lib.read_text(),
    )
    assert count == 1
    lib.write_text(text)
    builder = root / BINDING_TESTS / 'messages.py'
    builder.parent.mkdir(parents=True, exist_ok=True)
    builder.write_text(
        (smoke.REPO / BINDING_TESTS / 'messages.py').read_text() + NORMALIZING
    )
    return root


def _checkpoint_validates(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, numeric: bool
) -> tuple[Any, list[int], int]:
    """This checkout's ``test_a_query_is_validated_through_a_run`` as the
    suite stages it, over inputs whose builder normalizes, with a fake
    native module; with the contract version the inputs say."""
    source = (smoke.REPO / 'rust/crates/sdm-contract/src/lib.rs').read_text()
    found = re.search(r'pub const CONTRACT_VERSION: u32 = (\d+);', source)
    assert found
    version = int(found.group(1)) + 1
    inputs = _normalizing_inputs(tmp_path / 'inputs', version)
    tests = smoke.suite(smoke.REPO, inputs, _scratch(tmp_path))
    seen: list[int] = []
    _fake_native(monkeypatch, numeric, seen)
    _load(monkeypatch, 'messages', tests / 'messages.py')
    binding = _load(
        monkeypatch, 'wheel_smoke_seeded_binding', tests / 'test_binding.py'
    )
    return binding.test_a_query_is_validated_through_a_run, seen, version


def test_the_checkpoint_refuses_a_numeric_accepted_whatever_the_inputs_say(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex's seed, executed: a binding answering ``accepted`` as 0/1
    fails the checkpoint's own, unchanged assertion, though the inputs'
    message builder would read it as a bool."""
    check, seen, version = _checkpoint_validates(tmp_path, monkeypatch, True)
    with pytest.raises(AssertionError):
        check()
    assert seen and set(seen) == {version}


def test_the_checkpoint_takes_a_bool_accepted_from_the_same_suite(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed's control: the same suite and fake, answering a bool,
    passes, so the refusal above is the numeric ``accepted``'s; and the
    checkpoint's builder writes the inputs' version into each start."""
    check, seen, version = _checkpoint_validates(tmp_path, monkeypatch, False)
    check()
    assert seen == [version, version]


#: Each ``lib.rs`` refused: none or more than one ``CONTRACT_VERSION``.
LIB_FAULTS = {
    'no version': '//! bumps [`CONTRACT_VERSION`] on any change\n',
    'two versions': (
        'pub const CONTRACT_VERSION: u32 = 20;\n'
        'pub const CONTRACT_VERSION: u32 = 21;\n'
    ),
    'the version twice': (
        'pub const CONTRACT_VERSION: u32 = 20;\n'
        'pub const CONTRACT_VERSION: u32 = 20;\n'
    ),
}


@pytest.mark.parametrize('fault', list(LIB_FAULTS))
def test_the_inputs_lib_rs_without_one_version_is_refused_by_name(
    tmp_path: pathlib.Path, fault: str
) -> None:
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    (inputs / 'rust/crates/sdm-contract/src/lib.rs').write_text(
        LIB_FAULTS[fault]
    )
    with pytest.raises(smoke.SmokeFailed) as refused:
        smoke.suite(own, inputs, _scratch(tmp_path))
    assert 'lib.rs' in str(refused.value)


#: Each own ``messages.py`` refused: none or more than one line
#: ``CONTRACT = <digits>``.
BUILDER_FAULTS = {
    'no contract line': '# the builder of own\nimport json\n',
    'an annotated one only': 'CONTRACT: int = 19\n',
    'two contract lines': 'CONTRACT = 19\nimport json\nCONTRACT = 18\n',
    'the line twice': 'CONTRACT = 19\nCONTRACT = 19\n',
}


@pytest.mark.parametrize('fault', list(BUILDER_FAULTS))
def test_the_own_messages_py_without_one_contract_line_is_refused_by_name(
    tmp_path: pathlib.Path, fault: str
) -> None:
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    (own / BINDING_TESTS / 'messages.py').write_text(BUILDER_FAULTS[fault])
    with pytest.raises(smoke.SmokeFailed) as refused:
        smoke.suite(own, inputs, _scratch(tmp_path))
    assert 'messages.py' in str(refused.value)


SHA = '0123456789abcdef0123456789abcdef01234567'
OTHER = 'fedcba9876543210fedcba9876543210fedcba98'
#: A field an identity is written without.
DROP = object()


def _built(
    tmp_path: pathlib.Path, test_ops: bool, **changes: Any
) -> pathlib.Path:
    """A fake wheel of the build named, with its identity beside it as
    ``native_wheel.py`` writes it, changed by `changes`."""
    where = tmp_path / ('test-ops' if test_ops else 'shipped')
    where.mkdir(exist_ok=True)
    wheel = where / (
        'structured_data_models_client-1.0.0-cp310-abi3-'
        'manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
    )
    wheel.write_bytes(b'a wheel of ' + where.name.encode())
    identity = {
        'wheel': wheel.name,
        'sha256': smoke.sha256(wheel),
        'native_sha256': '0' * 64,
        'commit': SHA,
        'tree_clean': True,
        'features': ['test-ops'] if test_ops else [],
        'maturin': '1.0.0',
    } | changes
    record = {k: v for k, v in identity.items() if v is not DROP}
    (where / f'{wheel.name}.json').write_text(json.dumps(record, indent=2))
    return wheel


@pytest.mark.parametrize('test_ops', [False, True])
def test_a_wheel_of_the_commit_passes_its_provenance(
    tmp_path: pathlib.Path, test_ops: bool
) -> None:
    smoke.provenance(_built(tmp_path, test_ops), SHA, test_ops)


#: Each identity refused, and the field its refusal names (None: the
#: identity as a whole, named by the wheel).
PROVENANCE_FAULTS: dict[str, tuple[Any, str | None]] = {
    'no identity': (None, None),
    'unreadable JSON': ('{"commit": ', None),
    'not an object': (json.dumps([SHA]), None),
    'another commit': ({'commit': OTHER}, 'commit'),
    'a short commit': ({'commit': SHA[:12]}, 'commit'),
    'no commit': ({'commit': DROP}, 'commit'),
    'a dirty tree': ({'tree_clean': False}, 'tree_clean'),
    'tree_clean as text': ({'tree_clean': 'true'}, 'tree_clean'),
    'no tree_clean': ({'tree_clean': DROP}, 'tree_clean'),
    'another hash': ({'sha256': 'f' * 64}, 'sha256'),
    'no hash': ({'sha256': DROP}, 'sha256'),
    'the other build s features': ('swapped', 'features'),
    'a feature more': ('more', 'features'),
    'no features': ({'features': DROP}, 'features'),
}


@pytest.mark.parametrize('test_ops', [False, True])
@pytest.mark.parametrize('fault', list(PROVENANCE_FAULTS))
def test_a_wheel_s_provenance_is_refused_naming_the_field(
    tmp_path: pathlib.Path, fault: str, test_ops: bool
) -> None:
    change, field = PROVENANCE_FAULTS[fault]
    if change == 'swapped':
        change = {'features': [] if test_ops else ['test-ops']}
    elif change == 'more':
        change = {'features': ['test-ops', 'other'] if test_ops else ['other']}
    wheel = _built(
        tmp_path, test_ops, **(change if isinstance(change, dict) else {})
    )
    identity = wheel.parent / f'{wheel.name}.json'
    if change is None:
        identity.unlink()
    elif isinstance(change, str):
        identity.write_text(change)
    with pytest.raises(smoke.SmokeFailed) as refused:
        smoke.provenance(wheel, SHA, test_ops)
    assert wheel.name in str(refused.value)
    if field is not None:
        assert field in str(refused.value)


def test_a_short_commit_asked_for_is_refused(tmp_path: pathlib.Path) -> None:
    """The identity and the commit asked for agree, on 12 characters: the
    commit is the full 40, so it is refused."""
    wheel = _built(tmp_path, False, commit=SHA[:12])
    with pytest.raises(smoke.SmokeFailed) as refused:
        smoke.provenance(wheel, SHA[:12], False)
    assert 'commit' in str(refused.value)


def _main_arguments(
    tmp_path: pathlib.Path, shipped: pathlib.Path, test_ops: pathlib.Path
) -> list[str]:
    return [
        '--lane', 'floor', '--wheel', str(shipped),
        '--test-ops-wheel', str(test_ops), '--out', str(tmp_path / 'out'),
        '--python', sys.executable,
    ]  # fmt: skip


def _no_environment(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stands in for the smoke's environments: records that one was built."""
    built: list[str] = []

    def smoke_run(*arguments: Any, **keywords: Any) -> int:
        built.append('smoke')
        return 0

    def container(*arguments: Any, **keywords: Any) -> int:
        built.append('container')
        return 0

    monkeypatch.setattr(smoke, 'smoke', smoke_run)
    monkeypatch.setattr(smoke, 'in_container', container)
    return built


def test_the_wheels_provenance_is_held_before_any_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _no_environment(monkeypatch)
    # The inputs (this checkout, at no commit SHA names) are held by
    # their own tests below; here only the wheels' provenance is.
    monkeypatch.setattr(
        smoke, 'inputs_clean', lambda *a, **k: None, raising=False
    )
    shipped, test_ops = _built(tmp_path, False), _built(tmp_path, True)
    arguments = _main_arguments(tmp_path, shipped, test_ops)
    assert smoke.main([*arguments, '--commit', SHA]) == 0
    assert built == ['smoke']
    built.clear()
    # Another commit's wheels: refused, and no environment built.
    assert smoke.main([*arguments, '--commit', OTHER]) != 0
    assert built == []
    # The builds swapped: refused, and no environment built.
    swapped = _main_arguments(tmp_path, test_ops, shipped)
    assert smoke.main([*swapped, '--commit', SHA]) != 0
    assert built == []


def test_without_a_commit_no_provenance_is_asked(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _no_environment(monkeypatch)
    shipped, test_ops = _built(tmp_path, False), _built(tmp_path, True)
    for wheel in (shipped, test_ops):
        (wheel.parent / f'{wheel.name}.json').unlink()
    assert smoke.main(_main_arguments(tmp_path, shipped, test_ops)) == 0
    assert built == ['smoke']


def _option(argv: list[str], name: str) -> str:
    """The value of `name` in `argv`, given as ``NAME VALUE`` or
    ``NAME=VALUE``; exactly once."""
    found = [
        argv[i + 1] for i, word in enumerate(argv[:-1]) if word == name
    ] + [word.split('=', 1)[1] for word in argv if word.startswith(name + '=')]
    assert len(found) == 1, (name, argv)
    return found[0]


def test_the_container_mounts_the_inputs_read_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``--container``, the inputs tree is mounted read-only at
    ``/inputs``, as the checkout is at ``/repo``, and the run inside takes
    its inputs from there."""
    seen: list[list[str]] = []

    class Done:
        returncode = 0

    def run(command: list[str], **keywords: Any) -> Done:
        seen.append(command)
        return Done()

    monkeypatch.setattr(smoke, '_run', run)
    inputs = _tree(tmp_path / 'inputs', 20)
    shipped, test_ops = _built(tmp_path, False), _built(tmp_path, True)
    assert (
        smoke.main([
            '--lane', 'floor', '--wheel', str(shipped),
            '--test-ops-wheel', str(test_ops), '--out', str(tmp_path / 'out'),
            '--container', '--inputs', str(inputs),
        ])
        == 0
    )  # fmt: skip
    docker = [c for c in seen if c[:1] == ['docker']]
    assert len(docker) == 1, seen
    command = docker[0]
    mounts = [command[i + 1] for i, x in enumerate(command[:-1]) if x == '-v']
    at_inputs = [m for m in mounts if m.endswith(':/inputs:ro')]
    assert len(at_inputs) == 1, mounts
    assert pathlib.Path(at_inputs[0][: -len(':/inputs:ro')]).resolve() == (
        inputs.resolve()
    )
    assert any(m.endswith(':/repo:ro') for m in mounts), mounts
    inside = command[command.index('/repo/scripts/rust_core/wheel_smoke.py') :]
    assert _option(inside, '--inputs') == '/inputs'


# --- T1.20: a messages.py that patches pytest's report, in the inputs --------

#: What a candidate's message builder could add to pass a test it fails:
#: pytest's report of each test rewritten to ``passed``, at import (the
#: class's constructor of reports patched) or in a hook (the module
#: registering itself as a plugin from the running configuration).
REPORT_PATCHES = {
    'at import': """
import _pytest.reports as _reports

_made = _reports.TestReport.from_item_and_call.__func__


def _passed(cls, item, call):
    report = _made(cls, item, call)
    report.outcome, report.longrepr = 'passed', None
    return report


_reports.TestReport.from_item_and_call = classmethod(_passed)
""",
    'in a hook': """
import gc as _gc
import sys as _sys

import pytest as _pytest


@_pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    report.outcome, report.longrepr = 'passed', None


_config = next(o for o in _gc.get_objects() if isinstance(o, _pytest.Config))
_config.pluginmanager.register(_sys.modules[__name__])
""",
}


@pytest.mark.parametrize('patch', list(REPORT_PATCHES))
def test_a_report_patching_messages_py_in_the_inputs_changes_nothing(
    tmp_path: pathlib.Path, patch: str
) -> None:
    """The subagent's seed: the inputs' message builder rewrites pytest's
    reports to ``passed``. The suite's builder is the own tree's, but for
    its ``CONTRACT`` line, and holds nothing of the patch."""
    own = _tree(tmp_path / 'own', 19)
    inputs = _tree(tmp_path / 'inputs', 20)
    (inputs / BINDING_TESTS / 'messages.py').write_text(
        _builder('inputs', 20) + REPORT_PATCHES[patch]
    )
    tests = smoke.suite(own, inputs, _scratch(tmp_path))
    staged = (tests / 'messages.py').read_text()
    assert staged == _builder('own', 20)
    assert '_pytest' not in staged and 'outcome' not in staged
    for path in tests.rglob('*'):
        if path.is_file():
            assert "'passed'" not in path.read_text(errors='replace'), path


#: ``sdm_client`` as a stand-in package, imported by an inner pytest from
#: ``PYTHONPATH``: its ``Run`` validates a query as the binding does,
#: answering ``accepted`` as 0/1 when ``NUMERIC``, as a bool otherwise.
FAKE_NATIVE = """
import json

NUMERIC = {numeric}


class Run:
    @classmethod
    def start(cls, control, attachments):
        run = cls()
        good = json.loads(attachments[1]) == (
            'PREDICT users.age FOR EACH users.id'
        )
        run.accepted = int(good) if NUMERIC else good
        return run

    def next(self):
        step = {{
            'message': 'step',
            'step': 'done',
            'result': {{'op': 'validate_query', 'accepted': self.accepted}},
        }}
        return json.dumps(step).encode(), {{}}
"""


def _fake_client(where: pathlib.Path, numeric: bool) -> pathlib.Path:
    package = where / 'sdm_client'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / '_native.py').write_text(FAKE_NATIVE.format(numeric=numeric))
    return where


def _validate_in_pytest(
    tests: pathlib.Path, client: pathlib.Path, report: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    """The staged ``test_a_query_is_validated_through_a_run`` under a pytest
    of its own, as the smoke runs it (no ``PYTEST_*`` variable, no plugin
    autoload, an empty configuration), the fake client on the path; its
    JUnit report written to `report`."""
    ini = report.parent / 'pytest.ini'
    ini.write_text('[pytest]\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTEST_')}
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTHONPATH=str(client),
        PYTHONDONTWRITEBYTECODE='1',
    )
    root = tests.parents[3]
    return subprocess.run(
        [sys.executable, '-m', 'pytest',
         f'{tests / "test_binding.py"}::test_a_query_is_validated_through_a_run',
         '-c', str(ini), '--rootdir', str(root), '--confcutdir', str(tests),
         '-p', 'no:cacheprovider', '-q', '--native-build', 'production',
         f'--junitxml={report}'],
        cwd=report.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip


def _patching_inputs(root: pathlib.Path, patch: str) -> pathlib.Path:
    """A copy of this checkout's contract inputs at `root`, beside this
    checkout's message builder with `patch` added."""
    for path in INPUTS:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes((smoke.REPO / path).read_bytes())
    builder = root / BINDING_TESTS / 'messages.py'
    builder.parent.mkdir(parents=True, exist_ok=True)
    builder.write_text(
        (smoke.REPO / BINDING_TESTS / 'messages.py').read_text()
        + REPORT_PATCHES[patch]
    )
    return root


@pytest.mark.parametrize('patch', list(REPORT_PATCHES))
def test_a_failing_binding_still_fails_when_the_inputs_patch_pytest_s_report(
    tmp_path: pathlib.Path, patch: str
) -> None:
    """The seed, executed under pytest: a binding answering ``accepted`` as
    0/1 fails this checkout's test as the suite stages it, the report
    saying so, though the inputs' message builder rewrites reports to
    ``passed``. The controls: the same suite passes a binding answering a
    bool, so the failure is the binding's; and the same suite with the
    inputs' builder put in its place passes the failing binding, so the
    patch would have hidden it."""
    inputs = _patching_inputs(tmp_path / 'inputs', patch)
    staged = tmp_path / 'staged'
    staged.mkdir()
    tests = smoke.suite(smoke.REPO, inputs, staged)
    numeric = _fake_client(tmp_path / 'numeric', True)
    boolean = _fake_client(tmp_path / 'boolean', False)
    files = ('test_binding.py',)

    seeded = tmp_path / 'seeded'
    seeded.mkdir()
    done = _validate_in_pytest(tests, numeric, seeded / 'junit.xml')
    assert done.returncode == 1, done.stdout + done.stderr
    with pytest.raises(smoke.SmokeFailed):
        smoke.junit_verdict(seeded / 'junit.xml', files)
    cases = list(
        ElementTree.parse(seeded / 'junit.xml').getroot().iter('testcase')
    )
    assert [{c.tag for c in case} for case in cases] == [{'failure'}]

    control = tmp_path / 'control'
    control.mkdir()
    done = _validate_in_pytest(tests, boolean, control / 'junit.xml')
    assert done.returncode == 0, done.stdout + done.stderr
    counts = smoke.junit_verdict(control / 'junit.xml', files)
    assert counts['test_binding.py']['passed'] == 1

    # The inputs' builder in the suite's place: the patch hides the fault.
    swapped = tmp_path / 'swapped'
    shutil.copytree(staged, swapped)
    swapped_tests = swapped / BINDING_TESTS
    (swapped_tests / 'messages.py').write_bytes(
        (inputs / BINDING_TESTS / 'messages.py').read_bytes()
    )
    hidden = tmp_path / 'hidden'
    hidden.mkdir()
    done = _validate_in_pytest(swapped_tests, numeric, hidden / 'junit.xml')
    assert done.returncode == 0, done.stdout + done.stderr
    counts = smoke.junit_verdict(hidden / 'junit.xml', files)
    assert counts['test_binding.py'] == {
        'passed': 1, 'failed': 0, 'errors': 0, 'skipped': 0,
    }  # fmt: skip


# --- T1.20: the inputs tree is the commit's, clean -----------------------------

#: The paths of the inputs tree the smoke takes: the sibling distributions
#: and the contract inputs.
TAKEN = (
    'packages/structured-data-models-api',
    'packages/structured-data-models-pql',
    *INPUTS,
)


def _git(repo: pathlib.Path, *arguments: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), '-c', 'user.name=Smoke Test',
         '-c', 'user.email=smoke@example.com', '-c', 'commit.gpgsign=false',
         *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()  # fmt: skip


def _repository(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A real git checkout of a fake tree, committed clean, with a file
    under ``docs/`` and one ignored there; with its HEAD."""
    _tree(root, 20)
    (root / 'packages/structured-data-models-pql/src').mkdir(parents=True)
    (root / 'packages/structured-data-models-pql/src/pql.py').write_text('')
    (root / 'docs').mkdir()
    (root / 'docs/notes.md').write_text('# notes\n')
    (root / '.gitignore').write_text('docs/ignored.txt\n')
    _git(root, 'init', '-q')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '-m', 'the inputs')
    return root, _git(root, 'rev-parse', 'HEAD')


def test_a_clean_inputs_tree_at_the_commit_passes(
    tmp_path: pathlib.Path,
) -> None:
    inputs, head = _repository(tmp_path / 'inputs')
    assert smoke.inputs_clean(inputs, head) is None


def test_inputs_at_another_commit_are_refused(tmp_path: pathlib.Path) -> None:
    inputs, first = _repository(tmp_path / 'inputs')
    _git(inputs, 'commit', '-q', '--allow-empty', '-m', 'another')
    head = _git(inputs, 'rev-parse', 'HEAD')
    assert head != first
    smoke.inputs_clean(inputs, head)
    with pytest.raises(smoke.SmokeFailed) as refused:
        smoke.inputs_clean(inputs, first)
    assert first in str(refused.value) or head in str(refused.value)


def test_an_untracked_file_under_the_pql_distribution_is_refused_by_name(
    tmp_path: pathlib.Path,
) -> None:
    """In a directory of its own, so only a status listing every untracked
    file (not the directory alone) names it."""
    inputs, head = _repository(tmp_path / 'inputs')
    extra = inputs / 'packages/structured-data-models-pql/src/added/extra.py'
    extra.parent.mkdir()
    extra.write_text('print("taken into the sibling wheel")\n')
    with pytest.raises(smoke.SmokeFailed) as refused:
        smoke.inputs_clean(inputs, head)
    assert 'extra.py' in str(refused.value)


def _modify(path: pathlib.Path) -> None:
    path.write_text(path.read_text() + '// changed\n')


#: Each change under a path the smoke takes, and the file it names.
CHANGES = {
    'a modified lib.rs': ('rust/crates/sdm-contract/src/lib.rs', False),
    'a staged lib.rs': ('rust/crates/sdm-contract/src/lib.rs', True),
    'a modified error.rs': ('rust/crates/sdm-contract/src/error.rs', False),
    'a modified stub': (
        'packages/structured-data-models-client/src/sdm_client/_native.pyi',
        False,
    ),
    'a modified API distribution': (
        'packages/structured-data-models-api/pyproject.toml',
        False,
    ),
}


@pytest.mark.parametrize('change', list(CHANGES))
def test_a_changed_input_is_refused_by_name(
    tmp_path: pathlib.Path, change: str
) -> None:
    inputs, head = _repository(tmp_path / 'inputs')
    path, staged = CHANGES[change]
    _modify(inputs / path)
    if staged:
        _git(inputs, 'add', path)
    with pytest.raises(smoke.SmokeFailed) as refused:
        smoke.inputs_clean(inputs, head)
    assert pathlib.PurePosixPath(path).name in str(refused.value)


def test_an_inputs_directory_that_is_no_repository_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _tree(tmp_path / 'inputs', 20)
    with pytest.raises(smoke.SmokeFailed):
        smoke.inputs_clean(inputs, SHA)


#: Changes outside the paths the smoke takes, none of which refuses.
ELSEWHERE = {
    'an untracked file under docs': ('docs/untracked.md', 'new'),
    'an ignored file under docs': ('docs/ignored.txt', 'new'),
    'a modified file under docs': ('docs/notes.md', 'modified'),
}


@pytest.mark.parametrize('change', list(ELSEWHERE))
def test_a_change_elsewhere_does_not_refuse(
    tmp_path: pathlib.Path, change: str
) -> None:
    inputs, head = _repository(tmp_path / 'inputs')
    path, kind = ELSEWHERE[change]
    if kind == 'new':
        (inputs / path).write_text('elsewhere\n')
    else:
        _modify(inputs / path)
    assert _git(inputs, 'status', '--porcelain', '--ignored') != ''
    smoke.inputs_clean(inputs, head)


def _commit_arguments(
    tmp_path: pathlib.Path, inputs: pathlib.Path, commit: str, container: bool
) -> list[str]:
    """``main``'s arguments for wheels of `commit` and `inputs`, in the mode
    named; the wheels' identities say `commit`."""
    where = tmp_path / f'wheels-{commit[:8]}'
    where.mkdir(exist_ok=True)
    shipped = _built(where, False, commit=commit)
    test_ops = _built(where, True, commit=commit)
    arguments = [
        '--lane', 'floor', '--wheel', str(shipped),
        '--test-ops-wheel', str(test_ops), '--out', str(tmp_path / 'out'),
        '--inputs', str(inputs), '--commit', commit,
    ]  # fmt: skip
    return [*arguments, '--container'] if container else arguments


@pytest.mark.parametrize('container', [False, True])
def test_the_inputs_are_held_clean_at_the_commit_before_any_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, container: bool
) -> None:
    """With ``--commit``, in the direct mode and with ``--container``: a
    clean inputs tree at the commit builds the environment; an untracked
    file under a distribution the smoke takes, another commit, and an
    inputs directory that is no checkout are each refused before any
    environment is built."""
    built = _no_environment(monkeypatch)
    mode = 'container' if container else 'smoke'
    inputs, head = _repository(tmp_path / 'inputs')
    arguments = _commit_arguments(tmp_path, inputs, head, container)
    assert smoke.main(arguments) == 0
    assert built == [mode]
    built.clear()
    # Untracked under the PQL distribution.
    extra = inputs / 'packages/structured-data-models-pql/src/extra.py'
    extra.write_text('')
    assert smoke.main(arguments) != 0
    assert built == []
    extra.unlink()
    assert smoke.main(arguments) == 0
    assert built == [mode]
    built.clear()
    # Wheels of another commit than the inputs': their provenance holds,
    # the inputs do not.
    other = _commit_arguments(tmp_path, inputs, OTHER, container)
    assert smoke.main(other) != 0
    assert built == []
    # No checkout at all.
    plain = _tree(tmp_path / 'plain', 20)
    arguments = _commit_arguments(tmp_path, plain, head, container)
    assert smoke.main(arguments) != 0
    assert built == []


def test_without_a_commit_the_inputs_are_not_held(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``--commit``: an inputs directory that is no checkout is taken."""
    built = _no_environment(monkeypatch)
    inputs = _tree(tmp_path / 'inputs', 20)
    shipped, test_ops = _built(tmp_path, False), _built(tmp_path, True)
    arguments = _main_arguments(tmp_path, shipped, test_ops)
    assert smoke.main([*arguments, '--inputs', str(inputs)]) == 0
    assert built == ['smoke']


# --- T1.20: an ignored file never reaches a wheel or the suite ----------------

#: A file the commit holds although an ignore pattern matches it (added
#: with ``--force``): the export takes it, as the commit holds it.
FORCED = 'packages/structured-data-models-pql/src/predictive_query/kept.orig'

#: Each ignored file, how it is ignored, and the ignore rule: in the
#: committed ``.gitignore`` at the root, in a ``.gitignore`` the PQL
#: distribution commits, or in the checkout's own ``.git/info/exclude``.
#: The first four are under a sibling distribution the smoke builds into a
#: wheel; the last two beside the stub and the contract sources.
IGNORED: dict[str, tuple[str, str, str]] = {
    'a .py in the PQL distribution, by a pattern': (
        'packages/structured-data-models-pql/src/predictive_query/extra.py',
        '.gitignore',
        'extra.py',
    ),
    'a .py in an ignored directory of the API distribution': (
        'packages/structured-data-models-api/src/generated/api_extra.py',
        '.gitignore',
        'packages/structured-data-models-api/src/generated/',
    ),
    'a .py ignored by the PQL distribution s own .gitignore': (
        'packages/structured-data-models-pql/src/predictive_query/'
        'settings.local.py',
        'packages/structured-data-models-pql/.gitignore',
        '*.local.py',
    ),
    'a setup.py in the API distribution, by .git/info/exclude': (
        'packages/structured-data-models-api/setup.py',
        '.git/info/exclude',
        '/packages/structured-data-models-api/setup.py',
    ),
    'a copy beside the stub': (
        'packages/structured-data-models-client/src/sdm_client/_native.pyi.orig',
        '.gitignore',
        '*.orig',
    ),
    'a source beside the contract s': (
        'rust/crates/sdm-contract/src/extra.rs',
        '.gitignore',
        'rust/crates/sdm-contract/src/extra.rs',
    ),
}


def _ignoring_repository(
    root: pathlib.Path, ignored: str
) -> tuple[pathlib.Path, str, pathlib.Path]:
    """:func:`_repository`, with a tracked module beside each ignored file
    in the distributions, the ignore rule of `ignored` (committed, but for
    ``.git/info/exclude``), a file the commit holds although ``*.orig``
    ignores it, all committed; then the ignored file written, in the
    working tree only. Returns the checkout, its HEAD and the ignored
    file."""
    inputs, _ = _repository(root)
    path, rules, rule = IGNORED[ignored]
    for module in (
        'packages/structured-data-models-pql/src/predictive_query/__init__.py',
        'packages/structured-data-models-api/src/api.py',
    ):
        (inputs / module).parent.mkdir(parents=True, exist_ok=True)
        (inputs / module).write_text(f'# {module}, as committed\n')
    gitignore = inputs / '.gitignore'
    gitignore.write_text(gitignore.read_text() + '*.orig\n')
    target = inputs / rules
    target.parent.mkdir(parents=True, exist_ok=True)
    before = target.read_text() if target.is_file() else ''
    target.write_text(before + rule + '\n')
    _git(inputs, 'add', '-A')
    (inputs / FORCED).write_text('# held by the commit, though ignored\n')
    _git(inputs, 'add', '--force', FORCED)
    _git(inputs, 'commit', '-q', '-m', 'ignore rules')
    head = _git(inputs, 'rev-parse', 'HEAD')
    extra = inputs / path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text('raise SystemExit("not the commit s")\n')
    # Ignored: git sees it only when asked for ignored files.
    assert _git(inputs, 'status', '--porcelain', '--untracked-files=all') == ''
    assert path in _git(
        inputs, 'status', '--porcelain', '--ignored', '--untracked-files=all'
    )
    return inputs, head, extra


def _committed(repo: pathlib.Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ['git', '-C', str(repo), 'show', f'{commit}:{path}'],
        capture_output=True,
        check=True,
    ).stdout


def _held_as_committed(
    tree: pathlib.Path, repo: pathlib.Path, commit: str
) -> set[str]:
    """Asserts `tree` holds every file `commit` holds under :data:`TAKEN`
    and nothing `commit` does not hold, each file as `commit` holds it.
    Returns `tree`'s files."""
    files = {
        path.relative_to(tree).as_posix()
        for path in tree.rglob('*')
        if path.is_file() or path.is_symlink()
    }
    held = set(_git(repo, 'ls-tree', '-r', '--name-only', commit).splitlines())
    taken = set(
        _git(
            repo, 'ls-tree', '-r', '--name-only', commit, '--', *TAKEN
        ).splitlines()
    )
    assert FORCED in taken and INPUTS[0] in taken
    assert taken <= files, sorted(taken - files)
    assert files <= held, sorted(files - held)
    for name in files:
        assert (tree / name).read_bytes() == _committed(repo, commit, name), (
            name
        )
    return files


@pytest.mark.parametrize('ignored', list(IGNORED))
def test_an_ignored_file_passes_inputs_clean_and_is_never_exported(
    tmp_path: pathlib.Path, ignored: str
) -> None:
    """``inputs_clean`` leaves ignored files aside, so the export is what
    keeps them out: the tree ``exported`` makes holds every file the commit
    holds under the paths taken, as it holds them (an ignored file the
    commit holds among them), and not the ignored file."""
    inputs, head, extra = _ignoring_repository(tmp_path / 'inputs', ignored)
    smoke.inputs_clean(inputs, head)
    into = _scratch(tmp_path)
    tree = smoke.exported(inputs, head, into)
    assert tree == into
    files = _held_as_committed(tree, inputs, head)
    name = extra.relative_to(inputs).as_posix()
    assert name not in files
    assert not (tree / name).exists()
    assert not any(path.name == extra.name for path in tree.rglob('*'))


@pytest.mark.parametrize('ignored', list(IGNORED))
@pytest.mark.parametrize('container', [False, True])
def test_with_a_commit_the_smoke_runs_on_the_export_not_the_given_tree(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    container: bool,
    ignored: str,
) -> None:
    """Through ``main`` with ``--commit``, an ignored file in the inputs
    tree: the tree the run takes its inputs from (the one ``suite``
    receives directly, the one mounted at ``/inputs`` in the container) is
    not the given tree, and holds what the commit holds and not the
    ignored file. No environment is built: the direct run stops at its
    suite, the container's docker is never started."""
    inputs, head, extra = _ignoring_repository(tmp_path / 'inputs', ignored)
    name = extra.relative_to(inputs).as_posix()
    seen: list[pathlib.Path] = []

    def taken(tree: pathlib.Path) -> None:
        tree = tree.resolve()
        assert tree != inputs.resolve()
        assert not tree.is_relative_to(inputs.resolve())
        files = _held_as_committed(tree, inputs, head)
        assert name not in files
        assert not (tree / name).exists()
        seen.append(tree)

    def suite(
        own: pathlib.Path, given: pathlib.Path, into: pathlib.Path
    ) -> pathlib.Path:
        taken(given)
        raise smoke.SmokeFailed('stopped at the suite')

    def checked(command: list[str], **keywords: Any) -> str:
        raise smoke.SmokeFailed(f'no environment in this test: {command}')

    class Done:
        returncode = 0

    def run(command: list[str], **keywords: Any) -> Done:
        assert command[:1] == ['docker'], command
        mounts = [
            command[i + 1] for i, x in enumerate(command[:-1]) if x == '-v'
        ]
        at_inputs = [m for m in mounts if m.endswith(':/inputs:ro')]
        assert len(at_inputs) == 1, mounts
        taken(pathlib.Path(at_inputs[0][: -len(':/inputs:ro')]))
        return Done()

    monkeypatch.setattr(smoke, 'suite', suite)
    monkeypatch.setattr(smoke, '_checked', checked)
    monkeypatch.setattr(smoke, '_run', run)
    arguments = _commit_arguments(tmp_path, inputs, head, container)
    status = smoke.main(arguments)
    assert len(seen) == 1, seen
    assert status == (0 if container else 1)
    # The given tree is as it was: the ignored file still there, untouched.
    assert extra.read_text() == 'raise SystemExit("not the commit s")\n'
    # The tree the run took was a temporary one, gone with the run.
    assert not seen[0].exists()
