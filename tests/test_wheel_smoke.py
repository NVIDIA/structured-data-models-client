# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/rust_core/wheel_smoke.py``'s own rules (T1.17): its verdict on
a test report and on what resolved fails closed, its lanes are decision
E16's, and CI runs both of them.

The smoke itself runs in CI's ``wheel-smoke`` job, against the wheels."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import zipfile

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
