#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The client wheel as a user installs it, in a clean environment (T1.17).

CI's ``wheel-smoke`` job runs this once per lane, with the wheels the
``native-wheel`` and ``native-test-ops`` jobs built, never rebuilt and
never the source tree. Each lane is a pinned runtime of decision E16:

- ``floor``: Python 3.10 with pyarrow 14.0.1, pandas 2.2.0 and NumPy
  1.26.4, the floors every distribution declares;
- ``newest``: Python 3.14 with pandas 3, NumPy 2 and pyarrow at the newest
  releases those admit, resolved afresh on every run (no upper bound, a new
  environment, pip's newest).

With ``--container`` it runs itself in the lane's image, the official
Python image of the lane's version pinned by digest, which holds the
interpreter, pip and a C toolchain and no Python package, with the checkout
mounted read-only. Without, it runs in the interpreter ``--python`` names,
which must be the lane's Python, so a reviewer can run a lane against any
wheel of their own (a macOS build, in a local Python 3.10, say). Either way:

1. the API and PQL distributions are copied out of the checkout and built
   into wheels, so every distribution installed is a wheel
   (``--only-binary=:all:``; the toolchain builds nothing installed);
2. for each client wheel, the shipped one and the test-ops one, a virtual
   environment of its own installs the lane's pins, as requirements and
   constraints, pytest, the two sibling wheels and the client wheel; the
   test-ops environment is also constrained to everything the shipped one
   resolved, and the two must resolve alike;
3. it checks that ``sdm_client`` and ``sdm_client._native`` are imported
   from the environment, never the checkout, that ``_native`` is the build
   the wheel says (``TEST_OPS``), that ``pip check`` passes and that the
   lane's pins are what resolved (the floor's exactly; the newest lane's
   pandas 3 and NumPy 2);
4. it records ``pip freeze --all`` and runs the binding's tests from
   outside the checkout into a JUnit report: ``test_binding.py`` on the
   shipped wheel, and every file of ``rust/bindings/sdm-py/tests`` on the
   test-ops one (``test_arrow_lifetimes.py``: buffers shared and released
   exactly, admitted types, outputs after collection, error and
   destruction, busy and cancel); each file must report a passed test and
   nothing failed, errored or skipped, and a missing or unreadable report
   fails; pytest runs with no ``PYTEST_*`` variable, plugin autoload or
   configuration of the caller's, and a plugin of the smoke's records
   what was collected and deselected and where the test process imported
   ``sdm_client`` and its native module from: any deselection fails, as
   do a test of ``EXPECTED`` not collected, a report whose passed tests are
   not every test collected, and an import from anywhere but the
   environment;
5. last, the missing-library control: the shipped environment's client
   reinstalled from a copy of the wheel without its native library must
   fail the probe, with an import error naming ``sdm_client._native``,
   and the binding's tests, or the lane fails.

The binding's tests judge; what they read as the contract, its inputs,
comes from ``--inputs TREE`` (the checkout itself by default, as in CI's
own job): the API and PQL distributions built into wheels, the message
builder (``messages.py``), the contract's version and error codes
(``sdm-contract``'s ``lib.rs`` and ``error.rs``) and the native module's
stub. The gate's locked check 3 runs the checkpoint's copy of this script
on the candidate's wheels with ``--inputs`` the candidate's tree, so the
checkpoint's tests, ``EXPECTED``, guard and control judge a candidate at
the contract version it builds (T1.20, Codex P2): :func:`suite` puts them
together. ``--commit SHA`` holds each wheel to the identity
``native_wheel.py`` wrote beside it (:func:`provenance`): built from SHA,
from a clean tree, the file it describes, and the build it names, so a
wheel of another commit cannot stand in for the candidate's.

``environment.json`` in the output directory records the lane, the image,
the interpreter, the wheels' SHA-256, the versions resolved, each report's
counts, and the warnings pyarrow's own ``to_pandas()`` raises in the lane
before the client is imported (register HOST-3's evidence, recorded, not
judged).

    scripts/rust_core/wheel_smoke.py --lane floor|newest \\
        --wheel SHIPPED.whl --test-ops-wheel TEST_OPS.whl --out DIR \\
        [--inputs TREE] [--commit SHA] [--container | --python PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
import zipfile
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[2]
TESTS = 'rust/bindings/sdm-py/tests'
#: Every test of the binding's test files, by its name in the file (with
#: its parameters): a run must collect and pass each one, so a test removed
#: by any selection, an option, a configuration or a conftest hook, fails
#: the lane. ``tests/test_wheel_smoke.py`` holds this to the files' own test
#: functions, so a test added there is added here.
EXPECTED: dict[str, tuple[str, ...]] = {
    'test_arrow_lifetimes.py': (
        'test_the_canonical_set_is_every_type_of_the_closed_set',
        'test_canonical_inputs_cross_sharing_their_buffers',
        'test_the_run_holds_the_producer_s_memory_and_releases_it_exactly',
        'test_an_output_stays_valid_after_its_input_its_run_s_error_and_its_run',
        'test_admitted_types_match_the_expected_canonical_schema',
        'test_a_cancel_releases_what_the_run_held_and_refuses_what_follows',
        'test_a_cancel_drops_an_output_the_run_still_held',
        'test_a_cancel_while_another_call_is_in_the_run_is_busy',
    ),
    'test_binding.py': (
        'test_the_build_is_the_one_named',
        'test_the_stub_describes_the_module',
        'test_the_contract_version_is_the_contract_crate_s',
        'test_the_error_codes_are_the_contract_s_in_its_order',
        'test_a_query_is_validated_through_a_run',
        'test_an_error_the_run_returns_is_raised_as_core_error',
        'test_a_refused_start_quotes_nothing_it_was_given',
        'test_an_attachment_is_bytes_or_arrow',
        'test_an_object_with_only_pyarrow_s_private_export_is_refused',
        'test_a_start_with_an_unnamed_failing_stream_is_an_invalid_message',
        'test_only_the_test_ops_build_takes_a_test_operation',
        'test_importing_the_module_changes_nothing_in_the_process',
    ),
    'test_test_ops.py': (
        'test_another_thread_runs_and_logs_while_the_core_holds_a_call',
        'test_a_second_call_while_one_is_in_the_run_is_refused_as_busy',
        'test_a_handler_that_raises_is_reported_and_the_answer_delivered',
        'test_an_interrupt_in_a_handler_is_raised_and_the_step_is_gone[KeyboardInterrupt]',
        'test_an_interrupt_in_a_handler_is_raised_and_the_step_is_gone[SystemExit]',
        'test_a_handler_that_calls_into_its_own_run_is_refused_as_busy',
        'test_a_forced_abort_is_reported_and_never_recovered',
        'test_a_panic_is_contained_and_quotes_nothing',
        'test_an_error_past_the_cap_is_raised_whole_from_its_attachment',
        'test_data_that_does_not_match_its_request_is_refused_by_check_b',
        'test_a_stream_that_breaks_its_own_schema_fails_check_a',
        'test_a_producer_that_fails_is_the_provider_s_error',
        'test_an_unreadable_reply_s_attachments_are_checked_before_the_run_changes',
        'test_the_data_asked_for_is_taken_and_comes_back_out',
        'test_a_stream_is_exported_only_in_its_own_schema',
    ),
}
#: The binding's test files each build runs: the file any build passes on
#: the shipped wheel, and every file on the test-ops one.
FILES = {
    'production': ('test_binding.py',),
    'test-ops': tuple(sorted(EXPECTED)),
}
#: The modules whose origin the guard records inside the test process.
ORIGINS = ('sdm_client', 'sdm_client._native')
#: The sibling distributions the client pins, installed from the inputs.
SIBLINGS = ('structured-data-models-api', 'structured-data-models-pql')
#: What the binding's tests take from the tree under test, the inputs: the
#: message builder, and the files they read as the contract (its version,
#: its error codes) and as the native module's surface (its stub).
INPUTS = (
    f'{TESTS}/messages.py',
    'rust/crates/sdm-contract/src/lib.rs',
    'rust/crates/sdm-contract/src/error.rs',
    'packages/structured-data-models-client/src/sdm_client/_native.pyi',
)


@dataclasses.dataclass(frozen=True)
class Lane:
    python: str
    #: A Python image pinned by its index digest, from the public mirror of
    #: Docker's official images (Docker Hub's anonymous pull limit failed
    #: CI's pulls before).
    image: str
    #: What the lane installs as requirements and constraints.
    pins: tuple[str, ...]
    #: Each pinned distribution's version as it must resolve: exact, or
    #: its major version (`3.`), compared as a prefix.
    resolved: dict[str, str]


LANES = {
    'floor': Lane(
        python='3.10',
        image='public.ecr.aws/docker/library/python:3.10.22-trixie'
        '@sha256:c10791e6f69086d5216f7b5d3b839929c454e24305867ed9bc5152f0355b5423',
        pins=('pyarrow==14.0.1', 'pandas==2.2.0', 'numpy==1.26.4'),
        resolved={'pyarrow': '14.0.1', 'pandas': '2.2.0', 'numpy': '1.26.4'},
    ),
    'newest': Lane(
        python='3.14',
        image='public.ecr.aws/docker/library/python:3.14.8-trixie'
        '@sha256:446efb4225200a0a196707aa7953898b37962fd3b8859a636035de3af657081e',
        pins=('pandas>=3', 'numpy>=2', 'pyarrow'),
        resolved={'pandas': '3.', 'numpy': '2.'},
    ),
}

#: Run in each environment with -I: where the client was imported from,
#: which build it is, and what resolved. First, before anything of the
#: client is imported, the warnings pyarrow's own `to_pandas()` raises in
#: the lane, recorded and not judged: register HOST-3's evidence.
_PROBE = """
import importlib.metadata, json, pathlib, platform, sys, sysconfig, warnings
import pyarrow
with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter('always')
    pyarrow.table({'a': [1, None]}).to_pandas()
to_pandas = [f'{w.category.__name__}: {w.message}' for w in seen]
assert not any(name.startswith('sdm_') for name in sys.modules)
import sdm_client, sdm_client._native as native
site = {pathlib.Path(sysconfig.get_paths()[k]).resolve() for k in ('purelib', 'platlib')}
def where(module):
    path = pathlib.Path(module.__file__).resolve()
    return str(path), any(root in path.parents for root in site)
print(json.dumps({
    'python': platform.python_version(),
    'platform': platform.platform(),
    'libc': ' '.join(platform.libc_ver()).strip(),
    'client': where(sdm_client),
    'native': where(native),
    'test_ops': native.TEST_OPS,
    'to_pandas_warnings': to_pandas,
    'versions': {
        name: importlib.metadata.version(name)
        for name in ('pandas', 'numpy', 'pyarrow', 'pytest')
    },
}))
"""


class SmokeFailed(Exception):
    pass


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    shown = (word if '\n' not in word else '<script>' for word in command)
    print('+', ' '.join(shown), flush=True)
    return subprocess.run(command, text=True, **kwargs)


def _checked(command: list[str], **kwargs: Any) -> str:
    done = _run(command, capture_output=True, **kwargs)
    if done.returncode != 0:
        raise SmokeFailed(
            f'{command[0]} {" ".join(command[1:4])} ... failed '
            f'({done.returncode}):\n{done.stdout}\n{done.stderr}'
        )
    return done.stdout


def suite(
    own: pathlib.Path, inputs: pathlib.Path, into: pathlib.Path
) -> pathlib.Path:
    """The binding's tests as this smoke runs them, under `into`, laid out
    as the repository is: every file of `own`'s tests directory but the
    message builder (the test files, ``conftest.py``, the shim), which
    judge, and `inputs`' :data:`INPUTS` at their own paths, which say what
    the build under test is. Returns the tests directory. An input
    missing from `inputs` is refused by name."""
    missing = [name for name in INPUTS if not (inputs / name).is_file()]
    if missing:
        raise SmokeFailed(f'the inputs {inputs} lack {missing}')
    tests = into / TESTS
    tests.mkdir(parents=True, exist_ok=True)
    for path in sorted((own / TESTS).iterdir()):
        if path.is_file() and path.name != 'messages.py':
            shutil.copyfile(path, tests / path.name)
    for name in INPUTS:
        (into / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / name, into / name)
    return tests


def provenance(wheel: pathlib.Path, commit: str, test_ops: bool) -> None:
    """Holds `wheel` to its identity, ``<wheel>.json`` beside it, as
    ``native_wheel.py`` writes it: built from `commit` (in full), from a
    clean tree, its SHA-256 the file's own, and the build asked for
    (``features`` ``["test-ops"]`` for the test-ops wheel, none for the
    shipped one); refuses anything else by name."""
    identity = wheel.with_name(wheel.name + '.json')
    try:
        found = json.loads(identity.read_text())
    except (OSError, ValueError, RecursionError) as error:
        raise SmokeFailed(
            f'{wheel.name}: no readable identity ({error})'
        ) from error
    if not isinstance(found, dict):
        raise SmokeFailed(f'{wheel.name}: the identity is not an object')
    want = {
        'commit': commit,
        'tree_clean': True,
        'sha256': sha256(wheel),
        'features': ['test-ops'] if test_ops else [],
    }
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise SmokeFailed(f'--commit {commit!r} is not a full SHA')
    for field, value in want.items():
        if found.get(field) != value:
            raise SmokeFailed(
                f'{wheel.name}: identity {field} is {found.get(field)!r}, '
                f'not {value!r}'
            )


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def junit_verdict(
    report: pathlib.Path, files: tuple[str, ...]
) -> dict[str, Any]:
    """Each file's passed, failed, errored and skipped tests in a JUnit
    report, refused unless every file passed one or more and nothing
    failed, errored or was skipped. A report missing, unreadable or naming
    no test fails, so a run that collected nothing never passes."""
    try:
        root = ElementTree.parse(report).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise SmokeFailed(
            f'{report.name}: no readable JUnit report ({error})'
        ) from error
    counts = {
        name: {'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0}
        for name in files
    }
    for case in root.iter('testcase'):
        module = case.get('classname', '').rsplit('.', 1)[-1]
        name = f'{module}.py'
        if name not in counts:
            raise SmokeFailed(f'{report.name}: a test of {name}, not asked for')
        kinds = {child.tag for child in case}
        if 'failure' in kinds:
            counts[name]['failed'] += 1
        elif 'error' in kinds:
            counts[name]['errors'] += 1
        elif 'skipped' in kinds:
            counts[name]['skipped'] += 1
        else:
            counts[name]['passed'] += 1
    for name, count in counts.items():
        if count['passed'] == 0:
            raise SmokeFailed(f'{report.name}: {name} passed no test')
        bad = {k: v for k, v in count.items() if k != 'passed' and v}
        if bad:
            raise SmokeFailed(f'{report.name}: {name} {bad}')
    return counts


def check_resolved(lane: Lane, versions: dict[str, str]) -> None:
    """Refuses an environment whose pinned distributions resolved to
    other versions than the lane names; a missing one is refused too."""
    for name, wanted in lane.resolved.items():
        found = versions.get(name)
        ok = found is not None and (
            found.startswith(wanted)
            if wanted.endswith('.')
            else found == wanted
        )
        if not ok:
            raise SmokeFailed(f'{name} resolved to {found}, not {wanted}')


#: This repository's distributions, which a freeze names by their files.
OWN = {
    'structured-data-models-client',
    'structured-data-models-api',
    'structured-data-models-pql',
    'structured-data-models-connectors',
    'structured-data-models-relational-engine',
}


def third_party(freeze: str) -> list[str]:
    """The freeze's lines but this repository's own distributions."""
    return sorted(
        line
        for line in freeze.splitlines()
        if line.strip()
        and not line.startswith('-')
        and re.split(r'[=@ ]', line, maxsplit=1)[0].strip() not in OWN
    )


def _environment(
    lane: Lane,
    build: str,
    wheel: pathlib.Path,
    python: str,
    work: pathlib.Path,
    out: pathlib.Path,
    constraints: pathlib.Path,
    inputs: pathlib.Path,
    tests_root: pathlib.Path,
) -> dict[str, Any]:
    env_dir = work / build
    _checked([python, '-m', 'venv', str(env_dir)])
    py = str(env_dir / 'bin' / 'python')
    pip = [py, '-m', 'pip', '--disable-pip-version-check', '--no-input']
    siblings = sorted(str(p) for p in (work / 'wheels').glob('*.whl'))
    if len(siblings) != len(SIBLINGS):
        raise SmokeFailed(f'sibling wheels: {siblings}')
    _checked(
        [*pip, 'install', '--quiet', '--only-binary=:all:',
         '-c', str(constraints), *lane.pins, 'pytest', *siblings, str(wheel)],
        cwd=work,
    )  # fmt: skip
    probe = json.loads(_checked([py, '-I', '-c', _PROBE], cwd=work))
    for module in ('client', 'native'):
        path, installed = probe[module]
        if not installed or any(
            pathlib.Path(path).is_relative_to(tree) for tree in (REPO, inputs)
        ):
            raise SmokeFailed(f'{build}: {module} imported from {path}')
    if probe['test_ops'] != (build == 'test-ops'):
        raise SmokeFailed(f'{build}: TEST_OPS is {probe["test_ops"]}')
    if not probe['python'].startswith(f'{lane.python}.'):
        raise SmokeFailed(f'{build}: Python {probe["python"]}')
    check_resolved(lane, probe['versions'])
    _checked([*pip, 'check'], cwd=work)
    freeze = _checked([*pip, 'freeze', '--all'], cwd=work)
    (out / f'freeze-{build}.txt').write_text(freeze)
    files = FILES['production' if build == 'production' else 'test-ops']
    report = out / f'junit-{build}.xml'
    report.unlink(missing_ok=True)
    collection = out / f'collection-{build}.json'
    tests = _pytest(py, work, files, build, report, collection, tests_root)
    counts = junit_verdict(report, files)
    collection_verdict(collection, counts, env_dir, (inputs, tests_root))
    if tests.returncode != 0:
        raise SmokeFailed(f'{build}: pytest exited {tests.returncode}')
    return {
        'wheel': {'name': wheel.name, 'sha256': sha256(wheel)},
        'probe': probe,
        'freeze': f'freeze-{build}.txt',
        'report': report.name,
        'collection': collection.name,
        'tests': counts,
        'third_party': third_party(freeze),
    }


#: The pytest plugin every run of the binding's tests loads: it writes
#: each test collected and each test deselected to the file
#: ``SDM_SMOKE_COLLECTION`` names, so a test dropped by any selection, from
#: an option, a configuration or a plugin, is seen (JUnit names none).
_GUARD = """
import json, os

_items, _deselected = [], []


def pytest_deselected(items):
    _deselected.extend(item.nodeid for item in items)


def pytest_collection_finish(session):
    _items[:] = [item.nodeid for item in session.items]


def pytest_sessionfinish(session):
    import sys
    origins = {
        name: getattr(sys.modules.get(name), '__file__', None)
        for name in ('sdm_client', 'sdm_client._native')
    }
    with open(os.environ['SDM_SMOKE_COLLECTION'], 'w') as out:
        json.dump(
            {'items': _items, 'deselected': _deselected, 'origins': origins},
            out,
        )
"""

#: Runs pytest with the guard's directory first on the path, under -I.
_PYTEST = (
    'import sys; sys.path.insert(0, sys.argv.pop(1)); import pytest; '
    'sys.exit(pytest.main(sys.argv[1:]))'
)


def _pytest(
    py: str,
    work: pathlib.Path,
    files: tuple[str, ...],
    build: str,
    report: pathlib.Path,
    collection: pathlib.Path,
    root: pathlib.Path = REPO,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """The binding's tests in `files`, with nothing the caller's
    environment or a configuration file could add: every ``PYTEST_*``
    variable dropped (``PYTEST_ADDOPTS`` among them), plugins loaded only
    by name, an empty configuration file of the smoke's own, and conftest
    files read only from the tests' own directory; the guard plugin
    records what was collected and deselected in `collection`. `root` is
    the tree the tests run from (:func:`suite`'s)."""
    guard = work / 'guard'
    guard.mkdir(exist_ok=True)
    (guard / 'smoke_guard.py').write_text(_GUARD)
    (guard / 'pytest.ini').write_text('[pytest]\n')
    report.unlink(missing_ok=True)
    collection.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTEST_')}
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    env['SDM_SMOKE_COLLECTION'] = str(collection)
    return _run(
        [py, '-I', '-B', '-c', _PYTEST, str(guard),
         *(str(root / TESTS / name) for name in files),
         '-c', str(guard / 'pytest.ini'), '--rootdir', str(root),
         '--confcutdir', str(root / TESTS),
         '-p', 'smoke_guard', '-p', 'no:cacheprovider', '-q', '-rfEs',
         '--native-build', build, f'--junitxml={report}'],
        cwd=work,
        env=env,
        **kwargs,
    )  # fmt: skip


def collection_verdict(
    collection: pathlib.Path,
    counts: dict[str, dict[str, int]],
    environment: pathlib.Path,
    outside: tuple[pathlib.Path, ...] = (),
) -> None:
    """Refuses a run that deselected any test, that collected none of a
    test `EXPECTED` names for a file it ran, or whose report's passed tests
    are not, file by file, every test collected: so no test is dropped
    unseen, whatever selected it. Refuses too a run whose test process
    imported ``sdm_client`` or its native module from anywhere but
    `environment` (the checkout, and each tree of `outside`, included),
    and a record missing or unreadable."""
    try:
        record = json.loads(collection.read_text())
        items, deselected = record['items'], record['deselected']
        origins = record['origins']
    except (OSError, ValueError, RecursionError, KeyError, TypeError) as error:
        raise SmokeFailed(
            f'{collection.name}: no readable record ({error})'
        ) from error
    if not isinstance(items, list) or not isinstance(deselected, list):
        raise SmokeFailed(f'{collection.name}: not a record of tests')
    if deselected:
        raise SmokeFailed(f'{collection.name}: deselected {deselected}')
    collected: dict[str, int] = {}
    for nodeid in items:
        name = str(nodeid).split('::', 1)[0].rsplit('/', 1)[-1]
        collected[name] = collected.get(name, 0) + 1
    passed = {name: count['passed'] for name, count in counts.items()}
    if not items or collected != passed:
        raise SmokeFailed(
            f'{collection.name}: collected {collected}, passed {passed}'
        )
    names = {str(nodeid).rsplit('/', 1)[-1] for nodeid in items}
    missing = sorted(
        f'{name}::{test}'
        for name in counts
        for test in EXPECTED.get(name, ('(no record of its tests)',))
        if f'{name}::{test}' not in names
    )
    if missing:
        raise SmokeFailed(f'{collection.name}: not run: {missing}')
    if not isinstance(origins, dict):
        raise SmokeFailed(f'{collection.name}: no record of origins')
    home = environment.resolve()
    for module in ORIGINS:
        path = origins.get(module)
        where = pathlib.Path(str(path)).resolve() if path else None
        if (
            where is None
            or home not in where.parents
            or any(where.is_relative_to(tree) for tree in (REPO, *outside))
        ):
            raise SmokeFailed(
                f'{collection.name}: the tests imported {module} from {path}'
            )


def without_native(wheel: pathlib.Path, into: pathlib.Path) -> pathlib.Path:
    """A copy of `wheel` with the native library left out, as a broken
    build would ship it: every member but ``sdm_client/_native``'s
    extension module (its stub stays)."""
    into.mkdir(parents=True, exist_ok=True)
    doctored = into / wheel.name
    dropped = []
    with (
        zipfile.ZipFile(wheel) as source,
        zipfile.ZipFile(doctored, 'w', zipfile.ZIP_DEFLATED) as copy,
    ):
        for member in source.infolist():
            name = member.filename
            if name.startswith('sdm_client/_native.') and not name.endswith(
                '.pyi'
            ):
                dropped.append(name)
                continue
            copy.writestr(member, source.read(name))
    if len(dropped) != 1:
        raise SmokeFailed(f'{wheel.name}: native libraries {dropped}')
    return doctored


def missing_library_control(
    lane: Lane,
    wheel: pathlib.Path,
    work: pathlib.Path,
    out: pathlib.Path,
    tests_root: pathlib.Path = REPO,
) -> dict[str, Any]:
    """The control: the shipped environment, its client reinstalled from
    a copy of the wheel without its native library, must fail the probe
    and the binding's tests, or the smoke could pass with no core at all."""
    py = str(work / 'production' / 'bin' / 'python')
    doctored = without_native(wheel, work / 'control')
    _checked(
        [py, '-m', 'pip', '--disable-pip-version-check', '--no-input',
         'install', '--quiet', '--no-deps', '--force-reinstall',
         str(doctored)],
        cwd=work,
    )  # fmt: skip
    probe = _run([py, '-I', '-c', _PROBE], cwd=work, capture_output=True)
    report = out / 'junit-control.xml'
    report.unlink(missing_ok=True)
    files = FILES['production']
    tests = _pytest(
        py, work, files, 'production', report,
        out / 'collection-control.json', tests_root, capture_output=True,
    )  # fmt: skip
    try:
        junit_verdict(report, files)
        refused = False
    except SmokeFailed:
        refused = True
    found = {
        'probe_exit': probe.returncode,
        'probe_import_error': 'sdm_client._native' in probe.stderr,
        'tests_exit': tests.returncode,
        'tests_refused': refused,
    }
    if not (
        probe.returncode != 0
        and found['probe_import_error']
        and tests.returncode != 0
        and refused
    ):
        raise SmokeFailed(f'the missing-library control passed: {found}')
    return found


def smoke(
    lane_name: str,
    wheels: dict[str, pathlib.Path],
    out: pathlib.Path,
    python: str,
    image: str | None,
    inputs: pathlib.Path = REPO,
) -> int:
    lane = LANES[lane_name]
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        'lane': lane_name,
        'image': image,
        'pins': list(lane.pins),
        'inputs': str(inputs),
        'builds': {},
    }
    status = 0
    with tempfile.TemporaryDirectory(prefix='wheel-smoke-') as temporary:
        work = pathlib.Path(temporary)
        for name in SIBLINGS:
            shutil.copytree(
                inputs / 'packages' / name,
                work / 'src' / name,
                ignore=shutil.ignore_patterns(
                    'build', '*.egg-info', '__pycache__', 'tests'
                ),
            )
        constraints = work / 'constraints.txt'
        constraints.write_text('\n'.join(lane.pins) + '\n')
        try:
            tests_root = work / 'suite'
            suite(REPO, inputs, tests_root)
            # The siblings built once, each a wheel, so every distribution
            # an environment installs is a wheel (--only-binary=:all:).
            _checked([python, '-m', 'venv', str(work / 'build')])
            _checked(
                [str(work / 'build' / 'bin' / 'python'), '-m', 'pip',
                 '--disable-pip-version-check',
                 '--no-input', 'wheel', '--quiet', '--no-deps',
                 '--wheel-dir', str(work / 'wheels'),
                 *(str(work / 'src' / name) for name in SIBLINGS)],
                cwd=work,
            )  # fmt: skip
            for build in ('production', 'test-ops'):
                record['builds'][build] = _environment(
                    lane, build, wheels[build], python, work, out, constraints,
                    inputs, tests_root,
                )  # fmt: skip
                # The test-ops environment holds what the shipped one
                # resolved, so both lanes' wheels meet the same versions.
                constraints.write_text(
                    '\n'.join(
                        [*lane.pins, *record['builds'][build]['third_party']]
                    )
                    + '\n'
                )
            shipped, test_ops = (
                record['builds'][b].pop('third_party')
                for b in ('production', 'test-ops')
            )
            if shipped != test_ops:
                raise SmokeFailed(
                    'the two environments resolved differently: '
                    f'{sorted(set(shipped) ^ set(test_ops))}'
                )
            # Last, since it breaks the shipped environment.
            record['missing_library_control'] = missing_library_control(
                lane, wheels['production'], work, out, tests_root
            )
        except SmokeFailed as failure:
            record['failure'] = str(failure)
            print(f'wheel-smoke ({lane_name}): FAILED: {failure}', flush=True)
            status = 1
    for build in record['builds'].values():
        build.pop('third_party', None)
    (out / 'environment.json').write_text(json.dumps(record, indent=2) + '\n')
    if status == 0:
        versions = record['builds']['production']['probe']['versions']
        print(
            f'wheel-smoke ({lane_name}): passed on Python '
            f'{record["builds"]["production"]["probe"]["python"]} with '
            + ', '.join(f'{k} {v}' for k, v in sorted(versions.items()))
        )
    return status


def _docker_platform(wheel: pathlib.Path) -> str:
    """The container platform a Linux wheel's tag names."""
    tag = wheel.name.rsplit('-', 1)[-1]
    for arch, name in (('x86_64', 'linux/amd64'), ('aarch64', 'linux/arm64')):
        if arch in tag and 'linux' in tag:
            return name
    raise SmokeFailed(f'{wheel.name} is no Linux wheel a container runs')


def in_container(
    lane_name: str,
    wheels: dict[str, pathlib.Path],
    out: pathlib.Path,
    inputs: pathlib.Path = REPO,
) -> int:
    """Runs this script in the lane's image, the checkout, the inputs and
    the wheels mounted read-only, as the caller's user, writing only to
    `out`."""
    lane = LANES[lane_name]
    platforms = {_docker_platform(w) for w in wheels.values()}
    if len(platforms) != 1:
        raise SmokeFailed(f'the wheels are for {sorted(platforms)}')
    out.mkdir(parents=True, exist_ok=True)
    mounts = [
        '-v', f'{REPO}:/repo:ro',
        '-v', f'{inputs.resolve()}:/inputs:ro',
        '-v', f'{out.resolve()}:/out',
    ]  # fmt: skip
    inside = {}
    for build, wheel in wheels.items():
        mounts += ['-v', f'{wheel.resolve()}:/wheels/{build}/{wheel.name}:ro']
        inside[build] = f'/wheels/{build}/{wheel.name}'
    command = [
        'docker', 'run', '--rm', '--platform', platforms.pop(),
        '--user', f'{os.getuid()}:{os.getgid()}',
        '-e', 'HOME=/tmp/home',
        *mounts, '-w', '/tmp', lane.image,
        'python', '/repo/scripts/rust_core/wheel_smoke.py',
        '--lane', lane_name, '--wheel', inside['production'],
        '--test-ops-wheel', inside['test-ops'], '--out', '/out',
        '--image', lane.image, '--inputs', '/inputs',
    ]  # fmt: skip
    return _run(command).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n', 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--lane', choices=sorted(LANES), required=True)
    parser.add_argument('--wheel', type=pathlib.Path, required=True)
    parser.add_argument('--test-ops-wheel', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    where = parser.add_mutually_exclusive_group()
    where.add_argument('--container', action='store_true')
    where.add_argument('--python', default=sys.executable)
    # The tree whose contract the wheels build (the candidate's, under the
    # gate); by default this checkout.
    parser.add_argument('--inputs', type=pathlib.Path, default=REPO)
    # The commit the wheels must be built from, by their identity files.
    parser.add_argument('--commit')
    # Set by --container for the run inside, which records it.
    parser.add_argument('--image', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    wheels = {'production': args.wheel, 'test-ops': args.test_ops_wheel}
    for build, wheel in wheels.items():
        if not wheel.is_file() or wheel.suffix != '.whl':
            parser.error(f'{build}: {wheel} is not a wheel')
    if not args.inputs.is_dir():
        parser.error(f'--inputs {args.inputs} is not a directory')
    try:
        if args.commit is not None:
            for build, wheel in wheels.items():
                provenance(wheel, args.commit, build == 'test-ops')
        if args.container:
            return in_container(args.lane, wheels, args.out, args.inputs)
        return smoke(
            args.lane, wheels, args.out, args.python, args.image, args.inputs
        )
    except SmokeFailed as failure:
        print(f'wheel-smoke ({args.lane}): FAILED: {failure}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
