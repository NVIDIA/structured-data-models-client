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
   fails;
5. last, the missing-library control: the shipped environment's client
   reinstalled from a copy of the wheel without its native library must
   fail the probe, with an import error naming ``sdm_client._native``,
   and the binding's tests, or the lane fails.

``environment.json`` in the output directory records the lane, the image,
the interpreter, the wheels' SHA-256, the versions resolved, each report's
counts, and the warnings pyarrow's own ``to_pandas()`` raises in the lane
before the client is imported (register HOST-3's evidence, recorded, not
judged).

    scripts/rust_core/wheel_smoke.py --lane floor|newest \\
        --wheel SHIPPED.whl --test-ops-wheel TEST_OPS.whl --out DIR \\
        [--container | --python PATH]
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
#: The binding's test files each build runs: the file any build passes on
#: the shipped wheel, and every file on the test-ops one.
FILES = {
    'production': ('test_binding.py',),
    'test-ops': tuple(sorted(p.name for p in (REPO / TESTS).glob('test_*.py'))),
}
#: The sibling distributions the client pins, installed from the checkout.
SIBLINGS = ('structured-data-models-api', 'structured-data-models-pql')


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
        if not installed or pathlib.Path(path).is_relative_to(REPO):
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
    tests = _run(
        [py, '-I', '-B', '-m', 'pytest',
         *(str(REPO / TESTS / name) for name in files),
         '--rootdir', str(REPO), '-p', 'no:cacheprovider', '-q', '-rfEs',
         '--native-build', build, f'--junitxml={report}'],
        cwd=work,
    )  # fmt: skip
    counts = junit_verdict(report, files)
    if tests.returncode != 0:
        raise SmokeFailed(f'{build}: pytest exited {tests.returncode}')
    return {
        'wheel': {'name': wheel.name, 'sha256': sha256(wheel)},
        'probe': probe,
        'freeze': f'freeze-{build}.txt',
        'report': report.name,
        'tests': counts,
        'third_party': third_party(freeze),
    }


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
    lane: Lane, wheel: pathlib.Path, work: pathlib.Path, out: pathlib.Path
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
    tests = _run(
        [py, '-I', '-B', '-m', 'pytest',
         *(str(REPO / TESTS / name) for name in files),
         '--rootdir', str(REPO), '-p', 'no:cacheprovider', '-q',
         '--native-build', 'production', f'--junitxml={report}'],
        cwd=work,
        capture_output=True,
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
) -> int:
    lane = LANES[lane_name]
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        'lane': lane_name,
        'image': image,
        'pins': list(lane.pins),
        'builds': {},
    }
    status = 0
    with tempfile.TemporaryDirectory(prefix='wheel-smoke-') as temporary:
        work = pathlib.Path(temporary)
        for name in SIBLINGS:
            shutil.copytree(
                REPO / 'packages' / name,
                work / 'src' / name,
                ignore=shutil.ignore_patterns(
                    'build', '*.egg-info', '__pycache__', 'tests'
                ),
            )
        constraints = work / 'constraints.txt'
        constraints.write_text('\n'.join(lane.pins) + '\n')
        try:
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
                    lane, build, wheels[build], python, work, out, constraints
                )
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
                lane, wheels['production'], work, out
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
    lane_name: str, wheels: dict[str, pathlib.Path], out: pathlib.Path
) -> int:
    """Runs this script in the lane's image, the checkout and the wheels
    mounted read-only, as the caller's user, writing only to `out`."""
    lane = LANES[lane_name]
    platforms = {_docker_platform(w) for w in wheels.values()}
    if len(platforms) != 1:
        raise SmokeFailed(f'the wheels are for {sorted(platforms)}')
    out.mkdir(parents=True, exist_ok=True)
    mounts = ['-v', f'{REPO}:/repo:ro', '-v', f'{out.resolve()}:/out']
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
        '--image', lane.image,
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
    # Set by --container for the run inside, which records it.
    parser.add_argument('--image', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    wheels = {'production': args.wheel, 'test-ops': args.test_ops_wheel}
    for build, wheel in wheels.items():
        if not wheel.is_file() or wheel.suffix != '.whl':
            parser.error(f'{build}: {wheel} is not a wheel')
    try:
        if args.container:
            return in_container(args.lane, wheels, args.out)
        return smoke(args.lane, wheels, args.out, args.python, args.image)
    except SmokeFailed as failure:
        print(f'wheel-smoke ({args.lane}): FAILED: {failure}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
