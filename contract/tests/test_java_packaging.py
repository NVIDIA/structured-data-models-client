# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Java packaging, native loading and binding boundaries.

Set SDM_JAVA_JAR and JAVA_HOME to exercise packaging; SDM_REQUIRE_JAVA refuses a missing JAR.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]

CASES = ROOT / 'contract/cases'

S0 = CASES / 's0'

VECTORS = ROOT / 'contract/native-binding-vectors.json'

RUN_CLASS = 'com.nvidia.sdm.internal.Run'

FLAGS = (
    '--enable-native-access=ALL-UNNAMED',
    '--add-opens=java.base/java.nio=ALL-UNNAMED',
)

JDKS = (17, 21)

PLATFORM_TOKENS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    'linux-x86_64': (('linux',), ('x86_64', 'amd64', 'x86-64'), '.so'),
    'linux-aarch64': (('linux',), ('aarch64', 'arm64'), '.so'),
    'macos-arm64': (('macos', 'darwin', 'osx', 'mac'), ('arm64', 'aarch64'), '.dylib'),
    'macos-x86_64': (('macos', 'darwin', 'osx', 'mac'), ('x86_64', 'amd64', 'x86-64'), '.dylib'),
}  # fmt: skip

JAVA_17_MAJOR = 61


def _java_job() -> bool:
    r"""Whether this job has a JAR to test (``SDM_JAVA_JAR``); a job that
    says it must (``SDM_REQUIRE_JAVA``) fails without one."""
    if os.environ.get('SDM_JAVA_JAR'):
        return True
    assert not os.environ.get('SDM_REQUIRE_JAVA'), (
        'SDM_REQUIRE_JAVA is set but SDM_JAVA_JAR names no JAR'
    )
    return False


def _jar() -> Path:
    given = os.environ.get('SDM_JAVA_JAR', '')
    assert given, 'SDM_JAVA_JAR names no JAR'
    return Path(given).resolve()


def _java_homes() -> list[Path]:
    given = os.environ.get('SDM_JAVA_HOMES') or os.environ.get('JAVA_HOME', '')
    homes = [Path(each) for each in given.split(os.pathsep) if each]
    assert homes, 'neither SDM_JAVA_HOMES nor JAVA_HOME names a JDK'
    return homes


def _major(home: Path) -> int:
    done = subprocess.run(
        [str(home / 'bin/java'), '-XshowSettings:properties', '-version'],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    found = re.search(r'java\.specification\.version = (\d+)', done.stderr)
    assert found, done.stderr[-2000:]
    return int(found.group(1))


def _cases(directory: Path) -> dict[str, dict[str, Any]]:
    found = {}
    for path in sorted(directory.glob('*.json')):
        case = json.loads(path.read_text(encoding='utf-8'))
        found[case['id']] = case
    return found


def _here() -> str:
    r"""The current platform, as:data:`PLATFORM_TOKENS` names it."""
    system = {'Linux': 'linux', 'Darwin': 'macos'}[platform.system()]
    machine = platform.machine().lower()
    arch = 'aarch64' if machine in ('aarch64', 'arm64') else 'x86_64'
    if system == 'macos' and arch == 'aarch64':
        return 'macos-arm64'
    return f'{system}-{arch}'


def _word(text: str) -> str:
    r"""``text`` with x86_64's spellings as one word, so a path splits into
    its words on every other character."""
    return text.replace('x86-64', 'x86_64').replace('x86_64', 'x8664')


def _natives(jar: Path, name: str) -> list[str]:
    r"""The JAR's entries that are platform ``name``'s native library."""
    systems, arches, suffix = PLATFORM_TOKENS[name]
    found = []
    with zipfile.ZipFile(jar) as archive:
        for entry in archive.namelist():
            lower = entry.lower()
            if not lower.endswith(suffix) or 'sdm' not in lower:
                continue
            words = set(re.split(r'[^a-z0-9]+', _word(lower)))
            if any(_word(s) in words for s in systems) and any(
                _word(a) in words for a in arches
            ):
                found.append(entry)
    return found


PROBE = r"""
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Map;

public class PackagedJarProbe {
    public static void main(String[] args) throws Exception {
        Path at = Path.of(args[0]);
        byte[] start = Files.readAllBytes(at.resolve("start.bin"));
        Map<Integer, Object> attachments = Map.of(
            1, Files.readAllBytes(at.resolve("1.bin")),
            2, Files.readAllBytes(at.resolve("2.bin")));
        try {
            Class<?> run = Class.forName("%(run)s");
            Object handle = null;
            for (Method method : run.getMethods()) {
                if (method.getName().equals("start") && method.getParameterCount() == 2) {
                    handle = method.invoke(null, start, attachments);
                }
            }
            Object step = run.getMethod("next").invoke(handle);
            byte[] control = step instanceof byte[]
                ? (byte[]) step
                : (byte[]) step.getClass().getMethod("control").invoke(step);
            run.getMethod("close").invoke(handle);
            System.out.println("DONE " + HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256").digest(control)));
        } catch (java.lang.reflect.InvocationTargetException thrown) {
            Throwable cause = thrown.getCause();
            System.out.println("ERROR " + cause.getClass().getName() + ": " + cause.getMessage());
            System.exit(1);
        } catch (Throwable raised) {
            System.out.println("ERROR " + raised.getClass().getName() + ": " + raised.getMessage());
            System.exit(1);
        }
    }
}
"""


def _contract_version() -> int:
    source = (ROOT / 'rust/crates/sdm-contract/src/lib.rs').read_text(
        encoding='utf-8'
    )
    found = re.search(r'pub const CONTRACT_VERSION: u32 = (\d+);', source)
    assert found
    return int(found.group(1))


def _probe(
    home: Path, jar: Path, scratch: Path
) -> subprocess.CompletedProcess[str]:
    r"""The load probe on JDK ``home`` with the JAR alone, in a clean
    environment: a fresh home and working directory, a ``PATH`` holding
    the JDK and the system's directories but no Rust toolchain, an empty
    ``java.library.path``, and no ``LD_LIBRARY_PATH`` or
    ``JAVA_TOOL_OPTIONS``; the native can come from the JAR alone."""
    import base64

    vectors = json.loads(VECTORS.read_text(encoding='utf-8'))['binding']
    vector = vectors['validate']['s0.accept.binary-classification']
    scratch.mkdir(parents=True, exist_ok=True)
    start = {
        'message': 'start',
        'contract': _contract_version(),
        'facts': vectors['facts'],
        'operation': json.loads(vector['operation']),
    }
    (scratch / 'start.bin').write_bytes(json.dumps(start).encode())
    for key, value in vector['attachments'].items():
        (scratch / f'{key}.bin').write_bytes(base64.b64decode(value))
    source = scratch / 'PackagedJarProbe.java'
    source.write_text(PROBE % {'run': RUN_CLASS}, encoding='utf-8')
    empty = scratch / 'no-libraries'
    empty.mkdir(exist_ok=True)
    (scratch / 'home').mkdir(exist_ok=True)
    env = {
        'PATH': os.pathsep.join([str(home / 'bin'), '/usr/bin', '/bin']),
        'HOME': str(scratch / 'home'),
        'LANG': 'C.UTF-8',
        'JAVA_HOME': str(home),
    }
    return subprocess.run(
        [str(home / 'bin/java'), *FLAGS, f'-Djava.library.path={empty}',
         '-cp', str(jar), str(source), str(scratch)],
        cwd=scratch,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=600,
    )  # fmt: skip


@pytest.mark.skipif(not _java_job(), reason='SDM_JAVA_JAR is not configured')
def test_the_jar_loads_on_each_jdk_in_a_clean_environment(
    tmp_path: Path,
) -> None:
    # Exit 4: on each JDK given, the JAR alone, with the two flags and
    # nothing of the build around it, loads its native and answers S0's
    # first case as Python's binding does (the same done step, byte for
    # byte). CI runs it in a clean container per JDK and platform.
    jar = _jar()
    vectors = json.loads(VECTORS.read_text(encoding='utf-8'))['binding']
    expected = vectors['validate']['s0.accept.binary-classification']['digest']
    majors = []
    for home in _java_homes():
        majors.append(_major(home))
        done = _probe(home, jar, tmp_path / home.name)
        said = f'{home}:\n{done.stdout}\n{done.stderr}'[-4000:]
        assert done.returncode == 0, said
        assert f'DONE {expected}' in done.stdout, said
        assert not list((tmp_path / home.name).glob('hs_err*')), said
    assert set(majors) <= set(JDKS), majors


@pytest.mark.skipif(not _java_job(), reason='SDM_JAVA_JAR is not configured')
def test_the_jar_is_java_17_bytecode() -> None:
    # Choice 10: Java 17 bytecode, so the one JAR runs on JDK 17 and 21.
    jar = _jar()
    majors = set()
    with zipfile.ZipFile(jar) as archive:
        for entry in archive.namelist():
            if entry.startswith('com/nvidia/sdm/') and entry.endswith('.class'):
                header = archive.read(entry)[:8]
                majors.add(int.from_bytes(header[6:8], 'big'))
    assert majors == {JAVA_17_MAJOR}


@pytest.mark.skipif(not _java_job(), reason='SDM_JAVA_JAR is not configured')
def test_the_jar_holds_a_native_for_every_platform_named() -> None:
    # Exit 4 and check 14's control (a JAR missing one architecture's
    # native): each platform named holds exactly one native in the JAR.
    jar = _jar()
    named = [
        each.strip()
        for each in os.environ.get('SDM_JAVA_JAR_PLATFORMS', _here()).split(',')
        if each.strip()
    ]
    assert set(named) <= set(PLATFORM_TOKENS), named
    found = {name: _natives(jar, name) for name in named}
    assert {name: len(entries) for name, entries in found.items()} == {
        name: 1 for name in named
    }, found
    entries = [entry for each in found.values() for entry in each]
    assert len(set(entries)) == len(entries), found


@pytest.mark.skipif(not _java_job(), reason='SDM_JAVA_JAR is not configured')
def test_a_jar_without_this_platform_s_native_is_a_named_error(
    tmp_path: Path,
) -> None:
    # Brief, input axes: the native for the current platform missing is a
    # named error, the JVM ending normally with a message that names the
    # platform, never a crash or a bare UnsatisfiedLinkError.
    jar = _jar()
    here = _here()
    natives = set(_natives(jar, here))
    assert natives, f'the JAR holds no native for {here}'
    stripped = tmp_path / 'stripped.jar'
    with (
        zipfile.ZipFile(jar) as source,
        zipfile.ZipFile(stripped, 'w', zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            if info.filename not in natives:
                target.writestr(info, source.read(info.filename))
    home = _java_homes()[0]
    done = _probe(home, stripped, tmp_path / 'probe')
    said = f'{done.stdout}\n{done.stderr}'
    assert done.returncode == 1, said[-4000:]
    assert not list((tmp_path / 'probe').glob('hs_err*')), said[-4000:]
    error = next(
        (
            line
            for line in done.stdout.splitlines()
            if line.startswith('ERROR ')
        ),
        '',
    )
    assert 'UnsatisfiedLinkError' not in error.split(':')[0], said[-4000:]
    systems, arches, _ = PLATFORM_TOKENS[here]
    lower = error.lower()
    assert any(s in lower for s in systems), said[-4000:]
    assert any(a in lower for a in arches), said[-4000:]


def test_sdm_jni_depends_on_sdm_host_and_sdm_contract_and_no_pyo3() -> None:
    # Exit 5 on the real tree (seeded fixture holds dependency_policy.py's
    # rule): sdm-jni is a workspace member on jni, whose first-party edges
    # are sdm-host and sdm-contract only, with no pyo3 and no database
    # driver; and sdm-py names no jni.
    workspace = tomllib.loads(
        (ROOT / 'rust/Cargo.toml').read_text(encoding='utf-8')
    )
    assert 'bindings/sdm-jni' in workspace['workspace']['members']
    manifest = tomllib.loads(
        (ROOT / 'rust/bindings/sdm-jni/Cargo.toml').read_text(encoding='utf-8')
    )
    tables = [
        value
        for key, value in manifest.items()
        if key.endswith('dependencies') and isinstance(value, dict)
    ]
    tables += [
        value
        for target in manifest.get('target', {}).values()
        for key, value in target.items()
        if key.endswith('dependencies')
    ]
    named = {name for table in tables for name in table}
    first_party = {name for name in named if name.startswith('sdm-')}
    normal = manifest.get('dependencies', {})
    assert 'jni' in normal
    assert {name for name in normal if name.startswith('sdm-')} == {
        'sdm-host',
        'sdm-contract',
    }
    assert first_party <= {'sdm-host', 'sdm-contract'}
    assert not named & {'pyo3', 'pyo3-log', 'arrow-pyarrow', 'numpy', 'duckdb'}
    python = tomllib.loads(
        (ROOT / 'rust/bindings/sdm-py/Cargo.toml').read_text(encoding='utf-8')
    )
    assert 'jni' not in python.get('dependencies', {})


def test_the_vectors_cover_every_s0_case() -> None:
    # The recorder's input, current: every S0 case has its binding and
    # reference vectors, recorded from the case as it stands.
    document = json.loads(VECTORS.read_text(encoding='utf-8'))
    cases = _cases(S0)
    validate = document['binding']['validate']
    constructor = document['reference']['constructor']
    assert set(cases) <= set(validate)
    assert set(cases) <= set(constructor)
    for case, spec in cases.items():
        assert validate[case]['text'] == spec['input']['pql'], case
