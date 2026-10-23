# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.11's acceptance tests on the Java host: every S0 case through the
Java runner held to the reference, no S1 or S2 Java leg yet, the JAR on
each JDK in a clean environment, a native for every platform, a missing
native named, and ``sdm-jni``'s edges.

The JUnit side (``java/src/test/java/com/nvidia/sdm/``) holds the Java API
and the binding per S0 case, every error code, and a panic inside the core
in a JVM of its own (exit 3); ``tests/test_t311_java_ci.py`` holds the CI
jobs (exits 1 and 4), whose Java comparison judges the Java lines on their
own, so a difference only Java shows fails them (exit 2: the handoff stamp
runs no case since the owner cut it back to lint and the package's own
tests, 2026-10-23).

What these run, from the environment, as the brief and choice 10 name it
(design, "Architecture choices", 10):

- ``SDM_JAVA_JAR``: the JAR under test, as ``mvn package`` builds it under
  ``java/target/``, or as CI assembles it with every platform's native.
  Unset, these tests have nothing to hold and return, unless
  ``SDM_REQUIRE_JAVA`` says the job must have a JAR, when they fail.
- ``JAVA_HOME``: the JDK ``run_java.py`` runs on; ``SDM_JAVA_HOMES``
  (``os.pathsep``-separated): every JDK the load test loads the JAR on,
  ``JAVA_HOME`` alone when unset.
- ``SDM_JAVA_JAR_PLATFORMS`` (comma-separated, from
  :data:`PLATFORM_TOKENS`): the platforms whose natives the JAR must hold,
  the current one alone when unset; CI's load jobs name all four.

ASSUMED until the implementer relays them (only this block changes then):

- ``contract/runners/run_java.py`` runs as ``run_python.py`` does, from the
  tools' interpreter: ``python -I contract/runners/run_java.py --out FILE
  CASE_OR_DIRECTORY...``, the JAR from ``SDM_JAVA_JAR`` and the JDK from
  ``JAVA_HOME``; it exits 0 having written one observation per case it
  runs, ``identity.runner`` ``java``, and runs no case whose leg the Java
  host has no provider for (every S1 and S2 case at T3.11), as
  ``run_python.py`` runs no case on a database its ``databases`` leave
  out; its S0 lines are keyed by the bare case id and judged by
  ``compare.py`` as the other runners' are;
- the binding's low-level run is ``com.nvidia.sdm.internal.Run`` (see
  ``java/src/test/java/com/nvidia/sdm/T311Names.java``), which the load
  probe starts a ``validate_query`` on, so the native is loaded and used.

Gating (``contract/tests/t311_gate.py``): each test named in
``contract/tests/t311-pending.json``'s ``pending`` failed when written,
before the implementation, and must still fail until the implementer takes
it out of ``pending``; once it passes it fails, naming itself. ``pending``
stays within ``authored``, the test author's list.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import tomllib

ROOT = Path(__file__).resolve().parents[2]
#: ASSUMED: the Java runner.
RUN_JAVA = ROOT / 'contract/runners/run_java.py'
COMPARE = ROOT / 'scripts/rust_core/compare.py'
CASES = ROOT / 'contract/cases'
S0 = CASES / 's0'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
VECTORS = ROOT / 'contract/t311-java-vectors.json'
#: ASSUMED: the binding's low-level run (T311Names.RUN).
RUN_CLASS = 'com.nvidia.sdm.internal.Run'
#: Choice 10's runtime flags, which every Java test runs with.
FLAGS = (
    '--enable-native-access=ALL-UNNAMED',
    '--add-opens=java.base/java.nio=ALL-UNNAMED',
)
#: The JDKs choice 10 tests on (KO3-34).
JDKS = (17, 21)
#: Each platform the JAR holds a native for (KO3-32), as CI names it: the
#: words its entry's path holds (one of each group) and the library's
#: suffix.
PLATFORM_TOKENS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    'linux-x86_64': (('linux',), ('x86_64', 'amd64', 'x86-64'), '.so'),
    'linux-aarch64': (('linux',), ('aarch64', 'arm64'), '.so'),
    'macos-arm64': (('macos', 'darwin', 'osx', 'mac'), ('arm64', 'aarch64'), '.dylib'),
    'macos-x86_64': (('macos', 'darwin', 'osx', 'mac'), ('x86_64', 'amd64', 'x86-64'), '.dylib'),
}  # fmt: skip
#: A Java 17 class file's major version (choice 10: Java 17 bytecode).
JAVA_17_MAJOR = 61


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _module('t311_gate', ROOT / 'contract/tests/t311_gate.py')


def _java_job() -> bool:
    r"""Whether this job has a JAR to test (``SDM_JAVA_JAR``); a job that
    says it must (``SDM_REQUIRE_JAVA``) fails without one."""
    if os.environ.get('SDM_JAVA_JAR'):
        return True
    assert not os.environ.get('SDM_REQUIRE_JAVA'), (
        'SDM_REQUIRE_JAVA is set but SDM_JAVA_JAR names no JAR'
    )
    return False


#: A test of the JAR, held by the gate, run where there is one.
java = G.gated(_java_job)
#: A test held by the gate that needs nothing more.
held = G.gated()


def test_t311_the_pending_list_only_shrinks() -> None:
    assert G.only_shrinks() == []


# --- helpers -----------------------------------------------------------------


def _jar() -> Path:
    given = os.environ.get('SDM_JAVA_JAR', '')
    assert given, 'SDM_JAVA_JAR names no JAR'
    return Path(given)


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


def _reference() -> dict[str, dict[str, Any]]:
    lines = {}
    for line in REFERENCE.read_text(encoding='utf-8').splitlines():
        if line.strip():
            seen = json.loads(line)
            lines[seen['case']] = seen
    return lines


def _run_java(out: Path, *cases: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-I', str(RUN_JAVA), '--out', str(out),
         *map(str, cases)],
        cwd=ROOT,
        env={**os.environ, 'SDM_JAVA_JAR': str(_jar())},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=1800,
    )  # fmt: skip


def _observations(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _here() -> str:
    r"""The current platform, as :data:`PLATFORM_TOKENS` names it."""
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


#: The load probe: Java's single-file source launcher compiles and runs it
#: against the JAR alone, so it needs the JDK and the JAR and nothing of
#: this checkout's build. It starts the vector's validate_query on the
#: binding's run, which loads the native from the JAR, and prints the done
#: step's SHA-256, or the error that stopped it, class and message.
PROBE = r"""
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Map;

public class T311Load {
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
    source = scratch / 'T311Load.java'
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


# --- exit 1: every S0 case on the Java host, and no S1 or S2 leg -------------


@java
def test_t311_every_s0_case_on_the_java_host_equals_the_reference(
    tmp_path: Path,
) -> None:
    # Exit 1: run_java.py over contract/cases/s0, judged by the comparator
    # against the recorded old-client answers; and, read directly, each
    # acceptance case's outcome (succeeded or refused, and the code) the
    # reference's.
    out = tmp_path / 'java.jsonl'
    done = _run_java(out, S0)
    assert done.returncode == 0, (done.stdout + done.stderr)[-4000:]
    seen = {line['case']: line for line in _observations(out)}
    cases = _cases(S0)
    assert sorted(seen) == sorted(cases)
    assert {line['identity']['runner'] for line in seen.values()} == {'java'}
    reference = _reference()
    differing = {
        case: (seen[case]['outcome'], reference[case]['outcome'])
        for case, spec in cases.items()
        if spec['status'] == 'acceptance'
        and (
            seen[case]['outcome']['result'],
            seen[case]['outcome']['code'],
        )
        != (
            reference[case]['outcome']['result'],
            reference[case]['outcome']['code'],
        )
    }
    assert differing == {}
    # The reference's lines of these cases alone, as handoff.py holds them:
    # compare.py refuses a line of a case it was not given.
    held = tmp_path / 'reference.jsonl'
    held.write_text(
        ''.join(
            line + '\n'
            for line in REFERENCE.read_text(encoding='utf-8').splitlines()
            if line.strip() and json.loads(line)['case'] in cases
        ),
        encoding='utf-8',
    )
    compared = subprocess.run(
        [sys.executable, str(COMPARE), '--cases', str(S0),
         '--reference', str(held), '--register', str(REGISTER),
         str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=900,
    )  # fmt: skip
    assert compared.returncode == 0, (compared.stdout + compared.stderr)[-4000:]


@java
def test_t311_no_s1_or_s2_java_leg_is_admitted(tmp_path: Path) -> None:
    # Exit 1's second half: no Java provider exists at T3.11, so the runner
    # runs an S0 case and leaves an S1 and an S2 case out, and no committed
    # baseline holds a Java line of a case outside S0.
    _jar()
    s1 = CASES / 's1/base.regression.json'
    # Phase 3's warehouse cases, as they land (T3.4 to T3.7, T3.13).
    s2 = sorted((CASES / 's2').glob('*.json'))[:1]
    s0 = S0 / 's0.accept.sum-window.json'
    out = tmp_path / 'java.jsonl'
    done = _run_java(out, s0, s1, *s2)
    assert done.returncode == 0, (done.stdout + done.stderr)[-4000:]
    assert [line['case'] for line in _observations(out)] == [
        's0.accept.sum-window'
    ]
    slices = {
        case['id']: case['slice']
        for path in CASES.rglob('*.json')
        for case in [json.loads(path.read_text(encoding='utf-8'))]
        if isinstance(case, dict) and 'id' in case
    }
    admitted = []
    for baseline in (ROOT / 'contract/baselines').rglob('*.jsonl'):
        for line in baseline.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get('runner') == 'java':
                bare = entry['case'].split('@')[0]
                admitted.append(
                    (baseline.name, entry['case'], slices.get(bare))
                )
    assert [each for each in admitted if each[2] != 'S0'] == []


# --- exit 4: the JAR on every JDK, in a clean environment ----------------------


@java
def test_t311_the_jar_loads_on_each_jdk_in_a_clean_environment(
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


@java
def test_t311_the_jar_is_java_17_bytecode() -> None:
    # Choice 10: Java 17 bytecode, so the one JAR runs on JDK 17 and 21.
    jar = _jar()
    majors = set()
    with zipfile.ZipFile(jar) as archive:
        for entry in archive.namelist():
            if entry.startswith('com/nvidia/sdm/') and entry.endswith('.class'):
                header = archive.read(entry)[:8]
                majors.add(int.from_bytes(header[6:8], 'big'))
    assert majors == {JAVA_17_MAJOR}


@java
def test_t311_the_jar_holds_a_native_for_every_platform_named() -> None:
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


@java
def test_t311_a_jar_without_this_platform_s_native_is_a_named_error(
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


# --- the binding's edges (exit 5's crate rule, held on the real tree) ------------


@held
def test_t311_sdm_jni_depends_on_sdm_host_and_sdm_contract_and_no_pyo3() -> (
    None
):
    # Exit 5 on the real tree (T3.2's seeded fixture holds l0_deps.py's
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


def test_t311_the_vectors_cover_every_s0_case() -> None:
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
