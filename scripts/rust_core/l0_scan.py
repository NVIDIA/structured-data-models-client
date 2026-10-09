#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The L0 scans of the Rust core (Phase 1 design, task T1.10c).

Each check reads the tree and prints every violation it finds; the script
exits non-zero when any check found one. ``l0_deps.py`` checks the crate
graph; these check the text:

- ``vocabulary``: no Python-emulation vocabulary in the shipped code of the
  domain crates and ``sdm-host`` outside the modules on
  ``emulation-allowlist.txt`` (decisions H21, E18), which must all sit in the
  parser crate. Comments and ``#[cfg(test)]`` modules are not read; an
  identifier made of the words ``py``, ``python``, ``pandas``, ``numpy`` and
  the like, a ``py``-prefixed Python type name (``pyint``, ``pystr``), a
  dunder name, or a string literal naming Python, pandas, NumPy or pyarrow,
  is emulation vocabulary. The public names the allowlisted modules define
  may be used anywhere in their crate.
- ``sql``: no SQL fragment and no provider match in ``sdm-schema``,
  ``sdm-engine`` or ``sdm-request`` (migration plan, "Ownership"): a string
  literal holding a SQL clause, or an identifier naming a dialect, a
  backend or a SQL statement type.
- ``skips``: no ignored Rust test, pytest skip or expected-failure marker,
  and no CI step allowed to fail, under ``rust/``, ``scripts/rust_core/``,
  ``contract/``, the client tests that touch ``_rust`` or ``_native`` (and
  the ``conftest.py`` files over them), or the Rust and contract CI jobs
  (design, "Workspace rules"). The tokens are ``SKIP_TOKENS``, matched
  exactly (decision E24), so ``skip_while`` and ``#[schemars(skip)]`` pass;
  they are written in pieces here, since the ban covers this file.
- ``bans``: the clippy bans are configured for every domain and shipped
  crate (``rust/clippy.toml``, repeated in full by any crate's own file but
  ``sdm-testkit``'s), no lint table turns them off, and the only ``#[allow]``
  of them in shipped code is ``sdm-host``'s reading of the zone lookup
  roots (decision E24); a test file may allow them, with a reason.
- ``steps``: ``Step`` has exactly the three variants ``fetch``,
  ``call_model`` and ``done``, in the Rust source and in the locked schema:
  no step asks the host for language behavior (migration plan, L0).

``selftest.py`` holds each check to a seeded fixture it must fail on.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]

# The crates whose code ships and whose rules are the core's: the domain
# crates and the host layer. The binding, sdm-py, talks to Python by design,
# and sdm-testkit is dev-only (l0_deps.py classifies every crate).
CORE_CRATES = (
    'sdm-contract',
    'sdm-core',
    'sdm-pql-interim',
    'sdm-data',
    'sdm-schema',
    'sdm-engine',
    'sdm-wire',
    'sdm-request',
    'sdm-host',
)
PARSER_CRATE = 'sdm-pql-interim'
SQL_FREE_CRATES = ('sdm-schema', 'sdm-engine', 'sdm-request')
DEV_CRATES = ('sdm-testkit',)
ALLOWLIST = Path('scripts/rust_core/emulation-allowlist.txt')


@dataclass(frozen=True)
class Source:
    """A Rust file with its comments blanked.

    ``code`` keeps string literals; ``skeleton`` blanks their contents too.
    Both have the original's length and line breaks, so an offset in either
    is a position in the file.
    """

    code: str
    skeleton: str
    strings: tuple[tuple[int, int], ...]  # (start, end) of each literal

    def line(self, offset: int) -> int:
        return self.code.count('\n', 0, offset) + 1


def _blank(text: str) -> str:
    return ''.join('\n' if c == '\n' else ' ' for c in text)


_RAW_STRING = re.compile(r'b?r(#*)"')
_CHAR = re.compile(r"'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'\n])'")


def read_rust(text: str) -> Source:
    """Blank the comments of Rust source and find its string literals."""
    code: list[str] = []
    skeleton: list[str] = []
    strings: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            code.append(_blank(text[i:j]))
            skeleton.append(_blank(text[i:j]))
            i = j
            continue
        if text.startswith('/*', i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith('/*', j):
                    depth, j = depth + 1, j + 2
                elif text.startswith('*/', j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            code.append(_blank(text[i:j]))
            skeleton.append(_blank(text[i:j]))
            i = j
            continue
        raw = _RAW_STRING.match(text, i)
        if raw and (
            i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')
        ):
            close = '"' + raw.group(1)
            j = text.find(close, raw.end())
            j = n if j < 0 else j + len(close)
            code.append(text[i:j])
            skeleton.append(text[i : raw.end()] + _blank(text[raw.end() : j]))
            strings.append((i, j))
            i = j
            continue
        if text[i] == '"' or (
            text.startswith('b"', i)
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_'))
        ):
            start = i
            j = i + (2 if text[i] == 'b' else 1)
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            code.append(text[start:j])
            head = 2 if text[start] == 'b' else 1
            skeleton.append(
                text[start : start + head] + _blank(text[start + head : j])
            )
            strings.append((start, j))
            i = j
            continue
        if text[i] == "'":
            char = _CHAR.match(text, i)
            if char:
                code.append(_blank(char.group(0)))
                skeleton.append(_blank(char.group(0)))
                i = char.end()
                continue
        code.append(text[i])
        skeleton.append(text[i])
        i += 1
    return Source(''.join(code), ''.join(skeleton), tuple(strings))


_TEST_MODULE = re.compile(
    r'#\[cfg\(test\)\]\s*(?:pub(?:\([a-z]+\))?\s+)?mod\s+\w+\s*\{'
)


def without_test_modules(source: Source) -> Source:
    """Blank every ``#[cfg(test)] mod … { … }`` block."""
    code, skeleton = source.code, source.skeleton
    spans: list[tuple[int, int]] = []
    for match in _TEST_MODULE.finditer(skeleton):
        depth, j = 1, match.end()
        while j < len(skeleton) and depth:
            if skeleton[j] == '{':
                depth += 1
            elif skeleton[j] == '}':
                depth -= 1
            j += 1
        spans.append((match.start(), j))
    for start, end in spans:
        code = code[:start] + _blank(code[start:end]) + code[end:]
        skeleton = (
            skeleton[:start] + _blank(skeleton[start:end]) + skeleton[end:]
        )
    strings = tuple(
        s for s in source.strings if not any(a <= s[0] < b for a, b in spans)
    )
    return Source(code, skeleton, strings)


_IDENT = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')


def identifiers(source: Source) -> Iterable[tuple[int, str]]:
    for match in _IDENT.finditer(source.skeleton):
        yield match.start(), match.group(0)


def string_literals(source: Source) -> Iterable[tuple[int, str]]:
    for start, end in source.strings:
        yield start, source.code[start:end]


def words(identifier: str) -> list[str]:
    """``PyIntValue`` and ``py_int_value`` both give ``py``, ``int``, ``value``."""
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', identifier)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return [w.lower() for w in re.split(r'[\s_]+', spaced) if w]


def rust_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob('*.rs') if 'target' not in p.parts)


def crate_dir(root: Path, crate: str) -> Path:
    for parent in ('rust/crates', 'rust/bindings'):
        candidate = root / parent / crate
        if candidate.is_dir():
            return candidate
    return root / 'rust/crates' / crate


# --- vocabulary -------------------------------------------------------------

VOCABULARY_WORDS = frozenset(
    {
        'py',
        'python',
        'cpython',
        'pythonic',
        'pandas',
        'numpy',
        'pyarrow',
        'pyobject',
        'truthy',
        'truthiness',
        'timsort',
        'dumps',
        'datetime64',
        'timedelta64',
        'pythonhashseed',
    }
)
# A Python type or builtin behind a `py` prefix, as one word: pyint, pystr.
_PY_PREFIXED = re.compile(
    r'^py(int|float|str|repr|bool|bytes|list|dict|tuple|set|obj|object|hash|sort|round|format|len)'
)
_DUNDER = re.compile(r'^__[a-z0-9_]+__$')
_STRING_VOCABULARY = re.compile(
    r'(?i)\b(python|cpython|pandas|numpy|pyarrow|datetime64|timedelta64)\b|__[a-z]+__'
)
_PUB_ITEM = re.compile(
    r'\bpub(?:\([a-z]+\))?\s+(?:fn|struct|enum|const|type|trait|static|mod)\s+([A-Za-z_]\w*)'
)


def read_allowlist(root: Path) -> tuple[list[Path], list[str]]:
    entries: list[Path] = []
    problems: list[str] = []
    path = root / ALLOWLIST
    if not path.is_file():
        return [], [f'{ALLOWLIST}: missing']
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        entry = Path(line)
        if not (root / entry).is_file():
            problems.append(f'{ALLOWLIST}:{number}: {line} does not exist')
            continue
        parser_src = Path('rust/crates') / PARSER_CRATE / 'src'
        if parser_src not in entry.parents:
            problems.append(
                f'{ALLOWLIST}:{number}: {line} is outside {parser_src}; '
                'emulation stays in the parser crate (H21)'
            )
            continue
        entries.append(entry)
    return entries, problems


def vocabulary_term(identifier: str) -> str | None:
    if _DUNDER.match(identifier):
        return identifier
    for word in words(identifier):
        if word in VOCABULARY_WORDS or _PY_PREFIXED.match(word):
            return word
    return None


def check_vocabulary(root: Path) -> list[str]:
    allowed, problems = read_allowlist(root)
    allowed_set = {root / p for p in allowed}
    # The names the allowlisted modules define, and the modules themselves,
    # may be named anywhere in their crate.
    permitted = {p.stem for p in allowed}
    for path in allowed_set:
        permitted.update(
            _PUB_ITEM.findall(read_rust(path.read_text()).skeleton)
        )
    for crate in CORE_CRATES:
        src = crate_dir(root, crate) / 'src'
        for path in rust_files(src):
            if path in allowed_set:
                continue
            source = without_test_modules(read_rust(path.read_text()))
            rel = path.relative_to(root)
            for offset, name in identifiers(source):
                term = vocabulary_term(name)
                if term and not (crate == PARSER_CRATE and name in permitted):
                    problems.append(
                        f'{rel}:{source.line(offset)}: `{name}` is Python-emulation '
                        f'vocabulary ({term}); emulation lives only in the modules '
                        f'of {ALLOWLIST}'
                    )
            for offset, text in string_literals(source):
                match = _STRING_VOCABULARY.search(text)
                if match:
                    problems.append(
                        f'{rel}:{source.line(offset)}: a string names '
                        f'`{match.group(0)}`; the core takes typed data and '
                        'names no Python library or behavior'
                    )
    return problems


# --- SQL fragments ------------------------------------------------------------

_SQL_CLAUSE = re.compile(
    r'\b(SELECT|FROM|WHERE|JOIN|UNION|GROUP\s+BY|ORDER\s+BY|PARTITION\s+BY|'
    r'LIMIT|OFFSET|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|CAST|COALESCE|'
    r'ROW_NUMBER|OVER|QUALIFY|TABLESAMPLE|USING\s+SAMPLE|DESCRIBE|PRAGMA|'
    r'INFORMATION_SCHEMA|NULLS\s+(?:FIRST|LAST)|IS\s+(?:NOT\s+)?NULL|DISTINCT|'
    r'COUNT\s*\(|MIN\s*\(|MAX\s*\()(?![A-Za-z0-9_])'
)
_SQL_LOWER = re.compile(
    r'(?i)\bselect\b[^"]*\bfrom\b|\b(insert\s+into|create\s+(?:temp|temporary|table|view)|'
    r'drop\s+(?:table|view)|order\s+by|group\s+by|partition\s+by|nulls\s+(?:first|last)|'
    r'row_number\s*\(|information_schema|tablesample|using\s+sample)\b'
)
PROVIDER_WORDS = frozenset(
    {
        'sql',
        'dialect',
        'duckdb',
        'postgres',
        'postgresql',
        'snowflake',
        'databricks',
        'sqlite',
        'spark',
        'bigquery',
        'redshift',
        'lakebase',
    }
)


def check_sql(root: Path) -> list[str]:
    problems: list[str] = []
    for crate in SQL_FREE_CRATES:
        src = crate_dir(root, crate) / 'src'
        if not src.is_dir():
            problems.append(f'{src.relative_to(root)}: missing')
            continue
        for path in rust_files(src):
            source = read_rust(path.read_text())
            rel = path.relative_to(root)
            for offset, text in string_literals(source):
                match = _SQL_CLAUSE.search(text) or _SQL_LOWER.search(text)
                if match:
                    problems.append(
                        f'{rel}:{source.line(offset)}: a string holds the SQL '
                        f"`{match.group(0)}`; SQL is sdm-data's alone"
                    )
            for offset, name in identifiers(source):
                hit = PROVIDER_WORDS.intersection(words(name))
                if hit:
                    problems.append(
                        f'{rel}:{source.line(offset)}: `{name}` names a provider or '
                        f'SQL ({", ".join(sorted(hit))}); {crate} matches on none'
                    )
    return problems


# --- the skip and ignore ban ------------------------------------------------

# Exact tokens (decision E24). Written in pieces, so this file and
# selftest.py, which the ban covers, do not hold them whole.
_I, _S, _X = 'ignore', 'skip', 'fail'
SKIP_TOKENS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ('#[' + _I + ']', re.compile(r'#\[\s*' + _I + r'\b')),
    (
        'cfg_attr(…, ' + _I + ')',
        re.compile(r'cfg_attr\s*\([^\]]*\b' + _I + r'\b'),
    ),
    ('pytest.mark.' + _S, re.compile(r'pytest\.mark\.' + _S + r'\b')),
    ('pytest.' + _S, re.compile(r'pytest\.' + _S + r'\b')),
    (_S + 'if', re.compile(r'\b' + _S + r'if\b')),
    ('x' + _X, re.compile(r'\bx' + _X + r'\b')),
    ('importor' + _S, re.compile(r'\bimportor' + _S + r'\b')),
    ('unittest.' + _S, re.compile(r'unittest\.' + _S)),
    (_S.capitalize() + 'Test', re.compile(r'\b' + _S.capitalize() + r'Test\b')),
    (_S + 'Test', re.compile(r'\.' + _S + r'Test\b')),
    ('continue-on-' + 'error', re.compile(r'\bcontinue-on-' + r'error\b')),
)
CLIENT_TESTS = Path('packages/structured-data-models-client')
CI_FILE = Path('.github/workflows/ci.yml')
_CI_JOB = re.compile(r'^  ([A-Za-z0-9_-]+):\s*$')


def ci_job_in_scope(name: str, body: str) -> bool:
    """A Rust job, a contract job, or one that runs the contract's checks."""
    return (
        name.startswith(('rust-', 'contract-'))
        or name in ('native-wheel', 'wheel-smoke')
        or 'contract/' in body
        or 'scripts/rust_core/' in body
    )


def tracked_files(root: Path, *paths: str) -> list[Path]:
    """The files git tracks under ``paths``, or every file when not in git."""
    try:
        out = subprocess.run(
            ['git', 'ls-files', '-z', '--', *paths],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout.decode()
        files = [root / p for p in out.split('\0') if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = []
    if not files:
        files = [
            p
            for path in paths
            for p in sorted((root / path).rglob('*'))
            if p.is_file() and 'target' not in p.parts
        ]
    return [p for p in files if p.is_file()]


def _text(path: Path) -> str | None:
    try:
        return path.read_text()
    except (UnicodeDecodeError, OSError):
        return None


def skip_scope(root: Path) -> list[Path]:
    files = tracked_files(root, 'rust', 'scripts/rust_core', 'contract')
    client = [
        p
        for p in tracked_files(root, str(CLIENT_TESTS))
        if p.suffix == '.py'
        and p.relative_to(root / CLIENT_TESTS).parts[0].startswith('tests')
    ]
    touching = [
        p for p in client if re.search(r'\b_(rust|native)\b', _text(p) or '')
    ]
    conftests = {
        c
        for p in touching
        for c in client
        if c.name == 'conftest.py' and c.parent in (p.parent, *p.parents)
    }
    return sorted({*files, *touching, *conftests})


def find_skips(text: str) -> Iterable[tuple[int, str]]:
    for token, pattern in SKIP_TOKENS:
        for match in pattern.finditer(text):
            yield text.count('\n', 0, match.start()) + 1, token


def ci_jobs(text: str) -> dict[str, tuple[int, str]]:
    """Each job of a workflow, with its first line and its text."""
    lines = text.splitlines(keepends=True)
    jobs: dict[str, tuple[int, str]] = {}
    inside, name, start = False, '', 0
    body: list[str] = []
    for number, line in enumerate(lines, 1):
        if line.startswith('jobs:'):
            inside = True
            continue
        if not inside:
            continue
        if (
            line.strip()
            and not line.startswith(' ')
            and not line.startswith('#')
        ):
            break
        match = _CI_JOB.match(line)
        if match:
            if name:
                jobs[name] = (start, ''.join(body))
            name, start, body = match.group(1), number, []
        elif name:
            body.append(line)
    if name:
        jobs[name] = (start, ''.join(body))
    return jobs


def check_skips(root: Path) -> list[str]:
    problems: list[str] = []
    for path in skip_scope(root):
        text = _text(path)
        if text is None:
            continue
        for line, token in find_skips(text):
            problems.append(
                f'{path.relative_to(root)}:{line}: `{token}`; nothing in the Rust '
                'core\'s scope is skipped or ignored (design, "Workspace rules")'
            )
    ci = root / CI_FILE
    if not ci.is_file():
        return [*problems, f'{CI_FILE}: missing']
    for job, (start, body) in ci_jobs(ci.read_text()).items():
        if not ci_job_in_scope(job, body):
            continue
        for line, token in find_skips(body):
            problems.append(
                f'{CI_FILE}:{start + line}: `{token}` in job {job}; the Rust and '
                'contract jobs fail on every failure'
            )
    return problems


# --- the clippy bans ----------------------------------------------------------

REQUIRED_TYPES = frozenset(
    {
        'std::collections::HashMap',
        'std::collections::HashSet',
        'std::hash::RandomState',
        'std::time::SystemTime',
        'std::time::Instant',
        'std::fs::File',
        'std::fs::OpenOptions',
        'rand::rngs::StdRng',
        'rand::rngs::ThreadRng',
    }
)
REQUIRED_METHODS = frozenset(
    {
        'std::time::SystemTime::now',
        'std::time::Instant::now',
        'jiff::Timestamp::now',
        'jiff::Zoned::now',
        'std::env::var',
        'std::env::var_os',
        'std::env::vars',
        'std::env::vars_os',
        'std::env::current_dir',
        'std::fs::read',
        'std::fs::read_to_string',
        'std::fs::write',
        'std::fs::read_dir',
        'std::fs::metadata',
        'std::path::Path::exists',
        'std::path::Path::is_file',
        'std::path::Path::is_dir',
        'jiff::tz::db',
        'jiff::tz::TimeZoneDatabase::from_env',
        'jiff::tz::TimeZoneDatabase::from_dir',
        'jiff::tz::TimeZone::get',
        'jiff::tz::TimeZone::system',
    }
)
# The one place shipped code may allow a ban: sdm-host reads the zone lookup
# roots the host names (decision E24).
BAN_EXEMPTIONS = {
    (
        Path('rust/crates/sdm-host/src/zones.rs'),
        'clippy::disallowed_methods',
    ): 1,
}
BAN_LINTS = ('clippy::disallowed_methods', 'clippy::disallowed_types')
# Allowing these groups or names turns the bans off as well.
_BAN_SWITCHES = re.compile(
    r'\bclippy::(disallowed_methods|disallowed_types|disallowed_method|'
    r'disallowed_type|style|all)\b'
)
_ALLOW = re.compile(r'#(!?)\[\s*(?:allow|expect)\s*\(([^\]]*)\)\s*\]', re.S)


def clippy_files(root: Path, crate: str) -> list[Path]:
    """The configuration clippy reads for a crate: the nearest file up to rust/."""
    directory = crate_dir(root, crate)
    rust = root / 'rust'
    while True:
        found = [
            directory / n
            for n in ('clippy.toml', '.clippy.toml')
            if (directory / n).is_file()
        ]
        if found or directory == rust or rust not in directory.parents:
            return found
        directory = directory.parent


def ban_paths(config: dict[str, object], key: str) -> set[str]:
    entries = config.get(key, [])
    paths: set[str] = set()
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, str):
                paths.add(entry)
            elif isinstance(entry, dict) and isinstance(entry.get('path'), str):
                paths.add(entry['path'])
    return paths


def workspace_crates(root: Path) -> list[str]:
    names = []
    for manifest in sorted((root / 'rust').glob('*/*/Cargo.toml')):
        if 'target' in manifest.parts:
            continue
        package = tomllib.loads(manifest.read_text()).get('package', {})
        if isinstance(package.get('name'), str):
            names.append(package['name'])
    return names


def check_bans(root: Path) -> list[str]:
    problems: list[str] = []
    workspace = root / 'rust/clippy.toml'
    if not workspace.is_file():
        return ['rust/clippy.toml: missing']
    base = tomllib.loads(workspace.read_text())
    base_types = ban_paths(base, 'disallowed-types')
    base_methods = ban_paths(base, 'disallowed-methods')
    for missing in sorted(REQUIRED_TYPES - base_types):
        problems.append(
            f'rust/clippy.toml: the type ban on {missing} is missing'
        )
    for missing in sorted(REQUIRED_METHODS - base_methods):
        problems.append(
            f'rust/clippy.toml: the method ban on {missing} is missing'
        )
    for crate in workspace_crates(root):
        if crate in DEV_CRATES:
            continue
        configs = clippy_files(root, crate)
        if len(configs) != 1:
            problems.append(
                f'{crate}: clippy reads {len(configs)} configuration files, not one'
            )
            continue
        config = configs[0]
        if config == workspace:
            continue
        own = tomllib.loads(config.read_text())
        rel = config.relative_to(root)
        for missing in sorted(base_types - ban_paths(own, 'disallowed-types')):
            problems.append(
                f"{rel}: does not repeat the workspace's type ban on {missing}"
            )
        for missing in sorted(
            base_methods - ban_paths(own, 'disallowed-methods')
        ):
            problems.append(
                f"{rel}: does not repeat the workspace's method ban on {missing}"
            )
    # No lint table turns a ban off.
    for manifest in [
        root / 'rust/Cargo.toml',
        *sorted((root / 'rust').glob('*/*/Cargo.toml')),
    ]:
        if not manifest.is_file() or 'target' in manifest.parts:
            continue
        data = tomllib.loads(manifest.read_text())
        tables = [
            data.get('workspace', {}).get('lints', {}).get('clippy', {}),
            data.get('lints', {}).get('clippy', {}),
        ]
        for table in tables:
            for lint, level in table.items():
                setting = (
                    level.get('level') if isinstance(level, dict) else level
                )
                if lint in (
                    'disallowed_methods',
                    'disallowed_types',
                    'style',
                ) and setting in (
                    'allow',
                    'warn',
                ):
                    problems.append(
                        f'{manifest.relative_to(root)}: clippy::{lint} set to {setting}'
                    )
                if lint == 'all' and setting == 'allow':
                    problems.append(
                        f'{manifest.relative_to(root)}: clippy::all set to allow'
                    )
    # Shipped code allows a ban only at its exemption; tests give a reason.
    seen: dict[tuple[Path, str], int] = {}
    for crate in workspace_crates(root):
        if crate in DEV_CRATES:
            continue
        directory = crate_dir(root, crate)
        for path in rust_files(directory):
            source = read_rust(path.read_text())
            rel = path.relative_to(root)
            in_src = (directory / 'src') in path.parents
            for match in _ALLOW.finditer(source.code):
                inner, body = match.group(1), match.group(2)
                switches = sorted(set(_BAN_SWITCHES.findall(body)))
                if not switches:
                    continue
                line = source.line(match.start())
                if not in_src:
                    if 'reason' not in body:
                        problems.append(
                            f'{rel}:{line}: allows a clippy ban without a reason'
                        )
                    continue
                for switch in switches:
                    lint = f'clippy::{switch}'
                    key = (rel, lint)
                    if (
                        inner
                        or key not in BAN_EXEMPTIONS
                        or 'reason' not in body
                    ):
                        problems.append(
                            f'{rel}:{line}: shipped code allows {lint}; the one '
                            "exemption is sdm-host's zone lookup roots (E24)"
                        )
                        continue
                    seen[key] = seen.get(key, 0) + 1
    for key, count in seen.items():
        if count > BAN_EXEMPTIONS[key]:
            problems.append(
                f'{key[0]}: allows {key[1]} {count} times; the exemption is '
                f'{BAN_EXEMPTIONS[key]} site'
            )
    return problems


# --- the step variants --------------------------------------------------------

STEP_VARIANTS = ('fetch', 'call_model', 'done')
_STEP_ENUM = re.compile(r'\bpub\s+enum\s+Step\s*\{')
CONTRACT = Path('rust/crates/sdm-contract')


def _snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def step_variants_in_source(text: str) -> list[str] | None:
    source = read_rust(text)
    match = _STEP_ENUM.search(source.skeleton)
    if not match:
        return None
    depth, j, start = 1, match.end(), match.end()
    while j < len(source.skeleton) and depth:
        if source.skeleton[j] in '{(':
            depth += 1
        elif source.skeleton[j] in '})':
            depth -= 1
        j += 1
    body = source.skeleton[start : j - 1]
    body = re.sub(r'#\[[^\]]*\]', ' ', body)
    variants, depth, token = [], 0, ''
    for char in body:
        if char in '{(':
            if depth == 0 and token.strip():
                variants.append(token.strip())
            depth += 1
            token = ''
        elif char in '})':
            depth -= 1
        elif depth == 0:
            if char == ',':
                if token.strip():
                    variants.append(token.strip())
                token = ''
            else:
                token += char
    if token.strip():
        variants.append(token.strip())
    names = []
    for variant in variants:
        names.append(_snake(variant.split()[0]))
    return names


def step_variants_in_schema(schema: dict[str, Any]) -> list[str] | None:
    for branch in schema.get('oneOf', []):
        if not isinstance(branch, dict):
            continue
        message = branch.get('properties', {}).get('message', {})
        if message.get('const') != 'step':
            continue
        names = []
        for variant in branch.get('oneOf', []):
            const = variant.get('properties', {}).get('step', {}).get('const')
            if not isinstance(const, str):
                return None
            names.append(const)
        return names
    return None


def check_steps(root: Path) -> list[str]:
    problems: list[str] = []
    expected = sorted(STEP_VARIANTS)
    run = root / CONTRACT / 'src/run.rs'
    found = step_variants_in_source(run.read_text()) if run.is_file() else None
    if found is None:
        problems.append(f'{run.relative_to(root)}: no `pub enum Step`')
    elif sorted(found) != expected:
        problems.append(
            f'{run.relative_to(root)}: Step has {found}; it has exactly '
            f'{list(STEP_VARIANTS)}, and no step asks the host for language behavior'
        )
    schemas = sorted((root / CONTRACT / 'schema').glob('sdm-contract-*.json'))
    if not schemas:
        problems.append(f'{CONTRACT}/schema: no contract schema')
    for path in schemas:
        variants = step_variants_in_schema(json.loads(path.read_text()))
        if variants is None or sorted(variants) != expected:
            problems.append(
                f'{path.relative_to(root)}: the step message has {variants}; it '
                f'has exactly {list(STEP_VARIANTS)}'
            )
    return problems


CHECKS: dict[str, Callable[[Path], list[str]]] = {
    'vocabulary': check_vocabulary,
    'sql': check_sql,
    'skips': check_skips,
    'bans': check_bans,
    'steps': check_steps,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    # Checked here, not by argparse's choices, which before Python 3.14
    # refuse an empty list. No checks named means all of them.
    parser.add_argument(
        'checks', nargs='*', help=f'any of {", ".join(CHECKS)}; all by default'
    )
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    unknown = sorted(set(args.checks) - {*CHECKS, 'all'})
    if unknown:
        parser.error(f'unknown checks: {", ".join(unknown)}')
    names = (
        list(CHECKS) if not args.checks or 'all' in args.checks else args.checks
    )
    failed = False
    for name in names:
        problems = CHECKS[name](args.root)
        for problem in problems:
            print(f'{name}: {problem}')
        print(f'{name}: {"FAIL" if problems else "ok"}')
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
