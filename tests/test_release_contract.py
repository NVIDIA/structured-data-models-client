# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-package checks that no single distribution's suite can make.

Each package tests its own behaviour. Nothing tested that the five agree with
each other, that what they export stays what they exported, or that the claims
the configuration files make about themselves are true. Those are the failures
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import typing
from collections.abc import Sequence
from types import ModuleType
from typing import Annotated, NamedTuple

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # The packages support 3.10, where tomllib is not in the stdlib.
    import tomli as tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
PACKAGES = REPO / 'packages'
GOLDEN = REPO / 'tests' / 'golden'


def _split_rule(code: str) -> tuple[str, str]:
    """A ruff code as its linter prefix and its number, e.g. ``('FIX', '002')``."""
    match = re.match(r'([A-Z]+)(\d*)', code)
    assert match, f'not a ruff rule code: {code!r}'
    return match.group(1), match.group(2)


def _linter_of(code: str) -> str:
    return _split_rule(code)[0]


def _covered_by(rule: str, selectors: Sequence[str]) -> bool:
    """Whether any selector actually enables ``rule``.

    Plain string prefixes are wrong here: `'FIX002'.startswith('F')` is True,
    but ruff's `F` is Pyflakes and `FIX` is flake8-fixme, selecting one does
    not select the other, which is verifiable with
    """
    if 'ALL' in selectors:
        return True
    rule_linter, rule_number = _split_rule(rule)
    return any(
        selector_linter == rule_linter
        and rule_number.startswith(selector_number)
        for selector_linter, selector_number in map(_split_rule, selectors)
    )


def _pyproject(name: str) -> dict:
    return tomllib.loads((PACKAGES / name / 'pyproject.toml').read_text())


def _distributions() -> list[str]:
    return sorted(
        p.name for p in PACKAGES.iterdir() if (p / 'pyproject.toml').is_file()
    )


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO / 'scripts' / script), *arguments],
        capture_output=True,
        text=True,
    )


def _public_packages() -> list[ModuleType]:
    """The packages the snapshot records, which are the public ones."""
    return [
        importlib.import_module(name)
        for name in json.loads((GOLDEN / 'public_api.json').read_text())
    ]


#: Every top-level package this repository ships, public or not.
FIRST_PARTY = frozenset(
    {
        'predictive_query',
        'sdm_api',
        'sdm_client',
        'sdm_connectors',
        'sdm_relational_engine',
    }
)

#: Installed beside the public packages and not public themselves
#: (docs/reference/stability.md), so a public package may not re-export from
#: them or build a public class on one of their classes either.
INTERNAL_PACKAGES = frozenset({'predictive_query', 'sdm_relational_engine'})

#: Types a public name hands out that no public package exports. Each stays for
#: the reason beside it, and an entry nothing hands out any more fails, so the
#: list can only shrink.
HANDED_OUT_THOUGH_NOT_EXPORTED = frozenset(
    {
        # What `PredictionResult.items` holds, as its docstring says: the
        # contract's own response item, generated into `sdm_api` from the
        # OpenAPI document. Its fields are the wire's and move only with the
        # contract. Exporting it, or wrapping it in a type of the client's
        # own, is a decision about the result type rather than a leak to close.
        'sdm_api.generated.tfm_api.PredictionItem',
        # What `DatabricksSource.connection` is for a source built from a Spark
        # session: the session shaped like a Databricks SQL connection, so
        # sampling has one interface to run through. The connector's typing
        # check pins that type, so changing it is a change to sources.
        'sdm_connectors.backends.spark.SparkConnection',
        # What each `Source` overload says its connection must look like. They
        # are protocols, matched by shape: a caller passes a live connection
        # and never names, imports or builds one of these. Exporting seven
        # names to describe an argument nobody writes would be a larger
        # promise than the one it closes.
        'sdm_connectors._types.DatabricksLike',
        'sdm_connectors._types.DuckDBLike',
        'sdm_connectors._types.PostgresLike',
        'sdm_connectors._types.SQLiteLike',
        'sdm_connectors._types.SnowflakeLike',
        'sdm_connectors._types.SnowparkLike',
        'sdm_connectors._types.SparkLike',
    }
)

_UNKNOWN = object()


class _Unimported(NamedTuple):
    """A name in a first-party module that only its source can describe."""

    module: str
    name: str
    #: The text of the alias it is, or ``'class'``.
    definition: str


def _source_tree(module_name: str) -> ast.Module | None:
    """A module's source, parsed without importing it.

    Not importing matters: a backend imports its optional driver at the top,
    and a test run without the driver still has to say what the backend's
    names are.
    """
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith('.py'):
        return None
    return ast.parse(pathlib.Path(spec.origin).read_text())


def _top_level_imports(tree: ast.Module) -> dict[str, tuple[str, str | None]]:
    """What each name a module imports at its top level, or only under
    ``TYPE_CHECKING``, is imported from: ``(module, attribute)``.

    An annotation under ``from __future__ import annotations`` is a string, and
    a name imported only for the type checker is not there at runtime to look
    up, so the source says where it comes from.
    """
    found: dict[str, tuple[str, str | None]] = {}
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.If, ast.Try)):
            pending.extend(node.body)
            pending.extend(node.orelse)
        elif (
            isinstance(node, ast.ImportFrom) and node.module and not node.level
        ):
            for alias in node.names:
                found[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    found[alias.asname] = (alias.name, None)
                else:
                    head = alias.name.split('.')[0]
                    found[head] = (head, None)
    return found


def _defined_in_source(tree: ast.Module, name: str) -> str | None:
    """How a module's source defines ``name``, when it is not there to look
    up: the text of a type alias, ``'class'``, or ``None``.

    An alias over an optional driver exists only under ``TYPE_CHECKING``, and
    a module that cannot be imported has nothing to look up at all.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return 'class'
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.unparse(node.value)
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.unparse(node.value)
    return None


def _dotted(node: ast.expr) -> str | None:
    """``a.b.C`` for a name or attribute node, or ``None`` for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return '.'.join(reversed(parts))


#: What `type X = ...` builds, which stands for its value rather than being a
#: type of its own. Absent before Python 3.12.
_ALIAS = getattr(typing, 'TypeAliasType', None)

#: How to read the signatures an overloaded function offers. Absent before
#: Python 3.11, which keeps no registry of them, and `typing_extensions` does
#: not help: it records only what its own `overload` decorated, and the source
#: here uses `typing`'s. So on 3.10 the check reads every other place a type
#: is named and none of the overloads.
_OVERLOADS = getattr(typing, 'get_overloads', None)


def _quoted_within(annotation: object) -> list[str]:
    """The forward references inside an annotation that is already an object.

    Without postponed annotations `list['X']` is a generic alias holding the
    plain string `'X'`, which walking its arguments as types steps over. The
    strings are collected here and read as the names they are.

    The same three shapes the written form has to tell apart appear here too:
    a `Literal`'s arguments are values, an `Annotated`'s metadata is whatever
    the caller attached, and a `Callable` keeps its parameters in a plain list
    rather than among its arguments.
    """
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, str):
            found.append(node)
            return
        if isinstance(node, typing.ForwardRef):
            found.append(node.__forward_arg__)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if _ALIAS is not None and isinstance(node, _ALIAS):
            walk(node.__value__)
            return
        origin = typing.get_origin(node)
        if origin is typing.Literal:
            return
        arguments = typing.get_args(node)
        if origin is Annotated:
            arguments = arguments[:1]
        for argument in arguments:
            walk(argument)

    walk(annotation)
    return found


def _named_in(annotation: str) -> list[str]:
    """Every name an annotation written as a string refers to.

    A quoted string in an annotation is a forward reference and names a type.
    Under `Literal` it is a value and names nothing, and under `Annotated`
    only the first argument is a type, the rest being metadata the caller
    attached. Removing every quoted run instead, which is what this did, took
    the forward reference with it: a return written `-> 'Decomposition'` read
    as naming nothing at all, while the same annotation without the quotes
    was read.

    An annotation that will not parse falls back to every word in it, rather
    than to nothing, which would read as naming no type at all.
    """
    try:
        tree = ast.parse(annotation, mode='eval')
    except SyntaxError:
        stripped = re.sub(r"'[^']*'|\"[^\"]*\"", '', annotation)
        return re.findall(r'[A-Za-z_][\w.]*', stripped)

    found: list[str] = []

    def walk(node: ast.expr, in_literal: bool) -> None:
        if isinstance(node, ast.Subscript):
            base = (_dotted(node.value) or '').split('.')[-1]
            walk(node.value, in_literal)
            if base == 'Annotated':
                inside = node.slice
                first = (
                    inside.elts[0] if isinstance(inside, ast.Tuple) else inside
                )
                walk(first, in_literal)
                return
            walk(node.slice, in_literal or base == 'Literal')
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and not in_literal:
                found.extend(_named_in(node.value))
            return
        dotted = _dotted(node)
        if dotted is not None:
            found.append(dotted)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                walk(child, in_literal)

    walk(tree.body, False)
    return found


class _InternalTypes:
    """Every place a public package names or hands out a type that is not
    public.

    Public is a name in the ``__all__`` of a public package. A type from the
    standard library or a third-party package is not this project's to promise
    and passes. Read from every exported name: a function's signature, and a
    class's constructor, methods, properties, class-level annotations
    (dataclass fields among them) and the values of its public attributes,
    along every first-party class in its MRO. A class this project defines
    must be public itself to appear there.
    """

    def __init__(self, packages: Sequence[ModuleType]) -> None:
        self.packages = packages
        self.public = {
            id(getattr(module, name)): f'{module.__name__}.{name}'
            for module in packages
            for name in module.__all__
        }
        self.found: list[str] = []
        self.allowed: set[str] = set()
        self._walked: set[int] = set()
        self._trees: dict[str, ast.Module | None] = {}
        self._imported: dict[str, dict[str, tuple[str, str | None]]] = {}

    def run(self) -> list[str]:
        for module in self.packages:
            for name in module.__all__:
                obj = getattr(module, name)
                where = f'{module.__name__}.{name}'
                if _ALIAS is not None and isinstance(obj, _ALIAS):
                    self._type(where, obj)
                    continue
                if inspect.isclass(obj) or inspect.isroutine(obj):
                    self._type(where, obj)
                if inspect.isclass(obj):
                    self._class(where, obj)
                elif inspect.isroutine(obj):
                    self._signature(where, obj)
                else:
                    self._type(f'{where} holds', type(obj))
        return self.found

    def _type(self, where: str, obj: object) -> None:
        if obj is None or obj is Ellipsis or isinstance(obj, (str, int)):
            return  # `None`, or a `Literal` argument
        if isinstance(obj, typing.TypeVar):
            for inner in (obj.__bound__, *obj.__constraints__):
                self._type(where, inner)
            return
        if _ALIAS is not None and isinstance(obj, _ALIAS):
            self._type(where, obj.__value__)
            return
        origin = typing.get_origin(obj)
        if not inspect.isclass(obj) and origin is not None:
            self._type(where, origin)
            if origin is typing.Literal:
                return
            arguments = typing.get_args(obj)
            if origin is Annotated:
                arguments = arguments[:1]
            for argument in arguments:
                for one in (
                    argument if isinstance(argument, list) else [argument]
                ):
                    self._type(where, one)
            return
        module = getattr(obj, '__module__', None) or ''
        root = module.split('.')[0]
        if root not in FIRST_PARTY:
            return
        qualified = f'{module}.{getattr(obj, "__qualname__", obj)}'
        if id(obj) in self.public:
            if root in INTERNAL_PACKAGES:
                self.found.append(
                    f'{where}: {qualified} comes from {root}, which is not '
                    f'public'
                )
            return
        if qualified in HANDED_OUT_THOUGH_NOT_EXPORTED:
            self.allowed.add(qualified)
            return
        self.found.append(f'{where}: {qualified} is not public')

    def _annotation(self, where: str, annotation: object, module: str) -> None:
        if annotation is inspect.Parameter.empty:
            return
        if not isinstance(annotation, str):
            for quoted in _quoted_within(annotation):
                self._written(where, quoted, module)
            self._type(where, annotation)
            return
        self._written(where, annotation, module)

    def _written(self, where: str, annotation: str, module: str) -> None:
        """Read an annotation written as a string, in ``module``."""
        for dotted in _named_in(annotation):
            if dotted in {'None', 'True', 'False'}:
                continue
            resolved = self._resolve(dotted, module)
            if resolved is _UNKNOWN:
                self.found.append(
                    f'{where}: cannot tell what {dotted!r} is, so cannot '
                    f'tell whether it is public'
                )
            elif isinstance(resolved, _Unimported):
                self._unimported(where, resolved)
            elif resolved is not None:
                self._type(where, resolved)

    def _resolve(self, dotted: str, module_name: str) -> object:
        """What ``dotted`` names in ``module_name``: the object, ``None`` for
        a name from outside this project, an ``_Unimported`` for a name only
        the source can say anything about, or ``_UNKNOWN``.
        """
        loaded = sys.modules.get(module_name)
        head, *rest = dotted.split('.')
        imports = self._imports(module_name)
        if loaded is not None and head in vars(loaded):
            obj = vars(loaded)[head]
        elif head in imports:
            source, attribute = imports[head]
            if source.split('.')[0] not in FIRST_PARTY:
                return None
            if attribute is None:
                try:
                    obj = importlib.import_module(source)
                except ImportError:
                    return _UNKNOWN
            else:
                obj = self._lookup(source, attribute)
                if obj is _UNKNOWN or isinstance(obj, _Unimported):
                    return obj
        elif loaded is None and self._tree(module_name) is not None:
            return self._lookup(module_name, head)
        elif hasattr(builtins, head):
            return None
        else:
            return _UNKNOWN
        for part in rest:
            if isinstance(obj, ModuleType):
                if obj.__name__.split('.')[0] not in FIRST_PARTY:
                    return None
            obj = getattr(obj, part, _UNKNOWN)
            if obj is _UNKNOWN:
                return _UNKNOWN
        return None if isinstance(obj, ModuleType) else obj

    def _lookup(self, module_name: str, name: str) -> object:
        """``name`` in a first-party module, imported if it can be, and read
        from its source if it cannot or defines ``name`` only for the type
        checker."""
        try:
            module: ModuleType | None = importlib.import_module(module_name)
        except ImportError:
            module = None
        if module is not None and hasattr(module, name):
            return getattr(module, name)
        tree = self._tree(module_name)
        defined = None if tree is None else _defined_in_source(tree, name)
        if defined is None:
            return _UNKNOWN
        return _Unimported(module_name, name, defined)

    def _unimported(self, where: str, found: _Unimported) -> None:
        if found.definition != 'class':
            self._annotation(where, found.definition, found.module)
            return
        # A class whose module cannot be imported here cannot be exported by
        # a package that was, so it is not public.
        qualified = f'{found.module}.{found.name}'
        if qualified in HANDED_OUT_THOUGH_NOT_EXPORTED:
            self.allowed.add(qualified)
        else:
            self.found.append(f'{where}: {qualified} is not public')

    def _tree(self, module_name: str) -> ast.Module | None:
        if module_name not in self._trees:
            self._trees[module_name] = _source_tree(module_name)
        return self._trees[module_name]

    def _imports(self, module_name: str) -> dict[str, tuple[str, str | None]]:
        if module_name not in self._imported:
            tree = self._tree(module_name)
            self._imported[module_name] = (
                {} if tree is None else _top_level_imports(tree)
            )
        return self._imported[module_name]

    def _signature(self, where: str, function: object) -> None:
        """Read what a function declares, over every form a caller sees.

        An overloaded function's implementation says `Any`, which is what a
        caller is kept away from. The overloads are the signatures they are
        offered, and the types they name are the promise.
        """
        function = inspect.unwrap(function)  # type: ignore[arg-type]
        module = getattr(function, '__module__', None) or ''
        if module.split('.')[0] not in FIRST_PARTY:
            return
        offered = _OVERLOADS(function) if _OVERLOADS is not None else []
        for form in [function, *offered]:
            try:
                written = inspect.signature(form)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            for parameter in written.parameters.values():
                self._annotation(
                    f'{where}({parameter.name})', parameter.annotation, module
                )
            self._annotation(f'{where} ->', written.return_annotation, module)

    def _class(self, where: str, owner: type) -> None:
        if id(owner) in self._walked:
            return
        self._walked.add(id(owner))
        # A base of this project's own is read below as part of the class, so
        # a private one is an implementation detail. A base from a package
        # that is not public would tie a public class to it.
        for base in owner.__mro__[1:]:
            if (base.__module__ or '').split('.')[0] in INTERNAL_PACKAGES:
                self.found.append(
                    f'{where} inherits {base.__module__}.{base.__qualname__}'
                )
        for defined_on in owner.__mro__:
            module = defined_on.__module__ or ''
            if module.split('.')[0] not in FIRST_PARTY:
                continue
            annotations = vars(defined_on).get('__annotations__', {})
            for name, hint in annotations.items():
                if not name.startswith('_'):
                    self._annotation(f'{where}.{name}', hint, module)
            for name, member in vars(defined_on).items():
                dunder = name.startswith('__') and name.endswith('__')
                if name.startswith('_') and not dunder:
                    continue
                at = f'{where}.{name}'
                if isinstance(member, property):
                    for accessor in (member.fget, member.fset):
                        if accessor is not None:
                            self._signature(at, accessor)
                elif isinstance(member, (staticmethod, classmethod)):
                    self._signature(at, member.__func__)
                elif inspect.isfunction(member):
                    self._signature(at, member)
                elif inspect.isclass(member):
                    if not dunder:
                        self._type(at, member)
                elif not dunder:
                    self._type(f'{at} holds', type(member))


class TestPublicSurface:
    """What a caller can name, and what the endpoint receives."""

    def test_the_exported_api_matches_the_recorded_snapshot(self):
        result = _run('snapshot_public_api.py')
        assert result.returncode == 0, (
            'The public API changed. If that was intended, re-record it with '
            '`python scripts/snapshot_public_api.py --write` and say so in the '
            f'changelog.\n{result.stdout}{result.stderr}'
        )

    def test_no_public_name_names_a_type_that_is_not_public(self):
        """A type a caller is handed is one the caller can rely on.

        The snapshot records signatures as text, and records a property or an
        attribute by its name alone, so a public name could return or hold a
        type nobody exports, and the type would be public in all but name:
        `predictive_query.Profile` through `ModelSpec`, `ModelSpec` through a
        model handle's `spec`, `Decomposition` through `Request`. This reads
        the annotations and the values themselves.
        """
        finder = _InternalTypes(_public_packages())
        exposed = finder.run()
        assert not exposed, (
            'These public names name or hand out a type that is not public '
            '(docs/reference/stability.md). Make the name private, make the '
            'type public on purpose, or, if it truly must stay, add it to '
            'HANDED_OUT_THOUGH_NOT_EXPORTED with the reason:\n'
            + '\n'.join(exposed)
        )
        if _OVERLOADS is None:
            # Nothing an overload names was read, so nothing reached only
            # that way can be called unused.
            return
        stale = sorted(HANDED_OUT_THOUGH_NOT_EXPORTED - finder.allowed)
        assert not stale, (
            'Nothing public hands these out any more, so they no longer need '
            f'an exception in HANDED_OUT_THOUGH_NOT_EXPORTED: {stale}'
        )

    def test_the_check_finds_each_way_a_type_is_handed_out(
        self, tmp_path, monkeypatch
    ):
        """The check above passing means something only if it can fail.

        A probe module poses as part of `sdm_client` and hands out internal
        types each way the check reads, beside a function that names only
        public and third-party types and must pass.

        `quoted` writes its return as a string, which is a forward reference
        and names a type. `labelled` writes strings under `Literal`, which are
        values and name nothing. Removing every quoted run read the first as
        naming nothing at all.
        """
        probe = tmp_path / '_leak_probe.py'
        probe.write_text(
            'from __future__ import annotations\n'
            'import dataclasses\n'
            'from typing import TYPE_CHECKING, Literal\n'
            'import pandas as pd\n'
            'from predictive_query import Profile, validate\n'
            'from sdm_client.models import KUMO_TABULAR\n'
            'from sdm_client.request import Request\n'
            'if TYPE_CHECKING:\n'
            '    from sdm_client.base import ModelAdapter, ModelSpec\n'
            '    from sdm_client.decompose import Decomposition\n'
            '__all__ = [\n'
            "    'Field', 'Handle', 'Inherits', 'fine', 'labelled', 'mystery',\n"
            "    'quoted', 'rewritten',\n"
            "    'validate',\n"
            ']\n'
            'def rewritten() -> Decomposition | None: ...\n'
            'def quoted() -> "Decomposition": ...\n'
            "def labelled(x: Literal['a', 'b']) -> int: ...\n"
            'def mystery() -> NotDefinedAnywhere: ...\n'
            'def fine(frame: pd.DataFrame, n: int) -> list[Request]: ...\n'
            'class Handle:\n'
            '    spec = KUMO_TABULAR\n'
            '    def __init__(self, adapter: ModelAdapter) -> None: ...\n'
            '    @property\n'
            '    def described(self) -> ModelSpec: ...\n'
            '@dataclasses.dataclass\n'
            'class Field:\n'
            '    profile: Profile\n'
            'class Inherits(Profile):\n'
            '    pass\n'
        )
        name = 'sdm_client._leak_probe'
        spec = importlib.util.spec_from_file_location(name, probe)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)

        found = '\n'.join(_InternalTypes([*_public_packages(), module]).run())

        expected = [
            '_leak_probe.rewritten ->: sdm_client.decompose.Decomposition',
            '_leak_probe.quoted ->: sdm_client.decompose.Decomposition',
            "_leak_probe.mystery ->: cannot tell what 'NotDefinedAnywhere'",
            '_leak_probe.Handle.spec holds: sdm_client.base.ModelSpec',
            '_leak_probe.Handle.__init__(adapter): sdm_client.base.ModelAdapter',
            '_leak_probe.Handle.described ->: sdm_client.base.ModelSpec',
            '_leak_probe.Field.profile: predictive_query.profiles.Profile',
            '_leak_probe.Inherits inherits predictive_query',
            '_leak_probe.validate: predictive_query.api.validate comes from',
        ]
        missed = [line for line in expected if line not in found]
        assert not missed, f'not reported: {missed}\nreported:\n{found}'
        assert '_leak_probe.fine' not in found, found

    def test_the_wire_payloads_match_the_recorded_snapshot(self):
        result = _run('snapshot_wire_payloads.py')
        assert result.returncode == 0, (
            'A canonical request now serializes differently. If that was '
            'intended, re-record with `python scripts/snapshot_wire_payloads.py '
            f'--write`.\n{result.stdout}{result.stderr}'
        )


#: The contract to read the bodies against. CI clones it and says where, since
#: nothing is checked out beside the repository there; locally the sibling
#: checkout is the one being worked in.
CONTRACT = pathlib.Path(
    os.environ.get('SDM_CONTRACT_SPEC')
    or REPO.parent / 'structured-data-api' / 'nim-sd.openapi.yaml'
)

needs_the_contract = pytest.mark.skipif(
    not CONTRACT.is_file(),
    reason=(
        'needs the contract: set SDM_CONTRACT_SPEC, or check out '
        'structured-data-api beside this repository'
    ),
)


def _check_payloads(payloads: pathlib.Path | None = None):
    command = [
        sys.executable,
        str(REPO / 'scripts' / 'check_wire_payloads.py'),
        '--contract',
        str(CONTRACT),
    ]
    if payloads is not None:
        command += ['--payloads', str(payloads)]
    return subprocess.run(command, capture_output=True, text=True)


def _loaded(script: str):
    r"""One script as a module, by path.

    ``scripts`` is not a package, and importing it by name would rely on the
    repository root being on the path, which is true only for some ways of
    starting pytest.
    """
    spec = importlib.util.spec_from_file_location(
        script, REPO / 'scripts' / f'{script}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doctored(tmp_path: pathlib.Path, mutate) -> pathlib.Path:
    recorded = json.loads((GOLDEN / 'wire_payloads.json').read_text())
    entry = recorded['relational/pql-count-aggregation']
    mutate(entry[0] if isinstance(entry, list) else entry)
    written = tmp_path / 'wire_payloads.json'
    written.write_text(json.dumps(recorded))
    return written


class TestTheRequestBodiesAreContractual:
    """The bodies the client builds, read against the spec they are sent to.

    They are built as plain mappings rather than through the generated models,
    so nothing but this checks that what goes out is what the contract says.
    """

    @needs_the_contract
    def test_every_recorded_body_matches_the_contract(self):
        result = _check_payloads()
        assert result.returncode == 0, (
            f'A recorded request body is not what the contract declares.\n'
            f'{result.stdout}{result.stderr}'
        )

    @needs_the_contract
    @pytest.mark.parametrize(
        'mutate, said',
        [
            (lambda body: body.update(invented_key=1), 'invented_key'),
            (lambda body: body.update(modle=body.pop('model')), "'model'"),
            (
                lambda body: body['output'].update(fields=['predcition']),
                'predcition',
            ),
            (lambda body: body['task'].update(kind=7), 'not of type'),
            (lambda body: body.pop('schema'), "'schema'"),
            (lambda body: body['schema'].update(nonsense=1), 'nonsense'),
        ],
        ids=[
            'an invented top-level key',
            'a renamed required key',
            'a field off the enum',
            'a value of the wrong type',
            'a dropped section',
            'an invented key in a strict section',
        ],
    )
    def test_a_body_the_contract_does_not_declare_is_refused(
        self, tmp_path, mutate, said
    ):
        result = _check_payloads(_doctored(tmp_path, mutate))

        assert result.returncode == 1
        assert said in result.stderr

    @needs_the_contract
    @pytest.mark.parametrize(
        'sections, schema',
        [
            (
                ('model', 'task', 'schema', 'context', 'predict', 'output'),
                'PredictionRequest',
            ),
            (('model', 'task', 'schema', 'context'), 'SessionCreateRequest'),
            (('predict', 'output'), 'SessionPredictionRequest'),
        ],
        ids=['a prediction', 'a session', 'a session scoring call'],
    )
    def test_each_request_is_read_as_the_one_it_is(
        self, tmp_path, sections, schema
    ):
        """A session id travels in the path, so only the sections tell these
        apart. Reading one as another reports every field it wants as missing.
        """
        schema_for = _loaded('check_wire_payloads').schema_for

        recorded = json.loads((GOLDEN / 'wire_payloads.json').read_text())
        entry = recorded['relational/pql-count-aggregation']
        whole = entry[0] if isinstance(entry, list) else entry
        body = {key: whole[key] for key in sections if key in whole}

        assert schema_for(body) == schema

        written = tmp_path / 'wire_payloads.json'
        written.write_text(json.dumps({'case': body}))
        assert _check_payloads(written).returncode == 0

    @needs_the_contract
    def test_a_body_that_is_no_request_at_all_is_refused(self, tmp_path):
        written = tmp_path / 'wire_payloads.json'
        written.write_text(json.dumps({'case': {'model': 'kumo-relational'}}))

        result = _check_payloads(written)

        assert result.returncode == 1
        assert 'none of' in result.stderr

    @needs_the_contract
    @pytest.mark.parametrize(
        'recorded', [{}, {'case': []}], ids=['nothing', 'no bodies']
    )
    def test_a_recording_with_nothing_in_it_is_refused(
        self, tmp_path, recorded
    ):
        """A recording that lost its contents would otherwise report success."""
        written = tmp_path / 'wire_payloads.json'
        written.write_text(json.dumps(recorded))

        result = _check_payloads(written)

        assert result.returncode == 1
        assert 'records no request bodies' in result.stderr

    @needs_the_contract
    @pytest.mark.parametrize(
        'written, said',
        [
            (b'not json at all', 'is not JSON'),
            (b'[]', 'not calls by name'),
            (b'"text"', 'not calls by name'),
            (b'{"case": 1}', 'not a request body'),
            (b'{"case": [{}, 2]}', 'not a request body'),
            (b'\xff\xfe', 'is not UTF-8'),
            (None, 'cannot be read'),
            (b'{"case": ' + b'9' * 5000 + b'}', 'cannot be parsed'),
            (
                b'{"case": ' + b'[' * 10000 + b'0' + b']' * 10000 + b'}',
                'cannot be parsed',
            ),
            (b'{"case": {"model": NaN}}', 'not a JSON value'),
            (b'{"case": {"model": Infinity}}', 'not a JSON value'),
        ],
        ids=[
            'not json',
            'a list',
            'a string',
            'a scalar call',
            'a scalar among calls',
            'not utf-8',
            'no file at all',
            'an integer past the digit limit',
            'nesting past the recursion limit',
            'a not-a-number',
            'an infinity',
        ],
    )
    def test_a_recording_it_cannot_read_is_reported_not_raised(
        self, tmp_path, written, said
    ):
        """A malformed recording fails without a traceback."""
        recording = tmp_path / 'wire_payloads.json'
        if written is not None:
            recording.write_bytes(written)

        result = _check_payloads(recording)

        assert result.returncode == 1
        assert said in result.stderr
        assert 'Traceback' not in result.stderr

    @needs_the_contract
    def test_a_permissive_section_is_not_policed(self, tmp_path):
        """What this check cannot do, pinned so nobody assumes otherwise.

        Six of the schemas a request reaches declare
        `additionalProperties: true`, so a key invented inside one of them is
        contractual.
        """
        result = _check_payloads(
            _doctored(
                tmp_path,
                lambda body: body['output'].update(statisitc='median'),
            )
        )

        assert result.returncode == 0


_ENGINE_LICENSE = 'packages/structured-data-models-relational-engine/LICENSE'
_COMPONENTS = (
    PACKAGES
    / 'structured-data-models-relational-engine'
    / 'src'
    / 'sdm_relational_engine'
    / 'rfm'
    / 'assets'
    / 'MERMAID_COMPONENTS.json'
)
#: Mirrors scripts/mermaid_notices.py, where the reason for each is written.
_WITHOUT_A_NOTICE = frozenset(
    {
        'chevrotain',
        '@chevrotain/types',
        '@chevrotain/regexp-to-ast',
        '@chevrotain/gast',
        '@chevrotain/utils',
        '@chevrotain/cst-dts-gen',
        'robust-predicates',
    }
)


class TestTheNoticesAreComplete:
    """What an open-source release has to say about what it ships.

    The licenses of the bundled components ask for their copyright lines to be
    reproduced where they have one. Naming a component and its license is not
    that.
    """

    def test_the_license_files_carry_what_was_recorded(self):
        """The attribution is generated, so it cannot drift from its source.

        Hand-enumeration of the bundle did not converge: three rounds of it
        each missed more components, and the ones missed last were reached
        through other bundles rather than declared.
        """
        result = _run('mermaid_notices.py', '--check')

        assert result.returncode == 0, (
            'The license files no longer carry the attribution rendered from '
            'MERMAID_COMPONENTS.json. Run `python scripts/mermaid_notices.py '
            f'--write`.\n{result.stdout}{result.stderr}'
        )

    def test_every_recorded_component_has_a_notice_or_a_reason(self):
        """A license name is not a notice, and silence is not either."""
        recorded = json.loads(_COMPONENTS.read_text())['components']
        without = {
            component['name']
            for component in recorded.values()
            if not component['notices']
        }

        assert without <= _WITHOUT_A_NOTICE, (
            f'{sorted(without - _WITHOUT_A_NOTICE)} are attributed but carry '
            f'no copyright notice and are not recorded as having none'
        )
        assert len(recorded) > 50, (
            f'only {len(recorded)} components recorded, which is fewer than '
            f"mermaid's closure reaches; the refresh may have been truncated"
        )

    def test_the_readme_links_every_package_license(self):
        """A reader looking for the terms of one package should not have to
        guess that each directory holds its own copy.
        """
        readme = (REPO / 'README.md').read_text()
        packages = sorted(p.name for p in PACKAGES.iterdir() if p.is_dir())

        for package in packages:
            link = f'packages/{package}/LICENSE'
            assert link in readme, f'the README does not link {link}'
            assert (REPO / link).is_file(), f'{link} is linked but missing'

    def test_the_engine_license_says_what_the_root_one_says(self):
        """The engine ships the bundle, so its own copy carries the same
        notices. They differ only in the path each gives for the bundle.
        """

        def attribution(path):
            text = (REPO / path).read_text()
            text = text[text.index('THIRD-PARTY SOFTWARE BUNDLED') :]
            # Every line, not the ones that look like notices: a URL or an
            # "All rights reserved." is part of a notice too, and comparing
            # only the lines carrying the word let those drift apart.
            lines = text.splitlines()
            return [line for line in lines if not line.endswith('.min.js')]

        root = attribution('LICENSE')
        engine = attribution(_ENGINE_LICENSE)
        differing = [
            (a, b) for a, b in zip(root, engine, strict=False) if a != b
        ]

        assert len(root) == len(engine), (
            f'the two license files carry {len(root)} and {len(engine)} '
            f'lines of attribution; they should carry the same'
        )
        assert not [pair for pair in differing if not set(pair[0]) <= {'-'}], (
            f'the two license files disagree: {differing[:3]}'
        )


class TestErrorContract:
    """Exception identity is what callers branch on, so it is pinned."""

    def test_every_error_descends_from_one_base(self):
        import sdm_client
        from sdm_client import StructuredDataError

        errors = [
            getattr(sdm_client, name)
            for name in sdm_client.__all__
            if isinstance(getattr(sdm_client, name), type)
            and issubclass(getattr(sdm_client, name), Exception)
            and not issubclass(getattr(sdm_client, name), Warning)
        ]
        assert errors, 'no exception types are exported'
        for error in errors:
            assert issubclass(error, StructuredDataError), (
                f'{error.__name__} is raised at users but does not descend from '
                f'StructuredDataError, so `except StructuredDataError` misses it'
            )

    def test_the_warning_category_is_the_library_s_own(self):
        from sdm_client import SDMWarning

        assert issubclass(SDMWarning, Warning)
        assert not issubclass(SDMWarning, DeprecationWarning), (
            'SDMWarning must stay filterable on its own terms'
        )


class TestDistributionCoherence:
    """The five are released together, so they must agree."""

    @staticmethod
    def _declared_version() -> str:
        versions = TestDistributionCoherence._declared_versions()
        assert len(set(versions.values())) == 1, versions
        return next(iter(versions.values()))

    @staticmethod
    def _declared_versions() -> dict[str, str]:
        versions = {}
        for name in _distributions():
            project = _pyproject(name)['project']
            if 'version' in project:
                versions[name] = project['version']
                continue
            # Read off the tree rather than derived from the distribution
            # name: the two have never matched for predictive_query, and
            # since the rename they match for none of the five.
            roots = [
                child
                for child in (PACKAGES / name / 'src').iterdir()
                if child.is_dir() and (child / '__init__.py').exists()
            ]
            assert len(roots) == 1, f'{name} has import roots {roots}'
            source = roots[0] / '_version.py'
            text = source.read_text()
            versions[name] = text.split('=')[-1].strip().strip('\'"')
        return versions

    def test_every_distribution_declares_the_same_version(self):
        versions = self._declared_versions()
        assert len(set(versions.values())) == 1, (
            f'distributions disagree on the version: {versions}'
        )

    def test_every_first_party_pin_names_the_version_being_released(self):
        """The five pin each other by exact version, and those pins are literals.

        `test_every_distribution_declares_the_same_version` proves the five
        agree on what they *are*. It says nothing about what they *require*,
        and those are thirteen separate hard-coded strings across two files.
        """
        declared = self._declared_version()
        distributions = {name for name in _distributions()}

        # Any exactly pinned requirement, then judged. Reading only the five
        # real names would make a pin on a name nothing publishes -- a
        # survivor of the rename, or a typo -- invisible rather than wrong,
        # which is the regression this is here to catch. Everything first
        # party is spelled with one of these two stems, and nothing else
        # pinned here is.
        ours = re.compile(r'^(?:structured[-_]data[-_]models|sdm)[-_]', re.I)
        pattern = re.compile(
            r'(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)'
            r'(?:\[[^\]]*\])?==(?P<version>[0-9][^\'"\s,]*)'
        )
        wrong = []
        for source in sorted(PACKAGES.glob('*/pyproject.toml')) + sorted(
            PACKAGES.glob('*/setup.py')
        ):
            for match in pattern.finditer(source.read_text()):
                name = match.group('name')
                if not ours.match(name):
                    continue
                where = f'{source.relative_to(REPO)}: {match.group(0)}'
                if name not in distributions:
                    wrong.append(f'{where} names no distribution we publish')
                elif match.group('version') != declared:
                    wrong.append(f'{where} but the release is {declared}')
        assert not wrong, (
            'a first-party pin names a version, or a distribution, other '
            'than the one being released:\n' + '\n'.join(wrong)
        )

    def test_every_distribution_declares_the_same_licence(self):
        licences = {
            name: _pyproject(name)['project'].get('license')
            for name in _distributions()
        }
        assert set(licences.values()) == {'Apache-2.0'}, licences

    def test_every_distribution_ships_its_licence_files(self):
        for name in _distributions():
            declared = _pyproject(name)['project'].get('license-files')
            assert declared, f'{name} declares no license-files'
            for filename in declared:
                assert (PACKAGES / name / filename).is_file(), (
                    f'{name} declares {filename} but does not carry it'
                )

    def test_every_distribution_ships_a_typed_marker(self):
        for name in _distributions():
            markers = list((PACKAGES / name / 'src').glob('*/py.typed'))
            assert markers, f'{name} claims to be typed but ships no py.typed'

    def test_the_declared_python_floor_is_the_same_everywhere(self):
        floors = {}
        for name in _distributions():
            requires = _pyproject(name)['project']['requires-python']
            floors[name] = requires.split(',')[0].strip()
        assert len(set(floors.values())) == 1, (
            f'distributions disagree on the minimum Python: {floors}'
        )

    def test_every_advertised_extra_of_the_client_resolves_to_a_real_target(
        self,
    ):
        client = _pyproject('structured-data-models-client')['project']
        known = set(_distributions())
        for extra, requirements in client['optional-dependencies'].items():
            for requirement in requirements:
                name = (
                    requirement.split('[')[0]
                    .split('==')[0]
                    .split('>=')[0]
                    .strip()
                )
                # databricks-sdk is imported by the client's own serving
                # transport, so the client is the distribution that declares it.
                assert name in known or name in {
                    'pytest',
                    'requests-mock',
                    'databricks-sdk',
                }, (
                    f'extra {extra!r} requires {name!r}, which is not one of '
                    f"this repository's distributions"
                )

    def test_the_databricks_extra_brings_what_reading_a_metric_view_needs(
        self,
    ):
        """A metric view reached through `semantic_model` is parsed as YAML.

        The client's `databricks` extra reaches Databricks through the
        connectors' own, and Databricks Runtime 14.3 LTS has no YAML parser, so
        a metric view failed there with `No module named 'yaml'` until the
        connectors' extra carried one.
        """
        client = _pyproject('structured-data-models-client')['project'][
            'optional-dependencies'
        ]
        assert (
            'structured-data-models-connectors[databricks]==1.0.0'
            in client['databricks']
        )
        connectors = _pyproject('structured-data-models-connectors')['project']
        names = [
            requirement.split('>=')[0].split('==')[0].strip().lower()
            for requirement in connectors['optional-dependencies']['databricks']
        ]
        assert 'pyyaml' in names

    def test_the_lakebase_extra_brings_plain_psycopg_and_a_postgres_sdk(self):
        """`lakebase()` calls `WorkspaceClient.postgres`, added in 0.89.

        Plain psycopg, because the binary wheel's OpenSSL fails its FIPS
        self-test on Databricks' aarch64 hosts and aborts the process.
        """
        client = _pyproject('structured-data-models-client')['project'][
            'optional-dependencies'
        ]
        assert client['lakebase'] == [
            'structured-data-models-connectors[lakebase]==1.0.0'
        ]
        (everything,) = client['all']
        assert 'lakebase' in everything.partition('[')[2].rstrip(']').split(',')
        connectors = _pyproject('structured-data-models-connectors')['project']
        extra = connectors['optional-dependencies']['lakebase']
        assert sorted(extra) == ['databricks-sdk>=0.89,<1.0', 'psycopg>=3.1']


class TestConfigurationTellsTheTruth:
    """A comment that states a checkable fact is checked.

    The relational engine's lint configuration carried a note saying its ignored
    rules were enabled in two sibling packages. One of them was not: the rule
    belongs to a rule family the shared configuration never selected. Nothing
    """

    def test_the_shared_lint_baseline_is_not_narrowed_by_any_package(self):
        root = tomllib.loads((REPO / 'pyproject.toml').read_text())
        baseline = set(root['tool']['ruff']['lint']['select'])
        for name in _distributions():
            config = _pyproject(name).get('tool', {}).get('ruff', {})
            assert config.get('extend', '').endswith('pyproject.toml'), (
                f'{name} does not extend the shared ruff configuration, so it '
                f'can drift from the baseline silently'
            )
            narrowed = set(config.get('lint', {}).get('select', baseline))
            assert baseline <= narrowed or 'select' not in config.get(
                'lint', {}
            ), f'{name} narrows the shared lint baseline to {narrowed}'

    def test_the_ignore_list_sorts_itself_the_way_its_headings_say(self):
        """The engine's ignore list is written in two labelled blocks.

        One is rules the shared baseline really does select, so listing them
        here silences something; the other is a record of rules nothing selects
        anywhere. An earlier note called the whole list inert, which was wrong
        """
        source = (
            PACKAGES
            / 'structured-data-models-relational-engine'
            / 'pyproject.toml'
        ).read_text()
        root = tomllib.loads((REPO / 'pyproject.toml').read_text())
        selected = root['tool']['ruff']['lint']['select']
        engine = _pyproject('structured-data-models-relational-engine')['tool'][
            'ruff'
        ]['lint']
        effective = list(selected) + list(engine.get('extend-select', []))

        active_block, _, backlog_block = source.partition(
            '--- Recorded backlog'
        )
        _, _, active_block = active_block.partition('--- Active exemptions')
        assert active_block and backlog_block, (
            'the ignore list no longer carries both headings this test reads'
        )

        def listed(block: str) -> set[str]:
            return {
                rule
                for rule in engine.get('ignore', [])
                if f'"{rule}"' in block
            }

        for rule in listed(active_block):
            assert _covered_by(rule, effective), (
                f'{rule} sits under "Active exemptions" but nothing selects '
                f'it, so ignoring it silences nothing'
            )
        for rule in listed(backlog_block):
            assert not _covered_by(rule, selected), (
                f'{rule} sits under "Recorded backlog", which says the shared '
                f'baseline does not select it, but {selected} does'
            )

    def test_the_tabular_task_kinds_match_what_the_model_declares(self):
        """`TABULAR_TASK_KINDS` restates what `models.py` says, and says so.

        The materializer refuses a stated `task=` the tabular deployment does
        not serve, and it has to do that before a model handle is necessarily
        in hand, so the list is written twice. A comment claiming two things
        """
        from sdm_client.materializers.tabular import TABULAR_TASK_KINDS
        from sdm_client.models import spec_for

        assert TABULAR_TASK_KINDS == spec_for('kumo-tabular').runtime.tasks, (
            'the materializer and the model spec disagree about what the '
            'tabular deployment serves'
        )

    @staticmethod
    def _module_is_on_disk(dotted: str) -> bool | None:
        """Whether ``dotted`` exists as a file, without importing anything.

        Three times now this check has confused "absent" with something else,
        because it asked by importing. ``find_spec`` executes every parent
        ``__init__`` on the way down, so it reports a module as missing when a
        """
        import importlib.machinery
        import importlib.util

        top, *rest = dotted.split('.')
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):  # pragma: no cover - not installed
            return None
        if spec is None:
            return None
        if not rest:
            return True
        roots = spec.submodule_search_locations or []
        for root in roots:
            base = pathlib.Path(root, *rest[:-1])
            leaf = rest[-1]
            if (base / leaf / '__init__.py').is_file():
                return True
            if any(
                (base / f'{leaf}{suffix}').is_file()
                for suffix in importlib.machinery.all_suffixes()
            ):
                return True
            # A directory with no __init__.py is a namespace package and is
            # importable all the same. This tree has thirteen of them, so
            # requiring the file would call an existing package missing.
            if (base / leaf).is_dir():
                return True
        return False

    def test_every_module_the_type_baseline_names_still_exists(self):
        """A baseline entry for a module that is gone silences nothing.

        The engine was restructured on main and three entries kept naming
        modules that no longer exist, while the errors they covered moved to
        new ones. Nothing complains about an override that matches nothing, so
        """
        root = tomllib.loads((REPO / 'pyproject.toml').read_text())
        named = [
            module
            for override in root['tool']['mypy']['overrides']
            for module in (
                override['module']
                if isinstance(override['module'], list)
                else [override['module']]
            )
        ]
        assert named, 'the type-checking baseline vanished; update this test'

        checked, missing = 0, []
        for module in named:
            present = self._module_is_on_disk(module)
            if present is None:
                continue  # the distribution is absent from this job
            checked += 1
            if not present:
                missing.append(module)

        assert checked, (
            'no distribution the baseline names is installed here, so this '
            'checked nothing'
        )
        assert not missing, (
            f'the mypy baseline names modules that do not exist: {missing}. '
            f'Delete the entry, or point it at whatever the code became.'
        )

    def test_the_recorded_style_backlog_is_clean_where_the_note_says_it_is(
        self,
    ):
        """The engine's ignore list records a backlog and names where it is clean.

        That claim used to be wrong in both halves, the rules were not
        selected anywhere, and the two packages named did not pass clean under
        them. A comment cannot be wrong for long if a test reads it.
        """
        engine = _pyproject('structured-data-models-relational-engine')['tool'][
            'ruff'
        ]['lint']
        root = tomllib.loads((REPO / 'pyproject.toml').read_text())
        selected = root['tool']['ruff']['lint']['select']
        backlog = [
            rule
            for rule in engine.get('ignore', [])
            if _linter_of(rule) != 'D' and not _covered_by(rule, selected)
        ]
        assert backlog, 'the recorded style backlog vanished; update this test'

        ruff = pathlib.Path(sys.executable).parent / 'ruff'
        if not ruff.exists():  # pragma: no cover - ruff not installed
            import pytest

            pytest.skip('ruff is not installed in this environment')

        result = subprocess.run(
            [
                str(ruff),
                'check',
                '--no-cache',
                '--quiet',
                '--select',
                ','.join(backlog),
                str(PACKAGES / 'structured-data-models-client' / 'src'),
                str(PACKAGES / 'structured-data-models-connectors' / 'src'),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            'structured-data-models-client and structured-data-models-connectors no longer pass clean under the '
            'style backlog the engine records, so the note beside that ignore '
            f'list is now wrong:\n{result.stdout}{result.stderr}'
        )


class TestDocumentedExamples:
    """Examples are the most-read code in the repository and nothing ran them.

    Executing them needs a live NIM, so that belongs in the live suite. What can
    be checked with nothing running is still most of what goes wrong in a
    published example: it stopped being valid Python, or it names something the
    """

    @staticmethod
    def _python_blocks() -> list[tuple[str, str]]:
        import re

        blocks = []
        sources = [REPO / 'README.md'] + sorted((REPO / 'docs').rglob('*.md'))
        for path in sources:
            for match in re.finditer(
                r'^```python\n(.*?)^```', path.read_text(), re.S | re.M
            ):
                blocks.append((str(path.relative_to(REPO)), match.group(1)))
        return blocks

    def test_every_documented_python_block_is_valid_python(self):
        import ast

        broken = []
        for where, source in self._python_blocks():
            try:
                ast.parse(source)
            except SyntaxError as error:
                broken.append(f'{where}: {error}')
        assert not broken, (
            'a fenced ```python block no longer parses. Make it runnable, or '
            'fence it as ```text if it is deliberately a fragment:\n'
            + '\n'.join(broken)
        )

    def test_every_symbol_the_documentation_imports_still_exists(self):
        import ast
        import importlib

        missing = []
        for where, source in self._python_blocks():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith(
                    ('sdm_', 'predictive_query')
                ):
                    continue
                try:
                    module = importlib.import_module(node.module)
                except ImportError:  # an extra this environment lacks
                    continue
                missing += [
                    f'{where}: {node.module}.{alias.name}'
                    for alias in node.names
                    if not hasattr(module, alias.name)
                ]
        assert not missing, (
            'the documentation imports names the library no longer exports:\n'
            + '\n'.join(missing)
        )

    def test_the_readme_quickstart_imports_only_public_names(self):
        import ast
        import importlib

        readme = (REPO / 'README.md').read_text()
        import re

        for match in re.finditer(r'^```python\n(.*?)^```', readme, re.S | re.M):
            for node in ast.walk(ast.parse(match.group(1))):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == 'sdm_client'
                ):
                    exported = importlib.import_module('sdm_client').__all__
                    outside = [
                        alias.name
                        for alias in node.names
                        if alias.name not in exported
                    ]
                    assert not outside, (
                        f'the README imports {outside} from sdm_client, which '
                        f'is not in __all__, so the first thing a new user '
                        f'copies is an unsupported import'
                    )


class TestInterpreterSupportIsStated:
    """`requires-python` and the classifiers are two claims about one thing.

    They disagreed: every distribution admitted 3.14 through `requires-python`
    and none of them listed it, so an index would show a package as unsupported
    on an interpreter it runs on perfectly well.
    """

    KNOWN = ('3.10', '3.11', '3.12', '3.13', '3.14')

    @staticmethod
    def _parts(version: str) -> tuple[int, ...]:
        """Compare versions numerically.

        `'3.10' >= '3.9'` is False as strings, because `'1' < '9'`. Nothing
        here has a 3.9 floor today, so a lexical comparison happened to agree,
        and it would have started disagreeing silently at 3.20, in a helper
        every version comparison in this file runs through.
        """
        return tuple(int(part) for part in version.split('.'))

    @classmethod
    def _admitted(cls, requires: str) -> list[str]:
        floor = ceiling = None
        inclusive = False
        for clause in requires.split(','):
            clause = clause.strip()
            if clause.startswith('>='):
                floor = cls._parts(clause[2:].strip())
            elif clause.startswith('<='):
                ceiling = cls._parts(clause[2:].strip())
                inclusive = True
            elif clause.startswith('<'):
                ceiling = cls._parts(clause[1:].strip())
        assert floor is not None, f'no lower bound in {requires!r}'
        admitted = [v for v in cls.KNOWN if cls._parts(v) >= floor]
        if ceiling is not None:
            admitted = [
                v
                for v in admitted
                if (
                    cls._parts(v) <= ceiling
                    if inclusive
                    else cls._parts(v) < ceiling
                )
            ]
        return admitted

    def test_the_version_comparison_this_class_relies_on_is_numeric(self):
        """The helper above is the only thing standing between a metadata
        mistake and a release, so it is tested rather than trusted.
        """
        assert self._parts('3.10') > self._parts('3.9')
        assert self._parts('3.9') < self._parts('3.14')
        assert self._admitted('>=3.10') == list(self.KNOWN)
        assert self._admitted('>=3.10,<3.14') == [
            '3.10',
            '3.11',
            '3.12',
            '3.13',
        ]
        assert self._admitted('>=3.12') == ['3.12', '3.13', '3.14']
        assert self._admitted('>=3.10,<=3.13') == [
            '3.10',
            '3.11',
            '3.12',
            '3.13',
        ]

    def test_every_admitted_interpreter_is_also_a_classifier(self):
        for name in _distributions():
            project = _pyproject(name)['project']
            admitted = self._admitted(project['requires-python'])
            classified = {
                line.rsplit('::', 1)[-1].strip()
                for line in project.get('classifiers', [])
                if line.startswith('Programming Language :: Python :: 3.')
            }
            missing = [v for v in admitted if v not in classified]
            assert not missing, (
                f'{name} admits Python {missing} through requires-python '
                f'{project["requires-python"]!r} but does not classify them, so '
                f'an index reports it unsupported on an interpreter it runs on'
            )

    def test_an_extra_narrower_than_its_parent_says_so(self):
        """The relational extra does not resolve on every Python the client does.

        That is correct, the engine is a native build with a shorter wheel
        matrix, but pip's error names a distribution, not a reason. Anything
        that narrow has to be explained where someone reads it.
        """
        client = _pyproject('structured-data-models-client')
        engine = _pyproject('structured-data-models-relational-engine')
        if (
            client['project']['requires-python']
            == engine['project']['requires-python']
        ):
            return
        source = (
            PACKAGES / 'structured-data-models-client' / 'pyproject.toml'
        ).read_text()
        assert '3.14' in source and 'native build' in source, (
            'structured-data-models-client admits interpreters its [relational] extra cannot '
            'install on, and its pyproject does not explain that anywhere'
        )
        install_doc = (
            REPO / 'docs' / 'get-started' / 'installation.md'
        ).read_text()
        assert 'relational' in install_doc, (
            'the installation guide does not mention the relational extra'
        )
