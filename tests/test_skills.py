# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent skills must keep matching the shipped client.

A skill is read by something that cannot check it. Everything here is the
check: the names the skills import exist, the queries they teach resolve to the
task they claim, and the refusals they promise are the refusals that happen.

Most of it needs no endpoint: the tables are read out of the markdown and put
to the client, which answers without connecting. The rest is marked
``live_nim`` and needs one of each model, through ``SDM_TABULAR_URL`` and
``SDM_RELATIONAL_URL``. Those re-run every example that can be run -- see
``complete()`` -- and compare what it printed to the block pasted beneath it,
and probe the handful of claims the skills make about a deployment rather than
about this client.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import os
import re
import sys
import warnings
from pathlib import Path

import pytest
import sdm_client

if sys.version_info >= (3, 11):
    import tomllib
else:  # The packages support 3.10, where tomllib is not in the stdlib.
    import tomli as tomllib

SKILLS = Path(__file__).parents[1] / 'skills'

#: Arguments earlier generations of this client took. The retired *type* names
#: are covered repo-wide by ``scripts/check_repo_hygiene.py retired``; these two
#: are keyword arguments it does not look for, and both are easy to reach for:
#: an explanation is an output here, and a run mode is derived from the limits.
RETIRED = ('run_mode=', 'explain=')

FENCE = re.compile(r'^```(\w*)\n(.*?)^```', re.MULTILINE | re.DOTALL)
PREDICTS = re.compile(r'\b\w+\.predict\(')
LINK = re.compile(r'\]\(([^)]+)\)')


def documents() -> list[Path]:
    return sorted(SKILLS.rglob('*.md'))


def fences(path: Path, language: str) -> list[str]:
    return [
        body
        for tag, body in FENCE.findall(without_comments(path))
        if tag == language
    ]


def test_there_is_a_skill_for_each_model():
    assert (SKILLS / 'tabular' / 'SKILL.md').exists()
    assert (SKILLS / 'relational' / 'SKILL.md').exists()
    assert (SKILLS / 'README.md').exists()


def bound_by(source: str) -> set[str]:
    """Every name an example leaves behind for the ones after it."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def unbound_in(source: str, known: set[str]) -> set[str]:
    loaded = {
        node.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return loaded - known - set(dir(builtins))


def complete(path: Path) -> list[bool]:
    """For each example, whether it can run on its own by then.

    A reference about a warehouse has to show a Databricks session and a
    Snowflake one, and neither exists to run against; those blocks are code
    and are checked as code, but nothing can execute them. The rest are
    whole programs, and the difference is decidable here rather than
    declared: an example is complete when every name it reads was bound by
    an earlier one or by itself. A fragment that reuses a name an earlier
    example bound is run, and fails there rather than passing quietly, which
    is why the connections in those blocks are named for what they are.
    """
    known: set[str] = set()
    whether: list[bool] = []
    for source in fences(path, 'python'):
        whether.append(not unbound_in(source, known | bound_by(source)))
        if whether[-1]:
            known |= bound_by(source)
    return whether


@pytest.mark.parametrize('path', documents(), ids=str)
def test_an_example_that_predicts_can_be_run(path):
    """An incomplete example cannot be checked against an endpoint, so one
    that predicts would be the one claim nothing stands behind."""
    for index, (source, runs) in enumerate(
        zip(fences(path, 'python'), complete(path), strict=True)
    ):
        if PREDICTS.search(source):
            assert runs, (
                f'{path.name} example {index} predicts and reads '
                f'{sorted(unbound_in(source, bound_by(source)))}, which '
                f'nothing before it binds, so no endpoint can be asked to '
                f'confirm what it prints'
            )


@pytest.mark.parametrize('path', documents(), ids=str)
def test_an_example_that_predicts_shows_what_came_back(path):
    """An example with no answer under it leaves its shape to be guessed.

    Per example, not per document: the ranking's return shape -- two columns,
    one of them dicts -- was the least guessable thing in the set, and it sat
    in a file that already carried somebody else's verified block.
    """
    for index, (source, shown) in enumerate(examples(path)):
        if PREDICTS.search(source):
            assert shown is not None, (
                f'{path.name} example {index} predicts and shows no output; '
                f'paste what it printed under a **Verified** line'
            )


@pytest.mark.parametrize('name', ['tabular', 'relational'])
def test_a_skill_shows_output_it_has_run(name):
    """Each skill has to stake something on a live endpoint.

    The live suite compares every **Verified** block to what the example
    printed; a skill with none would pass it by having claimed nothing, and
    one whose blocks are all commented out would pass by claiming it where
    nobody reads.
    """
    body = without_comments(SKILLS / name / 'SKILL.md')
    assert '**Verified**' in body, f'{name} marks no output as verified'


@pytest.mark.parametrize('name', ['tabular', 'relational'])
def test_a_skill_announces_itself(name):
    """An agent finds a skill by its frontmatter, so it has to be there."""
    yaml = pytest.importorskip('yaml')
    body = (SKILLS / name / 'SKILL.md').read_text(encoding='utf-8')
    assert body.startswith('---\n'), f'{name} opens with no frontmatter'
    front = yaml.safe_load(body[4 : body.index('\n---\n', 3)])
    assert isinstance(front, dict), f'{name} frontmatter is not a mapping'
    assert front.get('name') == f'sdm-{name}', (
        f'{name} is announced as {front.get("name")!r}'
    )
    assert 'Triggers:' in front.get('description', ''), (
        f'{name} gives no triggers, so nothing would load it'
    )


@pytest.mark.parametrize('path', documents(), ids=str)
def test_a_document_is_written_in_the_one_markdown_this_file_reads(path):
    """Fences are ```` ``` ```` at the margin, and nothing is indented code.

    Markdown also accepts tilde fences, longer fences, fences indented up to
    three spaces, and four-space indented code blocks. This file reads one
    form of each, so rather than teach three regexes the whole of CommonMark,
    the documents are held to the form they already use. A table or an example
    written any other way would be read by a renderer and not by these tests,
    or the other way about: a ``py`` block would look like an example and run
    as nothing, and a fence inside a blockquote or a table without its outer
    pipes renders as itself while every check here steps over it.
    """
    body = path.read_text(encoding='utf-8')
    assert body.endswith('\n'), (
        f'{path.name} ends without a newline, and a fence closing there is '
        f'read as unclosed'
    )
    inside = False
    previous = ''
    said_last = ''
    for number, line in enumerate(body.splitlines(), start=1):
        where = f'{path.name}:{number}'
        if line == '```python':
            assert '**Verified**' not in said_last, (
                f'{where} gives verified output a python fence, which makes '
                f'it an example instead and pairs it with nothing'
            )
        if not inside and line.startswith('|'):
            assert previous == '' or previous.startswith('|'), (
                f'{where} begins a table with no blank line above it; under a '
                f'list or a paragraph it renders as more of that and is read '
                f'here as a table'
            )
        previous = line
        if line.strip():
            said_last = line
        if line.lstrip().startswith(('```', '~~~')):
            allowed = ('```',) if inside else ('```', '```python')
            assert line in allowed, (
                f'{where} {"closes" if inside else "opens"} a fence as '
                f'{line.strip()!r}; a fence is ``` at the margin, only an '
                f'opening one names a language, and python is the only name, '
                f'because every python block here is run'
            )
            inside = not inside
        elif not inside:
            assert not line.startswith(('    ', '\t')), (
                f'{where} is indented, by spaces or by a tab, which renders '
                f'as code and is read here as prose'
            )
            assert not line.lstrip().startswith('>'), (
                f'{where} is a blockquote, and a fence or a table inside one '
                f'renders as itself and is read here as prose'
            )
            assert '|' not in line or (
                line.startswith('|') and line.rstrip().endswith('|')
            ), (
                f'{where} holds a pipe and is not a table row; a table is '
                f'written with outer pipes, because one without them renders '
                f'as a table and is read here as prose'
            )
            assert '```' not in line, (
                f'{where} opens a fence part-way along a line, after a list '
                f'marker or otherwise; a fence begins its own line'
            )
            bare_comment = re.fullmatch(r'\s*(<!--.*|.*-->|<!--.*-->)\s*', line)
            assert not re.search(r'<[a-zA-Z/!?]', line) or (
                bare_comment and not re.search(r'<[a-zA-Z/]', line)
            ), (
                f'{where} holds raw HTML, which renders as a table or a code '
                f'block and is read here as nothing. It need not begin the '
                f'line to do that'
            )
            assert not re.search(
                r'&(#\d+|#[xX][0-9a-fA-F]+|[a-zA-Z]+);', line
            ), (
                f'{where} writes an HTML entity, which renders as a character '
                f'this file reads as the escape it is written with'
            )
            for abrupt in ('--!>', '<!-->', '<!--->'):
                assert abrupt not in line, (
                    f'{where} writes {abrupt}, which a renderer and this file '
                    f'disagree about the end of, so the document says one '
                    f'thing on the page and another here'
                )
            for marker in ('<!--', '-->'):
                assert (
                    marker not in line
                    or line.strip().startswith(marker)
                    or line.strip().endswith(marker)
                ), (
                    f'{where} writes {marker} part-way along a line, where a '
                    f'code span may render it and this file will not'
                )
    assert not inside, f'{path.name} leaves a fence open'


#: Every table each document holds, by its header row, and whether anything
#: here reads it. Exhaustive on purpose: an unlisted table can say the
#: opposite of a listed one and a reader cannot tell which was checked. The
#: two marked False are prose -- which skill to load, and a refusal beside its
#: cause -- with nothing in them to put to the client; listing them says so
#: rather than leaving them to look checked.
EVERY_TABLE = {
    'skills/README.md': [
        (['Skill', 'Model', 'Load it when'], False),
        (['Reference', ''], False),
    ],
    'skills/relational/SKILL.md': [
        (['', '`kumo-relational`'], True),
        (['Limit', 'Value'], True),
        (['Refusal', 'Cause'], False),
    ],
    'skills/relational/references/beyond-one-answer.md': [],
    'skills/relational/references/warehouses.md': [
        (['Source', 'Connection it takes', 'Namespace', 'Extra'], True)
    ],
    'skills/relational/pql.md': [
        (['Clause', 'Least', 'Most'], True),
        (['Target', 'Task', 'Calls'], True),
        (['Aggregation', ''], True),
        (['Comparison', ''], True),
    ],
    'skills/tabular/SKILL.md': [
        (['', '`kumo-tabular`'], True),
        (['Refusal', 'Cause'], False),
    ],
    'skills/tabular/references/pql.md': [(['Clause', 'On one table'], True)],
}


@pytest.mark.parametrize('path', documents(), ids=str)
def test_every_table_in_a_skill_is_accounted_for(path):
    """A new table has to be listed here before it can be shipped.

    Listed, and marked as read or not. Most carry something that can be put
    to the client and are; two are prose and say so, which is better than
    sitting among the others looking checked.
    """
    where = str(path.relative_to(SKILLS.parent))
    held = [one[0] for one in tables(path)]
    assert held == [head for head, _ in EVERY_TABLE[where]], (
        f'{path.name} holds tables {held} and this file accounts for '
        f'{[head for head, _ in EVERY_TABLE[where]]}'
    )


@pytest.mark.parametrize('path', documents(), ids=str)
def test_every_link_resolves(path):
    for target in LINK.findall(path.read_text(encoding='utf-8')):
        if target.startswith(('http://', 'https://', '#')):
            continue
        assert (path.parent / target).resolve().exists(), (
            f'{path.name} links to {target}, which is not there'
        )


@pytest.mark.parametrize('path', documents(), ids=str)
def test_verified_output_is_marked_one_way(path):
    """``**Verified**`` at the start of a line labels the block beneath it.

    Anything else that renders as an emphasised Verified -- ``__Verified__``,
    a heading, a bullet -- still reads as a label on the page and pairs with
    nothing here. The HTML and entity forms are refused by the markdown
    policy; these are the rest.
    """
    body = path.read_text(encoding='utf-8')
    assert '__Verified__' not in body, (
        f'{path.name} writes __Verified__, which renders as **Verified** and '
        f'pairs with no example'
    )
    assert not re.search(r'(?<!\*)\*Verified\*(?!\*)', body), (
        f'{path.name} writes *Verified*, which reads as a label and pairs '
        f'with no example'
    )
    for number, line in enumerate(body.splitlines(), start=1):
        if 'Verified' not in line:
            continue
        marker = line.startswith('**Verified**')
        prose = not set(line) & set('*[]`_')
        assert marker or prose, (
            f'{path.name}:{number} emphasises or links Verified somewhere '
            f'other than the start of a line, where it reads as a label and '
            f'pairs with nothing'
        )


def reaching_this_client(path: Path) -> set[str]:
    """Every name in a document's examples that holds something of ours.

    Imports, aliases, and whatever those get bound to, to a fixed point: a
    name may be bound after the thing that uses it.
    """
    reachable = set(sdm_client.__all__) | {'sdm_client', 'pandas'}
    trees: list[ast.Module] = []
    for source in fences(path, 'python'):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, 'module', None) or ''
                for alias in node.names:
                    named = module or alias.name
                    if named.split('.')[0] in {'sdm_client', 'pandas'}:
                        reachable.add(alias.asname or alias.name)
        trees.append(tree)
    grew = True
    while grew:
        grew = False
        for tree in trees:
            for node in ast.walk(tree):
                sources: list[ast.AST] = []
                targets: list[ast.AST] = []
                if isinstance(node, ast.Assign):
                    sources, targets = [node.value], list(node.targets)
                elif isinstance(node, ast.NamedExpr):
                    sources, targets = [node.value], [node.target]
                elif isinstance(node, ast.AnnAssign):
                    if node.value is None:
                        continue
                    sources, targets = [node.value], [node.target]
                elif isinstance(node, ast.With):
                    sources = [item.context_expr for item in node.items]
                    targets = [
                        item.optional_vars
                        for item in node.items
                        if item.optional_vars is not None
                    ]
                elif isinstance(node, ast.For):
                    sources, targets = [node.iter], [node.target]
                elif isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Attribute
                ):
                    sources = [
                        *node.args,
                        *(one.value for one in node.keywords),
                    ]
                    targets = [node.func.value]
                elif isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    sources = [*node.body, *node.decorator_list]
                    if isinstance(node, ast.ClassDef):
                        sources += [*node.bases, *node.keywords]
                    else:
                        sources += [
                            *node.args.defaults,
                            *[one for one in node.args.kw_defaults if one],
                        ]
                    targets = [ast.Name(id=node.name, ctx=ast.Store())]
                else:
                    continue
                if not any(
                    isinstance(inner, ast.Name) and inner.id in reachable
                    for source in sources
                    for inner in ast.walk(source)
                ):
                    continue
                bound = {
                    name.id
                    for target in targets
                    for name in ast.walk(target)
                    if isinstance(name, ast.Name)
                }
                grew = grew or not bound <= reachable
                reachable |= bound
    return reachable


@pytest.mark.parametrize('path', documents(), ids=str)
def test_no_example_hides_in_a_plain_fence(path):
    """A plain fence is for output and refusals, never for code.

    Only a ``python`` block is run, so moving an example into a plain fence
    would take it out of every check at once while still reading as an
    example. A plain fence that parses as Python and reaches for this client
    is that move.
    """
    exported = reaching_this_client(path)
    for index, source in enumerate(fences(path, '')):
        for giveaway in ('import ', '>>>', 'In ['):
            assert giveaway not in source, (
                f'{path.name} block {index} is in a plain fence and writes '
                f'{giveaway!r}; an example is a program in a python fence, '
                f'and anything else runs nowhere'
            )
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        reached = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                reached.add(node.id)
            elif isinstance(node, ast.Attribute):
                reached.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                reached.update(alias.name for alias in node.names)
                reached.add(getattr(node, 'module', None) or '')
        assert not reached & exported, (
            f'{path.name} block {index} is in a plain fence and uses '
            f'{sorted(reached & exported)}; tag it python so it is run'
        )


@pytest.mark.parametrize('path', documents(), ids=str)
def test_every_example_parses(path):
    for index, source in enumerate(fences(path, 'python')):
        try:
            ast.parse(source)
        except SyntaxError as error:  # pragma: no cover - failing is the point
            pytest.fail(f'{path.name} example {index} does not parse: {error}')


#: Every module of ours an example may import from, and where to look for the
#: names it takes. A fragment is never executed, so an import inside one is
#: checked only here.
OURS = ('sdm_client', 'sdm_connectors', 'sdm_connectors.databricks')


def exported_by(module: str) -> set[str]:
    imported = importlib.import_module(module)
    return set(
        getattr(imported, '__all__', None)
        or [name for name in dir(imported) if not name.startswith('_')]
    )


@pytest.mark.parametrize('path', documents(), ids=str)
@pytest.mark.parametrize('module', OURS)
def test_every_name_an_example_imports_is_exported(path, module):
    exported = exported_by(module)
    for source in fences(path, 'python'):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != module:
                continue
            missing = [
                alias.name for alias in node.names if alias.name not in exported
            ]
            assert not missing, (
                f'{path.name} imports {missing} from {module}, which does '
                f'not export them'
            )


@pytest.mark.parametrize('path', documents(), ids=str)
@pytest.mark.parametrize('retired', RETIRED)
def test_no_example_calls_a_retired_interface(path, retired):
    """Examples only. The prose names several of these to warn against them,
    and a check that cannot tell the two apart would forbid saying so."""
    for source in fences(path, 'python'):
        assert retired not in source, (
            f'{path.name} shows {retired!r} in an example, and this client '
            f'does not have it'
        )


@pytest.mark.parametrize('path', documents(), ids=str)
def test_no_example_trains_anything(path):
    """Both models are in-context learners: there is nothing to train, so an
    example reaching for a training API would be teaching a different product.
    ``fit()`` is the exception and is not training; it pins a context."""
    for source in fences(path, 'python'):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if not isinstance(called, ast.Attribute):
                continue
            assert called.attr not in {'train', 'fine_tune', 'evaluate'}, (
                f'{path.name} calls {called.attr}(), which this client has no '
                f'notion of'
            )
            if called.attr == 'fit':
                receiver = called.value
                assert isinstance(
                    receiver, ast.Name
                ) and receiver.id in reaching_this_client(path), (
                    f'{path.name} calls fit() on something other than a '
                    f'client, and only a client has one to call'
                )


# --- What the queries in the skills actually do ---------------------------


@pytest.fixture
def schema():
    pytest.importorskip('sdm_relational_engine')
    pd = pytest.importorskip('pandas')
    from sdm_client import Relationship, Schema

    users = pd.DataFrame({'user_id': range(12), 'segment': list('abc') * 4})
    orders = pd.DataFrame(
        {
            'order_id': range(120),
            'user_id': [i % 12 for i in range(120)],
            'item': [i % 4 for i in range(120)],
            'price': [10.0 + i for i in range(120)],
            'channel': ['web', 'app', 'store'] * 40,
            'ordered_at': pd.date_range('2026-01-01', periods=120, freq='D'),
        }
    )
    items = pd.DataFrame({'item': range(4), 'kind': list('abcd')})
    return Schema.from_data(
        {'users': users, 'orders': orders, 'items': items},
        primary_keys={
            'users': 'user_id',
            'orders': 'order_id',
            'items': 'item',
        },
        time_columns={'orders': 'ordered_at'},
        relationships=[
            Relationship('orders', 'user_id', 'users', 'user_id'),
            Relationship('orders', 'item', 'items', 'item'),
        ],
    )


@pytest.fixture
def schema_with_text():
    pytest.importorskip('sdm_relational_engine')
    pd = pytest.importorskip('pandas')
    from sdm_client import Relationship, Schema

    users = pd.DataFrame({'user_id': range(12)})
    orders = pd.DataFrame(
        {
            'order_id': range(120),
            'user_id': [i % 12 for i in range(120)],
            'note': [f'a free note number {i}' for i in range(120)],
            'ordered_at': pd.date_range('2026-01-01', periods=120, freq='D'),
        }
    )
    schema = Schema.from_data(
        {'users': users, 'orders': orders},
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={'orders': 'ordered_at'},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    assert str(schema.tables['orders'].columns['note'].stype) == 'text', (
        'the fixture no longer has a text column to ask about'
    )
    return schema


def resolve(query, schema, **extra):
    from sdm_client import KumoRelational, PredictiveQuery

    return PredictiveQuery(query, schema).materialize(
        indices=[1, 2], model=KumoRelational(), **extra
    )


PQL = SKILLS / 'relational' / 'pql.md'


FENCED = re.compile(r'^```[a-z]*\n.*?^```$', re.DOTALL | re.MULTILINE)


def without_comments(path: Path) -> str:
    """The document without what a reader never sees.

    A comment opened outside a fence runs until it closes, fences and all,
    which is what a renderer does with it; a ``<!--`` inside one is four
    characters on the page and stays. Scanned in one pass rather than by
    stripping between fences, because a comment that opens before a fence and
    closes after it belongs to neither side.
    """
    kept: list[str] = []
    hiding = False
    inside = False
    for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
        if hiding:
            if '-->' not in line:
                continue
            hiding = False
            line = line.split('-->', 1)[1]
        if not inside and '<!--' in line:  # noqa: SIM102
            rest = line
            while '<!--' in rest:
                before, rest = rest.split('<!--', 1)
                kept.append(before)
                if '-->' not in rest:
                    hiding = True
                    rest = ''
                    kept.append('\n')
                    break
                rest = rest.split('-->', 1)[1]
            kept.append(rest)
            continue
        if line.startswith('```'):
            inside = not inside
        kept.append(line)
    return ''.join(kept)


def rendered(path: Path) -> str:
    """The document as a reader sees it: no comments, no fenced blocks.

    A table inside either is not a table anyone reads, and checking one would
    be validating text that is not on the page.
    """
    return FENCED.sub('', without_comments(path))


def tables(path: Path) -> list[list[list[str]]]:
    """Every table in a document, each as its rows, each row as its cells.

    Grouped rather than flattened: two tables separated by a paragraph read as
    one otherwise, and a check for a row absent from the first then passes on
    finding it in the second.
    """
    found: list[list[list[str]]] = []
    current: list[list[str]] = []
    ruled = False
    for line in rendered(path).splitlines():
        stripped = line.strip()
        if not (stripped.startswith('|') and stripped.endswith('|')):
            if current:
                assert ruled, f'{path.name} has a table with no rule under it'
                found.append(current)
            current, ruled = [], False
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split('|')]
        if (
            not ruled
            and len(current) == 1
            and all(re.fullmatch(r':?-+:?', cell) for cell in cells)
        ):
            assert len(cells) == len(current[0]), (
                f'{path.name} rules {len(current[0])} columns with '
                f'{len(cells)}, which renders as a paragraph, not a table'
            )
            ruled = True
            continue
        assert not current or len(cells) == len(current[0]), (
            f'{path.name} gives a row {len(cells)} cells under a header of '
            f'{len(current[0])}, which renders ragged and is read here as '
            f'though it lined up'
        )
        current.append(cells)
    if current:
        assert ruled, f'{path.name} has a table with no rule under it'
        found.append(current)
    return found


def table(path: Path, label: str) -> dict[str, list[str]]:
    """The one table holding a row labelled ``label``, as label to cells."""
    matched = [
        one for one in tables(path) if any(row[0] == label for row in one)
    ]
    assert len(matched) == 1, (
        f'{path.name} has {len(matched)} tables with a {label!r} row'
    )
    collected: dict[str, list[str]] = {}
    for cells in matched[0]:
        assert cells[0] not in collected, (
            f'{path.name} lists {cells[0]!r} twice'
        )
        collected[cells[0]] = cells[1:]
    return collected


def heading(path: Path, label: str) -> str:
    """What the table holding ``label`` puts above its second column."""
    for one in tables(path):
        if any(row[0] == label for row in one):
            assert len(one[0]) == 2, (
                f'{path.name} gives the {label!r} table {len(one[0])} '
                f'columns, and only the second is read'
            )
            return one[0][1]
    raise AssertionError(f'{path.name} has no {label!r} row')


def backticked(cell: str) -> list[str]:
    """Every code span in a cell, whatever else the cell holds."""
    return re.findall(r'`([^`]+)`', cell)


def only(cell: str) -> str:
    """The one code span in a cell read for a single value."""
    found = listed(cell)
    assert len(found) == 1, (
        f'{cell!r} is read as one value and holds {len(found)}'
    )
    return found[0]


def listed(cell: str) -> list[str]:
    """The code spans in a cell that is nothing but code spans.

    Only the spans were read once, and the rest of the cell ignored, so a cell
    could name the right values and take them back in prose beside them. A
    cell this is read from is the list and nothing else.
    """
    found = backticked(cell)
    assert cell == ', '.join(f'`{value}`' for value in found), (
        f'{cell!r} is read as a list of code spans and is not one; write it '
        f'as `a`, `b`, `c` with nothing else in the cell'
    )
    assert len(found) == len(set(found)), (
        f'{cell!r} names something twice, and is read as a set'
    )
    return found


def task_table() -> list[tuple[str, str, int]]:
    """The target-decides-the-task table, read out of ``pql.md`` itself.

    Parsed rather than copied: a table that drifts from the client is the
    failure this file exists to catch, and a copy here would drift with it.
    """
    shown = table(PQL, 'Target')
    assert shown['Target'] == ['Task', 'Calls'], (
        f'pql.md heads the task table Target, {shown["Target"]}, and this '
        f'reads it as Target, Task, Calls'
    )
    del shown['Target']
    read = [
        (only(target), only(cells[0]), int(only(cells[1])))
        for target, cells in shown.items()
    ]
    assert len(read) == 6, f'pql.md task table has {len(read)} rows, not 6'
    return read


def split_table(heading: str, *rows: str) -> list[list[str]]:
    """A table in ``pql.md`` whose rows are exactly ``rows``, in order.

    Exactly: a row this does not read is a row that can say the opposite of
    the ones it does.
    """
    shown = table(PQL, heading)
    assert list(shown) == [heading, *rows], (
        f'pql.md gives the {heading.lower()} table the rows {list(shown)[1:]} '
        f'and this reads {list(rows)}'
    )
    return [listed(shown[row][0]) for row in rows]


@pytest.mark.parametrize(
    'target, kind, calls', task_table(), ids=lambda v: str(v)[:40]
)
def test_the_target_decides_the_task(target, kind, calls, schema):
    """Every row of the table in ``pql.md``, asked for real.

    The call count is read from the table too. A forecast and a ranking are
    several calls and the rest are one, which is the row that is easiest to
    write down wrongly and the hardest to notice.
    """
    pd = pytest.importorskip('pandas')
    extra = {'candidates': pd.Series(range(4))} if 'RANK' in target else {}
    request = resolve(
        f'PREDICT {target} FOR EACH users.user_id', schema, **extra
    )
    assert request.task.kind == kind
    assert len(request.parts or (request,)) == calls


def test_a_ranking_is_asked_as_a_classification_per_candidate(schema):
    pd = pytest.importorskip('pandas')
    request = resolve(
        'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.user_id',
        schema,
        candidates=pd.Series(range(4)),
    )
    assert request.task.kind == 'binary_classification'
    assert request.parts is not None
    assert len(request.parts) == 4


def test_a_forecast_is_asked_once_per_step(schema):
    request = resolve(
        'PREDICT COUNT(orders.*, 0, 30, days) FORECAST 3 TIMEFRAMES '
        'FOR EACH users.user_id',
        schema,
    )
    assert request.parts is not None
    assert len(request.parts) == 3


def test_a_single_question_is_one_part(schema):
    request = resolve(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
    )
    assert request.parts is None


ANSWERED, REFUSED = split_table(
    'Aggregation', 'Answered', 'Parsed, then refused'
)

ANY_VALUE, ORDERED, REFUSED_OPS = split_table(
    'Comparison',
    'Answered on any value',
    'Answered on numbers and times only',
    'Parsed, then refused',
)


@pytest.mark.parametrize('aggregation', ANSWERED)
def test_an_aggregation_the_skill_promises_is_answered(aggregation, schema):
    """A numeric aggregation of a numeric column. ``COUNT`` takes the rows and
    ``LIST_DISTINCT`` takes a key, so each is asked in the form it has."""
    pd = pytest.importorskip('pandas')
    if aggregation == 'COUNT':
        target, extra, kind = 'COUNT(orders.*, 0, 30, days)', {}, 'regression'
    elif aggregation == 'LIST_DISTINCT':
        target = 'LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2'
        extra = {'candidates': pd.Series(range(4))}
        kind = 'binary_classification'
    else:
        target = f'{aggregation}(orders.price, 0, 30, days)'
        extra, kind = {}, 'regression'
    query = f'PREDICT {target} FOR EACH users.user_id'
    assert resolve(query, schema, **extra).task.kind == kind


@pytest.mark.parametrize('aggregation', REFUSED)
def test_an_aggregation_the_skill_warns_about_is_refused(aggregation, schema):
    from sdm_client.errors import StructuredDataError

    query = (
        f'PREDICT {aggregation}(orders.price, 0, 30, days) '
        f'FOR EACH users.user_id'
    )
    with pytest.raises(StructuredDataError, match='not supported'):
        resolve(query, schema)


#: A condition using each comparison, against a number, a time and a string.
#: The operand matters: four of these are answered on the first two and
#: refused on the third, which is the distinction the table draws.
AS_A_CONDITION = {
    '=': (
        'orders.price = 50',
        'orders.ordered_at = 2026-02-01',
        "orders.channel = 'web'",
    ),
    '!=': (
        'orders.price != 50',
        'orders.ordered_at != 2026-02-01',
        "orders.channel != 'web'",
    ),
    '<': (
        'orders.price < 50',
        'orders.ordered_at < 2026-02-01',
        "orders.channel < 'web'",
    ),
    '<=': (
        'orders.price <= 50',
        'orders.ordered_at <= 2026-02-01',
        "orders.channel <= 'web'",
    ),
    '>': (
        'orders.price > 50',
        'orders.ordered_at > 2026-02-01',
        "orders.channel > 'web'",
    ),
    '>=': (
        'orders.price >= 50',
        'orders.ordered_at >= 2026-02-01',
        "orders.channel >= 'web'",
    ),
    'IS': (
        'orders.price IS 50',
        'orders.ordered_at IS 2026-02-01',
        "orders.channel IS 'web'",
    ),
    'IS NOT': (
        'orders.price IS NOT 50',
        'orders.ordered_at IS NOT 2026-02-01',
        "orders.channel IS NOT 'web'",
    ),
    'IN': (
        'orders.item IN (1, 2)',
        'orders.ordered_at IN (2026-02-01, 2026-03-01)',
        "orders.channel IN ('web', 'app')",
    ),
    'IS IN': (
        'orders.item IS IN (1, 2)',
        'orders.ordered_at IS IN (2026-02-01, 2026-03-01)',
        "orders.channel IS IN ('web', 'app')",
    ),
    'LIKE': (
        "orders.price LIKE 'w%'",
        "orders.ordered_at LIKE 'w%'",
        "orders.channel LIKE 'w%'",
    ),
    'NOT LIKE': (
        "orders.price NOT LIKE '%w%'",
        "orders.ordered_at NOT LIKE '%w%'",
        "orders.channel NOT LIKE '%w%'",
    ),
    'CONTAINS': (
        "orders.price CONTAINS 'e'",
        "orders.ordered_at CONTAINS 'e'",
        "orders.channel CONTAINS 'e'",
    ),
    'NOT CONTAINS': (
        "orders.price NOT CONTAINS 'e'",
        "orders.ordered_at NOT CONTAINS 'e'",
        "orders.channel NOT CONTAINS 'e'",
    ),
    'STARTS WITH': (
        "orders.price STARTS WITH 'w'",
        "orders.ordered_at STARTS WITH 'w'",
        "orders.channel STARTS WITH 'w'",
    ),
    'ENDS WITH': (
        "orders.price ENDS WITH 'b'",
        "orders.ordered_at ENDS WITH 'b'",
        "orders.channel ENDS WITH 'b'",
    ),
}


def filtered(comparison: str, against: str) -> str:
    assert comparison in AS_A_CONDITION, (
        f'pql.md lists {comparison!r} and this test does not know how to '
        f'write a condition with it'
    )
    side = ['a number', 'a time', 'a string'].index(against)
    return (
        f'PREDICT COUNT(orders.* WHERE '
        f'{AS_A_CONDITION[comparison][side]}, 0, 30, days) '
        f'FOR EACH users.user_id'
    )


#: Each row of the comparison table, and what being in it claims: whether a
#: condition using that operator is answered against a number, a time and a
#: string.
ROW_MEANS = {
    'Answered on any value': (True, True, True),
    'Answered on numbers and times only': (True, True, False),
    'Parsed, then refused': (False, False, False),
}


def refusal(comparison: str, against: str, schema) -> str | None:
    """Why a condition was refused, or ``None`` if it was answered.

    The reason matters. A condition refused for being malformed looks exactly
    like one refused for naming an operator the model will not answer, and
    only the second is what a row of that table claims.
    """
    from sdm_client.errors import StructuredDataError

    try:
        resolve(filtered(comparison, against), schema)
    except StructuredDataError as error:
        return str(error)
    return None


@pytest.mark.parametrize(
    'comparison, row',
    [
        (comparison, row)
        for row, names in (
            ('Answered on any value', ANY_VALUE),
            ('Answered on numbers and times only', ORDERED),
            ('Parsed, then refused', REFUSED_OPS),
        )
        for comparison in names
    ],
)
def test_a_comparison_is_in_the_row_it_belongs_in(comparison, row, schema):
    """Asked three ways, and the answers decide the row.

    Reading the table and testing each row on the operand that row talks
    about is circular: it confirms whatever the table says. Every operator is
    asked against a number, a time and a string, and the three answers pick
    the row, which is then compared with the row it is written in. Moving one
    operator between rows fails here.
    """
    why = {
        against: refusal(comparison, against, schema)
        for against in ('a number', 'a time', 'a string')
    }
    found = tuple(why[against] is None for against in why)
    belongs = [name for name, shape in ROW_MEANS.items() if shape == found]
    assert belongs, (
        f'{comparison} is answered {found} against a number, a time and a '
        f'string, and no row of the table says that'
    )
    assert belongs == [row], (
        f'pql.md lists {comparison} under {row!r} and it behaves like '
        f'{belongs[0]!r}'
    )
    if row == 'Parsed, then refused':
        assert re.search('not supported|deprecated', why['a string'] or ''), (
            f'{comparison} is listed as parsed and then refused, and what it '
            f'actually says is {why["a string"]!r}'
        )


#: The bounds pql.md gives the two clauses, read out of it.
def clause_table() -> dict[str, tuple[int, int]]:
    """The two clauses and the ends of their ranges, read out of ``pql.md``."""
    shown = table(PQL, 'Clause')
    assert list(shown) == [
        'Clause',
        '`RANK TOP k`',
        '`FORECAST n TIMEFRAMES`',
    ], f'pql.md gives the clause table the rows {list(shown)[1:]}'
    assert shown['Clause'] == ['Least', 'Most'], (
        f'pql.md heads the clause table Clause, {shown["Clause"]}'
    )
    return {
        only(row): (
            int(only(cells[0]).replace(',', '')),
            int(only(cells[1]).replace(',', '')),
        )
        for row, cells in shown.items()
        if row != 'Clause'
    }


CLAUSES = clause_table()


@pytest.mark.parametrize('clause', ['RANK TOP k', 'FORECAST n TIMEFRAMES'])
def test_a_clause_is_refused_outside_the_range_the_skill_gives(clause, schema):
    """Both ends, and each refusal has to name the bound written down.

    Asking past a bound that is too generous is refused anyway, by the real
    one, so a wrong number would pass a check that only looked for a refusal.
    """
    pd = pytest.importorskip('pandas')
    from sdm_client.errors import StructuredDataError

    low, high = CLAUSES[clause]
    if clause == 'RANK TOP k':
        target = 'LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP {}'
        extra = {'candidates': pd.Series(range(4))}
    else:
        target = 'COUNT(orders.*, 0, 30, days) FORECAST {} TIMEFRAMES'
        extra = {}

    def refused(value: int) -> str:
        with pytest.raises(StructuredDataError) as raised:
            resolve(
                f'PREDICT {target.format(value)} FOR EACH users.user_id',
                schema,
                **extra,
            )
        return str(raised.value)

    assert f'at least {low}' in refused(low - 1), (
        f'{clause} does not say {low} is its least'
    )
    said = refused(high + 1)
    assert re.search(rf'maximum[^.]*\b{high}\b', said), (
        f'{clause} does not say {high} is its most; it says {said!r}, which '
        f'may be the fan-out cap rather than the bound in the table'
    )


def test_a_fan_out_past_the_quoted_number_of_calls_is_refused(schema):
    """The cap a reader meets before either clause's own bound.

    A forecast of 10,000 steps is inside the language's range and is refused
    for what it would cost. A ranking costs a call per candidate, not per
    ``k``, so both are asked here: the number is read out of the prose rather
    than written twice.
    """
    pd = pytest.importorskip('pandas')
    from sdm_client.decompose import MOST_PARTS
    from sdm_client.errors import StructuredDataError

    said = re.search(r'past (\d+) of them', PQL.read_text(encoding='utf-8'))
    assert said and int(said[1]) == MOST_PARTS, (
        f'pql.md names {said and said[1]} calls and the client allows '
        f'{MOST_PARTS}'
    )
    with pytest.raises(StructuredDataError, match=str(MOST_PARTS)):
        resolve(
            f'PREDICT COUNT(orders.*, 0, 30, days) FORECAST '
            f'{MOST_PARTS + 1} TIMEFRAMES FOR EACH users.user_id',
            schema,
        )
    with pytest.raises(StructuredDataError, match=str(MOST_PARTS)):
        resolve(
            'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2 '
            'FOR EACH users.user_id',
            schema,
            candidates=pd.Series(range(MOST_PARTS + 1)),
        )


def transcripts() -> list[tuple[str, list[str]]]:
    """Every ``query → refusal`` block in ``pql.md``, as query and refusals.

    The refusals quoted through that file were pasted by hand. Executing the
    ones written beside their query turns them from a transcript into a claim
    the client has to keep making.
    """
    found = []
    for block in fences(PQL, ''):
        lines = block.splitlines()
        if not lines or not lines[0].startswith('PREDICT'):
            continue
        asked = [line for line in lines if not line.startswith(('→', ' '))]
        if len(asked) != 1 or not any(line.startswith('→') for line in lines):
            continue
        said, current = [], ''
        for line in lines[1:]:
            if line.startswith('→'):
                if current:
                    said.append(current)
                current = line[1:].strip()
            elif line.startswith(' '):
                current += ' ' + line.strip()
        if current:
            said.append(current)
        found.append((asked[0], said))
    assert len(found) == 8, f'pql.md shows {len(found)} transcripts, not 8'
    return found


def from_the_endpoint(said: list[str]) -> bool:
    """Whether a transcript quotes a deployment's refusal rather than this
    client's. The deployment's carry the status it answered with."""
    return any(re.match(r'\[?\d{3}\b', one) for one in said)


#: The one transcript whose query needs a column the fixture does not build.
NEEDS_MULTICATEGORICAL = 'PREDICT users.segment CLASSIFY FOR EACH users.user_id'


@pytest.mark.parametrize(
    'query, said',
    [one for one in transcripts() if not from_the_endpoint(one[1])],
    ids=lambda v: str(v)[:40],
)
def test_a_quoted_refusal_is_the_refusal(query, said, schema):
    """Run what pql.md shows being asked, and read back what it shows."""
    pd = pytest.importorskip('pandas')
    from sdm_client.errors import StructuredDataError

    if query == NEEDS_MULTICATEGORICAL:
        column = schema.tables['users'].columns['segment']
        schema.tables['users'].columns['segment'] = type(column)(
            'segment', dtype=column.dtype, stype='multicategorical'
        )
    extra = {'candidates': pd.Series(range(4))} if 'RANK' in query else {}
    with pytest.raises(StructuredDataError) as raised:
        resolve(query, schema, **extra)
    whole = ' '.join(str(raised.value).split())
    at = 0
    for quoted in said:
        parts = [
            ' '.join(one.split())
            for one in re.split(r'\s\.\.\.(?:\s|$)', quoted)
        ]
        for part in parts:
            if not part:
                continue
            found = whole.find(part, at)
            assert found >= 0, (
                f'pql.md shows {part!r} and the client says {whole!r}'
            )
            at = found + len(part)
        if not parts[-1]:
            continue
        rest = (
            whole[at:]
            if quoted is said[-1]
            else whole[
                at : whole.find(
                    ' '.join(
                        said[said.index(quoted) + 1].split(' ... ')[0].split()
                    ),
                    at,
                )
            ]
        )
        assert not re.sub(r'row \d+, column \d+:|[\s;.]', '', rest), (
            f'pql.md stops quoting at {parts[-1]!r} while the client goes on '
            f'to say {rest!r}; write ... to elide it'
        )


def test_the_anchor_defaults_to_the_latest_row(schema):
    """What "the next thirty days" is counted from, which is not today."""
    pd = pytest.importorskip('pandas')
    from sdm_client import KumoRelational, PredictiveQuery

    query = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
    asked = PredictiveQuery(query, schema).materialize(
        indices=[1], model=KumoRelational()
    )
    latest = max(
        pd.Timestamp(one).tz_localize(None)
        for payload in asked.payloads
        for one in anchors(payload['predict']['instance_table'])
    )
    newest = pd.date_range('2026-01-01', periods=120, freq='D').max()
    assert latest == newest, (
        f'the window is anchored at {latest} and the newest order is '
        f'{newest}; the skill says the anchor is the latest row'
    )
    elsewhere = PredictiveQuery(query, schema).materialize(
        indices=[1],
        model=KumoRelational(),
        anchor_time=pd.Timestamp('2026-09-01'),
    )
    assert asked.payloads != elsewhere.payloads, (
        'anchor_time changed nothing, and the skill says it moves the window'
    )


def anchors(table) -> list:
    """Every __kumo_anchor_time a sent table carries."""
    columns = table.get('columns') or []
    if '__kumo_anchor_time' not in columns:
        return []
    at = columns.index('__kumo_anchor_time')
    return [row[at] for row in table.get('rows') or []]


@pytest.mark.parametrize(
    'target, candidates',
    [
        ('COUNT(orders.*, 0, 30, days) FORECAST 2 TIMEFRAMES', 0),
        ('LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2', 3),
    ],
    ids=['forecast', 'ranking'],
)
def test_a_request_in_several_parts_cannot_be_fitted(
    target, candidates, schema
):
    """The sentence the skill adds beside sessions.

    Both are several questions, each wanting the context its own labels come
    from, and a session pins one. Refused where it is built, before any
    endpoint sees it.
    """
    pd = pytest.importorskip('pandas')
    from sdm_client import KumoRelational, SDMClient
    from sdm_client.errors import StructuredDataError

    extra = {'candidates': pd.Series(range(candidates))} if candidates else {}
    request = resolve(
        f'PREDICT {target} FOR EACH users.user_id', schema, **extra
    )
    with SDMClient(url='http://localhost:8000') as client:
        with pytest.raises(StructuredDataError, match='session'):
            client.fit(request, model=KumoRelational())


def test_a_category_is_multiclass_however_few_values_it_holds(schema):
    """The sentence under the task table.

    The fixture holds three values and the skill's own schema holds two, so
    counting them would read as the rule it is not: the stype decides.
    """
    pd = pytest.importorskip('pandas')

    from sdm_client import Relationship, Schema

    users = pd.DataFrame({'user_id': range(12), 'segment': ['a', 'b'] * 6})
    assert len(set(users['segment'])) == 2
    two = Schema.from_data(
        {'users': users, 'orders': schema.tables['orders'].data},
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={'orders': 'ordered_at'},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    kind = resolve(
        'PREDICT users.segment FOR EACH users.user_id', two
    ).payloads[0]['task']['kind']
    assert kind == 'multiclass_classification', (
        f'a two-value category resolved to {kind}, and pql.md says the way it '
        f'is stored decides rather than how many values it holds'
    )

    users['segment'] = [0, 1] * 6
    numbers = Schema.from_data(
        {'users': users, 'orders': schema.tables['orders'].data},
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={'orders': 'ordered_at'},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter('always')
        kind = resolve(
            'PREDICT users.segment FOR EACH users.user_id', numbers
        ).payloads[0]['task']['kind']
    assert any('two whole numbers' in str(one.message) for one in said), (
        'from_data asked a regression of ones and zeroes and said nothing, '
        'and pql.md says it warns that it is doing so'
    )
    assert kind == 'regression', (
        f'two whole numbers resolved to {kind}, and pql.md says a binary '
        f'question comes from a comparison and from nothing else'
    )

    numbers['users']['segment'].stype = 'categorical'
    kind = resolve(
        'PREDICT users.segment FOR EACH users.user_id', numbers
    ).payloads[0]['task']['kind']
    assert kind == 'multiclass_classification', (
        f'saying the column is a category left it as {kind}, and pql.md tells '
        f'a reader to say so that way'
    )


def test_a_warehouse_reads_the_same_whole_numbers_as_a_category():
    """The other half of that paragraph, and the reason it is written.

    The same column, built the two ways a schema can be built, resolves to two
    different tasks. An agent that read only the from_data half would ask a
    regression of a warehouse and get a multiclass question.
    """
    duckdb = pytest.importorskip('duckdb')
    pytest.importorskip('sdm_relational_engine')

    from sdm_client import Relationship, Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users AS SELECT i AS user_id, '
        '(i % 2)::INTEGER AS segment FROM range(60) x(i)'
    )
    connection.execute(
        'CREATE TABLE orders AS SELECT i AS order_id, (i % 60) AS user_id, '
        "TIMESTAMP '2026-01-01' + INTERVAL (i) DAY AS ordered_at "
        'FROM range(300) y(i)'
    )
    schema = Schema.from_source(
        DuckDBSource(connection),
        {'users': 'users', 'orders': 'orders'},
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={'orders': 'ordered_at'},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter('always')
        kind = resolve(
            'PREDICT users.segment FOR EACH users.user_id', schema
        ).payloads[0]['task']['kind']
    assert kind == 'multiclass_classification', (
        f'a warehouse column of ones and zeroes resolved to {kind}, and '
        f'pql.md says a source reads few distinct values as a category'
    )
    assert not [
        one for one in said if 'two whole numbers' in str(one.message)
    ], (
        'the warehouse path warned about whole numbers, and pql.md says it '
        'asks the multiclass question without warning'
    )


def test_asking_for_the_mean_changes_the_request(schema):
    """The other option the skill tells a reader to reach for.

    materialize takes it through **options, so a misspelling would be
    forwarded and ignored rather than refused; asking that it change the
    request is what catches that.
    """
    query = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
    assert (
        resolve(query, schema).payloads
        != resolve(query, schema, prediction_statistic='mean').payloads
    ), 'prediction_statistic changed nothing, and the skill says it chooses'


@pytest.mark.parametrize(
    'window', ['0, 0', '30, 0', '30, 30'], ids=['empty', 'backwards', 'same']
)
def test_a_window_must_start_before_it_ends(window, schema):
    """The rule beside non-negative, which it does not follow from."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='needs to be smaller'):
        resolve(
            f'PREDICT COUNT(orders.*, {window}, days) FOR EACH users.user_id',
            schema,
        )


def test_a_column_across_a_relationship_cannot_be_named(schema):
    """Aggregating over a join is the point; naming a column across one is
    refused, as a target and inside a filter alike."""
    from sdm_client.errors import StructuredDataError

    for query in (
        'PREDICT items.kind FOR EACH orders.order_id',
        "PREDICT COUNT(orders.* WHERE items.kind = 'a', 0, 30, days) "
        'FOR EACH users.user_id',
    ):
        with pytest.raises(StructuredDataError, match='Static references'):
            resolve(query, schema)


def test_a_ranking_takes_only_a_foreign_key(schema):
    """The shape a ranking's target has to be, as the skill quotes it."""
    pd = pytest.importorskip('pandas')
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='not multicategorical'):
        resolve(
            'PREDICT SUM(orders.price, 0, 30, days) RANK TOP 2 '
            'FOR EACH users.user_id',
            schema,
            candidates=pd.Series(range(4)),
        )


def test_a_forecast_takes_only_a_numerical_forward_target(schema):
    """The shape a forecast's target has to be, in both its refusals."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='numerical targets'):
        resolve(
            'PREDICT COUNT(orders.*, 0, 30, days) > 2 FORECAST 2 TIMEFRAMES '
            'FOR EACH users.user_id',
            schema,
        )
    with pytest.raises(StructuredDataError, match='forward-looking'):
        resolve(
            'PREDICT users.segment FORECAST 2 TIMEFRAMES FOR EACH '
            'users.user_id',
            schema,
        )


def test_a_text_column_cannot_be_compared_at_all(schema_with_text):
    """The rule beneath the comparison table.

    Past 64 distinct values a string is read as text rather than as a label,
    and then no comparison reaches it, which no row of that table would tell
    you.
    """
    from sdm_client.errors import StructuredDataError

    for condition in (
        "orders.note = 'x'",
        "orders.note != 'x'",
        "orders.note IN ('x', 'y')",
        'orders.note IS NULL',
    ):
        with pytest.raises(StructuredDataError, match='text column'):
            resolve(
                f'PREDICT COUNT(orders.* WHERE {condition}, 0, 30, days) '
                f'FOR EACH users.user_id',
                schema_with_text,
            )


def test_a_rankings_top_k_is_capped(schema):
    """The cap pql.md quotes on ``k``, which is how many come back ranked.

    Not on how many candidates are considered; that is the fan-out, and it is
    checked separately.
    """
    pd = pytest.importorskip('pandas')
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='maximum foundation model'):
        resolve(
            'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 21 '
            'FOR EACH users.user_id',
            schema,
            candidates=pd.Series(range(4)),
        )


def test_a_window_in_the_past_is_refused(schema):
    """The rule ``pql.md`` leads with, because it is the one people hit."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='in the past'):
        resolve(
            'PREDICT COUNT(orders.*, -30, 0, days) FOR EACH users.user_id',
            schema,
        )


def test_classify_is_refused_as_multilabel(schema):
    """The clause pql.md calls unreachable, and the reason it gives.

    The reason has been wrong twice: once as a target-type rule, once as the
    client refusing to send a list column. It is the task that is refused.
    """
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='not multicategorical'):
        resolve(
            'PREDICT COUNT(orders.*, 0, 30, days) CLASSIFY '
            'FOR EACH users.user_id',
            schema,
        )
    with pytest.raises(StructuredDataError, match='multilabel tasks'):
        resolve(
            'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) CLASSIFY '
            'FOR EACH users.user_id',
            schema,
        )
    column = schema.tables['users'].columns['segment']
    schema.tables['users'].columns['segment'] = type(column)(
        'segment', dtype=column.dtype, stype='multicategorical'
    )
    with pytest.raises(
        StructuredDataError, match='static multilabel classification'
    ):
        resolve('PREDICT users.segment CLASSIFY FOR EACH users.user_id', schema)


def test_a_window_without_a_unit_is_days(schema):
    """The second form pql.md gives an aggregation, and what it means.

    Asked without a unit the request has to be the one asked in days, which
    is the claim; that it merely resolves would not say which unit it took.
    """
    asked = {
        unit: resolve(
            f'PREDICT COUNT(orders.*, 0, 30{unit}) FOR EACH users.user_id',
            schema,
        ).payloads
        for unit in ('', ', days', ', hours')
    }
    assert asked[''] == asked[', days']
    assert asked[''] != asked[', hours']


def test_an_aggregation_without_a_window_is_refused(schema):
    """The language allows it; the model has no static aggregation."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='static aggregation'):
        resolve('PREDICT SUM(orders.price) FOR EACH users.user_id', schema)


def test_a_static_assumption_is_refused(schema):
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='no time range'):
        resolve(
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id '
            "ASSUMING users.segment = 'a'",
            schema,
        )


def test_a_temporal_assumption_is_answered(schema):
    request = resolve(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id '
        'ASSUMING SUM(orders.price, 0, 30, days) > 100',
        schema,
    )
    assert request.task.kind == 'regression'


TABULAR_PQL = SKILLS / 'tabular' / 'references' / 'pql.md'

#: Each row of the clause table in tabular/references/pql.md, written as a
#: query over one table. The claim is that every one is refused there, and
#: several of them are ordinary relational queries.
ON_ONE_TABLE = {
    '`COUNT`, `SUM`, any aggregation': (
        'PREDICT COUNT(customers.*, 0, 30, days) FOR EACH customers.customer_id'
    ),
    'a window, `0, 30, days`': (
        'PREDICT SUM(customers.spend, 0, 30, days) '
        'FOR EACH customers.customer_id'
    ),
    'a comparison target, `spend > 200`': (
        'PREDICT customers.spend > 200 FOR EACH customers.customer_id'
    ),
    '`RANK TOP k`': (
        'PREDICT customers.segment RANK TOP 2 FOR EACH customers.customer_id'
    ),
    '`FORECAST n TIMEFRAMES`': (
        'PREDICT customers.spend FORECAST 2 TIMEFRAMES '
        'FOR EACH customers.customer_id'
    ),
    '`WHERE` on the entity': (
        'PREDICT customers.churn FOR EACH customers.customer_id '
        "WHERE customers.segment = 'a'"
    ),
    '`ASSUMING`': (
        'PREDICT customers.churn FOR EACH customers.customer_id '
        "ASSUMING customers.segment = 'a'"
    ),
}


@pytest.fixture
def one_table():
    """The schema the tabular reference builds, run out of the reference.

    A fixture of its own would answer for columns the document does not
    define, and every refusal quoted there is about a column it names.
    """
    pytest.importorskip('pandas')
    code, _ = examples(TABULAR_PQL)[0]
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exec(code, namespace)  # noqa: S102 - running the documentation
    return namespace['schema']


def test_the_tabular_clause_table_is_every_clause_it_refuses(one_table):
    """Read the table out of the reference, and ask for every row of it."""
    shown = table(TABULAR_PQL, 'Clause')
    del shown['Clause']
    assert set(shown) == set(ON_ONE_TABLE), (
        f'tabular/references/pql.md lists {sorted(set(shown) ^ set(ON_ONE_TABLE))} '
        f'that this does not ask about'
    )
    assert all(cells == ['refused'] for cells in shown.values()), (
        'every row of that table says refused, and one of them does not'
    )


#: The four whose refusal names this model, which have to be refused in those
#: words. The other three are refused on a single table for more than one
#: reason at once, so a plain refusal is all they prove here; what makes all
#: seven the tabular profile's rather than the grammar's is read off the
#: profiles themselves, below.
ITS_OWN = {
    '`COUNT`, `SUM`, any aggregation': 'tabular model predicts',
    'a window, `0, 30, days`': 'tabular model predicts',
    'a comparison target, `spend > 200`': 'tabular model predicts',
    '`WHERE` on the entity': 'tabular model scores the rows it is given',
}


#: Each row of that table, and the capability the language switches off for
#: it. Two of them are the same switch: an aggregation carries a window.
BY_CAPABILITY = {
    '`COUNT`, `SUM`, any aggregation': 'temporal',
    'a window, `0, 30, days`': 'temporal',
    'a comparison target, `spend > 200`': 'derived_targets',
    '`RANK TOP k`': 'link_prediction',
    '`FORECAST n TIMEFRAMES`': 'forecasting',
    '`WHERE` on the entity': 'entity_filters',
    '`ASSUMING`': 'counterfactuals',
}


@pytest.mark.parametrize('clause', sorted(ON_ONE_TABLE), ids=lambda v: v[:28])
def test_a_clause_refused_on_one_table_is_the_tabular_profiles_doing(clause):
    """The sentence under that table: all seven are the profile refusing.

    The reference used to call three of them the grammar's, whatever the
    model, which the relational reference contradicts by teaching them. Asked
    of the profiles rather than of a query, because a query on a single table
    is refused for several reasons at once and the message names one.
    """
    from predictive_query.profiles import RELATIONAL, TABULAR

    capability = BY_CAPABILITY[clause]
    assert getattr(TABULAR.capabilities, capability) is False, (
        f'the tabular profile allows {capability}, and the reference says '
        f'{clause} is refused on one table'
    )
    assert getattr(RELATIONAL.capabilities, capability) is True, (
        f'the relational profile refuses {capability} too, and the reference '
        f'says the relational model takes all seven'
    )


@pytest.mark.parametrize('clause', sorted(ON_ONE_TABLE), ids=lambda v: v[:28])
def test_a_clause_the_tabular_reference_calls_refused_is(clause, one_table):
    from sdm_client import KumoTabular, PredictiveQuery
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(ON_ONE_TABLE[clause], one_table).materialize(
            indices=[1, 2], model=KumoTabular()
        )
    said = ITS_OWN.get(clause)
    if said:
        assert said in str(raised.value), (
            f'{clause} is refused by {str(raised.value)[:120]!r}, which is '
            f'not this model saying so'
        )


def test_for_without_a_selector_is_refused(one_table):
    """The grammar note: FOR EACH, or FOR ... IN, and FOR alone is neither."""
    from sdm_client import KumoTabular, PredictiveQuery
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='must specify one or'):
        PredictiveQuery(
            'PREDICT customers.churn FOR customers.customer_id', one_table
        ).materialize(indices=[1], model=KumoTabular())


def test_a_plain_column_is_what_the_tabular_reference_promises(one_table):
    """The one shape it says works, in all three of its forms."""
    from sdm_client import KumoTabular, PredictiveQuery

    for query, kind in (
        (
            'PREDICT customers.churn FOR EACH customers.customer_id',
            'classification',
        ),
        (
            'PREDICT customers.spend FOR EACH customers.customer_id',
            'regression',
        ),
        (
            'PREDICT customers.churn FOR customers.customer_id IN (1, 2)',
            'classification',
        ),
    ):
        request = PredictiveQuery(query, one_table).materialize(
            indices=[1, 2], model=KumoTabular()
        )
        assert request.task.kind == kind, (
            f'{query} resolved {request.task.kind}'
        )


WAREHOUSES = SKILLS / 'relational' / 'references' / 'warehouses.md'


def test_the_source_table_is_every_source_there_is():
    """Each row names a source, the connection it takes and its extra.

    Read from the package rather than copied: a backend added, renamed or
    given a different driver has to reach the reference.
    """
    import sdm_connectors

    shown = table(WAREHOUSES, 'Source')
    del shown['Source']
    named = {only(row) for row in shown}
    exported = {
        name
        for name in sdm_connectors.__all__
        if name.endswith('Source') and name != 'Source'
    }
    assert named == exported, (
        f'warehouses.md lists {sorted(named ^ exported)} that sdm_connectors '
        f'does not, or the other way about'
    )
    import inspect

    for row, cells in shown.items():
        source = getattr(sdm_connectors, only(row))
        takes, namespace, extra = cells
        wants = {
            name
            for name in inspect.signature(source).parameters
            if name != 'connection'
        }
        assert set(backticked(namespace)) == wants, (
            f'{only(row)} takes the namespace {sorted(wants)} and the table '
            f'says {backticked(namespace)}'
        )
        assert source._BACKEND == only(extra), (
            f'{only(row)} is the {source._BACKEND} backend and the table says '
            f'{only(extra)}'
        )
        driver = re.search(r'`([a-z_.]+\([^`]*\))`', takes)
        assert driver and driver[1].split('(')[0] in source._TAKES, (
            f'{only(row)} takes "{source._TAKES}" and the table says {takes!r}'
        )


def test_every_extra_the_warehouse_reference_names_is_one_you_can_install():
    """A `pip install` that names an extra nobody publishes teaches a dead end."""
    declared = set(
        tomllib.loads(
            (
                Path(__file__).parents[1]
                / 'packages'
                / 'structured-data-models-client'
                / 'pyproject.toml'
            ).read_text(encoding='utf-8')
        )['project']['optional-dependencies']
    )
    shown = table(WAREHOUSES, 'Source')
    del shown['Source']
    named = {only(cells[2]) for cells in shown.values()} | {'lakebase', 's3'}
    assert named <= declared, (
        f'warehouses.md names extras {sorted(named - declared)}, which '
        f'structured-data-models-client does not declare'
    )


def test_a_source_given_the_wrong_connection_says_which_it_wants():
    """The SQLite trap the reference leads with, and the message it quotes."""
    import sqlite3

    from sdm_connectors import ConnectorError, SQLiteSource

    with pytest.raises(ConnectorError, match='adbc_driver_sqlite'):
        SQLiteSource(sqlite3.connect(':memory:'))


def test_read_takes_the_formats_the_reference_names():
    """CSV and Parquet, and the refusal the reference quotes for anything else.

    Compression is the asymmetry the reference spells out: a suffix on a CSV
    is read through, and one on a Parquet is still read as Parquet and fails
    on the bytes.
    """
    import gzip
    import json
    import tempfile

    from sdm_connectors import ConnectorError, read

    pd = pytest.importorskip('pandas')
    pytest.importorskip('pyarrow')
    where = Path(tempfile.mkdtemp())
    frame = pd.DataFrame({'a': [1, 2, 3]})
    frame.to_csv(where / 'x.csv', index=False)
    frame.to_parquet(where / 'x.parquet')
    assert len(read(where / 'x.csv')) == 3
    assert len(read(where / 'x.parquet')) == 3
    (where / 'x.csv.gz').write_bytes(
        gzip.compress((where / 'x.csv').read_bytes())
    )
    assert len(read(where / 'x.csv.gz')) == 3
    (where / 'x.parquet.gz').write_bytes(
        gzip.compress((where / 'x.parquet').read_bytes())
    )
    with pytest.raises(ConnectorError) as raised:
        read(where / 'x.parquet.gz')
    assert 'unsupported file format' not in str(raised.value), (
        'the compressed Parquet was refused by its name, and the reference '
        'says it is taken as Parquet and fails on the file'
    )
    assert 'Parquet' in str(raised.value), (
        'the failure does not name Parquet, so nothing says the compressed '
        'file was dispatched to the Parquet reader rather than the CSV one'
    )
    (where / 'x.json').write_text(json.dumps([{'a': 1}]), encoding='utf-8')
    with pytest.raises(ConnectorError, match='unsupported file format'):
        read(where / 'x.json')


def test_lakebase_wants_the_database_the_reference_names():
    """Both keywords the reference shows, and which of them is required."""
    import inspect

    from sdm_connectors.databricks import lakebase

    taken = inspect.signature(lakebase).parameters
    assert taken['database'].kind is inspect.Parameter.KEYWORD_ONLY
    assert taken['database'].default is inspect.Parameter.empty, (
        'database has a default, and the reference says it is required'
    )
    assert taken['workspace'].kind is inspect.Parameter.KEYWORD_ONLY
    assert taken['workspace'].default is None, (
        'workspace is required, and the reference says it falls back to an '
        'ambient WorkspaceClient'
    )


def test_a_source_reads_the_catalogs_own_keys():
    """The reference says a source infers more than a frame can.

    Names no naming rule would resolve, declared as constraints: the source
    finds them and ``from_data`` over the same rows does not.
    """
    duckdb = pytest.importorskip('duckdb')
    # Schema.from_source needs the engine, which is the extra this very
    # reference says the connector alone does not bring.
    pytest.importorskip('sdm_relational_engine')
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute('CREATE TABLE a (ref INTEGER PRIMARY KEY, v INTEGER)')
    connection.execute(
        'CREATE TABLE b (own INTEGER PRIMARY KEY, points_at INTEGER '
        'REFERENCES a(ref))'
    )
    connection.execute('INSERT INTO a VALUES (1, 10), (2, 20)')
    connection.execute('INSERT INTO b VALUES (1, 1), (2, 2)')

    assert Schema.from_source(DuckDBSource(connection)).relationships, (
        'the catalog declares a foreign key and the source found none'
    )
    assert not Schema.from_data(
        {
            name: connection.execute(f'SELECT * FROM {name}').df()
            for name in ('a', 'b')
        }
    ).relationships, (
        'from_data found a relationship names alone do not give, so the '
        'reference is wrong that the catalog is what adds it'
    )


def test_describing_a_source_reads_a_thousand_rows_of_it():
    """What the reference says leaves the warehouse before any prediction.

    Counted by giving the source more rows than the sample takes and asking
    what was read: a table left in place is not a table nothing is read from.
    """
    duckdb = pytest.importorskip('duckdb')
    pytest.importorskip('sdm_relational_engine')
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource
    from sdm_relational_engine.rfm.base.table import Table

    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE t AS SELECT i AS id, i * 2 AS v FROM range(5000) x(i)'
    )
    schema = Schema.from_source(
        DuckDBSource(connection), {'t': 't'}, primary_keys={'t': 'id'}
    )
    assert Table._NUM_SAMPLE_ROWS == 1000, (
        f'the sample is {Table._NUM_SAMPLE_ROWS} rows and both files say a '
        f'thousand'
    )
    sampled = schema._engine_tables._tables['t']._source_sample_df
    read = len(sampled)
    assert read == 1000, (
        f'describing a 5,000-row source read {read} rows, and the reference '
        f'says up to a thousand of each leave the warehouse'
    )

    connection.execute(
        'CREATE TABLE small AS SELECT i AS id FROM range(12) y(i)'
    )
    smaller = Schema.from_source(
        DuckDBSource(connection),
        {'small': 'small'},
        primary_keys={'small': 'id'},
    )
    whole = len(smaller._engine_tables._tables['small']._source_sample_df)
    assert whole == 12, (
        f'a 12-row source gave {whole} rows, and the reference says all of a '
        f'table that small is what leaves'
    )


def test_an_empty_source_is_refused_in_the_words_the_reference_quotes():
    """The mistyped DuckDB path, which gives an empty database rather than none.

    The refusal names the source and not the path, which is the whole reason
    the reference tells a reader to pass ``read_only=True``.
    """
    duckdb = pytest.importorskip('duckdb')
    pytest.importorskip('sdm_relational_engine')

    from sdm_client import Schema
    from sdm_client.errors import StructuredDataError
    from sdm_connectors import DuckDBSource

    quoted = [
        block
        for block in fences(WAREHOUSES, '')
        if 'no table to describe' in block
    ]
    assert len(quoted) == 1, (
        'warehouses.md quotes that refusal once, and this found '
        f'{len(quoted)} of it'
    )
    said = quoted[0].strip()
    with pytest.raises(StructuredDataError) as raised:
        Schema.from_source(DuckDBSource(duckdb.connect()))
    assert said in ' '.join(str(raised.value).split()), (
        f'warehouses.md quotes {said!r} and the refusal is {raised.value}'
    )


def test_a_source_type_is_chosen_for_you():
    """``Source(connection)`` picks the backend, which the reference offers."""
    duckdb = pytest.importorskip('duckdb')
    from sdm_connectors import DuckDBSource, Source

    assert isinstance(Source(duckdb.connect()), DuckDBSource)


def test_a_connector_error_is_not_a_structured_data_error():
    """The reference says catching one does not catch the other."""
    from sdm_client.errors import StructuredDataError
    from sdm_connectors import ConnectorError, MissingBackendError

    assert not issubclass(ConnectorError, StructuredDataError)
    assert issubclass(MissingBackendError, ConnectorError)


def test_the_tabular_model_refuses_a_second_table(schema):
    """The sentence ``skills/tabular/SKILL.md`` quotes."""
    from sdm_client import KumoTabular, PredictiveQuery
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='single table'):
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
            schema,
        ).materialize(indices=[1, 2], model=KumoTabular())


# --- The examples, run ----------------------------------------------------


TABULAR_URL = os.environ.get('SDM_TABULAR_URL')
RELATIONAL_URL = os.environ.get('SDM_RELATIONAL_URL')

#: The placeholder every example points at, swapped for a real one below.
PLACEHOLDER = 'http://localhost:8000'


#: An example, and the block pasted beneath it if that block is marked
#: **Verified**. Pairing them is the point: comparing against everything the
#: document printed lets a block match output belonging to another example.
EXAMPLE = re.compile(
    r'^```python\n(?P<code>.*?)^```\n'
    r'(?:\n\*\*Verified\*\*[^\n]*\n\n```\n(?P<shown>.*?)^```\n)?',
    re.MULTILINE | re.DOTALL,
)


def examples(path: Path) -> list[tuple[str, str | None]]:
    """Runnable examples and what each is said to print, in document order.

    All of them. A block tagged ``python`` in a skill is a program that runs;
    anything that is only the shape of one is written in a plain fence, so
    there is no rule here about which blocks to skip and no way to write a
    block that quietly opts out. Read from the document with its comments
    stripped, because an example nobody can see is not evidence of anything.
    """
    return [
        (match['code'], match['shown'])
        for match in EXAMPLE.finditer(without_comments(path))
    ]


def assert_every_example_prints_what_it_shows(path: Path, url: str):
    """Output marked **Verified** has to be what that example printed.

    Pasted results are the first thing to rot, and a number that drifted is
    worse than no number: it reads as measured.
    """
    import io
    from contextlib import redirect_stdout

    pd = pytest.importorskip('pandas')
    namespace: dict[str, object] = {'pd': pd}
    checked = 0
    runs = complete(path)
    for index, (code, shown) in enumerate(examples(path)):
        if not runs[index]:
            assert shown is None, (
                f'{path.name} example {index} reads a name nothing binds and '
                f'shows output anyway; nothing could have printed it'
            )
            continue
        printed = io.StringIO()
        with redirect_stdout(printed):
            exec(  # noqa: S102 - running the documentation is the test
                compile(
                    code.replace(PLACEHOLDER, url), f'{path}#{index}', 'exec'
                ),
                namespace,
            )
        if shown is None:
            continue
        checked += 1
        assert printed.getvalue() == shown, (
            f'{path.name} example {index} shows\n{shown}\nand printed\n'
            f'{printed.getvalue()}'
        )
    marked = len(
        re.findall(r'^\*\*Verified\*\*', without_comments(path), re.MULTILINE)
    )
    assert checked == marked, (
        f'{path.name} marks {marked} outputs as verified and {checked} of '
        f'them sit where an example can be paired with them'
    )


@pytest.mark.live_nim
@pytest.mark.skipif(
    not RELATIONAL_URL,
    reason='set SDM_RELATIONAL_URL to a kumo-relational NIM',
)
@pytest.mark.parametrize(
    'query, said',
    [one for one in transcripts() if from_the_endpoint(one[1])],
    ids=lambda v: str(v)[:40],
)
def test_a_quoted_refusal_from_the_endpoint_is_the_refusal(query, said, schema):
    """The transcripts pql.md shows the deployment answering, asked of it.

    A link prediction is built here and refused there, so it takes an
    endpoint to see; the rest of the transcripts never leave this machine.
    """
    from sdm_client import KumoRelational, SDMClient
    from sdm_client.errors import StructuredDataError

    request = resolve(query, schema)
    with SDMClient(url=RELATIONAL_URL, timeout=600) as client:
        with pytest.raises(StructuredDataError) as raised:
            client.predict(request, model=KumoRelational())
    whole = ' '.join(str(raised.value).split())
    for quoted in said:
        assert ' '.join(quoted.split()) in whole, (
            f'pql.md shows {quoted!r} and the endpoint says {whole!r}'
        )


@pytest.mark.live_nim
@pytest.mark.skipif(
    not TABULAR_URL, reason='set SDM_TABULAR_URL to a kumo-tabular NIM'
)
@pytest.mark.parametrize(
    'name', ['tabular/SKILL.md', 'tabular/references/pql.md']
)
def test_the_tabular_examples_print_what_the_skill_shows(name):
    assert_every_example_prints_what_it_shows(SKILLS / name, TABULAR_URL)


@pytest.mark.live_nim
@pytest.mark.skipif(
    not RELATIONAL_URL, reason='set SDM_RELATIONAL_URL to a kumo-relational NIM'
)
@pytest.mark.parametrize(
    'name',
    [
        'relational/SKILL.md',
        'relational/pql.md',
        'relational/references/beyond-one-answer.md',
        'relational/references/warehouses.md',
        'README.md',
    ],
)
def test_the_relational_examples_print_what_the_skill_shows(name):
    pytest.importorskip('sdm_relational_engine')
    pytest.importorskip('duckdb')
    assert_every_example_prints_what_it_shows(SKILLS / name, RELATIONAL_URL)


# --- What the tables in the skills claim ----------------------------------


@pytest.mark.parametrize(
    'name, model_name',
    [('tabular', 'KumoTabular'), ('relational', 'KumoRelational')],
)
def test_the_capability_table_is_the_clients_own(name, model_name):
    """Every row of the table both skills open with, against the client."""
    from sdm_client import SDMClient

    model = getattr(sdm_client, model_name)()
    with SDMClient(url='http://localhost:8000') as client:
        capabilities = client.capabilities(model)
    path = SKILLS / name / 'SKILL.md'
    assert only(heading(path, 'Tasks')) == capabilities.model, (
        f'{name} heads its table with the wrong model'
    )
    shown = table(path, 'Tasks')
    shown.pop('', None)
    said: dict[str, object] = {
        'Tasks': set(capabilities.tasks),
        'Outputs': set(capabilities.outputs),
        'Several tables': capabilities.multiple_tables,
        'Relationships': capabilities.relationships,
        'Sessions': capabilities.sessions,
    }
    if name == 'tabular':
        from sdm_client.models import TABULAR_LIMITS

        said['Explanations'] = 'explanation' in capabilities.outputs
        said['Context rows'] = TABULAR_LIMITS['context_rows_max']
    assert set(shown) == set(said), (
        f'{name} shows rows nothing checks: {sorted(set(shown) ^ set(said))}'
    )
    for label, truth in said.items():
        assert len(shown[label]) == 1, (
            f'{name} gives {label!r} {len(shown[label])} columns, and only '
            f'the first is read'
        )
        cell = shown[label][0]
        if isinstance(truth, bool):
            assert cell == ('yes' if truth else 'no'), (
                f'{name} says {label} is {cell!r}'
            )
        elif isinstance(truth, int):
            assert int(cell.replace(',', '')) == truth, (
                f'{name} says {label} is {cell!r}'
            )
        else:
            assert set(listed(cell)) == truth, (
                f'{name} lists the wrong {label.lower()}'
            )


#: Each row of the limits table in ``skills/relational/SKILL.md``, against the
#: constant it reports. Every key is spoken for in both directions: a limit the
#: client enforces and the skill omits is a refusal an agent cannot see coming,
#: and a row naming no limit is a number nobody is holding to anything.
LIMIT_ROWS = {
    'context rows, by default': 'context_rows_fast',
    'context rows, with `num_context_samples`': 'context_rows_max',
    'rows scored at once': 'predict_rows_max',
    'rows scored for a link prediction': (
        'predict_rows_temporal_link_prediction'
    ),
    'rows scored with an explanation': 'explanation_predict_rows',
    'tables in the gathered graph': 'tables_max',
    'hops gathered': 'hops_max',
    'classes': 'classes_max',
}


def test_the_limits_table_is_the_shipped_limits():
    from sdm_client.models import RELATIONAL_LIMITS

    assert set(LIMIT_ROWS.values()) == set(RELATIONAL_LIMITS), (
        'the client enforces a limit this table does not account for'
    )
    path = SKILLS / 'relational' / 'SKILL.md'
    assert heading(path, 'Limit') == 'Value'
    shown = table(path, 'Limit')
    shown.pop('Limit')
    assert set(shown) == set(LIMIT_ROWS), (
        'the limits table and the limits no longer line up: '
        f'{sorted(set(shown) ^ set(LIMIT_ROWS))}'
    )
    for label, key in LIMIT_ROWS.items():
        value = int(only(shown[label][0]).replace(',', ''))
        assert value == RELATIONAL_LIMITS[key], (
            f'{label!r} says {value}, the client enforces '
            f'{RELATIONAL_LIMITS[key]}'
        )


def spelled_in_the_grammar(token: str) -> set[str]:
    """Every operator ``token`` covers, as the grammar spells it."""
    import predictive_query

    body = (
        Path(predictive_query.__file__).parent / 'grammar' / 'PQLGrammar.g4'
    ).read_text(encoding='utf-8')
    at = body.index(f'{token}:')
    block = body[at : body.index(';', at)]
    return {
        name for name in re.findall(r"'([^']+)'", block) if name == name.upper()
    }


@pytest.mark.parametrize(
    'token, answered, refused',
    [
        ('AGGR', ANSWERED, REFUSED),
        ('REL_OP', ANY_VALUE + ORDERED, REFUSED_OPS),
    ],
)
def test_the_table_accounts_for_the_whole_language(token, answered, refused):
    """Answered plus refused must be the language, or the skill omits one.

    Read off the grammar rather than a list here: an operator added to the
    language and to neither column is one an agent would never learn about,
    and one quietly dropped from a table would go unnoticed.
    """
    every = spelled_in_the_grammar(token)
    assert every == set(answered) | set(refused), (
        f'the grammar and pql.md disagree about {token}: '
        f'{sorted(every ^ (set(answered) | set(refused)))}'
    )


@pytest.mark.live_nim
@pytest.mark.skipif(
    not TABULAR_URL, reason='set SDM_TABULAR_URL to a kumo-tabular NIM'
)
def test_the_tabular_outputs_narrow_by_task_as_the_skill_says():
    """``quantiles`` on a regression, ``probabilities`` on a classification.

    The skill says the endpoint narrows what the client would send, so the
    narrowing is what is asked about here, not the client's own table. Both
    directions: an endpoint that began serving the wrong one would make the
    skill wrong, and should fail here rather than go unnoticed.
    """
    from sdm_client import KumoTabular, Request, SDMClient
    from sdm_client.errors import StructuredDataError

    pd = pytest.importorskip('pandas')
    rows_in = pd.DataFrame(
        {
            'age': [31.0, 44.5, 28.0, 51.0, 39.0, 35.0],
            'spend': [120.0, 430.0, 90.0, 510.0, 260.0, 300.0],
            'churn': [0, 1, 0, 1, 0, 1],
        }
    )
    to_score = pd.DataFrame({'age': [33.0, 47.0], 'spend': [150.0, 480.0]})
    with SDMClient(url=TABULAR_URL) as client:
        labelled = Request(
            context=rows_in,
            predict=to_score,
            target='churn',
            task='classification',
        )
        with pytest.raises(StructuredDataError, match='quantiles'):
            client.predict(labelled, model=KumoTabular(), outputs=['quantiles'])
        quantity = Request(
            context=rows_in.drop(columns='churn'),
            predict=to_score.drop(columns='spend'),
            target='spend',
            task='regression',
        )
        with pytest.raises(StructuredDataError, match='probabilities'):
            client.predict(
                quantity, model=KumoTabular(), outputs=['probabilities']
            )
        answer = client.predict(
            quantity, model=KumoTabular(), outputs=['prediction', 'quantiles']
        )
    deciles = [
        name
        for name in answer.to_pandas().columns
        if str(name).startswith('quantile_')
    ]
    assert len(deciles) == 9, f'quantiles came back as {deciles}'


#: The output matrix skills/relational/SKILL.md gives for the deployment:
#: what each task is served, and everything the client would send that it
#: will not. Asked of the endpoint, so a deployment that gains or loses one
#: fails here rather than making the skill quietly wrong.
SERVED = {
    'COUNT(orders.*, 0, 30, days)': (
        ('prediction', 'quantiles'),
        tuple(f'quantile_0.{one}' for one in range(1, 10)),
    ),
    'COUNT(orders.*, 0, 30, days) > 2': (
        ('prediction', 'probabilities'),
        ('probability_False', 'probability_True'),
    ),
    'users.segment': (
        ('prediction', 'probabilities'),
        ('probability_a', 'probability_b', 'probability_c'),
    ),
}


def not_served(outputs: tuple[str, ...]) -> list[str]:
    """Everything the client would send this task that the skill says the
    deployment will not serve: its whole advertised list, less what is.

    Derived rather than listed, so the set cannot quietly shrink to whatever
    the endpoint happens to allow.
    """
    advertised = set(
        listed(table(SKILLS / 'relational' / 'SKILL.md', 'Tasks')['Outputs'][0])
    )
    return sorted(advertised - set(outputs))


@pytest.mark.live_nim
@pytest.mark.skipif(
    not RELATIONAL_URL,
    reason='set SDM_RELATIONAL_URL to a kumo-relational NIM',
)
@pytest.mark.parametrize('target', sorted(SERVED), ids=lambda v: v[:30])
def test_the_relational_endpoint_serves_the_matrix_the_skill_gives(
    target, schema
):
    from sdm_client import KumoRelational, SDMClient
    from sdm_client.errors import StructuredDataError

    outputs, columns = SERVED[target]
    request = resolve(f'PREDICT {target} FOR EACH users.user_id', schema)
    with SDMClient(url=RELATIONAL_URL, timeout=600) as client:
        shown = client.predict(
            request, model=KumoRelational(), outputs=list(outputs)
        ).to_pandas()
        assert 'prediction' in shown.columns, f'{target} gave no prediction'
        named = {str(one) for one in shown.columns}
        assert set(columns) <= named, (
            f'{target} gave {sorted(named)}, and the skill shows {columns}'
        )
        for field in not_served(outputs):
            with pytest.raises(StructuredDataError, match=field):
                client.predict(request, model=KumoRelational(), outputs=[field])


@pytest.mark.live_nim
@pytest.mark.skipif(
    not RELATIONAL_URL,
    reason='set SDM_RELATIONAL_URL to a kumo-relational NIM',
)
@pytest.mark.parametrize('target', sorted(SERVED), ids=lambda v: v[:30])
def test_the_relational_endpoint_refuses_an_explanation(target, schema):
    """The gap the skill leads with: the client sends it, the endpoint will
    not serve it. If a deployment starts serving one, this fails and the
    skill is the thing to change."""
    from sdm_client import KumoRelational, SDMClient
    from sdm_client.errors import StructuredDataError

    request = resolve(f'PREDICT {target} FOR EACH users.user_id', schema)
    with SDMClient(url=RELATIONAL_URL, timeout=600) as client:
        with pytest.raises(StructuredDataError, match='explanation'):
            client.predict(
                request,
                model=KumoRelational(),
                outputs=['prediction', 'explanation'],
            )


@pytest.mark.live_nim
@pytest.mark.skipif(
    not RELATIONAL_URL,
    reason='set SDM_RELATIONAL_URL to a kumo-relational NIM',
)
def test_a_session_answers_only_the_question_it_pinned(schema):
    """The sentence beside the session example: one context is one question.

    It takes an endpoint because a session only exists once one has pinned
    it, and what the skill promises is the refusal naming which field differs.
    """
    from sdm_client import KumoRelational, SDMClient
    from sdm_client.errors import StructuredDataError

    pinned = resolve(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
    )
    other = resolve(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
    )
    with SDMClient(url=RELATIONAL_URL, timeout=600) as client:
        fitted = client.fit(pinned, model=KumoRelational())
        try:
            with pytest.raises(
                StructuredDataError, match='the query pinned'
            ) as raised:
                fitted.predict(other)
        finally:
            fitted.close()
    assert 'SUM(orders.price' in str(raised.value), (
        'the refusal does not name the question that was asked instead'
    )
