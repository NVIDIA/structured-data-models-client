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
``SDM_RELATIONAL_URL``. Those re-run every example and compare what it printed
to the block pasted beneath it, and probe the handful of claims the skills make
about a deployment rather than about this client.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest
import sdm_client

SKILLS = Path(__file__).parents[1] / 'skills'

#: Arguments earlier generations of this client took. The retired *type* names
#: are covered repo-wide by ``scripts/check_repo_hygiene.py retired``; these two
#: are keyword arguments it does not look for, and both are easy to reach for:
#: an explanation is an output here, and a run mode is derived from the limits.
RETIRED = ('run_mode=', 'explain=')

FENCE = re.compile(r'^```(\w*)\n(.*?)^```', re.MULTILINE | re.DOTALL)
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
    body = (SKILLS / name / 'SKILL.md').read_text(encoding='utf-8')
    assert body.startswith('---\n')
    front = body.split('---\n')[1]
    assert f'name: sdm-{name}' in front
    assert 'description:' in front
    assert 'Triggers:' in front, 'nothing would load this skill'


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
    for number, line in enumerate(body.splitlines(), start=1):
        where = f'{path.name}:{number}'
        if not inside and line.startswith('|'):
            assert previous == '' or previous.startswith('|'), (
                f'{where} begins a table with no blank line above it; under a '
                f'list or a paragraph it renders as more of that and is read '
                f'here as a table'
            )
        previous = line
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
            assert not re.search(r'&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);', line), (
                f'{where} writes an HTML entity, which renders as a character '
                f'this file reads as the escape it is written with'
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
    'skills/README.md': [(['Skill', 'Model', 'Use it when'], False)],
    'skills/relational/SKILL.md': [
        (['', '`kumo-relational`'], True),
        (['Limit', 'Value'], True),
        (['Refusal', 'Cause'], False),
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
    """``**Verified**`` is how a block is paired with the example above it.

    ``__Verified__`` and ``<strong>Verified</strong>`` render the same and
    pair with nothing, so a block marked either way would go unchecked. The
    word is counted rather than the spellings listed: there is no end to the
    ways to embolden a word, and only one of them is read here.
    """
    body = path.read_text(encoding='utf-8')
    assert body.count('Verified') == body.count('**Verified**'), (
        f'{path.name} writes "Verified" some way other than **Verified**, '
        f'which renders the same and pairs with no example'
    )


@pytest.mark.parametrize('path', documents(), ids=str)
def test_no_example_hides_in_a_plain_fence(path):
    """A plain fence is for output and refusals, never for code.

    Only a ``python`` block is run, so moving an example into a plain fence
    would take it out of every check at once while still reading as an
    example. A plain fence that parses as Python and reaches for this client
    is that move.
    """
    # Names the python blocks bind to something of this client's. The live
    # runner shares one namespace across a document, so an alias bound in one
    # block reaches a later one; a plain fence using it would otherwise look
    # like prose.
    reachable = set(sdm_client.__all__) | {'sdm_client'}
    trees: list[ast.Module] = []
    for source in fences(path, 'python'):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, 'module', None) or ''
                if module == 'sdm_client' or module.startswith('sdm_client.'):
                    reachable.update(
                        alias.asname or alias.name for alias in node.names
                    )
        # Not only what is imported: a name bound from one, by assignment or
        # by `with ... as`, reaches the next block too. Repeated to a fixed
        # point, because a name may be bound after the thing that uses it.
        trees.append(tree)
    grew = True
    while grew:
        grew = False
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    sources, targets = [node.value], node.targets
                elif isinstance(node, ast.NamedExpr):
                    sources, targets = [node.value], [node.target]
                elif isinstance(node, ast.AnnAssign):
                    sources, targets = [node.value], [node.target]
                elif isinstance(node, ast.With):
                    # The context expressions, not the body they wrap.
                    sources = [item.context_expr for item in node.items]
                    targets = [
                        item.optional_vars
                        for item in node.items
                        if item.optional_vars is not None
                    ]
                elif isinstance(node, ast.For):
                    sources, targets = [node.iter], [node.target]
                elif isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    # The body, and everything the definition itself names: a
                    # base class, a decorator, a default argument.
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
                    if source is not None
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
    exported = reachable
    for index, source in enumerate(fences(path, '')):
        # Whatever it parses as. A transcript -- `>>>`, `In [1]:` -- parses
        # as nothing and would otherwise read as output. A refusal quoted in
        # a plain fence may well name this client, so the giveaway is the
        # shape of a program rather than the mention.
        for giveaway in ('import ', '>>>', 'In ['):
            assert giveaway not in source, (
                f'{path.name} block {index} is in a plain fence and writes '
                f'{giveaway!r}; an example is a program in a python fence, '
                f'and anything else runs nowhere'
            )
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # output, a refusal, a query: not code, which is the rule
        reached = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                reached.add(node.id)
            elif isinstance(node, ast.Attribute):
                reached.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # The name as written, not as bound: an alias renames it.
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


@pytest.mark.parametrize('path', documents(), ids=str)
def test_every_name_an_example_imports_is_exported(path):
    for source in fences(path, 'python'):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != 'sdm_client':
                continue
            exported = set(sdm_client.__all__)
            missing = [
                alias.name for alias in node.names if alias.name not in exported
            ]
            assert not missing, (
                f'{path.name} imports {missing} from sdm_client, which does '
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
                assert isinstance(called.value, ast.Name), (
                    f'{path.name} calls fit() on something other than a client'
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
        'orders.price IS NULL',
        'orders.ordered_at IS NULL',
        'orders.channel IS NULL',
    ),
    'IS NOT': (
        'orders.price IS NOT NULL',
        'orders.ordered_at IS NOT NULL',
        'orders.channel IS NOT NULL',
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
        # On the operand it is for, and for being itself: a condition the
        # parser rejected as malformed would otherwise read as this row.
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

    # The refusal names the bound, which is what pins the number written
    # down: asking past a bound that is merely too generous is refused too.
    assert f'at least {low}' in refused(low - 1), (
        f'{clause} does not say {low} is its least'
    )
    assert str(high) in refused(high + 1), (
        f'{clause} does not say {high} is its most'
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

    with pytest.raises(StructuredDataError, match='multilabel tasks'):
        resolve(
            'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) CLASSIFY '
            'FOR EACH users.user_id',
            schema,
        )
    # The other way to write a target CLASSIFY would take: a column of its
    # own, declared the one stype the clause accepts. Refused in its own
    # words, which the skill quotes beside the first.
    column = schema.tables['users'].columns['segment']
    schema.tables['users'].columns['segment'] = type(column)(
        'segment', dtype=column.dtype, stype='multicategorical'
    )
    with pytest.raises(
        StructuredDataError, match='static multilabel classification'
    ):
        resolve('PREDICT users.segment CLASSIFY FOR EACH users.user_id', schema)


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
    for index, (code, shown) in enumerate(examples(path)):
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
        assert printed.getvalue().strip() == shown.strip(), (
            f'{path.name} example {index} shows\n{shown}\nand printed\n'
            f'{printed.getvalue()}'
        )
    marked = without_comments(path).count('**Verified**')
    assert checked == marked, (
        f'{path.name} marks {marked} outputs as verified and {checked} of '
        f'them sit where an example can be paired with them'
    )


@pytest.mark.live_nim
@pytest.mark.skipif(
    not TABULAR_URL, reason='set SDM_TABULAR_URL to a kumo-tabular NIM'
)
def test_the_tabular_examples_print_what_the_skill_shows():
    path = SKILLS / 'tabular' / 'SKILL.md'
    assert_every_example_prints_what_it_shows(path, TABULAR_URL)


@pytest.mark.live_nim
@pytest.mark.skipif(
    not RELATIONAL_URL, reason='set SDM_RELATIONAL_URL to a kumo-relational NIM'
)
@pytest.mark.parametrize('name', ['SKILL.md', 'pql.md'])
def test_the_relational_examples_print_what_the_skill_shows(name):
    pytest.importorskip('sdm_relational_engine')
    assert_every_example_prints_what_it_shows(
        SKILLS / 'relational' / name, RELATIONAL_URL
    )


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
    shown.pop('', None)  # the header, which labels the column not a row
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
    shown.pop('Limit')  # the header
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
    # The grammar spells each word operator twice, upper and lower. One
    # spelling is enough to compare against, and a symbol has only the one.
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


@pytest.mark.live_nim
@pytest.mark.skipif(
    not RELATIONAL_URL,
    reason='set SDM_RELATIONAL_URL to a kumo-relational NIM',
)
@pytest.mark.parametrize(
    'target',
    ['COUNT(orders.*, 0, 30, days)', 'COUNT(orders.*, 0, 30, days) > 2'],
)
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
