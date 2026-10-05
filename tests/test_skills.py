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
        for tag, body in FENCE.findall(path.read_text(encoding='utf-8'))
        if tag == language
    ]


def test_there_is_a_skill_for_each_model():
    assert (SKILLS / 'tabular' / 'SKILL.md').exists()
    assert (SKILLS / 'relational' / 'SKILL.md').exists()
    assert (SKILLS / 'README.md').exists()


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

    ``__Verified__`` renders identically and would pair with nothing, so a
    block marked that way would quietly go unchecked.
    """
    assert '__Verified__' not in path.read_text(encoding='utf-8'), (
        f'{path.name} marks an output with __Verified__; use **Verified**'
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


def resolve(query, schema, **extra):
    from sdm_client import KumoRelational, PredictiveQuery

    return PredictiveQuery(query, schema).materialize(
        indices=[1, 2], model=KumoRelational(), **extra
    )


PQL = SKILLS / 'relational' / 'pql.md'
ROW = re.compile(r'^\|(.+)\|$', re.MULTILINE)


HIDDEN = re.compile(r'<!--.*?-->|^```.*?^```', re.DOTALL | re.MULTILINE)


def rendered(path: Path) -> str:
    """The document as a reader sees it: no comments, no fenced blocks.

    A table inside either is not a table anyone reads, and checking one would
    be validating text that is not on the page.
    """
    return HIDDEN.sub('', path.read_text(encoding='utf-8'))


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
        if all(cell and set(cell) <= {'-', ':'} for cell in cells):
            assert len(current) == 1, (
                f'{path.name} rules a table under {len(current)} rows'
            )
            ruled = True
            continue
        current.append(cells)
    if current:
        assert ruled, f'{path.name} has a table with no rule under it'
        found.append(current)
    return found


def rows(path: Path) -> list[list[str]]:
    return [row for one in tables(path) for row in one]


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


def heading(path: Path, label: str) -> list[str]:
    """The header row of the table holding ``label``, as its cells."""
    for one in tables(path):
        if any(row[0] == label for row in one):
            return one[0]
    raise AssertionError(f'{path.name} has no {label!r} row')


def backticked(cell: str) -> list[str]:
    return re.findall(r'`([^`]+)`', cell)


def task_table() -> list[tuple[str, str]]:
    """The target-decides-the-task table, read out of ``pql.md`` itself.

    Parsed rather than copied: a table that drifts from the client is the
    failure this file exists to catch, and a copy here would drift with it.
    """
    table = [
        (backticked(cells[0]), backticked(cells[1]))
        for cells in rows(PQL)
        if len(cells) == 3 and backticked(cells[1])
    ]
    pairs = [(target[0], kind[0]) for target, kind in table if target and kind]
    assert len(pairs) == 6, f'pql.md task table has {len(pairs)} rows, not 6'
    return pairs


def answered_and_refused(heading: str) -> tuple[list[str], list[str]]:
    """The two halves of a table in ``pql.md`` split that way."""
    shown = table(PQL, heading)
    answered = backticked(shown['Answered'][0])
    refused = backticked(shown['Parsed, then refused'][0])
    assert answered and refused, f'pql.md lists no {heading.lower()}s'
    return answered, refused


@pytest.mark.parametrize('target, kind', task_table(), ids=lambda v: v[:40])
def test_the_target_decides_the_task(target, kind, schema):
    """Every row of the table in ``pql.md``, asked for real."""
    pd = pytest.importorskip('pandas')
    extra = {'candidates': pd.Series(range(4))} if 'RANK' in target else {}
    query = f'PREDICT {target} FOR EACH users.user_id'
    assert resolve(query, schema, **extra).task.kind == kind


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


ANSWERED, REFUSED = answered_and_refused('Aggregation')
ANSWERED_OPS, REFUSED_OPS = answered_and_refused('Comparison')


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


#: A condition using each comparison, written the way that operator takes one.
AS_A_CONDITION = {
    '=': 'orders.price = 50',
    '!=': 'orders.price != 50',
    '<': 'orders.price < 50',
    '<=': 'orders.price <= 50',
    '>': 'orders.price > 50',
    '>=': 'orders.price >= 50',
    'IS': 'orders.price IS NULL',
    'IS NOT': 'orders.price IS NOT NULL',
    'IN': 'orders.item IN (1, 2)',
    'IS IN': 'orders.item IS IN (1, 2)',
    'LIKE': "orders.channel LIKE 'w%'",
    'NOT LIKE': "orders.channel NOT LIKE '%w%'",
    'CONTAINS': "orders.channel CONTAINS 'e'",
    'NOT CONTAINS': "orders.channel NOT CONTAINS 'e'",
    'STARTS WITH': "orders.channel STARTS WITH 'w'",
    'ENDS WITH': "orders.channel ENDS WITH 'b'",
}


def filtered(comparison: str) -> str:
    assert comparison in AS_A_CONDITION, (
        f'pql.md lists {comparison!r} and this test does not know how to '
        f'write a condition with it'
    )
    return (
        f'PREDICT COUNT(orders.* WHERE {AS_A_CONDITION[comparison]}, 0, 30, '
        f'days) FOR EACH users.user_id'
    )


@pytest.mark.parametrize('comparison', ANSWERED_OPS)
def test_a_comparison_the_skill_promises_is_answered(comparison, schema):
    assert resolve(filtered(comparison), schema).task.kind == 'regression'


@pytest.mark.parametrize('comparison', REFUSED_OPS)
def test_a_comparison_the_skill_warns_about_is_refused(comparison, schema):
    """Every one of these parses and is then refused, which is why the skill
    lists them separately rather than leaving them out."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='not supported|deprecated'):
        resolve(filtered(comparison), schema)


def test_a_window_in_the_past_is_refused(schema):
    """The rule ``pql.md`` leads with, because it is the one people hit."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError, match='in the past'):
        resolve(
            'PREDICT COUNT(orders.*, -30, 0, days) FOR EACH users.user_id',
            schema,
        )


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

    Everything but a fence holding ``...``, which is a shape rather than a
    program. The capability blocks print nothing to compare, but they are run:
    they are the first thing an agent is told to do.
    """
    return [
        (match['code'], match['shown'])
        for match in EXAMPLE.finditer(path.read_text(encoding='utf-8'))
        if '...' not in match['code']
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
    marked = path.read_text(encoding='utf-8').count('**Verified**')
    assert marked, f'{path.name} marks nothing as verified'
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
def test_the_relational_examples_print_what_the_skill_shows():
    pytest.importorskip('sdm_relational_engine')
    assert_every_example_prints_what_it_shows(
        SKILLS / 'relational' / 'SKILL.md', RELATIONAL_URL
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
    assert backticked(heading(path, 'Tasks')[1]) == [capabilities.model], (
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
            assert set(backticked(cell)) == truth, (
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
    assert heading(path, 'Limit') == ['Limit', 'Value']
    shown = table(path, 'Limit')
    shown.pop('Limit')  # the header
    assert set(shown) == set(LIMIT_ROWS), (
        'the limits table and the limits no longer line up: '
        f'{sorted(set(shown) ^ set(LIMIT_ROWS))}'
    )
    for label, key in LIMIT_ROWS.items():
        cell = backticked(shown[label][0])
        assert len(cell) == 1, f'{label!r} shows {cell}'
        value = int(cell[0].replace(',', ''))
        assert value == RELATIONAL_LIMITS[key], (
            f'{label!r} says {value}, the client enforces '
            f'{RELATIONAL_LIMITS[key]}'
        )


def test_the_aggregations_are_all_of_them():
    """Answered plus refused must be the language, or the skill omits one.

    Read off the grammar rather than a list here: an aggregation added to the
    language and to neither column is one an agent would never learn about.
    """
    import predictive_query

    grammar = (
        Path(predictive_query.__file__).parent / 'grammar' / 'PQLGrammar.g4'
    )
    body = grammar.read_text(encoding='utf-8')
    block = body[body.index('AGGR:') : body.index(';', body.index('AGGR:'))]
    every = {name for name in re.findall(r"'(\w+)'", block) if name.isupper()}
    assert every == set(ANSWERED) | set(REFUSED), (
        'the grammar and pql.md disagree about what the aggregations are'
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
