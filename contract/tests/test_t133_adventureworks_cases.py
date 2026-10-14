# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32f's two AdventureWorks questions as cases (exit item (b)), folded
into T1.33.

    python -m pytest contract/tests/test_t133_adventureworks_cases.py -q

Written by the package's independent test author from the design's T1.32f
row ("the two further AdventureWorks questions committed as cases, churn
(binary) and ``products.Color`` (multiclass)"), the plan's exit 4 and the
notebook, before the implementation. The cases are
``contract/cases/s1/base.adventureworks.churn.json`` and
``base.adventureworks.color.json``.

- The questions are the notebook's (``examples/databricks/
  adventureworks_relational.ipynb``, ``QUESTIONS['inactivity_90d']`` and
  ``QUESTIONS['product_color']``): the query text as the client prints
  it once parsed (``to_string()``), no option the notebook does not pass
  (``inference_config`` among them), and the entities its
  ``frequent_customers`` and ``top_product`` queries answer over the
  fixture. Each has ``base.adventureworks``'s base, schema, edits
  and ``describe_source``, and the core operations its call starts.
- The cases' expectations: each case's ``expect`` is what ``python-ref``
  observed (``contract/reference/<commit>.jsonl``; the identity rules are
  ``tests/test_contract_reference.py``'s).
- Each request declares only classes its own context holds. The live
  endpoint refused ``Color`` on the full data with
  ``TASK_SCHEMA_MISMATCH`` (a declared class, ``Grey``, in no context row),
  which is why T1.32e's benchmark asks ``ProductLine``
  (``scripts/live/real_data_benchmark.py``); over this fixture the
  reference's request declares eight colours, each in the context, so
  ``Color`` stays, as the design row names it.
- Admission ("diagnostic cases run on ``python-core``, ``python-old`` and
  ``rust-host``" through the gate; admitted, they gate): each case is an
  acceptance case whose routes cover the DuckDB paths of its call, carrying
  to the core every operation up to the stage the reference answers at.
  Admitted, CI's ``rust-test`` step holds ``rust-host`` to the reference
  through ``compare.py``, and the ``cross-host`` step holds it to
  ``python-core``.
- ``python-core``, on the case's own routes once admitted and on
  ``base.adventureworks``'s until then, answers as ``python-ref`` does (the
  outcome, the public schema, the warning classes, each request, the
  result, as ``contract/tools/empty_null_record.py``'s ``project`` reads
  them), with the reference's statement counts per public call, or an
  approved register entry lists the case. ``python-old`` completes each
  case with the reference's answer.

Gating (``contract/tests/t133_gate.py``): a test whose id is in
``t133-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list, which may only shrink.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'contract/cases/s1'
IDS = ['base.adventureworks.churn', 'base.adventureworks.color']
BASE = 'base.adventureworks'
NOTEBOOK = ROOT / 'examples/databricks/adventureworks_relational.ipynb'
#: Each case's question in the notebook's ``QUESTIONS``.
QUESTION = {
    'base.adventureworks.churn': 'inactivity_90d',
    'base.adventureworks.color': 'product_color',
}
#: The notebook's entity queries (its ``frequent_customers`` and
#: ``top_product``), over the fixture's tables.
ENTITIES = {
    'base.adventureworks.churn': (
        'SELECT CustomerID FROM adventureworks_sales_order_headers '
        'GROUP BY CustomerID ORDER BY count(*) DESC, CustomerID LIMIT 25'
    ),
    'base.adventureworks.color': (
        'SELECT ProductID FROM adventureworks_sales_order_details '
        'GROUP BY ProductID ORDER BY sum(OrderQty) DESC, ProductID LIMIT 1'
    ),
}
#: The task each question is.
TASK = {
    'base.adventureworks.churn': 'binary_classification',
    'base.adventureworks.color': 'multiclass_classification',
}
#: The register's placeholder for the inventory's cases, which excuses none.
PLACEHOLDER = 'P1B-1'
#: The core operations that answer at each stage of a public call.
OPERATIONS = {
    'from_source': ('describe_source',),
    'materialize': ('describe_source', 'materialize'),
    'predict': ('describe_source', 'materialize', 'predict'),
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t133_gate', ROOT / 'contract/tests/t133_gate.py')
R = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')


def gated(request: pytest.FixtureRequest, check: Any) -> None:
    r"""Runs ``check`` held to the pending list, as ``py:<test id>``."""
    G.held('exits', f'py:{request.node.name}', check)


def per_case() -> list[Any]:
    return [pytest.param(case, id=case) for case in IDS]


def case_file(case: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (CASES / f'{case}.json').read_text(encoding='utf-8')
    )
    return loaded


def reference() -> dict[str, dict[str, Any]]:
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        out[observation['case']] = observation
    return out


def registered_case(case: str) -> str | None:
    r"""The approved entry, other than the placeholder, that lists ``case``
    in its ``cases``."""
    path = ROOT / 'docs/design/rust-core/register.yaml'
    for entry in yaml.safe_load(path.read_text(encoding='utf-8'))['entries']:
        if entry['id'] == PLACEHOLDER:
            continue
        if entry.get('status') == 'approved' and case in (
            entry.get('cases') or []
        ):
            return str(entry['id'])
    return None


def notebook_questions() -> dict[str, tuple[str | None, set[str]]]:
    r"""The notebook's ``QUESTIONS``: each question's query text and the
    names of the options it passes to ``materialize``."""
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        if 'QUESTIONS = ' not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'QUESTIONS'
            ):
                assert isinstance(node.value, ast.Dict)
                out = {}
                for key, value in zip(
                    node.value.keys, node.value.values, strict=True
                ):
                    assert isinstance(key, ast.Constant)
                    assert isinstance(value, ast.Tuple)
                    pql, options = value.elts
                    assert isinstance(options, ast.Dict)
                    # A query named by a variable (DEMAND) is no literal;
                    # neither question here is one.
                    text = (
                        ast.literal_eval(pql)
                        if isinstance(pql, ast.Constant)
                        else None
                    )
                    out[key.value] = (
                        text,
                        {ast.literal_eval(k) for k in options.keys if k},
                    )
                return out
    raise AssertionError('the notebook has no QUESTIONS')


def printed(pql: str) -> str:
    r"""``pql`` parsed and printed as the client prints a query."""
    from predictive_query.parser.parser import PQLParser

    text: str = PQLParser().to_parsed_predictive_query(pql).to_string()
    return text


def fixture() -> Any:
    import duckdb

    bases = yaml.safe_load(
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8')
    )
    con = duckdb.connect()
    for statement in bases['bases']['adventureworks']:
        con.execute(statement)
    return con


# --- the questions --------------------------------------------------------------


@pytest.mark.parametrize('case', per_case())
def test_each_case_asks_the_notebook_s_question(case: str) -> None:
    loaded = case_file(case)
    assert loaded['id'] == case
    pql, options = notebook_questions()[QUESTION[case]]
    # The notebook's query (its adjacent literals joined) in the form the
    # client prints it, query.to_string(), which python-core hands the core
    # and every acceptance case stores (churn's '=0' becomes ' = 0'): the
    # same question, parsed. The reference's request names the same text.
    assert pql is not None
    assert loaded['input']['pql'] == printed(pql)
    [payload] = reference()[case]['public']['payloads']
    assert payload['parsed']['metadata']['source_query'] == printed(pql)
    # Only the options the notebook passes: no inference_config.
    assert set(loaded['input']['materialize']) == options == {'indices'}
    # The entities the notebook's query answers over the fixture.
    wanted = [row[0] for row in fixture().execute(ENTITIES[case]).fetchall()]
    assert loaded['input']['materialize']['indices'] == wanted
    assert loaded['until'] == 'predict'


@pytest.mark.parametrize('case', per_case())
def test_each_case_is_base_adventureworks_s_journey(case: str) -> None:
    loaded, base = case_file(case), case_file(BASE)
    for key in ('base', 'from_source', 'edits'):
        assert loaded['input'][key] == base['input'][key], key
    core = loaded['core']
    assert core['describe_source'] == base['core']['describe_source']
    # materialize starts with the call's entities, as int64 scalars, no
    # anchor time and no seed (the call passes neither); predict follows.
    assert core['materialize'] == {
        'provider': base['core']['materialize']['provider'],
        'indices': [
            {'t': 'i64', 'v': str(index)}
            for index in loaded['input']['materialize']['indices']
        ],
        'anchor_time': None,
        'seed': None,
    }
    assert core['predict'] == base['core']['predict']


@pytest.mark.parametrize('case', per_case())
def test_each_case_expects_what_python_ref_observed(case: str) -> None:
    expect = case_file(case)['expect']
    observed = reference()[case]
    assert observed['identity']['runner'] == 'python-ref'
    outcome = observed['outcome']
    assert expect['outcome'] == outcome['result'] == 'success'
    assert expect['stage'] == outcome['stage'] == 'predict'


@pytest.mark.parametrize('case', per_case())
def test_each_request_declares_only_classes_its_context_holds(
    case: str,
) -> None:
    # The live endpoint's TASK_SCHEMA_MISMATCH on Color over the full data:
    # a declared class in no context row. Over the fixture the reference's
    # request declares each class its context holds, and no other.
    [payload] = reference()[case]['public']['payloads']
    request = payload['parsed']
    task = request['task']
    assert task['kind'] == TASK[case]
    target = task['target']
    context = request['context']['instance_table']
    column = context['columns'].index(target['column_name'])
    held = {json.dumps(row[column]) for row in context['rows']}
    declared = {
        json.dumps(_label(c, target['dtype'])) for c in target['classes']
    }
    assert declared == held - {'null'}, (declared, held)
    if case == 'base.adventureworks.color':
        assert len(declared) == 8


def _label(name: str, dtype: str) -> Any:
    r"""A declared class as the context's cell holds it."""
    return {'true': True, 'false': False}[name] if dtype == 'bool' else name


# --- admission ------------------------------------------------------------------


@pytest.mark.parametrize('case', per_case())
def test_each_case_is_admitted(
    case: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        loaded = case_file(case)
        assert loaded['status'] == 'acceptance', loaded['status']
        assert loaded['routes'], 'an admitted case names its routes'
        unrouted = [
            path
            for path in loaded['paths']
            if path.endswith('/duckdb') and path not in loaded['routes']
        ]
        assert unrouted == [], f'paths of the call not routed: {unrouted}'
        stage = reference()[case]['outcome']['stage']
        for operation in OPERATIONS[stage]:
            assert loaded['core'].get(operation), (
                f'the case carries {operation} to the core (the reference '
                f'answers at {stage})'
            )

    gated(request, check)


# --- the runs -------------------------------------------------------------------


def _forced(tmp: Path) -> Path:
    r"""Copies of the cases, each on its own routes, or on
    ``base.adventureworks``'s while it names none."""
    routes = case_file(BASE)['routes']
    out = tmp / 'cases'
    out.mkdir()
    for case in IDS:
        loaded = case_file(case)
        loaded['routes'] = loaded['routes'] or routes
        (out / f'{case}.json').write_text(json.dumps(loaded), encoding='utf-8')
    return out


def _execute(mode: str, cases: list[Path], out: Path) -> dict[str, Any]:
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            mode,
            '--out',
            str(out),
            *map(str, cases),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return {
        o['case']: o
        for o in (
            json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        )
    }


@pytest.fixture(scope='module')
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp('t133-aw')
    return {
        'python-core': _execute(
            'python-core', [_forced(tmp)], tmp / 'core.jsonl'
        ),
        'python-old': _execute(
            'python-old',
            [CASES / f'{case}.json' for case in IDS],
            tmp / 'old.jsonl',
        ),
    }


def answer_problem(
    case: str, ours: dict[str, Any], theirs: dict[str, Any]
) -> str | None:
    r"""Why an observation of ``case`` (``ours``) fails ``python-ref``'s
    (``theirs``), both as the recorder projects them; None when they agree
    or an approved entry lists the case."""
    mine, want = R.project(ours, {}), R.project(theirs, {})
    differs = sorted(k for k in want if mine.get(k) != want[k])
    if not differs or registered_case(case):
        return None
    return (
        f'{case}: {differs} differ from python-ref: {mine["outcome"]} '
        f'against {want["outcome"]}'
    )


@pytest.mark.parametrize('mode', ['python-core', 'python-old'])
@pytest.mark.parametrize('case', per_case())
def test_each_runner_answers_each_case_as_python_ref(
    case: str, mode: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        problem = answer_problem(case, runs[mode][case], reference()[case])
        assert problem is None, problem

    gated(request, check)


@pytest.mark.parametrize('case', per_case())
def test_python_core_starts_the_case_s_core_operations(
    case: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    # The core operations python-core's host started, recorded at the
    # boundary, include each the case says its call starts (the edits'
    # edit_schema besides).
    def check() -> None:
        operations = set(runs['python-core'][case]['identity']['operations'])
        missing = sorted(set(case_file(case)['core']) - operations)
        assert missing == [], f'not started on the core: {missing}'

    gated(request, check)


@pytest.mark.parametrize('case', per_case())
def test_statement_counts_equal_the_reference_s_or_are_registered(
    case: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')
        capture = _module(
            'capture_check', ROOT / 'contract/runners/capture_check.py'
        )
        ours = runs['python-core'][case]
        assert capture.complete(ours) == [], 'the capture is DuckDB query log'
        mine = compare.counts(ours)
        theirs = compare.counts(reference()[case])
        assert mine is not None and theirs is not None
        assert mine == theirs or registered_case(case), (mine, theirs)

    gated(request, check)


def test_the_answer_comparison_fails_closed() -> None:
    # On these cases' own evidence: the reference's observation agrees with
    # itself, and fails once any part of what the public calls gave is
    # changed: the outcome, the schema, a warning, the request (one context
    # label) and the result.
    case = 'base.adventureworks.churn'
    theirs = reference()[case]
    assert answer_problem(case, theirs, theirs) is None

    def changed(edit: Any) -> dict[str, Any]:
        copy = json.loads(json.dumps(theirs))
        edit(copy)
        return copy

    def a_label(o: dict[str, Any]) -> None:
        table = o['public']['payloads'][0]['parsed']['context'][
            'instance_table'
        ]
        column = table['columns'].index('TARGET')
        table['rows'][0][column] = not table['rows'][0][column]

    def a_stype(o: dict[str, Any]) -> None:
        o['public']['schema']['tables'][0]['columns'][-1]['stype'] = 'text'

    edits = {
        'outcome': lambda o: o['outcome'].update(
            result='refusal', code='INVALID_REQUEST'
        ),
        'schema': a_stype,
        'warnings': lambda o: o['host']['warnings'].append('SDMWarning'),
        'payload': a_label,
        'result': lambda o: o['public']['result'].update(rows=24),
    }
    for part, edit in edits.items():
        problem = answer_problem(case, changed(edit), theirs)
        assert problem is not None and part in problem, part
    assert registered_case(case) is None


# --- the gating -----------------------------------------------------------------


def test_the_pending_list_only_shrinks() -> None:
    assert G.only_shrinks() == []
    # Every authored id is a test of this package's two files.
    here = Path(__file__).read_text(encoding='utf-8') + (
        ROOT / 'contract/tests/test_t133_relbench_recipe.py'
    ).read_text(encoding='utf-8')
    for name in G.gating()['authored']['exits']:
        assert name.startswith('py:'), name
        assert f'def {name[3:].split("[")[0]}(' in here, name


def test_the_gate_holds_a_pending_test_to_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A listed test passes only while its check fails, and an unlisted one
    # only while its check passes.
    def passing() -> None:
        pass

    def failing() -> None:
        raise AssertionError('seeded')

    monkeypatch.setattr(G, 'pending', lambda group: {'seeded.listed'})
    G.held('exits', 'seeded.listed', failing)
    with pytest.raises(AssertionError, match='passes now'):
        G.held('exits', 'seeded.listed', passing)
    G.held('exits', 'seeded.unlisted', passing)
    with pytest.raises(AssertionError, match='seeded'):
        G.held('exits', 'seeded.unlisted', failing)
