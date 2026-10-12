# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records the reference's answers to the first attempt's 29 quirks and its
custom-profile and validator-option probes.

The inputs are the first attempt's tables (``archive/rust-core-attempt-1``,
``rust/crates/pql/tests/accept_quirks.json`` and ``custom_profiles.json``):
each row's query or date literal, schema, profile, capabilities or options.
This script asks the reference for each answer again and writes:

- ``quirks.json`` beside this file: each quirk's queries, answered as the
  corpus records a case (``tree`` or ``errors``, with ``warnings``), and its
  date literals, read as ``Constant.typed_value`` reads one
  (``pd.to_datetime``), with the warnings the reading gave;
- ``custom-profiles.json`` beside this file: queries validated with a
  ``Profile`` none of the five named ones is (each capability turned off,
  none, or only ``temporal`` and ``static``, in either mode);
- ``rust/crates/sdm-pql/tests/validator-options.json``: queries the
  reference's ``PredictiveQueryValidator`` validates with its
  ``allow_array_targets`` and ``allow_timestamp_arrays`` options, which no
  profile reaches: whether each is accepted, and its errors' codes and
  spans.

It prints how many answers differ from the ones the first attempt recorded.
The date literals' two-digit years follow the year of the clock it runs at.

Run it with the reference's interpreter, isolated, from the repository root:

    ~/code/sdmc-reference/.venv/bin/python -I \\
        rust/crates/sdm-core/tests/pql/record_quirks.py ~/code/sdmc-reference
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARCHIVE = 'archive/rust-core-attempt-1'
QUIRKS = HERE / 'quirks.json'
PROFILES = HERE / 'custom-profiles.json'
OPTIONS = REPO / 'rust/crates/sdm-pql/tests/validator-options.json'


def _guard(reference: Path) -> Any:
    path = reference / 'scripts/reference_guard.py'
    spec = importlib.util.spec_from_file_location('reference_guard', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archived(path: str) -> Any:
    text = subprocess.run(
        ['git', '-C', str(REPO), 'show', f'{ARCHIVE}:{path}'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(text)


def _given(caught: list[Any]) -> list[dict[str, str]]:
    return [
        {
            'category': f'{w.category.__module__}.{w.category.__qualname__}',
            'message': str(w.message),
        }
        for w in caught
        if issubclass(w.category, UserWarning)
    ]


def _span(span: Any) -> Any:
    if span is None:
        return None
    return {
        'start_row': span.start_row,
        'start_col': span.start_col,
        'end_row': span.end_row,
        'end_col': span.end_col,
    }


def _answer(validate: Any, query: str, schema: dict[str, Any]) -> Any:
    """``validate``'s answer to ``query``, as the corpus records one."""
    from predictive_query.schema import json_serde
    from predictive_query.schema.definition import SchemaDefinition

    definition = json_serde.from_json(schema, SchemaDefinition)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        resolved, response = validate(query, definition)
    if resolved is not None and response.ok:
        tree = json.loads(json.dumps(json_serde.to_json_dict(resolved)))
        _readings(tree)
        return {'tree': tree, 'warnings': _given(caught)}
    return {
        'errors': [
            {'code': e.code, 'span': _span(e.span), 'message': e.message}
            for e in response.errors
        ],
        'warnings': _given(caught),
    }


def _readings(node: Any) -> None:
    """Writes into each date constant under ``node`` the reference's
    reading of it, as ``record_readings.py`` does for the corpus."""
    if isinstance(node, dict):
        value = node.get('value')
        if node.get('dtype_maybe') in ('date', 'time') and isinstance(
            value, str
        ):
            read = _date(value)
            node['reading'] = {
                'value': read['value'],
                'offset': read['offset'],
                'nanos': 0,
            }
        for child in list(node.values()):
            _readings(child)
    elif isinstance(node, list):
        for child in node:
            _readings(child)


def _date(literal: str) -> dict[str, Any]:
    """``Constant.typed_value``'s reading of a date literal."""
    from predictive_query.ast.constant import Constant
    from predictive_query.schema.typing import Dtype

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            stamp = Constant(
                value=literal, dtype_maybe=Dtype.time
            ).typed_value()
        except Exception:  # noqa: BLE001 - every failure is a refusal
            stamp = None
    if stamp is None:
        return {'value': None, 'offset': None, 'warnings': _given(caught)}
    offset = stamp.utcoffset()
    assert stamp.microsecond == 0 and stamp.nanosecond == 0, literal
    return {
        'value': (
            f'{stamp.year:04d}-{stamp.month:02d}-{stamp.day:02d}T'
            f'{stamp.hour:02d}:{stamp.minute:02d}:{stamp.second:02d}'
        ),
        'offset': None if offset is None else int(offset.total_seconds()),
        'warnings': _given(caught),
    }


def _same(old: Any, new: Any) -> bool:
    """Whether the first attempt's answer is today's, messages aside."""

    def unread(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: unread(v) for k, v in node.items() if k != 'reading'}
        if isinstance(node, list):
            return [unread(v) for v in node]
        return node

    def bare(answer: Any) -> Any:
        if not isinstance(answer, dict):
            return answer
        out = {k: unread(v) for k, v in answer.items() if k != 'warnings'}
        if 'errors' in out:
            out['errors'] = [(e['code'], e['span']) for e in out['errors']]
        out['warned'] = len(answer.get('warnings', []))
        return out

    return bare(old) == bare(new)


def main(reference: Path) -> int:
    guard = _guard(reference)
    guard.require_isolation()

    import predictive_query
    from predictive_query import Capabilities, Mode, Profile
    from predictive_query.parser.parser import PQLParser
    from predictive_query.validator.predictive_query_validator import (
        PredictiveQueryValidator,
    )

    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    differ = 0

    table = _archived('rust/crates/pql/tests/accept_quirks.json')
    queries = []
    for row in table['queries']:
        profile = getattr(predictive_query, row['profile'])
        answer = _answer(
            lambda q, s, p=profile: predictive_query.validate(q, s, p),
            row['query'],
            table['schemas'][row['schema']],
        )
        differ += not _same(row['python'], answer)
        queries.append(
            {
                'id': row['id'],
                'quirk': row['quirk'],
                'query': row['query'],
                'profile': row['profile'],
                'schema': row['schema'],
                **answer,
            }
        )
    dates = []
    for row in table['dates']:
        answer = _date(row['literal'])
        old = row['python']
        archived = old['value']
        if archived is not None and len(archived) > 19:
            archived = archived[:19]
        # Constant reads a literal twice (built, then typed_value), so it
        # warns twice where the first attempt's pd.to_datetime warned once:
        # the warnings are compared as a set.
        said = {w['message'] for w in answer['warnings']}
        differ += archived != answer['value'] or said != {
            w['message'] for w in old['warnings']
        }
        dates.append(
            {
                'id': row['id'],
                'quirk': row['quirk'],
                'literal': row['literal'],
                **answer,
            }
        )
    source = (
        f'rust-core-reference at {commit}: the inputs of the first '
        f"attempt's {{}} ({ARCHIVE}, rust/crates/pql/tests), answered "
        'again by the reference, recorded by record_quirks.py'
    )
    QUIRKS.write_text(
        json.dumps(
            {
                'format': 'sdm-pql-quirks/1',
                'source': source.format('accept_quirks.json'),
                'quirks': table['quirks'],
                'schemas': table['schemas'],
                'queries': queries,
                'dates': dates,
            },
            indent=1,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )

    table = _archived('rust/crates/pql/tests/custom_profiles.json')

    def capabilities(variant: dict[str, Any]) -> Any:
        if variant['base'] == 'all':
            base = Capabilities(model=table['model'])
        else:
            base = Capabilities.nothing(table['model'])
        return base.also(**variant['set'])

    profiles = []
    for row in table['profiles']:
        profile = Profile(
            mode=Mode(row['mode']),
            capabilities=capabilities(
                table['capabilities'][row['capabilities']]
            ),
        )
        answer = _answer(
            lambda q, s, p=profile: predictive_query.validate(q, s, p),
            row['query'],
            table['schema'],
        )
        differ += not _same(row['python'], answer)
        profiles.append(
            {
                'id': row['id'],
                'capabilities': row['capabilities'],
                'mode': row['mode'],
                'query': row['query'],
                **answer,
            }
        )
    PROFILES.write_text(
        json.dumps(
            {
                'format': 'sdm-pql-custom-profiles/1',
                'source': source.format('custom_profiles.json'),
                'model': table['model'],
                'capabilities': table['capabilities'],
                'schema': table['schema'],
                'profiles': profiles,
            },
            indent=1,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )

    options = []
    for row in table['options']:

        def validate(text: str, schema: Any, row: dict[str, Any] = row) -> Any:
            mode = Mode(row['mode'])
            parsed = PQLParser(mode=mode).to_parsed_predictive_query(text)
            return PredictiveQueryValidator(
                schema,
                mode=mode,
                allow_array_targets=row['allow_array_targets'],
                allow_timestamp_arrays=row['allow_timestamp_arrays'],
            ).validate_predictive_query(parsed)

        answer = _answer(validate, row['query'], table['schema'])
        differ += not _same(row['python'], answer)
        options.append(
            {
                'id': row['id'],
                'allow_array_targets': row['allow_array_targets'],
                'allow_timestamp_arrays': row['allow_timestamp_arrays'],
                'mode': row['mode'],
                'query': row['query'],
                'accepted': 'tree' in answer,
                'errors': [
                    {'code': e['code'], 'span': e['span']}
                    for e in answer.get('errors', [])
                ],
            }
        )
    OPTIONS.write_text(
        json.dumps(
            {
                'format': 'sdm-pql-validator-options/1',
                'source': source.format('custom_profiles.json'),
                'schema': table['schema'],
                'options': options,
            },
            indent=1,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )
    guard.require_isolation()
    print(f"{differ} answers differ from the first attempt's")
    return 0


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1])))
