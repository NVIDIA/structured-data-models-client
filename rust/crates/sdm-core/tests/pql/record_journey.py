# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records the reference's PQL answer for every Phase 1b inventory query.

For each case of ``docs/design/rust-core/phase-0/phase-1b-inventory.yaml``,
builds the case's DuckDB tables, describes them with ``Schema.from_source``
and validates the case's query against that schema with the relational
profile, as ``materialize`` does (``PredictiveQuery._resolve``). Writes
``journey.json`` beside this file: per case, the schema in the contract's
form and whether the reference accepted the query, with each error's code
and span. A case whose ``from_source`` fails never reaches PQL: its id is
listed under ``skipped``, so the test can hold the count and the list.

Run it with the reference's interpreter, isolated, from the repository root:

    ~/code/sdmc-reference/.venv/bin/python -I \\
        rust/crates/sdm-core/tests/pql/record_journey.py ~/code/sdmc-reference
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
INVENTORY = ROOT / 'docs/design/rust-core/phase-0/phase-1b-inventory.yaml'


def _guard(reference: Path) -> Any:
    path = reference / 'scripts/reference_guard.py'
    spec = importlib.util.spec_from_file_location('reference_guard', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schema(schema: Any) -> dict[str, Any]:
    """A PQL schema definition in the contract's ``QuerySchema`` form."""
    tables = []
    for name, table in schema.tables.items():
        tables.append(
            {
                'name': name,
                'columns': [
                    {
                        'name': column.name,
                        'dtype': column.dtype.value,
                        'stype': column.stype.value,
                    }
                    for column in table.cols
                ],
                'primary_key': table.pkey,
                'time_column': table.time_col,
                'end_time_column': table.end_time_col,
            }
        )
    links = [
        [{'table': c.table_name, 'column': c.col_name} for c in group.columns]
        for group in schema.col_groups
    ]
    return {'tables': tables, 'links': links}


def _span(span: Any) -> dict[str, int] | None:
    if span is None:
        return None
    return {
        'start_row': span.start_row,
        'start_col': span.start_col,
        'end_row': span.end_row,
        'end_col': span.end_col,
    }


def main(reference: Path) -> int:
    guard = _guard(reference)
    guard.require_isolation()

    import duckdb
    import yaml
    from predictive_query import RELATIONAL, validate
    from sdm_client import Schema
    from sdm_client.schema import to_schema_definition
    from sdm_connectors import DuckDBSource

    inventory = yaml.safe_load(INVENTORY.read_text(encoding='utf-8'))
    cases = []
    skipped = []
    for case in inventory['cases']:
        connection = duckdb.connect()
        for statement in inventory['bases'][case['base']]:
            connection.execute(statement)
        for statement in case.get('setup', []):
            connection.execute(statement)
        how = case.get('from_source', {})
        try:
            schema = Schema.from_source(DuckDBSource(connection), **how)
        except Exception:  # noqa: BLE001 - such a case never reaches PQL
            skipped.append(case['id'])
            continue
        query = case.get('pql', inventory['defaults']['pql'])
        definition = to_schema_definition(schema)
        resolved, response = validate(query, definition, RELATIONAL)
        cases.append(
            {
                'id': case['id'],
                'query': query,
                'profile': 'RELATIONAL',
                'schema': _schema(definition),
                'accepted': resolved is not None and response.ok,
                'errors': [
                    {'code': e.code, 'span': _span(e.span)}
                    for e in response.errors
                ],
            }
        )
    guard.require_isolation()

    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    document = {
        'format': 'sdm-pql-journey/2',
        'source': (
            f'rust-core-reference at {commit}: PQL validation of each '
            f'Phase 1b inventory query against its Schema.from_source '
            f'schema, recorded by record_journey.py beside this file'
        ),
        'cases': cases,
        'skipped': skipped,
    }
    (HERE / 'journey.json').write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(
        f'{len(cases)} cases recorded, {len(skipped)} skipped, at {commit[:12]}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]).expanduser()))
