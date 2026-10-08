# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Ask questions over several temporal DuckDB tables through the public API,
and print every SQL statement they ran, for ``accept_fixes.py``.

``python _fixes_run.py`` prints one JSON document: the tables in the order
they were given, and for each question the statements ``Source._run`` ran.
``accept_fixes.py`` runs it under several ``PYTHONHASHSEED`` values.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

#: The tables, in the order the schema is given them: not alphabetical, and
#: each with a time column.
TABLES = ('visits', 'users', 'reviews', 'orders')

QUESTIONS = (
    # No anchor: the default anchor asks for the latest time first.
    ('PREDICT users.score FOR EACH users.user_id', {}),
    # An anchor: validating it asks for the earliest time first.
    (
        'PREDICT users.score FOR EACH users.user_id',
        {'anchor_time': '2024-02-01'},
    ),
)


def main() -> None:
    import duckdb
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource
    from sdm_connectors.source import Source

    seen: list[str] = []
    original = Source._run

    def recording(self: Any, sql: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(str(sql))
        return original(self, sql, *args, **kwargs)

    Source._run = recording  # type: ignore[method-assign]

    con = duckdb.connect()
    con.execute(
        'CREATE TABLE users AS SELECT range AS user_id, '
        "TIMESTAMP '2024-01-01' + to_days(CAST(range % 30 AS INTEGER)) "
        'AS signup, CAST(range % 7 AS DOUBLE) AS score FROM range(50)'
    )
    for name in ('visits', 'reviews', 'orders'):
        con.execute(
            f'CREATE TABLE {name} AS SELECT range AS {name}_id, '
            'range % 50 AS user_id, '
            "TIMESTAMP '2024-01-01' + to_days(CAST(range % 90 AS INTEGER)) "
            'AS at FROM range(300)'
        )
    out: dict[str, Any] = {'tables': list(TABLES), 'questions': []}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            DuckDBSource(con), {name: name for name in TABLES}
        )
        for query, options in QUESTIONS:
            seen.clear()
            given = dict(options)
            if 'anchor_time' in given:
                given['anchor_time'] = pd.Timestamp(given['anchor_time'])
            PredictiveQuery(query, schema).materialize(
                indices=[1], random_seed=7, **given
            )
            out['questions'].append(list(seen))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
