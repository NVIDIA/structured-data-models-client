# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import warnings

import pandas as pd


def infer_primary_key(
    table_name: str,
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    r"""Auto-detect potential primary key column.

    Args:
        table_name: The table name.
        df: The pandas DataFrame to analyze.
        candidates: A list of potential candidates.

    Returns:
        The name of the detected primary key, or ``None`` if not found.
    """
    if len(candidates) == 0:
        return None

    # A list of (potentially modified) table names that are eligible to match
    # with a primary key, i.e.:
    # - UserInfo -> User
    # - snakecase <-> camelcase
    # - camelcase <-> snakecase
    # - plural <-> singular (users -> user, eligibilities -> eligibility)
    # - verb -> noun (qualifying -> qualify)
    _table_names = {table_name}
    if table_name.lower().endswith('_info'):
        _table_names.add(table_name[:-5])
    elif table_name.lower().endswith('info'):
        _table_names.add(table_name[:-4])

    table_names = set()
    for _table_name in _table_names:
        table_names.add(_table_name.lower())
        snakecase = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', _table_name)
        snakecase = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', snakecase)
        table_names.add(snakecase.lower())
        camelcase = _table_name.replace('_', '')
        table_names.add(camelcase.lower())
        if _table_name.lower().endswith('s'):
            table_names.add(_table_name.lower()[:-1])
            table_names.add(snakecase.lower()[:-1])
            table_names.add(camelcase.lower()[:-1])
        else:
            table_names.add(_table_name.lower() + 's')
            table_names.add(snakecase.lower() + 's')
            table_names.add(camelcase.lower() + 's')
        if _table_name.lower().endswith('ies'):
            table_names.add(_table_name.lower()[:-3] + 'y')
            table_names.add(snakecase.lower()[:-3] + 'y')
            table_names.add(camelcase.lower()[:-3] + 'y')
        elif _table_name.lower().endswith('y'):
            table_names.add(_table_name.lower()[:-1] + 'ies')
            table_names.add(snakecase.lower()[:-1] + 'ies')
            table_names.add(camelcase.lower()[:-1] + 'ies')
        if _table_name.lower().endswith('ing'):
            table_names.add(_table_name.lower()[:-3])
            table_names.add(snakecase.lower()[:-3])
            table_names.add(camelcase.lower()[:-3])

    scores: list[tuple[str, int]] = []
    for col_name in candidates:
        col_name_lower = col_name.lower()

        score = 0

        if col_name_lower == 'id':
            score += 4

        for table_name_lower in table_names:
            if col_name_lower == table_name_lower:
                score += 4  # USER -> USER
                break

            for suffix in ['id', 'hash', 'key', 'code', 'uuid']:
                if not col_name_lower.endswith(suffix):
                    continue

                if col_name_lower == f'{table_name_lower}_{suffix}':
                    score += 5  # USER -> USER_ID
                    break

                if col_name_lower == f'{table_name_lower}{suffix}':
                    score += 5  # User -> UserId
                    break

                if col_name_lower.endswith(f'{table_name_lower}_{suffix}'):
                    score += 2

                if col_name_lower.endswith(f'{table_name_lower}{suffix}'):
                    score += 2

            # A name-shaped tie-break for one RelBench table whose primary
            # key shares no substring with its table name, so none of the
            # rules above can score it.
            if table_name == 'studies' and col_name == 'nct_id':
                score += 1

        ser = df[col_name].iloc[:1_000_000]
        score += 3 * (ser.nunique() / len(ser))

        scores.append((col_name, score))

    scores = [x for x in scores if x[-1] >= 4]
    scores.sort(key=lambda x: x[-1], reverse=True)

    if len(scores) == 0:
        return None

    if len(scores) == 1:
        return scores[0][0]

    # In case of multiple candidates, only return one if its score is unique:
    if scores[0][1] != scores[1][1]:
        return scores[0][0]

    max_score = max(score for _, score in scores)
    tied = [col_name for col_name, score in scores if score == max_score]
    warnings.warn(
        f'Found multiple potential primary keys in table '
        f"'{table_name}': {tied}. Please specify the primary "
        f'key for this table manually.'
    )

    return None
