# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.text: the Rust core normalizes text as ``_normalize_text`` does.

20,000 generated texts (seed 20261014): every character Python calls
whitespace (U+001C to U+001F and U+0085 among them), every character whose
case changes, in words, punctuation, ``<br>`` tags of several spellings,
long runs that the 300-character cut and the word limit trim, empty text and
nulls; under ``max_words`` 50 (the sampler's), 3 and ``None``.

The reference is the function itself, so its lower-casing is whatever its
pandas string column does.
"""

from __future__ import annotations

import pytest
from _sampling import as_list, native, python_normalize, report, short, texts


def test_the_texts_reach_every_character() -> None:
    from _sampling import _unicode_classes

    joined = ''.join(t for t in texts() if t)
    classes = _unicode_classes()
    assert set(classes['space']) <= set(joined)
    assert set(classes['cased']) <= set(joined)


@pytest.mark.parametrize('max_words', [50, 3, None])
def test_texts_normalize_as_today(max_words: int | None) -> None:
    module = native('normalize_text')
    values = texts()
    want = python_normalize(values, max_words)
    failures = []
    for start in range(0, len(values), 500):
        chunk = values[start : start + 500]
        got = module.normalize_text(chunk, max_words)
        got = [as_list(words) for words in as_list(got)]
        if len(got) != len(chunk):
            failures.append(
                f'texts {start}: {len(got)} results for {len(chunk)}'
            )
            continue
        for offset, (a, b) in enumerate(
            zip(want[start : start + 500], got, strict=True)
        ):
            if a != b:
                failures.append(
                    f'text {start + offset} {short(chunk[offset], 120)}: '
                    f'{short(b, 200)}, today {short(a, 200)}'
                )
    report(failures, len(values), 'texts')
