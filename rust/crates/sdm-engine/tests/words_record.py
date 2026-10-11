# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``words.json``, the reference sampler's words for a text feature
(``_normalize_text``, T1.26b), which ``words.rs`` holds ``sdm_engine::words``
to.

Run once, by hand, in the reference's environment (pandas 3.0.6, pyarrow
25.0.1, Python 3.12) from the reference checkout; CI never runs it, and the
vectors are not recorded again to pass a dependency bump (F4):

    python rust/crates/sdm-engine/tests/words_record.py OUT.json REFERENCE

``lower`` is, for every code point but the surrogates, its text after the
pipeline's own lower-casing (``fillna('')``, ``astype(str)``, ``str[:300]``,
``str.lower()``) wherever that is not the code point itself, as a list of
code points. ``values`` are texts, each with the words ``_normalize_text``
gives for it: every punctuation mark, every whitespace code point, the
line-break shapes, mixed case, long texts around the 300-character cut and
the 50-word limit, and random mixtures of those, drawn from a seeded
generator, so a second run writes the same file. ``null`` stands for a
missing value.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa


def main() -> None:
    out, reference = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(
        0,
        str(
            reference / 'packages/structured-data-models-relational-engine/src'
        ),
    )
    from sdm_relational_engine.rfm.base.sampler import _normalize_text

    def pipeline(texts: list[str | None]) -> list[list[str]]:
        series = pd.Series(texts, dtype=pd.ArrowDtype(pa.string()))
        return [list(words) for words in _normalize_text(series)]

    points = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    lowered = (
        pd.Series([chr(c) for c in points], dtype=pd.ArrowDtype(pa.string()))
        .fillna('')
        .astype(str)
        .str[:300]
        .str.lower()
    )
    lower = {
        str(c): [ord(x) for x in text]
        for c, text in zip(points, lowered, strict=True)
        if text != chr(c)
    }

    spaces = [chr(c) for c in points if chr(c).isspace()]
    marks = list('\'".,()!?;:') + ['-', '_', '/', '<', '>', '&', '#']
    pieces = (
        ['Word', 'ALPHA', 'beta', 'Ünïcödé', 'İstanbul', 'ΟΔΟΣ', 'ß', 'ǅ', 'Ω']
        + ['<br>', '<BR>', '<br/>', '<br />', '<br\t/>', '<br  >', '<br/ >']
        + ['<bR >', '<b r>', '<brx>', 'x<br>y', '​', '﻿', '᠎']
        + marks
        + spaces
    )
    rng = random.Random(2610)
    values: list[str | None] = [None, '', ' ', '...', 'Sport-100 Helmet, Red']
    values += pieces
    values += [f'a{s}b' for s in spaces]
    values += [f'{m}x{m}' for m in marks]
    for n in (49, 50, 51, 120):
        values.append(' '.join(f'w{i}' for i in range(n)))
    for n in (298, 299, 300, 301):
        values.append('a' * n + ' tail')
        values.append('b' * (n - 1) + 'İ' + 'c')
    for _ in range(600):
        size = rng.randint(1, 12)
        values.append(
            ''.join(
                rng.choice(pieces) + rng.choice(['', ' ', ''])
                for _ in range(size)
            )
        )
    for _ in range(40):
        size = rng.randint(250, 400)
        values.append(
            ''.join(
                rng.choice(['ab ', 'C.', ', ', '<br>', 'İ', 'Σ '])
                for _ in range(size // 3)
            )
        )
    words = pipeline(values)
    document = {
        'description': (
            "The reference sampler's words for a text feature "
            '(_normalize_text), recorded by words_record.py'
        ),
        'environment': {
            'python': sys.version.split()[0],
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
        },
        'lower': lower,
        'values': [[v, w] for v, w in zip(values, words, strict=True)],
    }
    Path(out).write_text(
        json.dumps(document, ensure_ascii=True, indent=0) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
