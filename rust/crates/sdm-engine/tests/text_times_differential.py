# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``text-times-differential.json``: the reference's readings of
text times against the core's, over the inputs T1.25 recorded pandas'
readings of (T1.25b), every difference classified.

Run once, by hand, under the reference's pandas (3.0.6), beside
``text_times_record.py``, whose reading of ``date-rules.md``'s "The core's
text formats" stands for the core's (``text_times.rs`` holds the core to
it, and to this report's examples); CI never runs it:

    git show b455b94:rust/crates/sdm-engine/tests/text-times.json > OLD.json
    python rust/crates/sdm-engine/tests/text_times_differential.py OLD.json OUT.json

Four outcomes are compared for each column, as the reference reaches them:
``materialize``, a SQL source's text time column read value by value
(``pd.to_datetime(values, unit='ns', errors='coerce')`` against the core's
reading); ``time_choice``, the time column's choice reading them to UTC
(``utc=True``); ``inference``, whether text is a timestamp by
``contains_timestamp``'s score, under a name that says time and one that
does not (``errors='coerce'`` over the column's non-empty values against
the core's share); and ``day_first``, the format inference warns of. Each
difference is one of three classes:

- ``a``: the core refuses by name (``text_time``) a column holding text
  outside the list, which the owner's T1.25b answer approves;
- ``b``: every value the outcome turns on is in the list, and the core
  reads, refuses or warns otherwise than the reference, which needs the
  owner's approval; its kinds are named (``b.order_open`` and the rest,
  below);
- ``c``: text outside the client's shapes, which the core counts as no
  time under a name that does not say time and never warns of, where the
  reference reads it as a time, and so decides or warns otherwise.

The report keeps each class's count and its three shortest examples.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
import warnings

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('record', HERE / 'text_times_record.py')
assert _spec and _spec.loader
record = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record)

NANOS = {'s': 10**9, 'ms': 10**6, 'us': 10**3, 'ns': 1}
DAY_FIRST = re.compile(r'Parsing dates in (.*) format when dayfirst=False')


def pandas(values: list[str | None], **options: object) -> object:
    """pandas' reading, each instant in UTC nanoseconds as text, or
    ``'MIXED'``, ``'ERR'``."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            parsed = pd.to_datetime(
                pd.Series(values, dtype=object), errors='coerce', **options
            )
        except ValueError as error:
            return 'MIXED' if 'Mixed timezones' in str(error) else 'ERR'
        except Exception:  # noqa: BLE001
            return 'ERR'
    out = []
    for stamp in parsed:
        if pd.isna(stamp):
            out.append(None)
            continue
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert('UTC').tz_localize(None)
        # In the stamp's own unit, past what nanoseconds in an int64 hold.
        out.append(str(int(stamp._value) * NANOS[stamp.unit]))
    return out


def pandas_day_first(values: list[str | None]) -> str | None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            pd.to_datetime(pd.Series(values, dtype=object), errors='coerce')
        except Exception:  # noqa: BLE001
            pass
    said = [DAY_FIRST.match(str(w.message)) for w in caught]
    formats = [m.group(1) for m in said if m]
    return formats[0] if formats else None


def kinds(values: list[str | None]) -> list[str]:
    """Each value's kind to the core: in the list, outside it but in the
    client's shapes, or outside both."""
    out = []
    mf, df = record.readings(values)
    for a, b in zip(mf, df):
        if a == record.OUTSIDE or b == record.OUTSIDE:
            out.append('unshaped')
        elif a == record.SHAPED or b == record.SHAPED:
            out.append('shaped')
        else:
            out.append('listed')
    return out


def alone(value: str) -> object:
    """pandas' reading of ``value`` on its own, value by value."""
    read = pandas([value], unit='ns')
    return read[0] if isinstance(read, list) else read


def classify_reading(values: list[str | None], ref: object, core: object) -> str | None:
    """The class of a difference in a reading of the values, or None:
    ``a``; ``b.order_open``, an order of day and month the values leave
    open, refused where the reference reads each value month first unless
    it cannot be; ``b.order_decided``, a value read in the order the values
    decide where the reference reads it otherwise; ``b.format_guess``, a
    value the reference reads as no time by the format it guessed from
    another (the time column's choice); ``b.offsets``, offsets mixed refused
    on one side only; ``b.range``, an instant past what the reference's
    nanoseconds hold."""
    if ref == core or (ref == 'MIXED' and core == 'mixed'):
        return None
    if ref == 'CLOCK':
        # The clock's words, read as the clock's time: outside the list.
        return 'a' if core == 'outside' else 'b.other'
    if core == 'outside':
        return 'a'
    if core == 'order':
        return 'b.order_open'
    if core == 'mixed' or ref == 'MIXED':
        return 'b.offsets'
    if not isinstance(ref, list) or not isinstance(core, list):
        return 'b.other'
    return '+'.join(
        sorted({why(v, r, c) for v, r, c in zip(values, ref, core) if r != c})
    )


def why(value: str | None, ref: str | None, core: str | None) -> str:
    """Why one value in the list reads otherwise: in another order, past
    the reference's nanoseconds, or by the reference's format guessed from
    another value (it reads the value alone as the core does)."""
    if ref is not None and core is not None:
        return 'b.order_decided'
    if ref is None and core is not None and not -(2**63) <= int(core) < 2**63:
        return 'b.range'
    if ref is None and core is not None and str(alone(value)) == core:
        return 'b.format_guess'
    return 'b.other'


def decision(score_of: float, named: bool) -> bool:
    return (0.3 if named else 0.0) + score_of >= 1.0


def core_decides(first: list[str], named: bool) -> object:
    """The core's answer to whether text is a timestamp: the share at
    fewest and at most, the most counting text outside the client's shapes
    under a name that says time."""
    share = record.share(first)
    if share == 'mixed':
        return 'text_time_zones'
    fewest, most, unshaped = share
    low = decision(fewest / len(first), named)
    high = decision((most + (unshaped if named else 0)) / len(first), named)
    return low if low == high else 'text_time'


def classify_inference(values: list[str | None], named: bool) -> tuple[str | None, object, object]:
    """The class of a difference in whether text is a timestamp: ``a``;
    ``b.offsets``, offsets mixed refused or raised on one side only;
    ``b.order_decided`` or ``b.format_guess``, values in the list counted
    otherwise, by the order the values decide or by the format the
    reference guessed from the first value; ``c``, text outside the
    client's shapes the reference reads as times."""
    first = [v for v in values if v is not None and v != ''][:100]
    if not first:
        return None, None, None
    ref = pandas(first)
    if ref == 'MIXED':
        ref_decision: object = 'refused (offsets mixed)'
    elif ref == 'ERR':
        ref_decision = 'error'
    else:
        # The clock's words read as the clock's time: counted, not dated.
        ref_decision = decision(sum(r is not None for r in ref) / len(first), named)
    core_decision = core_decides(first, named)
    if ref_decision == core_decision:
        return None, ref_decision, core_decision
    if ref_decision == 'refused (offsets mixed)' and core_decision == 'text_time_zones':
        return None, ref_decision, core_decision
    if core_decision == 'text_time':
        return 'a', ref_decision, core_decision
    if 'refused (offsets mixed)' in (ref_decision,) or core_decision == 'text_time_zones':
        return 'b.offsets', ref_decision, core_decision
    # Which values the two count differently: text outside the client's
    # shapes the reference reads, or values in the list.
    if isinstance(ref, list) and record.share(first) != 'mixed':
        mf, _ = record.readings(first)
        kind = kinds(first)
        differ = [
            (value, r, k)
            for value, r, c, k in zip(first, ref, mf, kind)
            if (r is not None) != (c[0] == 'time')
        ]
        if differ and all(k == 'unshaped' for _, _, k in differ):
            return 'c', ref_decision, core_decision
        if differ and all(k == 'listed' for _, _, k in differ):
            reasons = {
                why(v, r, str(c[1]) if c[0] == 'time' else None)
                for (v, r, _), c in zip(
                    differ, [mf[first.index(v)] for v, _, _ in differ]
                )
            }
            return '+'.join(sorted(reasons)), ref_decision, core_decision
    return 'b.other', ref_decision, core_decision


def main() -> None:
    with open(sys.argv[1], encoding='utf-8') as old:
        inputs = json.load(old)
    assert inputs['year'] == record.YEAR, inputs['year']
    columns = [c['values'] for c in inputs['columns']]
    outcomes: dict[str, dict] = {
        name: {'columns': 0, 'agree': 0, 'classes': {}}
        for name in ('materialize', 'time_choice', 'inference', 'day_first')
    }

    def note(name: str, cls: str | None, example: dict) -> None:
        outcome = outcomes[name]
        outcome['columns'] += 1
        if cls is None:
            outcome['agree'] += 1
            return
        entry = outcome['classes'].setdefault(cls, {'count': 0, 'examples': []})
        entry['count'] += 1
        entry['examples'].append(example)

    for values in columns:
        for name, options, utc in (
            ('materialize', {'unit': 'ns'}, False),
            ('time_choice', {'utc': True}, True),
        ):
            ref = (
                'CLOCK'
                if any(v in ('now', 'today') for v in values)
                else pandas(values, **options)
            )
            core = record.column(values, utc=utc)
            note(
                name,
                classify_reading(values, ref, core),
                {'values': values, 'reference': ref, 'core': core},
            )
        for named in (True, False):
            cls, ref, core = classify_inference(values, named)
            if ref is None:
                continue
            note(
                'inference',
                cls,
                {'values': values, 'named': named, 'reference': ref, 'core': core},
            )
        # The warning, where the core's inference decides under any name
        # (a refused description warns of nothing): every difference is in
        # how the order of day and month is decided, so in the list.
        first = [v for v in values if v is not None and v != ''][:100]
        if first and all(
            isinstance(core_decides(first, named), bool) for named in (True, False)
        ):
            ref = pandas_day_first(first)
            core = record.day_first(first)
            cls = None
            if ref != core:
                # The client's rule against the format guessed from the
                # first value; text outside the shapes the core never
                # warns of.
                cls = 'b.order_rule' if set(kinds(first)) == {'listed'} else 'c'
            note('day_first', cls, {'values': first, 'reference': ref, 'core': core})

    for outcome in outcomes.values():
        for entry in outcome['classes'].values():
            unique = {json.dumps(e, sort_keys=True): e for e in entry['examples']}
            entry['examples'] = [
                unique[k] for k in sorted(unique, key=lambda k: (len(k), k))
            ][:3]
        outcome['classes'] = dict(sorted(outcome['classes'].items()))
    document = {
        'format': 'sdm-text-times-differential/1',
        'reference': f'pandas {pd.__version__}',
        'inputs': (
            "the columns of text-times.json as T1.25 recorded them (b455b94), "
            f'{len(columns)} columns'
        ),
        'year': record.YEAR,
        'outcomes': outcomes,
    }
    with open(sys.argv[2], 'w', encoding='utf-8') as out:
        json.dump(document, out, indent=1, ensure_ascii=False)
        out.write('\n')


if __name__ == '__main__':
    main()
