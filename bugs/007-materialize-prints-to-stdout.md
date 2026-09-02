# 7. `materialize()` prints progress to stdout

**Severity: low** — but visible in the first ten seconds of using the SDK.

## Problem

Calling `PredictiveQuery.materialize()` writes to stdout:

```
✅ Materializing regression task (0.03s)
   ↳ Generated context of size 0.17MB
```

Nothing in the SDK's API asks for this and nothing turns it off.

The SDK already tries to suppress it — `_gather` constructs the engine model
with `verbose=False` — but that flag is a constructor argument, and
`materialize_task` has its own `verbose` parameter defaulting to `True` which
`_gather` never passes.

## Reproduced

```python
buffer = io.StringIO()
with redirect_stdout(buffer):
    query.materialize(indices=ids, num_context_samples=60)
buffer.getvalue()
# '✅ Materializing regression task (0.03s)\n   ↳ Generated context of size 0.17MB'
```

## Location

`packages/sdm-client/src/sdm_client/query.py:340` — `model.materialize_task(...)`
is called without `verbose`.

`packages/sdm-relational-engine/src/sdm_relational_engine/rfm/rfm.py:1183` —
`verbose: bool | ProgressLogger = True`.

`packages/sdm-relational-engine/src/sdm_relational_engine/utils/progress_logger.py:178-223`
— the `print()` calls.

## Why it needs fixing

A library writes to a logger; an application writes to stdout. Printing from
inside a function call breaks anything that reads this SDK's own stdout,
scrambles progress bars and structured logs, and puts ANSI escapes and emoji
into CI output and log aggregators. On an open-source release it is also the
first thing a reviewer notices.

The interesting part is that the intent is already there — `verbose=False` on
the constructor — so this reads as an oversight rather than a decision.

## Proposed fix

Pass `verbose=False` to `materialize_task` in `_gather`. One line.

If progress is wanted later, it belongs behind the standard mechanism: a
`logging.getLogger('sdm_client')` call the application configures, silent by
default.

Test to add: `materialize()` under `redirect_stdout` produces an empty string.
The probe above is the test.
