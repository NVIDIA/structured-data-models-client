# 11. A numpy scalar in `inference` raises a raw `TypeError` from the JSON encoder

**Severity: medium** — an unhandled exception type, from a value that came out
of pandas.

## Problem

`Request(inference=...)` is copied into the payload untouched and handed to
`json`. A numpy scalar — which is what indexing a DataFrame gives you — has no
JSON encoding, so the request dies inside the encoder:

```python
Request(context=..., predict=..., target='y', task='classification',
        inference={'seed': np.int64(4)})
client.predict(request, model=KumoTabular())
# TypeError: Object of type int64 is not JSON serializable
```

`TypeError` is not part of this SDK's error contract, and the message names
neither the key, nor `inference`, nor the request. A caller who wrote

```python
inference={'seed': df['seed'].iloc[0]}
```

gets a bare encoder error with nothing pointing at what they did.

The relational path is fine: the engine's config construction casts to Python
ints on the way through.

## Reproduced

Live, against the tabular NIM. The exception escapes `client.predict`.

## Location

`packages/sdm-client/src/sdm_client/adapters/tabular.py:177-178` — `inference`
is copied into the payload as given.

`packages/sdm-client/src/sdm_client/core/transport.py:604-613` — `_post` passes
`json=payload` to `requests` and catches transport errors, not encoding ones.

## Why it needs fixing

The value comes from pandas, so this is not an exotic input — it is what
happens when a caller reads a parameter out of their own data. And it fails
late, inside a third-party encoder, with a message that mentions neither the
SDK nor the offending key.

It is the same class as [3](003-engine-exceptions-escape-the-error-surface.md):
an exception type outside the documented surface reaching the caller.

## Proposed fix

Two changes, both small:

- Coerce numpy scalars when building the payload: `np.generic` has `.item()`,
  which gives the Python equivalent. Apply it to `inference` values the way
  cell values are already normalised in `wire/dtypes.py`.
- Wrap the encode in `_post` so that anything unencodable becomes a
  `StructuredDataError` naming the offending key and type, rather than a
  `TypeError` naming neither. That is a backstop for every value that reaches
  the payload, not only `inference`.

Tests to add: `inference={'seed': np.int64(4)}` succeeds and sends `4`;
`inference={'x': object()}` raises `StructuredDataError` naming `x`; the
relational path keeps working.
