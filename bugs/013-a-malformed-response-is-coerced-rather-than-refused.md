# 13. A malformed `200` response is coerced rather than refused

**Severity: low** — needs a misbehaving endpoint, but the coercion is silent
and can attribute a prediction to the wrong row.

## Problem

The generated response parser coerces where it should validate:

```python
response = {'id': 'r', 'model': 'kumo-tabular', 'predictions': [
    {'row_index': 1.2, 'prediction': 'yes', 'probabilities': {'yes': nan}}],
    'metadata': {}}

TabularAdapter().normalize(response, ('prediction', 'probabilities')).to_pandas()
#    row_index prediction  probability_yes
# 0          1        yes              NaN
```

Three things happened without comment:

- `row_index = 1.2` became `1`. A prediction meant for no row at all is now
  attributed to row 1, and the caller's `merge(on='row_index')` will join it.
- `row_index = True` also becomes `1` — a bool is an `int` in Python, so the
  same `int()` accepts it.
- A `NaN` probability passes through into the result frame.

`_require_int` calls `int(value)`, which accepts anything with an `__int__`;
`_float_dict` calls `float(value)`, which accepts `nan`.

## Location

`packages/sdm-api/src/sdm_api/generated/tfm_api.py`

- `:85` — `row_index` parsed through `_require_int`
- `:143` — `_require_int`, whose body is `int(value)`
- `:173` — `_float_dict`, which does not reject non-finite values

This file is generated, so the fix belongs in the generator
(`scripts/generate_sdm_api.py`) and not in the output.

## Why it needs fixing

The SDK is careful in the other direction — `_require_finite` refuses to *send*
`inf`, `serialize_cell` refuses to send a float that cannot be an `int64` — so
accepting the same shapes on the way back is inconsistent. The name
`_require_int` says it validates.

The row-index case is the one with teeth. Correlation is the SDK's own
responsibility: `_named_entities` goes to considerable lengths to refuse a
relational response that does not line up (missing row_index, out of range,
duplicated, unanswered). A tabular `row_index` of `1.2` silently becoming `1`
is the same class of error, unchecked.

Severity is low only because it needs a server that is already wrong. It should
not be the client that makes that invisible.

## Proposed fix

In the generator:

- `_require_int`: accept `int` and integral floats only; reject `bool`
  explicitly (`isinstance(value, bool)` before the numeric test) and reject a
  float with a fractional part, with a message naming the field and value, as
  `INVALID_RESPONSE`.
- `_float_dict`: reject non-finite values, matching `_require_finite` on the
  send side.
- Regenerate, and re-pin the provenance digests
  (`scripts/vendor_manifest.py`, `scripts/check_contract_pin.py`).

Tests to add: `row_index` of `1.2`, `True`, `'1'` and `None` each raise
`INVALID_RESPONSE` naming the field; a `NaN` probability raises; a well-formed
response is unchanged.
