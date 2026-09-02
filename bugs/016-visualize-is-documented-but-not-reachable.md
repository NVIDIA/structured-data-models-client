# 16. `Schema` cannot draw itself, though that is what the vendored mermaid bundle is for

**Severity: medium** — the documented call raises `AttributeError` on the
object the SDK asks you to build.

## Problem

The README says:

> `mermaid.min.js` is bundled in the relational engine so `Graph.visualize()`
> draws offline: Snowflake and Databricks notebooks block external scripts, so
> a diagram fetched from a CDN would not draw.

`docs/about/architecture.md` records the same decision, and adds that
`to_mermaid()` returns the source. The asset is vendored, SHA-256 pinned, its
components enumerated in `LICENSE`, and its digest asserted by a test.

Neither method is reachable:

```python
schema.visualize()    # AttributeError: 'Schema' object has no attribute 'visualize'
schema.to_mermaid()   # AttributeError: 'Schema' object has no attribute 'to_mermaid'
```

Both live on the engine's `Graph`. That class *is* exported
(`sdm_relational_engine.rfm.Graph`), so the methods are not unreachable in
general — but they are unreachable from a `Schema`, which is the object a
caller of this SDK actually has. Turning one into the other is a private
helper:

```python
from sdm_client.query import _to_engine_graph, _engine
_to_engine_graph(_engine(), schema).to_mermaid()   # works
```

`Graph` is also not the name the SDK uses. It was renamed to `Schema` for this
client, so a reader following the README looks for a class the public API does
not have.

## Reproduced

```
to_mermaid FAILED AttributeError 'Schema' object has no attribute 'to_mermaid'
visualize  FAILED AttributeError 'Schema' object has no attribute 'visualize'
```

Through the private path, both work — `to_mermaid()` renders all four tables
including the one whose name has a space.

## Location

`packages/sdm-relational-engine/src/sdm_relational_engine/rfm/graph.py:2623`
— `Graph.visualize`

`packages/sdm-relational-engine/src/sdm_relational_engine/rfm/graph.py:2706`
— `Graph.to_mermaid`

`packages/sdm-client/src/sdm_client/schema.py` — `Schema` has neither.

`README.md:91-94` and `docs/about/architecture.md:586` — both describe the
feature as available.

## Why it needs fixing

Three separate costs, and they compound:

1. A user follows the README, calls it on the object they have, and gets an
   `AttributeError`. On a public repository that is the first impression.
2. The vendored bundle is justified *by* this feature. If it is unreachable,
   the SDK carries a third-party JavaScript bundle, its licence obligations and
   its provenance pinning for nothing — and that is exactly what an OSRB
   reviewer will ask about.
3. It is the one tool that would let a user see the fabricated relationship in
   [15](015-the-schemas-relationships-are-additive-only.md). The graph is wrong
   and the way to look at it is missing.

## Proposed fix

Add both methods to `Schema`, delegating to the engine's graph:

```python
def to_mermaid(self, show_columns: bool = True) -> str: ...
def visualize(self, ...): ...
```

They should render **the schema**, not the engine's re-inferred graph — which
matters more once [15](015-the-schemas-relationships-are-additive-only.md) is
fixed and the two agree. Until then, rendering the engine graph is arguably the
more useful of the two, because it shows what was actually sent.

Note the extras boundary: the engine is an optional dependency, so `visualize`
should raise `MissingExtraError` when it is not installed, the way the
relational path already does.

Update `README.md` and `docs/about/architecture.md` to say `Schema.visualize()`
rather than `Graph.visualize()` once it exists.

Tests to add: `Schema.to_mermaid()` returns source naming every table,
including one whose name has a space; `visualize()` on a schema with no engine
installed raises `MissingExtraError`; the README's example runs.
