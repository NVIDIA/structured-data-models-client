# 15. A schema's `relationships` are additive only, so the model is shown a graph the caller did not describe

**Severity: high** — the request is computed over a graph that differs from the
one the `Schema` object reports, and the documented way to correct it does not
work.

## Problem

`_to_engine_graph` hands the frames to the engine, lets the engine infer a
graph, and then **adds** any relationship the schema has that the engine
missed. It never removes one the engine invented and the schema does not have.

The two inferences are not the same, so they disagree on real data:

```python
schema = Schema.from_data(northwind)      # Customers, Orders, "Order Details", Products
len(schema.relationships)                 # 3
len(_to_engine_graph(engine, schema).edges)   # 4
```

The fourth edge is fabricated:

```
Edge(src_table='Products', fkey='UnitsOnOrder', dst_table='Orders')
```

`Products.UnitsOnOrder` is a stock quantity. It is not a foreign key to
anything. Every prediction on this schema is computed over a join that does not
exist.

## The documented control does not work

`Schema.from_data`'s docstring says of `relationships`:

> Overrides for the inferred relationships. **Pass an empty sequence to assert
> there are none.**

It does not:

```python
schema = Schema.from_data(northwind, relationships=[])
len(schema.relationships)                     # 0
len(_to_engine_graph(engine, schema).edges)   # 4

schema = Schema.from_data(northwind, relationships=[Relationship('Orders','CustomerID','Customers','CustomerID')])
len(schema.relationships)                     # 1
len(_to_engine_graph(engine, schema).edges)   # 4
```

The argument is write-only. A caller who inspects `schema.relationships`, sees
what they intended, and proceeds, is looking at something that has no bearing on
the request.

## Location

`packages/sdm-client/src/sdm_client/query.py` — `_to_engine_graph`:

```python
graph = engine.Graph.from_data(frames, verbose=False)
...
# The engine infers edges from the same frames, so most of these are
# already there. ... the schema is the authority only for the ones
# inference missed.
existing = {(edge.src_table, edge.fkey, edge.dst_table) for edge in graph.edges}
for link in schema.relationships:
    if (link.source_table, link.source_column, link.target_table) not in existing:
        graph.link(...)
```

The premise is stated in the docstring — *"The engine infers a graph from
frames in the same way"* — and it is the premise that is wrong. `Schema`'s
`_infer_relationships` matches a column against another table's primary key;
the engine's inference is broader, and finds edges `Schema` does not.

The comment *"the schema is the authority only for the ones inference missed"*
describes the bug precisely: the schema is documented as the authority for all
of them.

## Why it needs fixing

`Schema` is the object this SDK asks the user to build, inspect, correct and
trust. Its `__repr__` reports a relationship count. The docs describe editing it
as the way to fix inference. If what it says is not what is sent, none of that
holds.

The consequence is a silently different model input. A spurious edge changes
which rows are sampled as neighbours and therefore what the model is shown —
not an error, a different answer. And it is not rare: it happens on Northwind,
with default arguments, on the first call.

There is also no diagnostic. `visualize()`/`to_mermaid()` would show the fourth
edge, but they are on the engine's graph and are not reachable from the public
API (see [16](016-visualize-is-documented-but-not-reachable.md)), so the one
tool that would reveal this is unavailable.

## Proposed fix

Make the schema authoritative in both directions. Build the engine graph from
the schema rather than re-inferring:

- Construct the engine graph **without** inference where the engine allows it,
  and add exactly the schema's relationships.
- If the engine's constructor always infers, then reconcile after: remove every
  edge not in `schema.relationships` as well as adding every one that is
  missing. `Graph` supports `link`; it needs an `unlink` — or rebuild the edge
  list wholesale.
- Assert the result: after reconciliation, the engine graph's edge set must
  equal the schema's relationship set. That assertion is the test.

A separate question worth deciding once: should `Schema`'s inference be widened
to match the engine's, or the engine's narrowed to match `Schema`'s? They
should not both exist. `Schema`'s is the more conservative and the one the
docs describe, which argues for making it the only one.

Tests to add: `relationships=[]` yields an engine graph with no edges; an
explicit single relationship yields exactly that edge; Northwind inferred
yields three edges and not four; the engine graph's edges equal
`schema.relationships` for every fixture in the suite.
