# Changelog

## Unreleased

The first release is being assembled. Nothing here has shipped yet.

### Behaviour worth knowing before you rely on it

**`materialize()` takes the model it is gathering for.** It used to pick by
counting tables, and the model was named later at `predict()`. For a one-table
schema those two decisions could disagree with no way to reconcile them: the
request was always built for `kumo-tabular`, so `predict(model='kumo-relational')`
refused it and no argument could say otherwise. `materialize(model=…)` now takes
the same handle or id `predict()` does. Left unsaid, the shape still decides —
one table tabular, several relational — so existing calls are unchanged.

Gathering and sending for different models is now refused where it used to raise
a bare `KeyError` from inside the adapter: a request carrying relational payloads
is not shaped into a tabular envelope, and the message names the model it was
gathered for.

**`id` holds the key you asked about, on both models.** The relational path
already returned the caller's own key; the tabular path returned what the
endpoint echoed, which is the stringified list of primary-key values — `'[1]'`
where you asked about `1`. Both now return `'1'`, so a result joins back onto
your rows whichever model answered. As with the relational path, a response that
names a row outside the batch, or answers one twice, is refused rather than
silently re-keyed. A composite key, or rows supplied without the key column, are
left as the endpoint sent them: there is no single value to put back.

**A `DECIMAL` column travels as the number it is, and a fractional one rounds.**
A warehouse `DECIMAL`/`NUMBER` arrives as `Decimal`, which is exact and
unbounded, and the wire has no such type. So: a whole-number decimal is sent as
`int64`; one too wide for `int64` is sent as digits in a string, which keeps
every one of them; and a **fractional** decimal is sent as `float64`, which
rounds it. Below 2**53 that rounding is ordinary float behaviour and is
accepted silently. Above it a float cannot hold even the integer part, so the
value is refused rather than quietly mangled. If you need a fractional decimal
kept exactly, send the column as strings.

This applies however the column reaches us. Read through Arrow -- which the
connectors prefer, because it is the path that keeps a wide integer exact -- a
`DECIMAL` used to be announced as `string`, so an id column stopped being an id
to the model. It is now named the same way on both paths.

**A CSV column with blanks keeps its type instead of becoming `float64`.**
`read_table` on a local CSV now reads through pandas' nullable dtypes, so an
integer column containing a blank comes back `Int64` rather than being promoted
to `float64` and losing exactness past 2**53. Columns with no nulls are
narrowed back to the plain numpy dtypes, so the common case is unchanged, but a
column **with** nulls now arrives as `Int64`, `string` or `boolean` where it
previously arrived as `float64` or `object`. Code that tests `frame.dtypes`
against numpy types for such a column will see the difference.

### Notes for a reviewer

**Graph visualization is kept, and so is its provenance.** `Graph.visualize()`
renders a diagram offline because the notebook environments this client is used
from -- Snowflake and Databricks -- block external scripts, so a diagram that
fetched from a CDN would not draw at all. The mermaid bundle that makes that
possible is vendored as upstream bytes with its SHA-256 pinned, its components
enumerated in `LICENSE`, and a test that re-checks the digest. That test exists
because a repo-wide rename once rewrote two identifiers inside the minified
file and silently turned a vendored artifact into a modified one.
`Graph.to_mermaid()` returns the diagram source for anyone who would rather
render it themselves.

**The relational engine's example scripts and notebooks were removed.** They
came across with the port written against `RelationalClient` and
`client.relational(...)`, an interface this client does not export, and they
set a `run_mode` that is no longer a knob. `scripts/check_repo_hygiene.py
retired` now fails on any shipped file naming a retired interface, notebooks
and per-package example directories included. The one quickstart this
repository ships was written against the shipped API and every cell was
executed against a live `kumo-tabular` and `kumo-relational` NIM.
