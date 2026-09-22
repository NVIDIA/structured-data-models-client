# Changelog

Kept in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) shape, and
versioned by [semantic versioning](https://semver.org) as
[`docs/reference/stability.md`](docs/reference/stability.md) defines it for
these packages. All five distributions share one version and one entry.

## Unreleased

The first release is being assembled. Nothing here has shipped yet.

### Added

- `docs/reference/stability.md`: the compatibility contract — what is public,
  what a version bump means, and the deprecation path. Enforced rather than
  stated: `tests/golden/` records the exported surface of every distribution and
  the wire payloads of four canonical requests, and `tests/test_release_contract.py`
  compares them on every run.
- `RELEASING.md`, and a publish workflow using PyPI Trusted Publishing with PEP
  740 attestations and a CycloneDX SBOM. No upload token exists.
- A blocking `mypy` job. All five distributions ship `py.typed`; nothing checked
  them until now.
- Branch coverage, measured to be read rather than gated on a number.
- CI jobs that run the suite against the *installed wheel*, audit dependencies,
  and generate an SBOM.
- `RuntimeContract.unread_feature_stypes`: the semantic types a deployment
  accepts and its model never reads. The relational model has no text
  processor, so a `text` column is carried, dropped at the model, and
  contributes nothing; naming the types makes that sayable rather than silent.
- `FittedModel` carries the question its context was pinned for, so a session
  can refuse a request that asks something else rather than answering the
  pinned question in its place.

### Fixed

- **A column the model cannot read is named rather than silently dropped.**
  A `text` feature column reaches the relational model and is ignored there.
  Holding content fixed and varying only the semantic type, the same
  determining column scores 1.00 as `categorical` and chance as `text`. The
  request still succeeds, because a model that answers from graph structure
  alone should not refuse over an unreadable attribute, but the columns that
  carried nothing are now named.

- **A time column the caller never named can no longer be acquired.** The
  engine guessed keys and time columns from column *names*, and the guess could
  fail while it was being made: a string column of dates called `release` was
  parsed during graph construction and overflowed, before any schema could undo
  it. Nothing is inferred now; the schema decides every key, time column and
  semantic type.

- **Selecting no rows is the same no-op however it is written.** `indices=[]`
  was refused while an empty `predict` frame was answered with zero rows, so a
  filter that matched nothing had to be handled two ways. Naming no candidates
  to rank stays refused: that is a question with no answer, not an empty one.

- **A failed request no longer names a cause the client cannot know.** Every
  5xx and every dropped connection reported GPU memory pressure. Measured, the
  GPU was at one percent while that appeared, six identical retries failed, and
  only a smaller neighbourhood sample succeeded. The NIM's own detail is quoted
  instead, and both remedies are named.

- **A binary target asked through a predictive query is a classification.**
  `PREDICT users.churn FOR EACH users.user_id`, over a column of 0 and 1, was
  gathered as a regression: 0 and 1 are integers, integers are numerical, and a
  numerical target meant regression. The endpoint answered it — with continuous
  values near 0.5, which read like probabilities and are not. Nothing warned.

  A target holding exactly two whole numbers is now asked as a classification,
  and the client says so. It is the values that decide, not how pandas holds
  them — a 0/1 column with one missing value arrives as `float64` and is still
  a label. Only that case changes: three whole numbers may be a count, and a
  fractional pair like 0.5/1.5 is not a label, so both remain regressions.

  **This changes what goes over the wire** for such a request — `task.kind`
  becomes `classification`, and the target column is declared `categorical`
  rather than `numerical`, which is the correction the tabular adapter already
  made for classification targets and could not make while the task was wrong.
  If you were relying on the old reading, pass `task='regression'` to keep it.

  **Say plainly what this costs:** the rule is two whole numbers, so a genuine
  count that happens to hold two — `items_bought` in {0, 1}, `visits` in
  {1, 2}, `rating` in {7, 9} — is now asked as a classification and its request
  moves on the wire. That is a change of default rather than a repair: those
  targets were being read correctly before. The warning names the other
  reading and `task='regression'` restores it, and the judgement is that
  reading every 0/1 label as a quantity was the more common and the more
  damaging mistake.

- **A context that cannot teach is refused where the endpoint would refuse it.**
  Score every row of one class and the context left behind holds only the
  other. A classification over that is rejected by the endpoint
  (`TASK_SCHEMA_MISMATCH`), so it is now rejected locally in the same terms,
  before a table is encoded and sent. A regression over a constant context is
  accepted by the endpoint, so it is still only warned about. Previously the
  same degeneracy produced a working request on one path and a guaranteed
  server error on two others.
- **`materialize()` takes `task=`.** A query names a target, not what is asked
  of it, and the single-table path had no way to say. It is authoritative, and
  stating it silences the warning above. A relational query resolves to its own
  kind in the engine, so passing `task=` with one is refused rather than
  silently ignored.
- `predictive_query` attached no logging handler, so its warnings reached the
  stderr of any application that had not configured logging. It now installs a
  `NullHandler`; applications that *have* configured logging still receive the
  records, unchanged.
- `sdm-api`, `sdm-pql` and `sdm-client` each declared their version in two
  places. There is now one literal per distribution, in `_version.py`, and a
  test that the five agree.
- The four pure-Python distributions admitted Python 3.14 through
  `requires-python` but did not classify it. They now do — and the narrower
  range of the `[relational]` extra, which is a native build, is documented
  where users meet it rather than surfacing as an unexplained pip resolver
  error.

### Behaviour worth knowing before you rely on it

**`materialize()` takes the model it is gathering for.** It used to pick by
counting tables, and the model was named later at `predict()`. For a one-table
schema those two decisions could disagree with no way to reconcile them: the
request was always built for `kumo-tabular`, so `predict(model='kumo-relational')`
refused it and no argument could say otherwise. `materialize(model=…)` now takes
the same handle or id `predict()` does. Left unsaid, the shape still decides,
so existing calls are unchanged: one table with no relationships is tabular, and
anything wider is relational. That now counts a relationship as well as a second
table, matching `Request.is_relational` — a lone table pointing at itself used to
be gathered tabularly and then refused by the model for needing relationships.

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

A session answers the same way, which needed the adapter to be handed the whole
request rather than the half that just went over the wire. That also settles a
latent one: a relational session could not tell a forecast's repeated rows from
a duplicate answer, because the task kind was pinned at fit time and the adapter
only ever saw the scoring call. The wire is unchanged -- a session's scoring
call still carries only what the contract allows there.

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
