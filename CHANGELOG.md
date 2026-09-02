# Changelog

## Unreleased

The first release is being assembled. Nothing here has shipped yet.

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
