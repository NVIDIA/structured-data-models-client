# Changelog

## Unreleased

The first release is being assembled. Nothing here has shipped yet.

### Removed before it could mislead anyone

The relational engine's example scripts and AdventureWorks notebooks came
across with the port and were written against `RelationalClient` and
`client.relational(...)` — an interface that belongs to a different repository
and that this client does not export. They also set a `run_mode`, which is no
longer a knob here.

Rather than ship examples nobody could run, they are removed. The one
quickstart this repository ships, `examples/sdm_client_quickstart.ipynb`, was
written against the shipped API and every cell was executed against a live
`kumo-tabular` and `kumo-relational` NIM.

`scripts/check_repo_hygiene.py retired` now fails on any shipped file naming a
retired interface, notebooks and per-package example directories included. The
earlier sweep looked only at `src/` and the top-level `examples/`, which is how
a whole tree of them stayed invisible.
