# Vendored assets

## mermaid.min.js

- Project: https://github.com/mermaid-js/mermaid
- Version: 11.15.0 (browser IIFE bundle, `dist/mermaid.min.js`)
- Source: https://cdn.jsdelivr.net/npm/mermaid@11.15.0/dist/mermaid.min.js
- License: MIT (see `MERMAID_LICENSE`)
- SHA-256: `70137e77bb273bb2ef972b86e8b0400cca8be53cb25bfc45911a186dc98665de`

Bundled so that `Graph.visualize()` renders entirely offline: no system
executables (graphviz `dot`), no CDN fetch, and no calls to the mermaid.ink
web service are needed to display a graph in a notebook. This matters for
restricted notebook environments (Snowflake, Databricks) whose content
security policies or egress rules block external scripts and services.

The bundle is upstream bytes, unmodified. The SHA-256 above is the digest of
the file as published, and `test_vendored_mermaid.py` re-checks it. That test
exists because a repo-wide identifier rename once rewrote two strings inside
this file, which silently turned a vendored artifact into a modified one and
invalidated the third-party attribution built from upstream's manifest.

To upgrade: replace `mermaid.min.js` and `MERMAID_LICENSE` from the same
jsdelivr path at the new version, update the version and SHA-256 here, re-run
the visualization tests, then re-record the attribution:

    python scripts/mermaid_notices.py --refresh   # reads npm, needs network
    python scripts/mermaid_notices.py --write     # updates both LICENSE files

`MERMAID_COMPONENTS.json` beside this file is what those write and read. Do
not edit the attribution in `LICENSE` by hand: enumerating this bundle by hand
was tried and did not converge, which is why it is generated.

## relbench_registry.json

- Project: https://github.com/stanford-star/relbench
- Data: the Hugging Face dataset repositories `stanford-star/relbench-v1`,
  `stanford-star/relbench-v2-extra`, `stanford-star/dbinfer` and
  `stanford-star/tgb`, each at the commit the file records under `revisions`.
  RelBench's other repositories, such as the CTU datasets in
  `stanford-star/redelex`, are not pinned; add one to the script to offer it.
- Written by `scripts/pin_relbench.py`; regenerate it rather than editing it.

What `Graph.from_relbench()` may load, and the bytes it must load.
For each dataset it records the tables, their keys and time columns as the
dataset's `manifest.yaml` declares them, and the SHA-256 of each table's
parquet file.

RelBench 3 moved its data from `relbench.stanford.edu` to Hugging Face and
deleted the `hashes.json` the loader used to fetch from upstream's `main`, so
every load failed with a 404.
The Hub publishes no hash file and serves `main` unless told otherwise.
Recording the commit and the digests here means a dataset changing upstream
fails its check rather than silently changing what a graph holds.

How the digests were obtained: each parquet file is stored in Git LFS, whose
pointer at the pinned commit names the SHA-256 of the file's content, and the
script records that.
Each `manifest.yaml` is a plain Git blob; the script downloads it at the pinned
commit and checks it against its Git object id before reading it.
When first pinned, `rel-f1` and `rel-salt` were downloaded in full, every file
matched its recorded digest, and every table equalled the one in that
dataset's `db.zip` from `relbench.stanford.edu`, itself checked against the
last `hashes.json` upstream published (commit `10d9cfe414`).

To update: run `python scripts/pin_relbench.py`, which pins each repository's
current `main`, or pass `repository=commit` to pin a given commit.
Review the diff, then run the engine's tests with `--runintegration`.
||||||| 3c684406
the visualization tests, and re-enumerate the bundled components in the root
`LICENSE` from the new version's package manifest.
