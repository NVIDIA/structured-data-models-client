# QA probes

The end-to-end pass these fixes came out of, and now their regression suite.
Every probe here was written against a real failure against a live NIM; each
one that passes is a bug that stays fixed.

These are not part of the test suite. They need live endpoints, real
credentials and datasets that are downloaded rather than vendored, so they are
run by hand.

## Setup

```bash
mkdir -p ~/sdm-qa/data && cd ~/sdm-qa/data
curl -sSLO https://archive.ics.uci.edu/static/public/2/adult.zip
curl -sSLO https://archive.ics.uci.edu/static/public/186/wine+quality.zip
curl -sSLO https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
curl -sSLO https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
curl -sSLO "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
curl -sSL -o northwind.sqlite \
  https://github.com/jpwhite3/northwind-SQLite3/raw/main/dist/northwind.db

unzip -q adult.zip -d adult
unzip -q wine+quality.zip -d wine
unzip -q bank+marketing.zip -d bank && unzip -q bank/bank-additional.zip -d bank
unzip -q "predict+students+dropout+and+academic+success.zip" -d students
```

Each is there for something it is awkward about: Adult for `?`-as-missing and
mixed types, Wine for an ordinal multiclass target, Titanic for nulls and bool
columns, Bank Marketing for an 11% positive rate and `'unknown'` as a real
level, Students for 29 column names carrying spaces, slashes, parentheses and a
trailing tab, and Northwind for a relational schema with string keys, real
dates and a table whose name contains a space.

## Running

```bash
export SDM_TABULAR_URL=http://localhost:8000
export SDM_RELATIONAL_URL=http://localhost:8001

python probe_tabular.py       # kumo-tabular on Adult, Wine, Titanic, Bank, Students
python probe_relational.py    # kumo-relational on Northwind
python probe_connectors.py    # local files, SQLite, DuckDB
python probe_edges.py         # awkward values, shapes, the error contract
python probe_cloud.py         # S3   (needs AWS credentials and /tmp/qa_bucket)
python probe_warehouse.py     # Snowflake and Databricks (needs SNOWFLAKE_*, DATABRICKS_*)
```

The relational live-NIM contract tests have a runner that creates its own
environment and defaults to the smoke suite:

```bash
qa/run_rfm_nim_live_tests.sh --url http://localhost:8001
qa/run_rfm_nim_live_tests.sh --url http://localhost:8001 --full
```

Each writes a JSON record to `~/sdm-qa/findings/` and prints a summary. A probe
never stops the run: it records `PASS`, `FAIL` or `RAISED` and continues, so one
broken thing does not hide the next.

`probe_cloud.py` and `probe_warehouse.py` create objects and tables and remove
them before they exit.

## Status

All 139 probes pass. Each was written against a failure that is now fixed, so
the suite is the regression check for those fixes: point it at a live NIM and a
warehouse, and anything that comes back red is something that used to be
broken.

| Suite | Probes |
|---|---|
| `probe_tabular.py` | 24 |
| `probe_relational.py` | 27 |
| `probe_connectors.py` | 29 |
| `probe_edges.py` | 27 |
| `probe_warehouse.py` | 23 |
| `probe_cloud.py` | 9 |

## Release shape

Not a probe suite, but a check that the thing users install is the thing that was
tested. Run after building, from a venv with no source tree on `sys.path`:

```bash
pip install build
for pkg in sdm-api sdm-pql sdm-connectors sdm-client; do
  python -m build --wheel --sdist -o dist/ packages/$pkg
done
python -m build --wheel -o dist/ packages/sdm-relational-engine   # native, wheel only

python -m venv /tmp/relenv
/tmp/relenv/bin/pip install --find-links dist/ 'sdm-client'
```

What that proved on this branch: all five distributions build; a base install
pulls `sdm-api`, `sdm-pql`, pandas, numpy, requests, urllib3 and the ANTLR
runtime and nothing else; `sdm_relational_engine` is **not** importable, and the
relational path raises `MissingExtraError` naming the extra to install;
`[relational]`, `[duckdb]` and `[s3]` each resolve on their own; and the fixes
are present in the installed bytes, not only in the working tree.
