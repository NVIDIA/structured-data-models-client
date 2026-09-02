# 6. A Parquet prefix on S3 is refused unless it ends in `/`, while the same local directory works

**Severity: medium** — the standard S3 data layout is rejected by a one
character difference.

## Problem

`_resolve_format` decides a path is a Parquet dataset if it ends in a separator
**or** `os.path.isdir(path)` is true. The second test is a local filesystem
call, and it is always false for an `s3://` URI. So a partitioned Parquet
prefix is refused unless the caller happens to write a trailing slash:

```python
read('s3', path='s3://bucket/qa/parts')     # ConnectorError: unsupported file format
read('s3', path='s3://bucket/qa/parts/')    # 400 rows
read('local', path='/tmp/qa/parts')         # 400 rows
```

The error text says the reader supports "Parquet directories", so a user who
reads it concludes their prefix is not one.

## Reproduced

Live, against a real bucket. Two `part-*.parquet` objects under one prefix:

```
no trailing slash: REFUSED unsupported file format 's3://…/qa/parts'
trailing slash   : OK 400 rows
format=parquet   : OK 400 rows
LOCAL no slash   : OK 400 rows
```

## Location

`packages/sdm-connectors/src/sdm_connectors/reader.py:340`

```python
if path.endswith(('/', os.sep)) or os.path.isdir(path):
    return 'parquet'
```

`os.path.isdir` cannot answer for a remote URI, and the function has no other
way to tell a prefix from an object.

## Why it needs fixing

Partitioned Parquet under a prefix is *the* way tabular data lands in S3 —
every Spark, Glue, Athena and `to_parquet(partition_cols=...)` write produces
exactly this. A user pasting a prefix from the console gets a refusal that
names formats rather than the real problem, and the workaround is a trailing
slash that nothing tells them about.

It is also an inconsistency between two connectors for identical data: the same
directory is readable locally and not on S3.

## Proposed fix

Stop asking the local filesystem about a remote path. Two parts:

- Treat a path with **no recognised suffix** as a Parquet dataset for remote
  schemes rather than refusing it. `pd.read_parquet` already resolves a prefix
  through `s3fs`, and a wrong guess surfaces as a clear read error rather than
  silently parsing junk — which is the concern the docstring raises about
  falling back to CSV, and does not apply to Parquet.
- Or, more precisely: when the scheme is not local, ask the filesystem that
  owns the path (`fsspec.filesystem(scheme).isdir(path)`) instead of
  `os.path.isdir`. This is exact, at the cost of one network call and an
  `s3fs` import inside the branch.

Either way, keep `format='parquet'` working as the explicit override, and
mention the trailing-slash equivalence nowhere — it should stop mattering.

Tests to add: a prefix with and without a trailing slash both read; a
single object still reads; a `.csv` object is still CSV; an unreadable prefix
gives a read error naming the path.
