# sdm-pql

The Predictive Query Language, vendored for the NVIDIA Structured Data Models
client.

```sql
PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id
```

Text goes in, a validated syntax tree comes out. This package runs no model,
reads no data and opens no socket.

It installs the `predictive_query` import package. That is deliberate: this is
a verbatim copy of the upstream distribution, carried only until upstream
publishes to an index. See [`SOURCE.md`](SOURCE.md) for the pinned revision,
the rules that keep the copy honest, and how to replace it with the real
dependency.

**Do not install this alongside `predictive-query-language`.** Both provide
`predictive_query`.
