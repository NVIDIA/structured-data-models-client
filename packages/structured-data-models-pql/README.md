# structured-data-models-pql

The Predictive Query Language for the NVIDIA Structured Data Models client.

```sql
PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id
```

Text goes in, a validated syntax tree comes out. This package runs no model,
reads no data and opens no socket.

It is an internal dependency of `structured-data-models-client`, not a supported API: its names
can change in any release. Write PQL with `sdm_client.PredictiveQuery`.

It installs the `predictive_query` import package. The name is inherited from
the separate repository this tree was first carried from, which has since been
retired; the language is maintained here now, under the same Apache-2.0 licence
as the rest of the repository.
