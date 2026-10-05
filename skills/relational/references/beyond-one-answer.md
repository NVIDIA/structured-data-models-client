# Rankings, forecasts and explanations

Three questions that are not one prediction per entity: a ranking is a
call per candidate, a forecast a call per step, and an explanation covers
one scored row. [`../SKILL.md`](../SKILL.md) is the plain path; this is
what sits beyond it.

A ranking asks which of a set of candidates an entity will reach, and comes
back as a list in one row rather than a row per candidate, because expanding it
would stop the key identifying a row:

```python
request = PredictiveQuery(
    'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2 '
    'FOR EACH users.user_id',
    schema,
).materialize(indices=[1, 2], candidates=items['item'], model=KumoRelational())
```

A candidate whose gathered context holds only one answer cannot be scored, so
it is left out of the ranking and named on `result.unscored_candidates`, with a
warning. That happens when the labels do not vary — every entity in the sample
reached that candidate, or none did — and the second is the common one. The ranking still fills `TOP k` from
whatever was scored, and comes back shorter only when fewer than `k` candidates
survive — so a short list means check `result.unscored_candidates`, and a full
one does not mean every candidate was considered.

A forecast asks the same question over several future windows, one call per
step, and gives a row per entity per step. The clause goes after the target:

```python
request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FORECAST 3 TIMEFRAMES '
    'FOR EACH users.user_id',
    schema,
).materialize(indices=[1, 2], model=KumoRelational())

with SDMClient(url='http://localhost:8000', timeout=600) as client:
    print(client.predict(request, model=KumoRelational()).to_pandas())
```

**Verified** against a live endpoint:

```
   row_index  id  forecast_step  prediction
0          0   1              1    2.173828
1          0   1              2    2.929688
2          0   1              3    2.144531
3          1   2              1    2.181641
4          1   2              2    2.912109
5          1   2              3    2.185547
```

`forecast_step` counts from one, and `row_index` repeats, so neither alone
identifies a row: a forecast is keyed by the pair.

Neither can be fitted into a session. Each part needs the context its own
labels come from, so a ranking and a forecast are asked with `predict()` and
`fit()` refuses them.

An explanation is asked for as an output, `outputs=[..., 'explanation']`, never
an `explain=` argument, and read from `result.explanation`. It covers one
scored row at a time. A ranking refuses it locally, and the shipping endpoint
refuses it everywhere else, so treat `result.explanation` as possibly `None`.
