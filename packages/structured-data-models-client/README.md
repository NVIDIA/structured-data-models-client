# structured-data-models-client

One Python client for NVIDIA's structured-data foundation models. Ask a
predictive question of one table, or of a whole relational schema, without
training a model for each dataset.

```bash
pip install structured-data-models-client
```

Inference runs in a NIM, so this package downloads no weights and needs no GPU.

```python
import pandas as pd
from sdm_client import KumoTabular, Request, SDMClient

labelled = pd.DataFrame(
    {
        'age': [31.0, 44.5, 28.0, 51.0, 39.0],
        'spend': [120.0, 430.0, 90.0, 510.0, 260.0],
        'churn': [0, 1, 0, 1, 0],
    }
)
to_score = pd.DataFrame({'age': [33.0, 47.0], 'spend': [150.0, 480.0]})

request = Request(
    context=labelled, predict=to_score, target='churn', task='classification'
)

with SDMClient(url='http://localhost:8000') as client:
    print(client.predict(request, model=KumoTabular()).to_pandas())
```

The model is an argument, not a method name. For several related tables, ask in
the Predictive Query Language and pass `KumoRelational()` instead; that needs
`pip install 'structured-data-models-client[relational]'`.

Documentation, the other four packages and the examples are in the repository:
<https://github.com/NVIDIA/structured-data-models-client>

Apache-2.0. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
