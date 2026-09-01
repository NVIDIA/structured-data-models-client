# sdm-client

One Python client for NVIDIA's structured-data foundation model NIMs.

```python
from sdm_client import Request, StructuredDataClient, KumoTabular

request = Request(
    context=labelled, predict=to_score, target='churn', task='classification'
)

with StructuredDataClient(url='http://localhost:8000') as client:
    print(client.predict(request, model=KumoTabular()).to_pandas())
```

The model is an argument, not a method name. See the repository README for the
full picture.
