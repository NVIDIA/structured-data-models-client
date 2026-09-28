# Prerequisites

- **Python** 3.10 to 3.14 for the client, the language, the wire types and the
  connectors. **3.10 to 3.13 for the relational engine**, which is a native
  build with a shorter wheel matrix — so `pip install sdm-client[relational]`
  does not resolve on 3.14 while plain `pip install sdm-client` does. pip
  reports that as `No matching distribution found for sdm-relational-engine`,
  which names the package and not the reason.
- **A reachable NIM** serving `kumo-tabular` or `kumo-relational`. The client
  runs no model and downloads no weights.
- **No GPU** on the client. The NIM owns that.

## Platforms

The client, the language and the connectors are pure Python and install
anywhere. The relational engine is a native build: it compiles a neighbour
sampler, and published wheels are Linux only (x86-64 and aarch64). So `kumo-tabular` works on
a laptop; `kumo-relational` needs Linux.

## Getting a NIM

Two ways.

**A hosted endpoint.** Find the model in the NVIDIA API catalog, generate a
key, and pass the endpoint and key to `SDMClient`.

**The container, run yourself.** Pull the NIM and run it on your own GPU host,
then point the client at it. Use this when the data cannot leave your
environment.

Either way the client needs only the URL and, for a hosted endpoint, the key.
It reads nothing from the environment.
