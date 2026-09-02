# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Self-contained Mermaid rendering for graph visualization.

Display never requires system executables (graphviz ``dot``), CDN scripts,
or the mermaid.ink web service: the bundled ``assets/mermaid.min.js`` is
inlined into a standalone HTML document that the notebook front end renders
locally. Only ``render_image`` (PNG/SVG export) talks to mermaid.ink, since
rasterizing Mermaid requires a browser engine.
"""

from __future__ import annotations

import base64
import html
import json
from functools import lru_cache
from importlib import resources

MERMAID_INK_URL = 'https://mermaid.ink'

_MERMAID_CONFIG = {
    'startOnLoad': True,
    'theme': 'neutral',
    'er': {
        'useMaxWidth': False,
    },
}


@lru_cache
def _mermaid_js() -> str:
    path = (
        resources.files('sdm_relational_engine.rfm')
        / 'assets'
        / 'mermaid.min.js'
    )
    return path.read_text(encoding='utf-8')


def to_html(source: str) -> str:
    r"""Returns a standalone HTML document rendering the Mermaid ``source``.

    The document embeds the bundled mermaid.js, so it renders in any browser
    or notebook iframe without network access.
    """
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ margin: 0; background: #ffffff; }}
pre.mermaid {{ margin: 8px; }}
</style>
</head>
<body>
<pre class="mermaid">
{html.escape(source)}
</pre>
<script>{_mermaid_js()}</script>
<script>mermaid.initialize({json.dumps(_MERMAID_CONFIG)});</script>
</body>
</html>"""


def to_iframe(source: str, height: int = 540) -> str:
    r"""Returns an ``<iframe srcdoc=...>`` snippet wrapping :meth:`to_html`.

    Notebook front ends (Jupyter, Databricks, Colab, VS Code) sanitize or
    scope scripts in raw HTML output; a sandboxed ``srcdoc`` iframe executes
    the inlined mermaid.js reliably across all of them.
    """
    doc = html.escape(to_html(source), quote=True)
    return (
        f'<iframe srcdoc="{doc}" width="100%" height="{height}" '
        f'style="border:none;" sandbox="allow-scripts"></iframe>'
    )


def render_image(source: str, format: str, timeout: int = 30) -> bytes:
    r"""Renders Mermaid ``source`` to ``'png'`` or ``'svg'`` bytes via the
    mermaid.ink web service (requires network access).
    """
    if format not in ('png', 'svg'):
        raise ValueError(
            f"Unsupported image format '{format}'. Expected "
            f"either 'png' or 'svg'."
        )

    import requests

    state = {'code': source, 'mermaid': {'theme': 'neutral'}}
    encoded = base64.urlsafe_b64encode(
        json.dumps(state).encode('utf-8')
    ).decode('ascii')
    route = 'img' if format == 'png' else 'svg'
    url = f'{MERMAID_INK_URL}/{route}/base64:{encoded}'
    if format == 'png':
        url += '?type=png'

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not render the graph to '{format}' because the "
            f'mermaid.ink web service is unreachable (image export requires '
            f"network access). Use `visualize(path='graph.html')` for a "
            f'fully offline rendering instead. Error: {e}'
        ) from e

    return response.content
