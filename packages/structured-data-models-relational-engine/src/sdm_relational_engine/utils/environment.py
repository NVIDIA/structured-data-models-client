# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from functools import lru_cache


@lru_cache
def in_streamlit_notebook() -> bool:
    try:
        import streamlit  # noqa: F401
        from snowflake.snowpark.context import get_active_session

        get_active_session()
        return True
    except Exception:
        return False


@lru_cache
def in_jupyter_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if 'google.colab' in str(shell.__class__):
            return True
        if 'DatabricksShell' in str(shell.__class__):
            return True
        return shell.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


@lru_cache
def in_vnext_notebook() -> bool:
    try:
        from snowflake.snowpark.context import get_active_session

        get_active_session()
        return in_jupyter_notebook()
    except Exception:
        return False


@lru_cache
def in_notebook() -> bool:
    return in_streamlit_notebook() or in_jupyter_notebook()


@lru_cache
def in_tmux() -> bool:
    r"""Whether output should avoid emoji.

    Terminal multiplexers commonly mis-measure the display width of an emoji
    and corrupt the rest of the line, so the table and graph banners fall back
    to plain text. Each name below marks one such environment; add another only
    alongside a terminal that shows the same defect.
    """
    return 'TMUX' in os.environ or 'HERDR_ENV' in os.environ
