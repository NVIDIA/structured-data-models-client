# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from collections.abc import Sequence

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
from sdm_relational_engine.utils.environment import (
    in_jupyter_notebook,
    in_notebook,
    in_streamlit_notebook,
    in_vnext_notebook,
)

_console = Console(legacy_windows=True)


def message(msg: str) -> None:
    if in_streamlit_notebook():
        import streamlit as st

        st.markdown(msg)
    elif in_jupyter_notebook():
        from IPython.display import Markdown, display

        display(Markdown(msg))
    else:
        print(msg.replace('`', "'"))


def title(msg: str) -> None:
    if in_notebook():
        message(f'### {msg}')
    else:
        msg = msg.replace('`', "'")
        _console.print(f'[bold]{msg}[/bold]', highlight=False)


def italic(msg: str) -> None:
    if in_notebook():
        message(f'*{msg}*')
    else:
        msg = msg.replace('`', "'")
        _console.print(
            f'[italic]{msg}[/italic]',
            highlight=False,
            style='dim',
        )


def unordered_list(items: Sequence[str]) -> None:
    if in_notebook():
        msg = '\n'.join([f'- {item}' for item in items])
        message(msg)
    else:
        text = Text('\n').join(
            Text.assemble(
                Text(' • ', style='yellow'),
                Text(item.replace('`', '')),
            )
            for item in items
        )
        _console.print(text, highlight=False)


def dataframe(df: pd.DataFrame) -> None:
    if in_streamlit_notebook():
        import streamlit as st

        st.dataframe(df, hide_index=True)
    elif in_vnext_notebook():
        from IPython.display import display

        display(df.reset_index(drop=True))
    elif in_jupyter_notebook():
        from IPython.display import display

        if hasattr(df.style, 'hide'):
            display(df.style.hide(axis='index'))  # pandas=2
        else:
            display(df.style.hide_index())  # pandas<1.3
    else:
        _console.print(to_rich_table(df))


def to_rich_table(df: pd.DataFrame) -> Table:
    table = Table(box=box.ROUNDED)
    for column in df.columns:
        table.add_column(str(column))
    for _, row in df.iterrows():
        values: list[str | Text] = []
        for value in row:
            if str(value) == 'True':
                values.append('✅')
            elif str(value) in {'False', '-'}:
                values.append(Text('-', style='dim'))
            else:
                values.append(str(value))
        table.add_row(*values)
    return table
