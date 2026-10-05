# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pandas as pd
from predictive_query.schema.common import StrEnum
from predictive_query.schema.table import Column as ColumnDefinition
from predictive_query.schema.table import TableDefinition
from predictive_query.schema.typing import Dtype, Stype
from predictive_query.time_columns import (
    ColumnRange,
    TableRange,
    TimeColumnChoice,
    choose_time_column,
    taken_unchecked,
)
from sdm_connectors import quote_ident
from typing_extensions import Self

import sdm_relational_engine.utils.display as display
from sdm_relational_engine.rfm.base import (
    Column,
    ColumnSpec,
    ColumnSpecType,
    DataBackend,
    SourceColumn,
    SourceForeignKey,
    composite_key,
)
from sdm_relational_engine.rfm.base.utils import to_datetime
from sdm_relational_engine.rfm.metadata_inference import (
    infer_dtype,
    infer_primary_key,
    infer_stype,
)
from sdm_relational_engine.utils.environment import in_tmux

_DERIVED_KEY_PREFIX = composite_key.DERIVED_PREFIX


class TimeColumnWarning(UserWarning):
    r"""Warn that a time column was taken on the rows read alone.

    When the query over the whole table fails, a column the rows read hold
    one value in, or mostly no value, is kept, since the first rows of a
    table kept in time order often share one day. The category lets a
    caller that says this itself, such as ``sdm_client.Schema``, keep the
    engine from saying it again.
    """


class MissingType(StrEnum):
    r"""Sentinel for a table attribute the caller did not supply.

    Distinguishes "not given, so infer it" from an explicit ``None``, which
    means "this table has no such column".
    """

    VALUE = '???'


def _is_key_like(ser: pd.Series) -> bool:
    r"""Whether ``ser`` could serve as a primary key: fully populated and
    without a repeated value.
    """
    if len(ser) == 0:
        return False
    return bool(ser.notna().all()) and ser.nunique() == len(ser)


class Table(ABC):
    r"""A :class:`Table` fully specifies the relevant metadata of a single
    table, *i.e.* its selected columns, data types, semantic types, primary
    keys and time columns.

    Args:
        name: The name of this table.
        source_name: The source name of this table. If set to ``None``,
            ``name`` is being used.
        columns: The selected columns of this table.
        primary_key: The name of the primary key of this table, if it exists.
        time_column: The name of the time column of this table, if it exists.
        end_time_column: The name of the end time column of this table, if it
            exists.
    """

    _NUM_SAMPLE_ROWS = 1_000

    def __init__(
        self,
        name: str,
        source_name: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:

        self._name = name
        self._source_name = source_name or name
        self._column_dict: dict[str, Column] = {}
        self._primary_key: str | None = None
        self._primary_key_columns: tuple[str, ...] | None = None
        self._time_column: str | None = None
        self._end_time_column: str | None = None
        self._declined_primary_keys: tuple[str, ...] = ()
        self._declined_primary_key_rows: int | None = None
        self._time_column_choice: TimeColumnChoice | None = None
        # Whether the time column, or its absence, is still the one
        # inference chose: any assignment from outside it says otherwise.
        self._time_column_inferred = False
        self._expr_sample_df = pd.DataFrame(index=range(self._NUM_SAMPLE_ROWS))

        if columns is None:
            columns = list(self._source_column_dict.keys())

        self.add_columns(columns)

        if isinstance(primary_key, MissingType):
            # Infer primary key from source metadata, but only set it in case
            # it is already part of the column set (don't magically add it):
            if any(column.is_source for column in self.columns):
                primary_key = self._source_primary_key
                if (
                    primary_key is not None
                    and primary_key in self
                    and self[primary_key].is_source
                ):
                    self.primary_key = primary_key
        elif primary_key is not None:
            if primary_key not in self:
                self.add_column(primary_key)
            self.primary_key = primary_key

        if time_column is not None:
            if time_column not in self:
                self.add_column(time_column)
            self.time_column = time_column

        if end_time_column is not None:
            if end_time_column not in self:
                self.add_column(end_time_column)
            self.end_time_column = end_time_column

    @property
    def name(self) -> str:
        r"""The name of this table."""
        return self._name

    @property
    def source_name(self) -> str:
        r"""The source name of this table."""
        return self._source_name

    @property
    def source_foreign_keys(self) -> tuple[SourceForeignKey, ...]:
        r"""The foreign keys the source declares on this table.

        Each names its target table the way :attr:`source_name` names this
        one, not by the name a graph gives that table.
        """
        return tuple(self._source_foreign_key_dict.values())

    @property
    def _quoted_source_name(self) -> str:
        return quote_ident(self._source_name)

    def copy(self, name: str | None = None) -> Self:
        r"""A copy of this table, under ``name`` if given.

        The copy shares what was read from the source, such as its sample
        rows and the keys the source declares, but not what a later write
        sets: a column, key or time column set or removed on one leaves the
        other as it was.

        Args:
            name: The name of the copy. If set to ``None``, the copy keeps
                this table's name.
        """
        copied = copy.copy(self)
        copied._column_dict = {
            key: copy.copy(column) for key, column in self._column_dict.items()
        }
        if name is not None:
            copied._name = name
        return copied

    # Column ##################################################################

    def has_column(self, name: str) -> bool:
        r"""Returns ``True`` if this table holds a column with name ``name``;
        ``False`` otherwise.
        """
        return name in self._column_dict

    def column(self, name: str) -> Column:
        r"""Returns the data column named with name ``name`` in this table.

        Args:
            name: The name of the column.

        Raises:
            KeyError: If ``name`` is not present in this table.
        """
        if not self.has_column(name):
            raise KeyError(f"Column '{name}' not found in table '{self.name}'")
        return self._column_dict[name]

    @property
    def columns(self) -> list[Column]:
        r"""Returns a list of :class:`Column` objects that represent the
        columns in this table.
        """
        return list(self._column_dict.values())

    def add_columns(self, columns: Sequence[ColumnSpecType]) -> None:
        r"""Adds a set of columns to this table.

        Args:
            columns: The columns to add.

        Raises:
            KeyError: If any of the column names already exist in this table.
        """
        if len(columns) == 0:
            return

        column_specs = [ColumnSpec.coerce(column) for column in columns]

        # Obtain a batch-wise sample for all column expressions:
        expr_specs = [spec for spec in column_specs if not spec.is_source]
        if len(expr_specs) > 0:
            if any(
                spec.dtype is None or spec.stype is None for spec in expr_specs
            ):
                self._update_expr_sample_df(expr_specs)
            else:  # Remove out-dated columns:
                self._expr_sample_df = self._expr_sample_df.drop(
                    columns=[spec.name for spec in expr_specs],
                    errors='ignore',
                )

        for column_spec in column_specs:
            if column_spec.name in self:
                raise KeyError(
                    f"Column '{column_spec.name}' already exists "
                    f"in table '{self.name}'"
                )

            dtype = column_spec.dtype
            stype = column_spec.stype

            if column_spec.is_source:
                if column_spec.name not in self._source_column_dict:
                    raise ValueError(
                        f"Column '{column_spec.name}' does not exist in the "
                        f'underlying source table'
                    )

                if dtype is None:
                    dtype = self._source_column_dict[column_spec.name].dtype

                if dtype == Dtype.unsupported:
                    raise ValueError(
                        f'Encountered unsupported data type for column '
                        f"'{column_spec.name}' in table '{self.name}'. Please "
                        f"either change the column's data type or remove the "
                        f'column from this table.'
                    )

            if dtype is None:
                if column_spec.is_source:
                    ser = self._source_sample_df[column_spec.name]
                else:
                    ser = self._expr_sample_df[column_spec.name]
                try:
                    dtype = infer_dtype(ser)
                except Exception as e:
                    raise RuntimeError(
                        f"Encountered unsupported data type '{ser.dtype}' for "
                        f"column '{column_spec.name}' in table '{self.name}'. "
                        f'Please either cast the column to a supported type '
                        f'before building the table (for example '
                        f"df['{column_spec.name}'] = "
                        f"df['{column_spec.name}'].astype('int64')) or remove "
                        f'the column from this table.'
                    ) from e

            if stype is None:
                if column_spec.is_source:
                    ser = self._source_sample_df[column_spec.name]
                else:
                    ser = self._expr_sample_df[column_spec.name]
                try:
                    stype = infer_stype(ser, column_spec.name, dtype)
                except Exception as e:
                    raise RuntimeError(
                        f'Could not determine semantic type for column '
                        f"'{column_spec.name}' with data type '{dtype}' in "
                        f"table '{self.name}'. Please either change the "
                        f"column's data type or remove the column from this "
                        f'table.'
                    ) from e

            self._column_dict[column_spec.name] = Column(
                name=column_spec.name,
                expr=column_spec.expr,
                dtype=dtype,
                stype=stype,
            )

    def add_column(self, column: ColumnSpecType) -> Column:
        r"""Adds a column to this table.

        Args:
            column: The column to add.

        Raises:
            KeyError: If the column name already exists in this table.
        """
        column_spec = ColumnSpec.coerce(column)
        self.add_columns([column_spec])
        return self[column_spec.name]

    def remove_column(self, name: str) -> Self:
        r"""Removes a column from this table.

        A :class:`~sdm_relational_engine.rfm.Table` does not know which graph
        holds it. This clears the table's primary-key and time-column pointers
        but leaves any graph edge using ``name`` as its foreign key. Call
        :meth:`~sdm_relational_engine.rfm.Graph.unlink` before removing a
        linked foreign key. Otherwise,
        :meth:`~sdm_relational_engine.rfm.Graph.validate` rejects the graph.

        Args:
            name: The name of the column.

        Raises:
            KeyError: If ``name`` is not present in this table.
        """
        if name not in self:
            raise KeyError(f"Column '{name}' not found in table '{self.name}'")

        if self._primary_key == name:
            self.primary_key = None
        if self._time_column == name:
            self.time_column = None
        if self._end_time_column == name:
            self.end_time_column = None
        del self._column_dict[name]

        return self

    # Primary key #############################################################

    def has_primary_key(self) -> bool:
        r"""Returns ``True``` if this table has a primary key; ``False``
        otherwise.
        """
        return self._primary_key is not None

    @property
    def primary_key(self) -> Column | None:
        r"""The primary key column of this table.

        The getter returns the primary key column of this table, or ``None`` if
        no such primary key is present.

        The setter marks a column as this table's primary key, accepting
        either one name or a sequence of them for an identity spread
        across several columns. A single name the data frame does not
        carry raises :class:`KeyError`; a missing component of a
        composite one raises :class:`ValueError`, as does naming a column
        already serving as a time or end time column.
        """
        if self._primary_key is None:
            return None
        return self[self._primary_key]

    @property
    def primary_key_columns(self) -> tuple[str, ...]:
        r"""The columns this table's identity is spread across.

        A single-column key reports one name. A composite key reports the
        columns the caller declared, not the derived column that carries their
        folded value.
        """
        if self._primary_key_columns is not None:
            return self._primary_key_columns
        if self._primary_key is None:
            return ()
        return (self._primary_key,)

    @property
    def has_composite_primary_key(self) -> bool:
        r"""Whether this table's identity is spread across several columns."""
        return self._primary_key_columns is not None

    def _set_composite_primary_key(self, names: Sequence[str]) -> None:
        r"""Folds *names* into one derived identity column and keys on it.

        Links, the sampler, the payload, and the serving contract's
        ``instance_id`` each identify a row by one value. A derived column
        therefore carries a multi-column identity.
        """
        names = tuple(names)
        missing = [name for name in names if name not in self]
        if missing:
            raise ValueError(
                f'Cannot use {list(names)} as a composite primary key of '
                f"table '{self.name}': column(s) {missing} are not present"
            )
        for name in names:
            if name in (self._time_column, self._end_time_column):
                raise ValueError(
                    f"Cannot use column '{name}' in a composite primary key "
                    f'since it is already defined to be a time column'
                )
        composite_key.refuse_unfoldable_dtypes(
            [(name, self[name].dtype) for name in names]
        )

        derived = self._derived_key_column_name(names)
        self._materialize_derived_key(derived, names)
        self.primary_key = derived
        self._primary_key_columns = names

    def _derived_key_column_name(self, names: Sequence[str]) -> str:
        r"""Names the column carrying the folded identity of *names*.

        The same columns always give the same name, so declaring a key twice,
        or linking twice, reuses one column rather than accumulating a new one
        each time and hiding the duplicate from the checks that reject it.
        Different columns always give a different name, so two references from
        one table cannot land on each other's values.
        """
        derived = _DERIVED_KEY_PREFIX + composite_key.encode_identity(names)
        if derived in self and not self._is_derived_key_column(derived):
            raise ValueError(
                f'Cannot fold {list(names)} into an identity for table '
                f"'{self.name}': it already holds a column named "
                f"'{derived}'. Rename that column."
            )
        return derived

    def _is_derived_key_column(self, name: str) -> bool:
        return name.startswith(_DERIVED_KEY_PREFIX)

    def _drop_derived_key(self, derived: str | None) -> None:
        r"""Forgets a derived identity column this table no longer keys on."""
        if derived is None or not derived.startswith(_DERIVED_KEY_PREFIX):
            return
        if derived in self:
            self._column_dict.pop(derived, None)
        self._forget_derived_key(derived)

    def _forget_derived_key(self, derived: str) -> None:  # noqa: B027
        r"""Drop any backend state cached for a derived composite key.

        An optional hook rather than an abstract method: a backend that
        materialises the derived column has something to forget, and one that
        computes it per query does not. Left concrete and empty so the latter
        does not have to declare an override that does nothing.
        """

    _SQL_TEXT_TYPE: str | None = None
    _SQL_CHR_FUNCTION = 'CHR'

    def _quote_key_column(self, name: str) -> str:
        return quote_ident(name)

    def _materialize_derived_key(
        self,
        derived: str,
        names: Sequence[str],
    ) -> None:
        if self._SQL_TEXT_TYPE is None:
            raise ValueError(
                f'Composite primary keys are not supported on '
                f"'{self.__class__.__name__}' yet. The identity of table "
                f"'{self.name}' is spread across {list(names)}; add a column "
                f'holding that identity to the source and declare it '
                f'instead.'
            )

        expression = composite_key.sql_expression(
            names,
            quote=self._quote_key_column,
            text_type=self._SQL_TEXT_TYPE,
            chr_function=self._SQL_CHR_FUNCTION,
        )
        self.add_column(ColumnSpec(name=derived, expr=expression))
        self[derived].stype = Stype.ID
        self._assert_warehouse_folds_like_pandas(derived, names)

    def _sample_values(self, name: str) -> pd.Series | None:
        r"""Return a sample of source or derived column *name*."""
        source = self._source_sample_df
        if name in source.columns:
            return source[name]
        if name in self._expr_sample_df.columns:
            return self._expr_sample_df[name]
        return None

    def _assert_warehouse_folds_like_pandas(
        self,
        derived: str,
        names: Sequence[str],
    ) -> None:
        r"""Checks the warehouse renders an identity exactly as pandas does.

        Seeds are folded here while the warehouse folds the rows it joins
        against. A dialect that renders a number, date, or decimal's trailing
        zeros differently yields no matched rows instead of an error. Comparing
        a sample turns that into a refusal when the key is declared.
        """
        folded = self._sample_values(derived)
        source = self._source_sample_df
        if folded is None or any(name not in source.columns for name in names):
            return
        warehouse = set(folded.dropna())
        locally = set(composite_key.encode_frame(source, list(names)))
        if not warehouse or warehouse == locally:
            return
        differing = sorted(warehouse - locally)[:1]
        raise ValueError(
            f"'{self.name}' folds {list(names)} into an identity differently "
            f'in the warehouse than this client does, so a prediction seeded '
            f'here would match no row: the warehouse produced '
            f'{differing!r}. Add a column holding the identity you intend '
            f'and declare that instead.'
        )

    @primary_key.setter
    def primary_key(self, name: str | Sequence[str] | None) -> None:
        if name is not None and not isinstance(name, str):
            names = tuple(name)
            if len(names) == 0:
                raise ValueError(
                    f'Cannot use an empty composite primary key on table '
                    f"'{self.name}'"
                )
            if len(names) > 1:
                self._set_composite_primary_key(names)
                return
            name = names[0]
        if name is not None and name == self._time_column:
            raise ValueError(
                f"Cannot specify column '{name}' as a primary "
                f'key since it is already defined to be a time '
                f'column'
            )
        if name is not None and name == self._end_time_column:
            raise ValueError(
                f"Cannot specify column '{name}' as a primary "
                f'key since it is already defined to be an end '
                f'time column'
            )
        if name is not None and name not in self:
            raise KeyError(f"Column '{name}' not found in table '{self.name}'")

        stale_derived: str | None = None
        if self._primary_key_columns is not None and name != self._primary_key:
            stale_derived = self._primary_key
            self._primary_key = None
        self._primary_key_columns = None

        if self.primary_key is not None:
            self.primary_key._is_primary_key = False

        if name is None:
            self._primary_key = None
            self._drop_derived_key(stale_derived)
            return

        self[name].stype = Stype.ID
        self[name]._is_primary_key = True
        self._primary_key = name
        self._drop_derived_key(stale_derived)

    # Time column #############################################################

    def has_time_column(self) -> bool:
        r"""Returns ``True`` if this table has a time column; ``False``
        otherwise.
        """
        return self._time_column is not None

    @property
    def time_column(self) -> Column | None:
        r"""The time column of this table.

        The getter returns the time column of this table, or ``None`` if no
        such time column is present.

        The setter marks a column as this table's time column. A name the
        data frame does not carry raises :class:`KeyError`, and a dtype
        that cannot mean a timestamp, such as an integer, raises
        :class:`ValueError`. A column that can carry a timestamp, such as text
        waiting to be parsed, is accepted and its semantic type is rewritten.
        """
        if self._time_column is None:
            return None
        return self[self._time_column]

    @time_column.setter
    def time_column(self, name: str | None) -> None:
        # Every check runs before anything changes, so a refused name leaves
        # the table as it was, an inferred column still marked inferred.
        if name is not None and name == self._primary_key:
            raise ValueError(
                f"Cannot specify column '{name}' as a time "
                f'column since it is already defined to be a '
                f'primary key'
            )
        if name is not None and name == self._end_time_column:
            raise ValueError(
                f"Cannot specify column '{name}' as a time "
                f'column since it is already defined to be an '
                f'end time column'
            )
        if name is not None:
            self[name].stype = Stype.timestamp

        self._time_column_inferred = False
        if self.time_column is not None:
            self.time_column._is_time_column = False

        if name is None:
            self._time_column = None
            return

        self[name]._is_time_column = True
        self._time_column = name

    # End Time column #########################################################

    def has_end_time_column(self) -> bool:
        r"""Returns ``True`` if this table has an end time column; ``False``
        otherwise.
        """
        return self._end_time_column is not None

    @property
    def end_time_column(self) -> Column | None:
        r"""The end time column of this table.

        The getter returns the end time column of this table, or ``None`` if no
        such end time column is present.

        The setter marks a column as this table's end time column. A name
        the data frame does not carry raises :class:`KeyError`, and a
        column whose type is not timestamp-compatible raises
        :class:`ValueError`.
        """
        if self._end_time_column is None:
            return None
        return self[self._end_time_column]

    @end_time_column.setter
    def end_time_column(self, name: str | None) -> None:
        if name is not None and name == self._primary_key:
            raise ValueError(
                f"Cannot specify column '{name}' as an end time "
                f'column since it is already defined to be a '
                f'primary key'
            )
        if name is not None and name == self._time_column:
            raise ValueError(
                f"Cannot specify column '{name}' as an end time "
                f'column since it is already defined to be a '
                f'time column'
            )

        if self.end_time_column is not None:
            self.end_time_column._is_end_time_column = False

        if name is None:
            self._end_time_column = None
            return

        self[name].stype = Stype.timestamp
        self[name]._is_end_time_column = True
        self._end_time_column = name

    # Metadata ################################################################

    @property
    def metadata(self) -> pd.DataFrame:
        r"""Returns a :class:`pandas.DataFrame` object containing metadata
        information about the columns in this table.

        The returned dataframe has columns ``"Name"``, ``"Data Type"``,
        ``"Semantic Type"``, ``"Primary Key"``, ``"Time Column"`` and
        ``"End Time Column"``, which provide an aggregated view of the
        properties of the columns of this table.

        Example:
            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm
            >>> table = rfm.LocalTable(df=..., name=...).infer_metadata()
            >>> table.metadata.iloc[0].to_dict()
            {'Name': 'CustomerID', 'Data Type': 'float64',
             'Semantic Type': 'ID', 'Primary Key': True,
             'Time Column': False, 'End Time Column': False}
        """
        cols = self.columns

        return pd.DataFrame(
            {
                'Name': pd.Series(dtype=str, data=[c.name for c in cols]),
                'Data Type': pd.Series(dtype=str, data=[c.dtype for c in cols]),
                'Semantic Type': pd.Series(
                    dtype=str, data=[c.stype for c in cols]
                ),
                'Primary Key': pd.Series(
                    dtype=bool,
                    data=[self._primary_key == c.name for c in cols],
                ),
                'Time Column': pd.Series(
                    dtype=bool,
                    data=[self._time_column == c.name for c in cols],
                ),
                'End Time Column': pd.Series(
                    dtype=bool,
                    data=[self._end_time_column == c.name for c in cols],
                ),
            }
        )

    def print_metadata(self) -> None:
        r"""Prints the :meth:`~metadata` of this table."""
        msg = f'Metadata of Table `{self.name}`'
        if not in_tmux():
            msg = f'🏷️ {msg}'
        if num := self._num_rows:
            msg += ' (1 row)' if num == 1 else f' ({num:,} rows)'

        display.title(msg)
        display.dataframe(self.metadata)

    def infer_primary_key(self, verbose: bool = True) -> Self:
        r"""Infers the primary key in this table.

        Args:
            verbose: Whether to print verbose output.
        """
        if self.has_primary_key():
            return self

        def _set_primary_key(primary_key: str) -> None:
            self.primary_key = primary_key
            if verbose:
                display.message(
                    f'Inferred primary key `{primary_key}` for '
                    f'table `{self.name}`'
                )

        # Inference from source column metadata:
        if any(column.is_source for column in self.columns):
            primary_key = self._source_primary_key
            if (
                primary_key is not None
                and primary_key in self
                and self[primary_key].is_source
            ):
                _set_primary_key(primary_key)
                return self

            unique_keys = [
                column.name
                for column in self._source_column_dict.values()
                if column.is_unique_key
            ]
            if (
                len(unique_keys) == 1  # NOTE No composite keys yet.
                and unique_keys[0] in self
                and self[unique_keys[0]].is_source
            ):
                _set_primary_key(unique_keys[0])
                return self

        # Heuristic-based inference:
        candidates = [
            column.name for column in self.columns if column.stype == Stype.ID
        ]
        if len(candidates) == 0:
            for column in self.columns:
                if self.name.lower() == column.name.lower() or (
                    self.name.lower().endswith('s')
                    and self.name.lower()[:-1] == column.name.lower()
                ):
                    candidates.append(column.name)

        df = self._get_sample_df()
        if primary_key := infer_primary_key(
            table_name=self.name,
            df=df,
            candidates=candidates,
        ):
            _set_primary_key(primary_key)
            return self

        # Nothing scored high enough. Record the candidates that would have
        # served as a key so `Graph.infer_links` can report the ones link
        # inference does not go on to explain. At this point, a viable key and
        # a coincidentally unique foreign key look identical, and only the
        # edges tell them apart.
        #
        # Record how far that evidence reaches too. `df` is capped at
        # `_NUM_SAMPLE_ROWS` on every SQL backend, so a foreign key that merely
        # happens to be distinct in the first rows read looks exactly like a
        # key candidate, and the diagnostic must not state as fact something
        # only the sample supports.
        self._declined_primary_keys = tuple(
            name for name in candidates if _is_key_like(df[name])
        )
        self._declined_primary_key_rows = (
            len(df) if len(df) == self._NUM_SAMPLE_ROWS else None
        )

        return self

    def infer_time_column(self, verbose: bool = True) -> Self:
        r"""Infers the time column in this table.

        Args:
            verbose: Whether to print verbose output.
        """
        if self.has_time_column():
            return self

        # Heuristic-based inference:
        candidates = [
            column.name
            for column in self.columns
            if column.stype == Stype.timestamp
            and column.name != self._end_time_column
        ]

        if time_column := self._infer_time_column_among(candidates):
            if verbose:
                display.message(
                    f'Inferred time column `{time_column}` for '
                    f'table `{self.name}`'
                )

        return self

    def _infer_time_column_among(self, candidates: list[str]) -> str | None:
        r"""Set the time column inferred among ``candidates``, keeping how it
        was chosen and what was refused, and return it.
        """
        df = self._get_sample_df()
        # Only a capped read can hide a second value past its last row.
        truncated = len(df) == self._NUM_SAMPLE_ROWS
        self._time_column_choice = choose_time_column(
            df,
            candidates,
            full_range=self._full_range if truncated else None,
        )
        if time_column := self._time_column_choice.column:
            self.time_column = time_column
        self._time_column_inferred = True
        for doubt in self._time_column_choice.unconfirmed:
            if doubt.column == time_column:
                warnings.warn(
                    taken_unchecked(
                        self.name,
                        doubt,
                        f'graph[{self.name!r}].time_column = None',
                    ),
                    TimeColumnWarning,
                    stacklevel=2,
                )
        return time_column

    @property
    def time_column_choice(self) -> TimeColumnChoice | None:
        r"""How :meth:`infer_time_column` chose the time column and which
        timestamps it refused, so a caller can say why. ``None`` until it
        has run, or when the time column was given.
        """
        return self._time_column_choice

    def _settle_time_column(self, linked: bool) -> None:
        r"""Take or refuse a lone edit stamp by whether another table links
        to this one, once the graph's edges say. A time column set by hand
        since inference, to a column or to ``None``, is left alone, and so
        is a stamp since removed or made the end time column or key.
        """
        choice = self._time_column_choice
        if choice is None or not self._time_column_inferred:
            return
        settled = choice.for_links(linked)
        if settled.column is not None and (
            settled.column not in self
            or settled.column in (self._primary_key, self._end_time_column)
        ):
            return
        if settled != choice:
            self._time_column_choice = settled
            self.time_column = settled.column
            self._time_column_inferred = True

    def _full_range(self, names: Sequence[str]) -> TableRange | None:
        r"""The row count, and ``MIN``, ``MAX`` and ``COUNT`` of each column
        of ``names``, over every row in one query, or ``None`` when this
        table cannot ask its source.
        """
        selects = ['COUNT(*)']
        for name in names:
            held = self[name]
            if held.is_source:
                column = self._quote_key_column(name)
            else:
                # A metric view's dimension is an expression, read as the
                # sample reads it.
                column = f'({held.expr})'
            selects.append(f'MIN({column}), MAX({column}), COUNT({column})')
        try:
            rows = self._fetch_rows(
                f'SELECT {", ".join(selects)}\nFROM {self._quoted_source_name}'
            )
        except Exception:
            # A timeout on a large view, most often. The rows read then say
            # what they can, and a column they would refuse is kept and
            # named as unconfirmed.
            return None
        count, *ranges = rows[0]
        return TableRange(
            int(count),
            {
                name: ColumnRange(
                    ranges[3 * i], ranges[3 * i + 1], int(ranges[3 * i + 2])
                )
                for i, name in enumerate(names)
            },
        )

    def infer_metadata(self, verbose: bool = True) -> Self:
        r"""Infers metadata, *i.e.*, primary keys and time columns, in this
        table.

        Args:
            verbose: Whether to print verbose output.
        """
        logs = []

        if not self.has_primary_key():
            self.infer_primary_key(verbose=False)
            if self.has_primary_key():
                logs.append(f'primary key `{self._primary_key}`')

        if not self.has_time_column():
            self.infer_time_column(verbose=False)
            if self.has_time_column():
                logs.append(f'time column `{self._time_column}`')

        if verbose and len(logs) > 0:
            display.message(
                f'Inferred {" and ".join(logs)} for table `{self.name}`'
            )

        return self

    # Helpers #################################################################

    def _to_api_table_definition(self) -> TableDefinition:
        return TableDefinition(
            cols=[
                ColumnDefinition(col.name, col.stype, col.dtype)
                for col in self.columns
            ],
            pkey=self._primary_key,
            time_col=self._time_column,
            end_time_col=self._end_time_column,
        )

    @cached_property
    def _source_column_dict(self) -> dict[str, SourceColumn]:
        source_columns = self._get_source_columns()
        if len(source_columns) == 0:
            raise ValueError(f"Table '{self.name}' has no columns")
        return {column.name: column for column in source_columns}

    @cached_property
    def _source_primary_key(self) -> str | None:
        primary_keys = [
            column.name
            for column in self._source_column_dict.values()
            if column.is_primary_key
        ]
        # NOTE No composite keys yet.
        return primary_keys[0] if len(primary_keys) == 1 else None

    @cached_property
    def _source_foreign_key_dict(self) -> dict[str, SourceForeignKey]:
        return {key.name: key for key in self._get_source_foreign_keys()}

    @cached_property
    def _source_sample_df(self) -> pd.DataFrame:
        return self._get_source_sample_df().reset_index(drop=True)

    def _update_expr_sample_df(
        self,
        columns: Sequence[ColumnSpec | Column],
    ) -> None:
        if len(columns) == 0:
            return

        dfs = [
            self._expr_sample_df,
            self._get_expr_sample_df(columns).reset_index(drop=True),
        ]
        size = min(map(len, dfs))
        df = pd.concat([dfs[0].iloc[:size], dfs[1].iloc[:size]], axis=1)
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        self._expr_sample_df = df

    @cached_property
    def _num_rows(self) -> int | None:
        return self._get_num_rows()

    def _local_row_count(self) -> int | None:
        r"""Row count, only when it costs nothing to ask.

        Used to notice that a table's rows changed between predictions. A
        backend that would have to query for this returns ``None`` instead:
        the caller runs this on every prediction, and a ``COUNT(*)`` per
        prediction would cost more than the work it saves. Deliberately not
        :attr:`_num_rows`, which is cached and so cannot report a change.
        """
        return None

    def _get_sample_df(self) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []

        if any(column.is_source for column in self.columns):
            dfs.append(self._source_sample_df)

        if any(not column.is_source for column in self.columns):
            self._update_expr_sample_df(
                [
                    column
                    for column in self.columns
                    if not column.is_source
                    and column.name not in self._expr_sample_df
                ]
            )
            dfs.append(self._expr_sample_df)

        if len(dfs) == 0:
            return pd.DataFrame(index=range(self._NUM_SAMPLE_ROWS))
        if len(dfs) == 1:
            return dfs[0]

        size = min(map(len, dfs))
        df = pd.concat([dfs[0].iloc[:size], dfs[1].iloc[:size]], axis=1)
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        return df

    @staticmethod
    def _sanitize(
        df: pd.DataFrame,
        dtype_dict: Mapping[str, Dtype | None] | None = None,
        stype_dict: Mapping[str, Stype | None] | None = None,
    ) -> pd.DataFrame:
        r"""Sanitzes a :class:`pandas.DataFrame` in-place such that its data
        types match table data and semantic type specification.
        """

        def _to_list(ser: pd.Series, dtype: Dtype | None) -> pd.Series:
            if pd.api.types.is_string_dtype(ser) and dtype in {
                Dtype.intlist,
                Dtype.floatlist,
            }:
                try:
                    ser = ser.map(
                        lambda row: (
                            np.fromstring(
                                row.strip('[]'),
                                sep=',',
                                dtype=int
                                if dtype == Dtype.intlist
                                else np.float32,
                            )
                            if row is not None
                            else None
                        )
                    )
                except Exception:
                    pass

            if pd.api.types.is_string_dtype(ser):
                if find_spec('orjson') is not None:
                    json = import_module('orjson')
                else:
                    json = import_module('json')
                try:
                    ser = ser.map(
                        lambda row: json.loads(row) if row is not None else None
                    )
                except Exception:
                    pass

            return ser

        for column_name in df.columns:
            dtype = (dtype_dict or {}).get(column_name)
            stype = (stype_dict or {}).get(column_name)

            if dtype == Dtype.time or stype == Stype.timestamp:
                df[column_name] = to_datetime(df[column_name], column_name)
            elif dtype is not None and dtype.is_list():
                df[column_name] = _to_list(df[column_name], dtype)
            elif stype == Stype.sequence:
                df[column_name] = _to_list(df[column_name], Dtype.floatlist)

        return df

    # Python builtins #########################################################

    def __hash__(self) -> int:
        metadata = [
            self._primary_key,
            self._time_column,
            self._end_time_column,
        ]
        return hash(tuple(self.columns + metadata))

    def __contains__(self, name: str) -> bool:
        return self.has_column(name)

    def __getitem__(self, name: str) -> Column:
        return self.column(name)

    def __delitem__(self, name: str) -> None:
        self.remove_column(name)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'  name={self.name},\n'
            f'  num_columns={len(self.columns)},\n'
            f'  primary_key={self._primary_key},\n'
            f'  time_column={self._time_column},\n'
            f'  end_time_column={self._end_time_column},\n'
            f')'
        )

    # Key counts ##############################################################

    #: The column sets the source refuses to repeat, set while reading the
    #: columns by a backend that enforces its primary and unique keys.
    _unique_constraints: tuple[frozenset[str], ...] = ()

    def _declares_key(self, names: Sequence[str]) -> bool:
        r"""Whether the source's catalog names ``names`` as a key of this
        table, as a primary key or a unique constraint.
        """
        if len(names) != 1:
            return False
        column = self._source_column_dict.get(names[0])
        return column is not None and (
            column.is_primary_key or column.is_unique_key
        )

    @classmethod  # noqa: B027
    def _before_counting_keys(
        cls, keys: Sequence[tuple['Table', Sequence[str]]]
    ) -> None:
        r"""Read once, for all of ``keys``, what deciding which of them the
        source enforces needs. Most backends read it with the columns.
        """

    def _enforces_key(self, names: Sequence[str]) -> tuple[bool, bool]:
        r"""Whether the source refuses a second row with the same ``names``,
        and whether it refuses a row missing any of them.

        Unique and never null together leave nothing to count.
        """
        columns = self._source_column_dict
        if any(name not in columns for name in names):
            return False, False
        unique = any(
            constraint <= set(names) for constraint in self._unique_constraints
        )
        never_null = all(not columns[name].is_nullable for name in names)
        return unique, never_null

    def _key_count_sql(
        self, label: int, names: Sequence[str], *, unique: bool
    ) -> str:
        r"""One ``SELECT`` of a ``UNION ALL``: this table's rows, the distinct
        values of its key, and the rows missing part of the key.
        """
        quoted = [self._quote_key_column(name) for name in names]
        missing = ' OR '.join(f'{column} IS NULL' for column in quoted)
        nulls = f'COUNT(CASE WHEN {missing} THEN 1 END)'
        # A key the source keeps unique has one distinct value per row that
        # holds one, so only the nulls cost anything to count.
        distinct = (
            f'COUNT(*) - {nulls}' if unique else self._count_distinct(quoted)
        )
        return (
            f'SELECT {label} AS k, COUNT(*) AS n, {distinct} AS d, '
            f'{nulls} AS z\nFROM {self._quoted_source_name}'
        )

    def _count_distinct(self, quoted: Sequence[str]) -> str:
        r"""The distinct values of a key, leaving out a row missing any part.

        Spark and Snowflake count several columns and skip such a row
        themselves. A dialect that does neither overrides this.
        """
        return f'COUNT(DISTINCT {", ".join(quoted)})'

    def _repeated_key_sql(self, names: Sequence[str]) -> str:
        r"""One value of the key that more than one row holds."""
        quoted = ', '.join(self._quote_key_column(name) for name in names)
        present = ' AND '.join(
            f'{self._quote_key_column(name)} IS NOT NULL' for name in names
        )
        return (
            f'SELECT {quoted}\nFROM {self._quoted_source_name}\n'
            f'WHERE {present}\nGROUP BY {quoted}\nHAVING COUNT(*) > 1\nLIMIT 1'
        )

    def _fetch_rows(self, sql: str) -> list[tuple]:
        r"""Run one statement against the source and return its rows."""
        raise NotImplementedError(
            f"'{self.__class__.__name__}' cannot run SQL against its source"
        )

    # Abstract Methods ########################################################

    @property
    @abstractmethod
    def backend(self) -> DataBackend:
        r"""The data backend of this table."""

    @abstractmethod
    def _get_source_columns(self) -> list[SourceColumn]:
        pass

    @abstractmethod
    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        pass

    @abstractmethod
    def _get_source_sample_df(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _get_expr_sample_df(
        self,
        columns: Sequence[ColumnSpec | Column],
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def _get_num_rows(self) -> int | None:
        pass
