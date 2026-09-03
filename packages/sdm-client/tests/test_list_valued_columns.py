# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A column of lists is refused, in the same words, wherever it enters.

SDM's own `Task.classification` covers "binary, multi-class, or multi-label",
so it is reasonable to try a multi-label target. The wire does not: `CellValue`
is documented as excluding array-valued prediction scalars *on purpose*, so a
set of labels has no shape to come back in.

Before this, the same input produced two different wrong answers. Building a
schema raised `TypeError: unhashable type: 'list'` from inside pandas, because
`_stype_of` counts distinct values. And a list that reached the wire without a
schema -- the rows being scored are not described by one -- was written by
`serialize_cell`'s `str(value)` fallback as the repr of a Python list, so the
model was asked about the text `"['a', 'b']"`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_client import Request, Schema
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.errors import StructuredDataError

TAGS = [['a', 'b'], ['b'], ['a'], ['a', 'b'], ['b'], ['a']]


def context(**columns) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            'row_id': range(6),
            'v': [float(index) for index in range(6)],
            'y': [index % 2 for index in range(6)],
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def scored(**columns) -> pd.DataFrame:
    frame = pd.DataFrame({'row_id': [900], 'v': [1.0]})
    for name, values in columns.items():
        frame[name] = values
    return frame


# --- the target ------------------------------------------------------------


def test_a_multi_label_target_is_refused_as_one():
    """Named for what it is, not as a typing problem."""
    with pytest.raises(StructuredDataError) as error:
        Request(
            context=context(tags=TAGS),
            predict=scored(),
            target='tags',
            task='classification',
        )
    message = str(error.value)
    assert error.value.code == 'UNSUPPORTED_REQUEST'
    assert 'multi-label' in message
    assert 'tags' in message


# --- a feature, which is a different sentence ------------------------------


def test_a_list_feature_in_the_context_is_refused_by_the_schema():
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data({'t': context(tags=TAGS)})
    message = str(error.value)
    assert 't.tags' in message
    assert 'list' in message


def test_a_list_present_only_in_the_rows_to_score_is_refused():
    """No schema describes those rows, so the wire is the only guard."""
    request = Request(
        context=context(tags=['a', 'b', 'a', 'b', 'a', 'b']),
        predict=scored(tags=[['a', 'b']]),
        target='y',
        task='classification',
    )
    with pytest.raises(StructuredDataError) as error:
        list(TabularAdapter().prediction_plan(request, ('prediction',)))
    assert 'tags' in str(error.value)


def test_a_list_is_never_written_as_the_repr_of_one():
    """The failure this replaces: `"['a', 'b']"` as an ordinary category."""
    from sdm_client.wire import serialize_column

    with pytest.raises(StructuredDataError):
        serialize_column(pd.Series(TAGS, name='tags'), 'string')


# --- every shape of sequence, not just `list` ------------------------------


@pytest.mark.parametrize(
    'value',
    [
        ['a', 'b'],
        ('a', 'b'),
        {'a', 'b'},
        frozenset({'a'}),
        {'a': 1},
        np.array([1, 2]),
        range(2),
        b'ab',
        memoryview(b'ab'),
        pd.Series([1, 2]),
        pd.DataFrame({'a': [1]}),
    ],
    ids=[
        'list',
        'tuple',
        'set',
        'frozenset',
        'dict',
        'ndarray',
        'range',
        'bytes',
        'memoryview',
        'Series',
        'DataFrame',
    ],
)
def test_any_container_cell_is_refused(value):
    """Every one of these was written as its own repr by the `str()` fallback.

    Asked as a property rather than a list of types, because the list is how a
    tuple, then a mapping, then a `range` each turned out to be missing.
    """
    from sdm_client.wire.dtypes import require_scalar_cells

    with pytest.raises(StructuredDataError):
        require_scalar_cells(pd.Series([value], dtype=object, name='x'), 'x')


@pytest.mark.parametrize(
    'value',
    [
        'abc',
        1,
        1.5,
        True,
        None,
        np.int64(3),
        __import__('decimal').Decimal('1.5'),
        __import__('uuid').uuid4(),
        __import__('datetime').date(2024, 1, 1),
        __import__('datetime').datetime(2024, 1, 1),
    ],
    ids=[
        'str',
        'int',
        'float',
        'bool',
        'None',
        'np.int64',
        'Decimal',
        'UUID',
        'date',
        'datetime',
    ],
)
def test_everything_a_scalar_column_carries_still_passes(value):
    """The guard must not start refusing what a text column legitimately holds."""
    from sdm_client.wire.dtypes import require_scalar_cells

    require_scalar_cells(pd.Series([value], dtype=object, name='x'), 'x')


def test_an_object_with_a_custom_iter_is_refused():
    class Custom:
        def __iter__(self):
            return iter([1, 2])

    from sdm_client.wire.dtypes import require_scalar_cells

    with pytest.raises(StructuredDataError):
        require_scalar_cells(pd.Series([Custom()], dtype=object, name='x'), 'x')


def test_a_string_is_not_a_sequence_for_this_purpose():
    """A string is iterable, and is exactly what a categorical column holds."""
    schema = Schema.from_data(
        {'t': context(tag=['a', 'b', 'a', 'b', 'a', 'b'])}
    )
    assert schema['t']['tag'].stype == 'categorical'


# --- nothing ordinary changed ----------------------------------------------


def test_an_ordinary_frame_still_builds():
    schema = Schema.from_data({'t': context()})
    assert sorted(schema['t']) == ['row_id', 'v', 'y']


def test_an_ordinary_request_still_encodes():
    request = Request(
        context=context(), predict=scored(), target='y', task='classification'
    )
    plan = next(
        iter(TabularAdapter().prediction_plan(request, ('prediction',)))
    )
    assert plan['predict']['instance_table']['rows'] == [[900, 1.0]]


def test_a_null_beside_a_list_is_still_a_null():
    frame = context()
    frame['maybe'] = [None] * 6
    assert Schema.from_data({'t': frame})['t']['maybe'].dtype == 'string'


# --- the dtype gate must not become a way past the check --------------------


def test_an_arrow_backed_list_column_is_refused_too():
    """`pd.read_parquet(dtype_backend='pyarrow')` gives this for a list column.

    The guard skips a column whose dtype cannot hold a sequence. Asking
    "is it object?" instead would let this one through, because an Arrow list
    dtype is not object -- and the cells would go out as `"['a', 'b']"` again.
    """
    pa = pytest.importorskip('pyarrow')
    series = pd.Series(
        TAGS[:2], dtype=pd.ArrowDtype(pa.list_(pa.string())), name='tags'
    )
    assert not pd.api.types.is_object_dtype(series.dtype)
    from sdm_client.wire import serialize_column

    with pytest.raises(StructuredDataError):
        serialize_column(series, 'string')


@pytest.mark.parametrize(
    'dtype', ['int64', 'float64', 'bool', 'datetime64[ns]']
)
def test_a_dtype_that_cannot_hold_a_sequence_is_not_walked(dtype):
    """The check is on the hot path, so the common columns must cost nothing."""
    from sdm_client.wire.dtypes import _holds_only_scalars

    assert _holds_only_scalars(pd.Series([], dtype=dtype).dtype)


def test_an_arrow_scalar_column_is_not_walked():
    """A Parquet-heavy frame is Arrow-backed throughout, list column or not."""
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    assert _holds_only_scalars(pd.ArrowDtype(pa.int64()))
    assert _holds_only_scalars(pd.ArrowDtype(pa.string()))
    assert _holds_only_scalars(pd.ArrowDtype(pa.timestamp('us')))


def test_a_dictionary_of_scalars_is_still_scalar():
    """Unwrapping must not make every encoded column pay for the walk."""
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    assert _holds_only_scalars(
        pd.ArrowDtype(pa.dictionary(pa.int8(), pa.string()))
    )


@pytest.mark.parametrize(
    'arrow',
    [
        'list',
        'list_view',
        'large_list',
        'large_list_view',
        'fixed_size_list',
        'struct',
        'map',
        'dictionary_of_list',
        'run_end_encoded_list',
    ],
    ids=str,
)
def test_every_arrow_type_that_can_hold_a_sequence_is_walked(arrow):
    """Enumerating the nested types let a list *view* through once already.

    `pa.types.is_nested` answers for the containers. The encodings that wrap
    one -- a dictionary or a run-end encoding of a list -- are still columns of
    lists, so they are unwrapped before the question is asked.
    """
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    nested = {
        'list': pa.list_(pa.string()),
        'list_view': pa.list_view(pa.string()),
        'large_list': pa.large_list(pa.string()),
        'large_list_view': pa.large_list_view(pa.string()),
        'fixed_size_list': pa.list_(pa.string(), 2),
        'struct': pa.struct([('a', pa.int64())]),
        'map': pa.map_(pa.string(), pa.int64()),
        'dictionary_of_list': pa.dictionary(pa.int8(), pa.list_(pa.string())),
        'run_end_encoded_list': pa.run_end_encoded(
            pa.int16(), pa.list_(pa.string())
        ),
    }[arrow]
    assert not _holds_only_scalars(pd.ArrowDtype(nested))


def test_a_categorical_of_sequences_is_refused():
    """`Categorical.from_codes` builds one the ordinary constructor refuses."""
    from sdm_client.wire.dtypes import require_scalar_cells

    # Built by hand: `pd.Index([('a', 'b'), ...])` becomes a MultiIndex.
    values = np.empty(2, dtype=object)
    values[0], values[1] = ('a', 'b'), ('b',)
    series = pd.Series(
        pd.Categorical.from_codes([0, 1, 0], categories=pd.Index(values))
    )
    with pytest.raises(StructuredDataError):
        require_scalar_cells(series, 'x')


def test_an_extension_type_over_a_list_is_walked():
    """`is_nested` is False for the extension type; its storage is a list."""
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    class ListExtension(pa.ExtensionType):
        def __init__(self) -> None:
            super().__init__(pa.list_(pa.string()), 'sdm.test.listext')

        def __arrow_ext_serialize__(self) -> bytes:
            return b''

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    assert not pa.types.is_nested(ListExtension())
    assert not _holds_only_scalars(pd.ArrowDtype(ListExtension()))


def test_an_extension_type_over_a_scalar_is_not_walked():
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    class StringExtension(pa.ExtensionType):
        def __init__(self) -> None:
            super().__init__(pa.string(), 'sdm.test.strext')

        def __arrow_ext_serialize__(self) -> bytes:
            return b''

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    assert _holds_only_scalars(pd.ArrowDtype(StringExtension()))


def test_a_canonical_arrow_extension_over_a_list_is_walked():
    """`fixed_shape_tensor` is a `BaseExtensionType`, not an `ExtensionType`.

    Its storage is a fixed-size list and pandas yields Python lists for it, so
    checking only for `ExtensionType` let it through.
    """
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    tensor = pa.fixed_shape_tensor(pa.float32(), (2,))
    assert not isinstance(tensor, pa.ExtensionType)
    assert isinstance(tensor, pa.BaseExtensionType)
    assert not _holds_only_scalars(pd.ArrowDtype(tensor))


def test_a_canonical_arrow_extension_over_a_scalar_is_not_walked():
    """`json_` stores a string, so the gate can skip it outright."""
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    factory = getattr(pa, 'json_', None)
    if factory is None:
        pytest.skip('pyarrow has no json_')
    assert _holds_only_scalars(pd.ArrowDtype(factory()))


def test_a_uuid_column_is_walked_and_passes():
    """`pa.uuid()` stores a fixed-size binary, which the gate cannot clear.

    So it is walked -- and its cells are `uuid.UUID` objects, not bytes, so
    they pass. The gate is conservative and the cell rule is the authority;
    the cost of that is one walk over a column that turns out to be fine.
    """
    pa = pytest.importorskip('pyarrow')
    import uuid as uuid_module

    from sdm_client.wire import serialize_column
    from sdm_client.wire.dtypes import _holds_only_scalars

    value = uuid_module.uuid4()
    series = pd.Series(
        pa.array([value.bytes], type=pa.uuid()).to_pandas(),
        dtype=pd.ArrowDtype(pa.uuid()),
    )
    assert not _holds_only_scalars(series.dtype)
    assert serialize_column(series, 'string') == [str(value)]


@pytest.mark.parametrize(
    'name',
    [
        'binary',
        'large_binary',
        'binary_view',
        'fixed_size_binary',
        'month_day_nano_interval',
        'list',
        'large_list',
        'list_view',
        'large_list_view',
        'fixed_size_list',
        'struct',
        'map',
    ],
    ids=str,
)
def test_every_arrow_type_whose_cells_are_not_scalars_is_walked(name):
    """The gate names what is safe to skip, and walks everything else.

    Listing the unsafe types is how a list view, then binary, then a *binary*
    view each got through in turn: a new Arrow type defaulted to the fast path,
    and the fast path is the one that can be wrong.
    """
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    builders = {
        'binary': lambda: pa.binary(),
        'large_binary': lambda: pa.large_binary(),
        'binary_view': lambda: pa.binary_view(),
        'fixed_size_binary': lambda: pa.binary(4),
        # Its cell is a `MonthDayNano`, which is a NamedTuple and so a tuple.
        'month_day_nano_interval': lambda: pa.month_day_nano_interval(),
        'list': lambda: pa.list_(pa.string()),
        'large_list': lambda: pa.large_list(pa.string()),
        'list_view': lambda: pa.list_view(pa.string()),
        'large_list_view': lambda: pa.large_list_view(pa.string()),
        'fixed_size_list': lambda: pa.list_(pa.string(), 2),
        'struct': lambda: pa.struct([('a', pa.int64())]),
        'map': lambda: pa.map_(pa.string(), pa.int64()),
    }
    build = builders[name]
    try:
        arrow_type = build()
    except AttributeError:  # pragma: no cover - older pyarrow
        pytest.skip(f'pyarrow has no {name}')
    assert not _holds_only_scalars(pd.ArrowDtype(arrow_type))


@pytest.mark.parametrize(
    'name',
    [
        'int64',
        'float64',
        'bool_',
        'string',
        'large_string',
        'string_view',
        'date32',
        'timestamp',
        'duration',
        'null',
    ],
    ids=str,
)
def test_every_arrow_scalar_type_stays_on_the_fast_path(name):
    """The other direction: skipping must not quietly stop happening."""
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    factory = getattr(pa, name, None)
    if factory is None:
        pytest.skip(f'pyarrow has no {name}')
    arrow_type = (
        factory('us') if name in {'timestamp', 'duration'} else factory()
    )
    assert _holds_only_scalars(pd.ArrowDtype(arrow_type))


def test_a_wrapped_binary_view_is_walked_too():
    """The wrappers unwrap to it, so the same answer has to come back."""
    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars

    view = pa.binary_view()
    assert not _holds_only_scalars(
        pd.ArrowDtype(pa.run_end_encoded(pa.int16(), view))
    )


def test_a_container_target_is_refused_as_a_multi_label_one():
    """The target guard asks the same question the wire does, not its own."""
    frame = context()
    frame['tgt'] = [range(2)] * 6
    with pytest.raises(StructuredDataError) as error:
        Request(
            context=frame, predict=scored(), target='tgt', task='classification'
        )
    assert 'multi-label' in str(error.value)


def test_a_mapping_cell_is_refused_too():
    """An Arrow struct column yields dicts, which stringified the same way."""
    from sdm_client.wire import serialize_column

    with pytest.raises(StructuredDataError) as error:
        serialize_column(pd.Series([{'a': 1}, {'a': 2}], name='meta'), 'string')
    assert 'mapping' in str(error.value)


def test_a_frozenset_is_refused():
    from sdm_client.wire.dtypes import require_scalar_cells

    with pytest.raises(StructuredDataError):
        require_scalar_cells(pd.Series([frozenset({'a'})], name='x'), 'x')


def test_no_arrow_type_is_skipped_whose_cells_would_be_refused():
    """The invariant, checked against every Arrow type pyarrow exposes.

    Enumerating types is what went wrong four times, so this asserts the
    property instead: whatever the gate lets past, the cell rule must accept.
    A type the gate walks is safe by construction and is not examined here.
    """
    import inspect

    pa = pytest.importorskip('pyarrow')
    from sdm_client.wire.dtypes import _holds_only_scalars, is_container

    def build(name: str):
        factory = getattr(pa, name, None)
        if factory is None or not callable(factory):
            return None
        try:
            required = [
                parameter
                for parameter in inspect.signature(factory).parameters.values()
                if parameter.default is parameter.empty
                and parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            if required:
                return None
            return factory()
        except Exception:  # noqa: BLE001 - not a type factory, or unbuildable
            return None

    offenders = []
    for name in dir(pa):
        arrow_type = build(name)
        if arrow_type is None:
            continue
        if not isinstance(arrow_type, pa.DataType):
            continue
        try:
            dtype = pd.ArrowDtype(arrow_type)
        except Exception:  # noqa: BLE001 - pandas cannot represent it
            continue
        if not _holds_only_scalars(dtype):
            continue
        for value in (None, 1, 1.5, True, 'a', b'ab'):
            try:
                cell = pd.Series(
                    pa.array([value], type=arrow_type), dtype=dtype
                ).iloc[0]
            except Exception:  # noqa: BLE001 - the type cannot hold this value
                continue
            if is_container(cell):
                offenders.append((name, type(cell).__name__))
                break
    assert not offenders
