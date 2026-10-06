# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import difflib
import json
import warnings
from functools import lru_cache
from importlib import import_module, resources
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet
from predictive_query.schema.typing import Stype

from sdm_relational_engine.exceptions import UnknownDatasetError
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.local import LocalTable

PREFIX = 'rel-'
HUB = 'https://huggingface.co'

EXCLUDE = {
    'hm': {
        'article': [  # Duplicated columns:
            'product_type_name',
            'graphical_appearance_name',
            'colour_group_name',
            'perceived_colour_value_name',
            'perceived_colour_master_name',
            'department_name',
            'index_name',
            'index_group_name',
            'section_name',
            'garment_group_name',
            'product_code',
        ],
    },
    'amazon': {  # Useless text columns:
        'customer': ['customer_name'],
        'product': ['brand', 'title', 'description'],
        'review': ['review_text', 'summary'],
    },
}


def _pooch() -> Any:
    r"""Import ``pooch``, naming the extra that provides it when it is absent.

    Kept lazy: ``pooch`` is only needed by this module, so importing
    :mod:`sdm_relational_engine.rfm` must not require the ``relbench`` extra.
    """
    try:
        return import_module('pooch')
    except ModuleNotFoundError as error:
        if error.name != 'pooch':
            raise
        raise ModuleNotFoundError(
            "Loading a RelBench dataset requires 'pooch'. Install it via the "
            "'relbench' extra, e.g. `pip install 'structured-data-models-relational-engine[relbench]'` "
            "(or `pip install 'structured-data-models-relational-engine[relbench]'`).",
            name='pooch',
        ) from error


def cache_dir() -> Path:
    r"""The directory RelBench tables are downloaded to."""
    return Path(_pooch().os_cache('relbench'))


@lru_cache
def get_registry() -> dict[str, Any]:
    r"""The RelBench datasets this loader knows, pinned to one commit each.

    RelBench publishes its datasets as Hugging Face dataset repositories that
    carry no hash file and move with ``main``. ``assets/relbench_registry.json``
    records a commit per repository and, for every dataset at that commit,
    its tables' keys and time columns and the SHA-256 of each table's parquet
    file. ``scripts/pin_relbench.py`` writes it; ``assets/VENDORED.md`` says
    how the digests were obtained. A file whose bytes differ from the recorded
    digest is refused, so what loads is what was pinned.
    """
    path = (
        resources.files('sdm_relational_engine.rfm')
        / 'assets'
        / 'relbench_registry.json'
    )
    registry: dict[str, Any] = json.loads(path.read_text(encoding='utf-8'))
    return registry


def _fetch(archive: str, verbose: bool) -> Path:
    r"""Download ``archive``'s tables, check each digest, and return its
    directory.

    Files are read at the pinned commit rather than at ``main``, so the URL
    names the same bytes the digest was recorded from. ``pooch`` checks the
    digest of a cached file on every call too, and fetches it again if it
    differs.
    """
    registry = get_registry()
    dataset = registry['datasets'][archive]
    repository = dataset['repository']
    revision = registry['revisions'][repository]
    pooch = _pooch()
    store = pooch.create(
        path=cache_dir() / repository / revision,
        base_url=f'{HUB}/datasets/{repository}/resolve/{revision}/',
        registry={
            f'{archive}/db/{table}.parquet': f'sha256:{spec["sha256"]}'
            for table, spec in dataset['tables'].items()
        },
        # A dropped connection should not cost a multi-gigabyte restart by hand.
        retry_if_failed=2,
    )
    # pooch logs a line per file it downloads, and a dataset is now many files,
    # so verbose=False silences that as well as the progress bars.
    logger = pooch.get_logger()
    level = logger.level
    if not verbose:
        logger.setLevel('WARNING')
    try:
        for key in store.registry:
            store.fetch(key, progressbar=verbose)
    finally:
        logger.setLevel(level)
    return Path(store.abspath) / archive


def from_relbench(dataset: str, verbose: bool = True) -> Graph:
    r"""Load a RelBench dataset into a :class:`Graph`.

    On first use this downloads the dataset's tables (hundreds of MB for the
    larger datasets) from RelBench's Hugging Face repositories into a per-user
    cache directory (:func:`cache_dir`), and checks each against the digest
    pinned in :func:`get_registry`.

    ``dataset`` is a name from :func:`get_registry`, with or without its
    family prefix: both ``'f1'`` and ``'rel-f1'`` resolve. Only some of the
    published datasets carry a ``rel-`` prefix, so the published name rather
    than a stripped one is what identifies a dataset here.

    Requires the ``relbench`` extra
    (``pip install 'structured-data-models-relational-engine[relbench]'``).
    """
    dataset = dataset.lower()
    archives = get_registry()['datasets']

    archive = next(
        (name for name in (dataset, f'{PREFIX}{dataset}') if name in archives),
        None,
    )
    if archive is None:
        names = sorted({name.removeprefix(PREFIX) for name in archives})
        matches = difflib.get_close_matches(dataset, names, n=1)
        hint = f" Did you mean '{matches[0]}'?" if len(matches) > 0 else ''
        raise UnknownDatasetError(
            f"Unknown RelBench dataset '{dataset}'.{hint} Valid "
            f'datasets are {str(names)[1:-1]}.'
        )

    dataset = archive.removeprefix(PREFIX)
    directory = _fetch(archive, verbose)

    graph = Graph(tables=[])
    edges: list[tuple[str, str, str]] = []
    for name, spec in archives[archive]['tables'].items():
        path = directory / 'db' / f'{name}.parquet'
        schema = pa.parquet.read_schema(path)
        exclude = EXCLUDE.get(dataset, {}).get(name, [])
        columns = [column for column in schema.names if column not in exclude]
        data = pa.parquet.read_table(path, columns=columns)

        graph.add_table(
            LocalTable(
                df=data.to_pandas(),
                name=name,
                primary_key=spec['primary_key'],
                time_column=spec['time_column'],
            )
        )

        edges.extend(
            (name, fkey, dst_table)
            for fkey, dst_table in spec['foreign_keys'].items()
        )

    for edge in edges:
        graph.link(*edge)

    _warn_ambiguous_links(edges)

    if dataset == 'salt':
        # Correct some categorical columns misclassified as numerical
        table = graph['salesdocument']
        table['SALESOFFICE'].stype = Stype.categorical
        table['SALESGROUP'].stype = Stype.categorical
        table['CUSTOMERPAYMENTTERMS'].stype = Stype.categorical
        table['SHIPPINGCONDITION'].stype = Stype.categorical
        table['SALESDOCUMENTTYPE'].stype = Stype.categorical
        table['SALESORGANIZATION'].stype = Stype.categorical
        table['DISTRIBUTIONCHANNEL'].stype = Stype.categorical
        table['ORGANIZATIONDIVISION'].stype = Stype.categorical
        table['BILLINGCOMPANYCODE'].stype = Stype.categorical
        table['TRANSACTIONCURRENCY'].stype = Stype.categorical
        table['HEADERINCOTERMSCLASSIFICATION'].stype = Stype.categorical
        table['CREATIONTIMESTAMP'].stype = Stype.timestamp

        table = graph['salesdocumentitem']
        table['SALESDOCUMENTITEM'].stype = Stype.categorical
        table['PLANT'].stype = Stype.categorical
        table['SHIPPINGPOINT'].stype = Stype.categorical
        table['SALESDOCUMENTITEMCATEGORY'].stype = Stype.categorical
        table['PRODUCT'].stype = Stype.categorical
        table['ITEMINCOTERMSCLASSIFICATION'].stype = Stype.categorical
        table['CREATIONTIMESTAMP'].stype = Stype.timestamp

        table = graph['address']
        table['COUNTRY'].stype = Stype.categorical
        table['REGION'].stype = Stype.categorical

    return graph


def _warn_ambiguous_links(edges: list[tuple[str, str, str]]) -> None:
    r"""Warn when a table reaches another through more than one foreign key.

    RelBench records every foreign key, and several datasets point at the same
    table repeatedly: rel-salt links a sales document item to a customer four
    times, as sold-to, ship-to, bill-to and payer. Each is a real relationship,
    so the loader keeps them all; picking one would silently decide what the
    graph means.

    PQL, though, refuses to aggregate across a link it cannot resolve to one
    key, and it only says so once a query fails, which reads as a problem with
    the query rather than the graph it was written against. Saying it at load
    time puts the warning where the cause is.
    """
    pairs: dict[tuple[str, str], list[str]] = {}
    for src_table, fkey, dst_table in edges:
        pairs.setdefault((src_table, dst_table), []).append(fkey)

    ambiguous = {pair: keys for pair, keys in pairs.items() if len(keys) > 1}
    if not ambiguous:
        return

    described = '; '.join(
        f"'{src}' -> '{dst}' via {sorted(keys)}"
        for (src, dst), keys in sorted(ambiguous.items())
    )
    # Every duplicate but one has to go, so spell out the whole set rather
    # than a single call that leaves the graph just as ambiguous as before.
    (src, dst), keys = sorted(ambiguous.items())[0]
    kept, *dropped = sorted(keys)
    drops = '; '.join(
        f"graph.unlink('{src}', '{key}', '{dst}')" for key in dropped
    )
    warnings.warn(
        f'This dataset links some tables through more than one foreign key '
        f'({described}). A predictive query that aggregates across one of '
        f'these will be rejected as ambiguous. Keep the key your question '
        f"means and drop the rest; to keep '{kept}', run: {drops}",
        stacklevel=3,
    )
