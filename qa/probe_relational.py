# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""kumo-relational against Northwind, a real relational schema.

Northwind is a good adversary: a table name with a space in it, string primary
keys, real dates, and 40-odd columns of which many are free text.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import data
import pandas as pd
from harness import RELATIONAL, fail, note, probe, save
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    StructuredDataError,
)

AREA = 'relational'


def schema():
    """Northwind, with the carrier as a label rather than a number.

    A predictive query takes the task from the target's dtype, so an integer
    carrier id reads as a regression. Casting it to a string is what a caller
    would do, and it is what makes this a classification.
    """
    built = data.northwind_ready()
    orders = built['Orders'].data
    orders['ShipVia'] = orders['ShipVia'].astype('string')
    built['Orders']['ShipVia'].dtype = 'string'
    built['Orders']['ShipVia'].stype = 'categorical'
    return built


def customer_ids(schema, n: int = 3):
    return list(schema['Customers'].data['CustomerID'].head(n))


def order_ids(schema, n: int = 3):
    return list(schema['Orders'].data['OrderID'].head(n))


def run(client):
    with probe(
        AREA, 'the endpoint serves kumo-relational', 'the id is listed'
    ) as e:
        if 'kumo-relational' not in client.models():
            fail(e, f'models: {client.models()}')

    with probe(
        AREA,
        'northwind schema is inferred from the frames alone',
        'three links, right way round',
    ) as e:
        built = Schema.from_data(data.northwind_graph())
        links = {
            (r.source_table, r.source_column, r.target_table)
            for r in built.relationships
        }
        want = {
            ('Orders', 'CustomerID', 'Customers'),
            ('Order Details', 'OrderID', 'Orders'),
            ('Order Details', 'ProductID', 'Products'),
        }
        if links != want:
            fail(e, f'inferred {links}')
        elif built['Orders'].time_column != 'OrderDate':
            fail(e, f'time column {built["Orders"].time_column}')
        else:
            note(e, f'pkeys={ {n: built[n].primary_key for n in built} }')

    # KNOWN BUG bugs/017
    with probe(
        AREA,
        'a primary key that is not a column',
        'refused at the point of assignment',
    ) as e:
        built = schema()
        try:
            built['Customers'].primary_key = 'NoSuchColumn'
            fail(
                e,
                f'accepted; primary_key is now '
                f'{built["Customers"].primary_key!r}',
            )
        except StructuredDataError as exc:
            note(e, str(exc)[:120])

    with probe(
        AREA,
        'a numeric column assigned as the time column',
        'refused at the point of assignment',
    ) as e:
        built = schema()
        try:
            built['Orders'].time_column = 'Freight'
            fail(e, 'a numeric column was accepted as the time column')
        except StructuredDataError as exc:
            note(e, str(exc)[:120])

    # KNOWN BUG bugs/002
    with probe(
        AREA,
        'a non-unique primary key assigned directly',
        'refused; inference declined it for a reason',
    ) as e:
        tables = dict(data.northwind_graph())
        tables['Customers'] = pd.concat(
            [tables['Customers'], tables['Customers'].head(1)],
            ignore_index=True,
        )
        built = Schema.from_data(tables)
        if built['Customers'].primary_key is not None:
            fail(e, 'inference nominated a non-unique key')
        try:
            built['Customers'].primary_key = 'CustomerID'
            fail(e, 'a non-unique column was accepted as the primary key')
        except StructuredDataError as exc:
            note(e, str(exc)[:120])

    # KNOWN BUG bugs/015
    with probe(
        AREA,
        'the graph sent matches the schema',
        'no relationship the schema does not have',
    ) as e:
        from sdm_client.query import _engine, _to_engine_graph

        built = schema()
        graph = _to_engine_graph(_engine(), built)
        sent = {
            (edge.src_table, edge.fkey, edge.dst_table) for edge in graph.edges
        }
        described = {
            (link.source_table, link.source_column, link.target_table)
            for link in built.relationships
        }
        if sent != described:
            fail(
                e,
                f'sent but not described: {sorted(sent - described)}; '
                f'described but not sent: {sorted(described - sent)}',
            )

    # KNOWN BUG bugs/015
    with probe(
        AREA,
        'asserting there are no relationships',
        'an empty sequence means empty',
    ) as e:
        from sdm_client.query import _engine, _to_engine_graph

        built = Schema.from_data(data.northwind_graph(), relationships=[])
        edges = _to_engine_graph(_engine(), built).edges
        if edges:
            fail(e, f'relationships=[] still sent {len(edges)} edges')

    with probe(
        AREA,
        'a table name with a space is carried through',
        'the wire names it as given',
    ) as e:
        built = schema()
        if 'Order Details' not in built:
            fail(e, 'the spaced table name was dropped or rewritten')
            return
        request = PredictiveQuery(
            'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
            built,
        ).materialize(indices=customer_ids(built, 2), num_context_samples=40)
        names = set(request.payloads[0]['context'])
        note(e, f'context tables on the wire: {sorted(names)}')

    # --- static classification over the graph -----------------------------
    with probe(
        AREA, 'classify an order from its related tables', 'a carrier per order'
    ) as e:
        built = schema()
        ids = order_ids(built, 3)
        request = PredictiveQuery(
            'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
        ).materialize(indices=ids, num_context_samples=200)
        frame = client.predict(request, model=KumoRelational()).to_pandas()
        carriers = set(built['Orders'].data['ShipVia'].dropna().unique())
        if len(frame) != 3:
            fail(e, f'expected 3 rows, got {len(frame)}')
        elif (
            not set(frame['prediction'])
            <= {str(c) for c in carriers} | carriers
        ):
            fail(
                e,
                f'predicted outside the label set: {set(frame["prediction"])}',
            )
        else:
            note(
                e,
                f'ids={list(frame["id"])} predictions={list(frame["prediction"])}',
            )

    with probe(
        AREA,
        'probabilities come back for a classification',
        'one column per class, summing to 1',
    ) as e:
        built = schema()
        request = PredictiveQuery(
            'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
        ).materialize(indices=order_ids(built, 3), num_context_samples=200)
        frame = client.predict(
            request,
            model=KumoRelational(),
            outputs=['prediction', 'probabilities'],
        ).to_pandas()
        cols = [c for c in frame.columns if c.startswith('probability_')]
        if not cols:
            fail(e, f'no probability columns: {list(frame.columns)}')
        elif not frame[cols].sum(axis=1).sub(1).abs().lt(1e-4).all():
            fail(
                e,
                f'probabilities do not sum to 1: {frame[cols].sum(axis=1).tolist()}',
            )
        else:
            note(e, f'{len(cols)} classes')

    # --- the temporal aggregation that is the point of a relational model --
    with probe(
        AREA,
        'a temporal aggregation over a related table',
        'a number per entity',
    ) as e:
        built = schema()
        request = PredictiveQuery(
            'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
            built,
        ).materialize(indices=customer_ids(built, 3), num_context_samples=200)
        frame = client.predict(request, model=KumoRelational()).to_pandas()
        if len(frame) != 3:
            fail(e, f'expected 3 rows, got {len(frame)}')
        elif (
            not pd.to_numeric(frame['prediction'], errors='coerce')
            .notna()
            .all()
        ):
            fail(
                e, f'non-numeric regression output: {list(frame["prediction"])}'
            )
        else:
            note(e, f'predictions={list(frame["prediction"])}')

    with probe(
        AREA, 'a sum aggregation over a numeric column', 'a number per entity'
    ) as e:
        built = schema()
        request = PredictiveQuery(
            'PREDICT SUM(Orders.Freight, 0, 90, days) FOR EACH Customers.CustomerID',
            built,
        ).materialize(indices=customer_ids(built, 3), num_context_samples=200)
        frame = client.predict(request, model=KumoRelational()).to_pandas()
        if len(frame) != 3:
            fail(e, f'expected 3 rows, got {len(frame)}')
        else:
            note(e, f'predictions={list(frame["prediction"])}')

    # --- knobs ------------------------------------------------------------
    with probe(
        AREA,
        'num_context_samples is honoured on real data',
        'gathered == asked',
    ) as e:
        built = schema()
        query = PredictiveQuery(
            'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
        )
        sizes = {}
        for ask in (5, 40, 300):
            request = query.materialize(
                indices=order_ids(built, 2), num_context_samples=ask
            )
            sizes[ask] = len(
                request.payloads[0]['context']['instance_table']['rows']
            )
            client.predict(request, model=KumoRelational()).to_pandas()
        wrong = {k: v for k, v in sizes.items() if v != k}
        if wrong:
            fail(e, f'asked -> gathered {sizes}')
        else:
            note(e, str(sizes))

    with probe(
        AREA,
        'num_neighbors changes what is gathered',
        'a wider fanout gathers more',
    ) as e:
        built = schema()
        query = PredictiveQuery(
            'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
            built,
        )

        def rows(request):
            context = request.payloads[0]['context']
            total = len(context['instance_table'].get('rows', []))
            for table in context.get('related_tables', {}).values():
                total += len(table.get('rows', []))
            return total

        small = query.materialize(
            indices=customer_ids(built, 2),
            num_context_samples=60,
            num_neighbors=[1, 1],
        )
        large = query.materialize(
            indices=customer_ids(built, 2),
            num_context_samples=60,
            num_neighbors=[16, 16],
        )
        if rows(small) >= rows(large):
            fail(e, f'fanout did nothing: {rows(small)} vs {rows(large)}')
        else:
            note(e, f'{rows(small)} rows at [1,1], {rows(large)} at [16,16]')
        client.predict(small, model=KumoRelational()).to_pandas()

    with probe(
        AREA, 'an anchor time inside the data', 'accepted and scored'
    ) as e:
        built = schema()
        anchor = built['Orders'].data['OrderDate'].max()
        request = PredictiveQuery(
            'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
            built,
        ).materialize(
            indices=customer_ids(built, 2),
            num_context_samples=60,
            anchor_time=anchor,
        )
        frame = client.predict(request, model=KumoRelational()).to_pandas()
        if len(frame) != 2:
            fail(e, f'expected 2 rows, got {len(frame)}')

    with probe(
        AREA,
        'an anchor time before the data starts',
        'a StructuredDataError, not a bare ValueError',
    ) as e:
        built = schema()
        try:
            PredictiveQuery(
                'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
                built,
            ).materialize(
                indices=customer_ids(built, 2),
                num_context_samples=60,
                anchor_time=pd.Timestamp('1990-01-01'),
            )
            fail(e, 'an impossible anchor was accepted')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    # --- limits the endpoint enforces -------------------------------------
    with probe(
        AREA,
        'more entities than the endpoint scores',
        'a StructuredDataError naming this SDK',
    ) as e:
        built = schema()
        many = list(built['Orders'].data['OrderID'].head(1_500))
        try:
            PredictiveQuery(
                'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
            ).materialize(indices=many, num_context_samples=60)
            note(e, 'accepted; batching happens downstream')
        except StructuredDataError as exc:
            note(e, str(exc)[:160])
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    with probe(
        AREA,
        'a target with more classes than the model takes',
        'refused locally, naming the count',
    ) as e:
        built = schema()
        try:
            request = PredictiveQuery(
                'PREDICT Orders.ShipCountry FOR EACH Orders.OrderID', built
            ).materialize(indices=order_ids(built, 2), num_context_samples=300)
            client.predict(request, model=KumoRelational()).to_pandas()
            note(e, '21 classes were accepted')
        except StructuredDataError as exc:
            note(e, f'{type(exc).__name__}: {str(exc)[:160]}')

    with probe(
        AREA, 'a free-text column', 'refused locally, naming the column'
    ) as e:
        built = data.northwind_ready()
        built['Customers']['CompanyName'].stype = 'text'
        try:
            request = PredictiveQuery(
                'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
            ).materialize(indices=order_ids(built, 2), num_context_samples=200)
            client.predict(request, model=KumoRelational()).to_pandas()
            note(e, 'text was accepted')
        except StructuredDataError as exc:
            message = str(exc)
            if 'CompanyName' not in message:
                fail(e, f'refused without naming the column: {message[:160]}')
            else:
                note(e, message[:160])

    with probe(
        AREA, 'an explanation over several entities', 'refused before the wire'
    ) as e:
        built = schema()
        request = PredictiveQuery(
            'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
        ).materialize(indices=order_ids(built, 4), num_context_samples=200)
        try:
            client.predict(
                request,
                model=KumoRelational(),
                outputs=['prediction', 'explanation'],
            )
            fail(e, 'four entities were accepted for an explanation')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])

    with probe(
        AREA,
        'an explanation for one entity',
        'served, or refused for a stated reason',
    ) as e:
        built = schema()
        request = PredictiveQuery(
            'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
        ).materialize(indices=order_ids(built, 1), num_context_samples=200)
        try:
            result = client.predict(
                request,
                model=KumoRelational(),
                outputs=['prediction', 'explanation'],
            )
            note(e, f'served: {list(result.to_pandas().columns)}')
        except StructuredDataError as exc:
            note(e, f'refused: {str(exc)[:160]}')

    # --- sessions ---------------------------------------------------------
    with probe(
        AREA,
        'a relational session serves repeat calls',
        'the pinned context is reused',
    ) as e:
        built = schema()
        request = PredictiveQuery(
            'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
        ).materialize(indices=order_ids(built, 2), num_context_samples=200)
        try:
            with client.fit(request, model=KumoRelational()) as fitted:
                first = fitted.predict(request).to_pandas()
                second = fitted.predict(request).to_pandas()
                if len(first) != 2 or len(second) != 2:
                    fail(e, f'{len(first)} then {len(second)}')
        except StructuredDataError as exc:
            note(e, f'refused: {str(exc)[:160]}')

    # --- correlation and privacy -------------------------------------------
    with probe(
        AREA,
        'entity values never travel raw',
        'the wire carries transport keys only',
    ) as e:
        built = schema()
        ids = order_ids(built, 3)
        request = PredictiveQuery(
            'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
        ).materialize(indices=ids, num_context_samples=200)
        predict_table = request.payloads[0]['predict']['instance_table']
        columns = predict_table.get('columns', [])
        frame = client.predict(request, model=KumoRelational()).to_pandas()
        if 'instance_id' not in columns:
            fail(e, f'no transport key on the wire: {columns}')
        elif sorted(str(i) for i in frame['id']) != sorted(str(i) for i in ids):
            fail(e, f'got ids {sorted(frame["id"])}, asked {sorted(ids)}')
        else:
            note(e, f'wire columns {columns}')

    with probe(
        AREA,
        'a string primary key correlates',
        'CustomerIDs come back, not row numbers',
    ) as e:
        built = schema()
        ids = customer_ids(built, 3)
        request = PredictiveQuery(
            'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
            built,
        ).materialize(indices=ids, num_context_samples=200)
        frame = client.predict(request, model=KumoRelational()).to_pandas()
        if sorted(frame['id']) != sorted(ids):
            fail(e, f'got {sorted(frame["id"])}, asked {sorted(ids)}')

    # --- bad input --------------------------------------------------------
    with probe(
        AREA,
        'an entity id that is not in the table',
        'a StructuredDataError, not an engine type',
    ) as e:
        built = schema()
        try:
            PredictiveQuery(
                'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
            ).materialize(indices=[-1, -2], num_context_samples=60)
            fail(e, 'ids that are not in the table were accepted')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    with probe(
        AREA,
        'a duplicated primary key',
        'a StructuredDataError, not a silent mis-join',
    ) as e:
        tables = dict(data.northwind_graph())
        tables['Customers'] = pd.concat(
            [tables['Customers'], tables['Customers'].head(1)],
            ignore_index=True,
        )
        try:
            built = Schema.from_data(
                tables,
                primary_keys={
                    'Customers': 'CustomerID',
                    'Orders': 'OrderID',
                    'Products': 'ProductID',
                },
                time_columns={'Orders': 'OrderDate'},
            )
            for table in built:
                for column in built[table]:
                    if built[table][column].stype == 'text':
                        built[table][column].stype = 'categorical'
            request = PredictiveQuery(
                'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
                built,
            ).materialize(
                indices=customer_ids(built, 1), num_context_samples=60
            )
            client.predict(request, model=KumoRelational()).to_pandas()
            fail(e, 'a duplicated primary key was accepted')
        except StructuredDataError as exc:
            note(e, str(exc)[:140])
        except Exception as exc:  # noqa: BLE001
            fail(
                e, f'{type(exc).__name__} escaped the SDK error surface: {exc}'
            )

    # --- a library should not print ---------------------------------------
    with probe(
        AREA,
        'materializing prints nothing to stdout',
        'a library writes to a logger, not stdout',
    ) as e:
        built = schema()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            PredictiveQuery(
                'PREDICT Orders.ShipVia FOR EACH Orders.OrderID', built
            ).materialize(indices=order_ids(built, 2), num_context_samples=60)
        printed = buffer.getvalue()
        if printed.strip():
            fail(e, f'printed {printed.strip()[:160]!r}')


if __name__ == '__main__':
    with SDMClient(url=RELATIONAL, timeout=900.0) as client:
        run(client)
    save('relational.json')
