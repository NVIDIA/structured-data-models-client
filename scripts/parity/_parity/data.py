# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The scenarios' data, generated in code from a name and a seed.

No data file is checked in. A transcript's ``data`` names a generator here
and its seed, and a replay of an in-memory scenario builds the same frames
again. Every draw comes from numpy's legacy ``RandomState`` (MT19937), whose
streams NEP 19 freezes, so a generator gives the same rows on every numpy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Frames = dict[str, pd.DataFrame]

#: The day the shop opens; every order falls in the year after it.
SHOP_START = pd.Timestamp('2024-01-01')


def shop(seed: int) -> Frames:
    r"""A small shop: users, products, and a year of orders.

    * ``users``: 40 users with an age and a country.
    * ``products``: 12 products in three categories, with a price.
    * ``orders``: order lines. A basket is one to three lines a user placed at
      the same hour, so many lines tie on ``ordered_at``, within a user and
      across users. A quarter of the users stop ordering partway through the
      year, which is what an inactivity question finds.

    Order ids are given per user, so the table is not stored in time order.
    """
    rng = np.random.RandomState(seed)
    num_users, num_products = 40, 12
    users = pd.DataFrame(
        {
            'user_id': np.arange(1, num_users + 1, dtype='int64'),
            'age': rng.randint(18, 70, num_users).astype('int64'),
            'country': rng.choice(['US', 'DE', 'IN'], num_users).tolist(),
        }
    )
    categories = ['books', 'games', 'tools']
    products = pd.DataFrame(
        {
            'product_id': np.arange(1, num_products + 1, dtype='int64'),
            'category': [
                categories[i % 3] if i < 3 else str(rng.choice(categories))
                for i in range(num_products)
            ],
            'price': np.round(rng.uniform(5.0, 100.0, num_products), 2),
        }
    )
    lines = []
    for user in range(1, num_users + 1):
        first = int(rng.randint(0, 60))
        last = 365 if user % 4 else int(rng.randint(150, 270))
        for _ in range(int(rng.poisson(14))):
            day = int(rng.randint(first, last))
            hour = int(rng.randint(8, 20))
            at = SHOP_START + pd.Timedelta(days=day, hours=hour)
            for product in rng.choice(
                np.arange(1, num_products + 1),
                int(rng.randint(1, 4)),
                replace=False,
            ):
                lines.append((user, int(product), int(rng.randint(1, 5)), at))
    orders = pd.DataFrame(
        lines, columns=['user_id', 'product_id', 'quantity', 'ordered_at']
    )
    orders.insert(0, 'order_id', np.arange(1, len(orders) + 1, dtype='int64'))
    orders = orders.astype(
        {'user_id': 'int64', 'product_id': 'int64', 'quantity': 'int64'}
    )
    orders['ordered_at'] = orders['ordered_at'].astype('datetime64[us]')
    return {'users': users, 'products': products, 'orders': orders}


def shop_catalog(seed: int) -> Frames:
    r"""The shop, with a supplier's catalog keyed by the shop's product ids.

    ``catalog`` holds one row per product: its ``sku``, which is the shop's
    ``product_id``, and the ``supplier`` that lists it. A schema may link
    ``orders.product_id`` to both ``products`` and ``catalog``, which makes a
    ``LIST_DISTINCT`` over it a multilabel question rather than a link
    prediction.
    """
    frames = shop(seed)
    rng = np.random.RandomState(seed + 1)
    skus = frames['products']['product_id'].to_numpy()
    catalog = pd.DataFrame(
        {
            'sku': skus.astype('int64'),
            'supplier': rng.choice(['acme', 'globex'], len(skus)).tolist(),
        }
    )
    return {**frames, 'catalog': catalog}


def shop_referrals(seed: int) -> Frames:
    r"""The shop, with who referred each user and when each joined.

    ``users`` gains ``referrer_id``, the id of another user, and
    ``joined_at``, a day in the shop's first two months. A schema may link
    ``users.referrer_id`` to ``users.user_id``: a table linked to itself.
    """
    frames = shop(seed)
    rng = np.random.RandomState(seed + 2)
    users = frames['users']
    num_users = len(users)
    ids = users['user_id'].to_numpy()
    offsets = rng.randint(1, num_users, num_users)
    users['referrer_id'] = ((ids - 1 + offsets) % num_users + 1).astype('int64')
    days = rng.randint(0, 60, num_users)
    users['joined_at'] = pd.Series(
        SHOP_START + pd.to_timedelta(days, unit='D')
    ).astype('datetime64[us]')
    return frames


def shop_categorical_keys(seed: int) -> Frames:
    r"""The shop, with the user ids held as a pandas ``category``.

    ``users.user_id`` and ``orders.user_id`` keep their values, as
    categories of the users' ids. Casting a value that is none of them to
    the column's dtype gives a missing value rather than an error.
    """
    frames = shop(seed)
    ids = frames['users']['user_id'].to_numpy()
    for table in ('users', 'orders'):
        frames[table]['user_id'] = pd.Categorical(
            frames[table]['user_id'], categories=ids
        )
    return frames


def wire_snapshot(seed: int) -> Frames:
    r"""The two tables ``scripts/snapshot_wire_payloads.py`` asks about.

    A copy of its ``_people`` and ``_orders``, so the tabular cases recorded
    here put exactly what ``tests/golden/wire_payloads.json`` holds on the
    wire. Copied rather than imported, so the corpus does not move when the
    script does. They draw nothing; ``seed`` must be 0.
    """
    if seed != 0:
        raise ValueError('the wire snapshot tables draw nothing; seed is 0')
    n = 12
    people = pd.DataFrame(
        {
            'user_id': range(n),
            'age': [20.0 + i for i in range(n)],
            'segment': ['north' if i % 3 else 'south' for i in range(n)],
            'churn': [i % 2 for i in range(n)],
        }
    )
    m = 240
    orders = pd.DataFrame(
        {
            'order_id': range(m),
            'user_id': [i % 12 for i in range(m)],
            'price': [10.0 + i for i in range(m)],
            'ordered_at': pd.date_range('2026-01-01', periods=m, freq='D'),
        }
    )
    return {'people': people, 'orders': orders}


def marketplace(seed: int) -> Frames:
    r"""A small marketplace, for what the shop never makes the sampler do.

    * ``customers``: 60 customers with a segment, who signed up on one of 12
      Mondays, so many share a ``signup_at``: an anchor taken from the entity
      ties.
    * ``items``: 8 items in two categories, with a price.
    * ``reviews``: 36 to 79 reviews of each item, without a time column, so
      a hop from an item to its reviews has more of them than a hop samples
      and the in-memory sampler draws which. Each has a body of 8 to 29
      words, which is inferred as text and sent normalized.
    * ``stores``: 5 stores keyed by a code, text rather than a number.
    * ``purchases``: a year of purchases after each customer signed up, of an
      amount in cents, so a sum of them is a fraction ``float32`` rounds, at
      one of the stores.

    Two tables, ``customers`` and ``purchases``, have a time column.
    """
    rng = np.random.RandomState(seed)
    num_customers, num_items = 60, 8
    signup_days = rng.randint(0, 12, num_customers) * 7
    customers = pd.DataFrame(
        {
            'customer_id': np.arange(1, num_customers + 1, dtype='int64'),
            'segment': rng.choice(
                ['retail', 'business', 'student'], num_customers
            ).tolist(),
            'signup_at': pd.Series(
                SHOP_START + pd.to_timedelta(signup_days, unit='D')
            ).astype('datetime64[us]'),
        }
    )
    items = pd.DataFrame(
        {
            'item_id': np.arange(1, num_items + 1, dtype='int64'),
            'category': rng.choice(['garden', 'kitchen'], num_items).tolist(),
            'price': np.round(rng.uniform(1.0, 50.0, num_items), 2),
        }
    )
    rows = []
    for item in range(1, num_items + 1):
        for _ in range(int(rng.randint(36, 80))):
            rows.append((item, int(rng.randint(1, 6))))
    reviews = pd.DataFrame(rows, columns=['item_id', 'rating'])
    reviews.insert(
        0, 'review_id', np.arange(1, len(reviews) + 1, dtype='int64')
    )
    words = (
        'great value broke fast shipping quality poor love returned again '
        'would buy not recommend sturdy cheap solid gift size color zip pack '
        'box late early nice bad ok meh wow'
    ).split()
    reviews['body'] = [
        ' '.join(rng.choice(words, int(rng.randint(8, 30)))).capitalize() + '.'
        for _ in range(len(reviews))
    ]
    stores = pd.DataFrame(
        {
            'store_code': [f'S{i:02d}' for i in range(1, 6)],
            'region': ['north', 'south', 'east', 'west', 'north'],
        }
    )
    lines = []
    for customer in range(1, num_customers + 1):
        first = int(signup_days[customer - 1])
        for _ in range(int(rng.poisson(12))):
            day = int(rng.randint(first, 366))
            hour = int(rng.randint(6, 22))
            at = SHOP_START + pd.Timedelta(days=day, hours=hour)
            item = int(rng.randint(1, num_items + 1))
            amount = round(float(rng.uniform(0.5, 99.99)), 2)
            store = f'S{int(rng.randint(1, 6)):02d}'
            lines.append((customer, item, amount, at, store))
    purchases = pd.DataFrame(
        lines,
        columns=[
            'customer_id',
            'item_id',
            'amount',
            'purchased_at',
            'store_code',
        ],
    )
    purchases.insert(
        0, 'purchase_id', np.arange(1, len(purchases) + 1, dtype='int64')
    )
    purchases['purchased_at'] = purchases['purchased_at'].astype(
        'datetime64[us]'
    )
    return {
        'customers': customers,
        'items': items,
        'reviews': reviews,
        'stores': stores,
        'purchases': purchases,
    }


def marketplace_garbled(seed: int) -> Frames:
    r"""The marketplace, with ``purchases.purchased_at`` stored as text.

    Times are written ``YYYY-MM-DD HH:MM:SS``, as a database without a time
    type keeps them, and every 25th purchase between February and November
    is at the end of its day, spelled ``24:00:00``. DuckDB and PostgreSQL
    read that as the next midnight, and SQLite compares it as text among the
    others, so each finds real earliest and latest times; pandas does not
    read it, so the engine drops those values as ``NaT`` and warns of the
    column it read them from.
    """
    frames = marketplace(seed)
    purchases = frames['purchases']
    text = [
        f'{at:%Y-%m-%d} 24:00:00'
        if i % 25 == 7 and 2 <= at.month <= 11
        else f'{at:%Y-%m-%d %H:%M:%S}'
        for i, at in enumerate(purchases['purchased_at'])
    ]
    purchases['purchased_at'] = text
    return frames


def marketplace_large(seed: int) -> Frames:
    r"""A marketplace too large for one batch of the sampler's candidates.

    * ``customers``: 12,000 customers who signed up on one of 12 Mondays,
      in the marketplace's segments; about one in 200 is a ``'vip'``, so a
      question about them finds few among many candidates.
    * ``items``: 8 items in two categories, with a price.
    * ``stores``: the marketplace's 5 stores, keyed by a text code.
    * ``purchases``: none to a few purchases per customer, after signing up,
      of an amount in cents, at one of the stores.

    Every draw is vectorized, so the tables are made in well under a second.
    """
    rng = np.random.RandomState(seed)
    num_customers, num_items = 12_000, 8
    signup_days = rng.randint(0, 12, num_customers) * 7
    segment = rng.choice(['retail', 'business', 'student'], num_customers)
    segment = np.where(rng.rand(num_customers) < 0.005, 'vip', segment)
    customers = pd.DataFrame(
        {
            'customer_id': np.arange(1, num_customers + 1, dtype='int64'),
            'segment': segment.tolist(),
            'signup_at': pd.Series(
                SHOP_START + pd.to_timedelta(signup_days, unit='D')
            ).astype('datetime64[us]'),
        }
    )
    items = pd.DataFrame(
        {
            'item_id': np.arange(1, num_items + 1, dtype='int64'),
            'category': rng.choice(['garden', 'kitchen'], num_items).tolist(),
            'price': np.round(rng.uniform(1.0, 50.0, num_items), 2),
        }
    )
    stores = pd.DataFrame(
        {
            'store_code': [f'S{i:02d}' for i in range(1, 6)],
            'region': ['north', 'south', 'east', 'west', 'north'],
        }
    )
    counts = rng.poisson(1.5, num_customers)
    customer = np.repeat(np.arange(1, num_customers + 1), counts)
    first = signup_days[customer - 1]
    day = first + (rng.rand(len(customer)) * (366 - first)).astype('int64')
    hour = rng.randint(6, 22, len(customer))
    purchases = pd.DataFrame(
        {
            'purchase_id': np.arange(1, len(customer) + 1, dtype='int64'),
            'customer_id': customer.astype('int64'),
            'item_id': rng.randint(1, num_items + 1, len(customer)).astype(
                'int64'
            ),
            'amount': np.round(rng.uniform(0.5, 99.99, len(customer)), 2),
            'purchased_at': pd.Series(
                SHOP_START
                + pd.to_timedelta(day, unit='D')
                + pd.to_timedelta(hour, unit='h')
            ).astype('datetime64[us]'),
            'store_code': [
                f'S{i:02d}' for i in rng.randint(1, 6, len(customer))
            ],
        }
    )
    return {
        'customers': customers,
        'items': items,
        'stores': stores,
        'purchases': purchases,
    }


def marketplace_extras(seed: int) -> Frames:
    r"""The marketplace, with what some schemas have besides.

    * ``items.dims``: each item's dimensions as text, ``'[w, h, d]'``, as a
      database without a list type keeps a sequence; one is ``'[]'``.
    * ``stock``: how many of each item each store holds, keyed by the pair
      of ``store_code`` and ``item_id``.
    * ``cards``: loyalty cards, which hold nothing but their number and their
      customer.
    """
    frames = marketplace(seed)
    rng = np.random.RandomState(seed + 1)
    items = frames['items']
    dims = []
    for item in items['item_id']:
        if item == 4:
            dims.append('[]')
            continue
        w, h, d = np.round(rng.uniform(1.0, 60.0, 3), 1)
        dims.append(f'[{w}, {h}, {d}]')
    items['dims'] = dims
    stores = frames['stores']['store_code'].tolist()
    stock = pd.DataFrame(
        [
            (store, int(item), int(rng.randint(0, 40)))
            for store in stores
            for item in items['item_id']
        ],
        columns=['store_code', 'item_id', 'units'],
    )
    customers = frames['customers']['customer_id']
    cards = pd.DataFrame(
        {
            'card_no': (1000 + customers[customers % 3 == 0]).astype('int64'),
            'customer_id': customers[customers % 3 == 0].astype('int64'),
        }
    ).reset_index(drop=True)
    return {**frames, 'stock': stock, 'cards': cards}


def marketplace_wide(seed: int) -> Frames:
    r"""The marketplace, with twelve more tables beside its customers.

    ``notes_01`` to ``notes_12`` each hold a note or two about some of the
    customers: a number, and the customer. Two hops from a customer reach
    more tables than a request may carry.
    """
    frames = marketplace(seed)
    rng = np.random.RandomState(seed + 2)
    customers = frames['customers']['customer_id'].to_numpy()
    for n in range(1, 13):
        about = np.sort(rng.choice(customers, 30))
        frames[f'notes_{n:02d}'] = pd.DataFrame(
            {
                f'note_{n:02d}_id': np.arange(1, len(about) + 1, dtype='int64'),
                'customer_id': about.astype('int64'),
                'score': rng.randint(0, 10, len(about)).astype('int64'),
            }
        )
    return frames


def marketplace_untimed(seed: int) -> Frames:
    r"""The marketplace, with purchases nobody recorded the time of.

    ``purchases`` loses its ``purchase_id``, so it has no key, and every
    ``purchased_at`` is missing, so its time column holds no time at all.
    """
    frames = marketplace(seed)
    purchases = frames['purchases'].drop(columns=['purchase_id'])
    purchases['purchased_at'] = pd.Series(
        [pd.NaT] * len(purchases), dtype='datetime64[us]'
    )
    frames['purchases'] = purchases
    return frames


GENERATORS: dict[str, Callable[[int], Frames]] = {
    'shop': shop,
    'shop-catalog': shop_catalog,
    'shop-referrals': shop_referrals,
    'shop-categorical-keys': shop_categorical_keys,
    'wire-snapshot': wire_snapshot,
    'marketplace': marketplace,
    'marketplace-garbled': marketplace_garbled,
    'marketplace-large': marketplace_large,
    'marketplace-extras': marketplace_extras,
    'marketplace-wide': marketplace_wide,
    'marketplace-untimed': marketplace_untimed,
}


def generate(name: str, seed: int) -> Frames:
    return GENERATORS[name](seed)
