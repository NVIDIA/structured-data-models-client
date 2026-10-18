-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
-- All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- The provider conformance kit's fixture dataset (T2.5, T2.6): built in
-- DuckDB from this file and, for SQLite and Postgres, loaded from DuckDB by
-- the one contract loader (contract/runners/databases.py), so every provider
-- reads the same rows. expected.json beside it holds the kit's cases and
-- their answers.
--
-- Small on purpose: integers, text and one float column, exactly
-- representable, with no time column, so an answer reads the same on every
-- provider whatever Arrow type its driver gives. orders.seq ties within
-- customer 1 (orders 12 and 17), so an order needs its second term;
-- orders.customer_id has one null (order 16), which joins no seed and
-- counts as missing; orders.amount has one null (order 14), carried as it
-- is; customers.region repeats one value ('north') and holds one null.

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    region VARCHAR
);

INSERT INTO customers VALUES
    (1, 'ann', 'north'),
    (2, 'bob', 'south'),
    (3, 'cy', NULL),
    (4, 'dee', 'north'),
    (5, 'eve', 'west');

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers (id),
    seq INTEGER NOT NULL,
    amount DOUBLE
);

INSERT INTO orders VALUES
    (10, 1, 1, 2.5),
    (11, 1, 2, 4.0),
    (12, 1, 3, 1.5),
    (13, 2, 1, 8.0),
    (14, 2, 2, NULL),
    (15, 4, 1, 3.0),
    (16, NULL, 1, 6.0),
    (17, 1, 3, 0.5),
    (18, 4, 2, 7.25);

-- Round 1's regressions (T2.5). zeros holds a DOUBLE key as -0.0 and 0.0,
-- one value to every SQL engine, so its key counts one distinct value and
-- the value repeats. wide holds ten rows of 2,048 characters, so a page of
-- even one row is past a fallback's 1,000 bytes.

CREATE TABLE zeros (k DOUBLE);

INSERT INTO zeros VALUES (CAST('-0.0' AS DOUBLE)), (CAST('0.0' AS DOUBLE));

CREATE TABLE wide AS
SELECT CAST(i AS INTEGER) AS id, repeat(chr(CAST(97 + i AS INTEGER)), 2048) AS v
FROM range(10) AS t (i);
