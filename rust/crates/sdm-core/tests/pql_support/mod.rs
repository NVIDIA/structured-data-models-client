// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small authored schema inputs shared by PQL behavior tests.
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    ColumnRef, Dtype, ProfileName, QueryColumn, QuerySchema, QueryTable, Stype,
};
use sdm_contract::{HostFacts, Scalar, Unit};
use sdm_core::profiles::Profile;
pub struct NoZones;

impl sdm_core::Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, sdm_core::UnknownZone> {
        Err(sdm_core::UnknownZone)
    }
}

/// Fixed clock for deterministic literal interpretation.
pub fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
        date_literal_range: None,
    }
}

pub fn profile(name: &str) -> Profile {
    Profile::named(match name {
        "RELATIONAL" => ProfileName::Relational,
        "FINE_TUNING" => ProfileName::FineTuning,
        "TABULAR" => ProfileName::Tabular,
        "TIMESERIES" => ProfileName::Timeseries,
        "ANOMALY" => ProfileName::Anomaly,
        _ => panic!("unknown test profile {name}"),
    })
}

fn table(
    name: &str,
    key: &str,
    time: Option<&str>,
    columns: &[(&str, Dtype, Stype)],
) -> QueryTable {
    QueryTable {
        name: name.into(),
        primary_key: Some(key.into()),
        time_column: time.map(str::to_owned),
        end_time_column: None,
        columns: columns
            .iter()
            .map(|(name, dtype, stype)| QueryColumn {
                name: (*name).into(),
                dtype: *dtype,
                stype: *stype,
            })
            .collect(),
    }
}
fn link(columns: &[(&str, &str)]) -> Vec<ColumnRef> {
    columns
        .iter()
        .map(|(table, column)| ColumnRef {
            table: (*table).into(),
            column: (*column).into(),
        })
        .collect()
}

pub fn schema(name: &str) -> QuerySchema {
    match name {
        "schema-1" => QuerySchema {
            tables: vec![
                table(
                    "users",
                    "user_id",
                    None,
                    &[
                        ("user_id", Dtype::Int, Stype::Id),
                        ("age", Dtype::Int, Stype::Numerical),
                        ("country", Dtype::String, Stype::Categorical),
                        ("vip", Dtype::Bool, Stype::Categorical),
                        ("bio", Dtype::String, Stype::Text),
                        ("signup", Dtype::Time, Stype::Timestamp),
                    ],
                ),
                table(
                    "products",
                    "product_id",
                    None,
                    &[
                        ("product_id", Dtype::Int, Stype::Id),
                        ("category", Dtype::String, Stype::Categorical),
                        ("price", Dtype::Float, Stype::Numerical),
                        ("tags", Dtype::Stringlist, Stype::Multicategorical),
                    ],
                ),
                table(
                    "orders",
                    "order_id",
                    Some("ordered_at"),
                    &[
                        ("order_id", Dtype::Int, Stype::Id),
                        ("user_id", Dtype::Int, Stype::Id),
                        ("product_id", Dtype::Int, Stype::Id),
                        ("quantity", Dtype::Int, Stype::Numerical),
                        ("amount", Dtype::Float, Stype::Numerical),
                        ("status", Dtype::String, Stype::Categorical),
                        ("ordered_at", Dtype::Time, Stype::Timestamp),
                    ],
                ),
                table(
                    "reviews",
                    "review_id",
                    Some("created_at"),
                    &[
                        ("review_id", Dtype::Int, Stype::Id),
                        ("user_id", Dtype::Int, Stype::Id),
                        ("product_id", Dtype::Int, Stype::Id),
                        ("rating", Dtype::Int, Stype::Numerical),
                        ("created_at", Dtype::Time, Stype::Timestamp),
                    ],
                ),
            ],
            links: vec![
                link(&[
                    ("orders", "user_id"),
                    ("reviews", "user_id"),
                    ("users", "user_id"),
                ]),
                link(&[
                    ("orders", "product_id"),
                    ("products", "product_id"),
                    ("reviews", "product_id"),
                ]),
            ],
        },
        "schema-2" => QuerySchema {
            tables: vec![
                table(
                    "Order Lines",
                    "Line ID",
                    Some("Order Date"),
                    &[
                        ("Line ID", Dtype::Int, Stype::Id),
                        ("Customer ID", Dtype::Int, Stype::Id),
                        ("Unit Price", Dtype::Float, Stype::Numerical),
                        ("Order Date", Dtype::Time, Stype::Timestamp),
                    ],
                ),
                table(
                    "Customers",
                    "Customer ID",
                    None,
                    &[
                        ("Customer ID", Dtype::Int, Stype::Id),
                        ("Region", Dtype::String, Stype::Categorical),
                    ],
                ),
            ],
            links: vec![link(&[
                ("Customers", "Customer ID"),
                ("Order Lines", "Customer ID"),
            ])],
        },
        "schema-3" => QuerySchema {
            tables: vec![table(
                "people",
                "person_id",
                None,
                &[
                    ("person_id", Dtype::Int, Stype::Id),
                    ("age", Dtype::Float, Stype::Numerical),
                    ("churn", Dtype::Int, Stype::Numerical),
                    ("segment", Dtype::String, Stype::Categorical),
                    ("joined", Dtype::Time, Stype::Timestamp),
                ],
            )],
            links: vec![],
        },
        "schema-4" => QuerySchema {
            tables: vec![
                table(
                    "people",
                    "person_id",
                    None,
                    &[("person_id", Dtype::Int, Stype::Id)],
                ),
                table(
                    "trips",
                    "trip_id",
                    Some("started_at"),
                    &[
                        ("trip_id", Dtype::Int, Stype::Id),
                        ("rider_id", Dtype::Int, Stype::Id),
                        ("driver_id", Dtype::Int, Stype::Id),
                        ("fare", Dtype::Float, Stype::Numerical),
                        ("started_at", Dtype::Time, Stype::Timestamp),
                    ],
                ),
            ],
            links: vec![
                link(&[("people", "person_id"), ("trips", "rider_id")]),
                link(&[("people", "person_id"), ("trips", "driver_id")]),
            ],
        },
        "schema-5" => QuerySchema {
            tables: vec![
                table(
                    "customers",
                    "customer_id",
                    None,
                    &[
                        ("customer_id", Dtype::Int, Stype::Id),
                        ("region", Dtype::String, Stype::Categorical),
                    ],
                ),
                table(
                    "orders",
                    "order_id",
                    Some("placed_at"),
                    &[
                        ("order_id", Dtype::Int, Stype::Id),
                        ("customer_id", Dtype::Int, Stype::Id),
                        ("total", Dtype::Float, Stype::Numerical),
                        ("placed_at", Dtype::Time, Stype::Timestamp),
                    ],
                ),
                table(
                    "items",
                    "item_id",
                    Some("added_at"),
                    &[
                        ("item_id", Dtype::Int, Stype::Id),
                        ("order_id", Dtype::Int, Stype::Id),
                        ("price", Dtype::Float, Stype::Numerical),
                        ("added_at", Dtype::Time, Stype::Timestamp),
                    ],
                ),
            ],
            links: vec![
                link(&[("customers", "customer_id"), ("orders", "customer_id")]),
                link(&[("items", "order_id"), ("orders", "order_id")]),
            ],
        },
        "schema-6" => QuerySchema {
            tables: vec![table(
                "employees",
                "employee_id",
                Some("hired_at"),
                &[
                    ("employee_id", Dtype::Int, Stype::Id),
                    ("manager_id", Dtype::Int, Stype::Id),
                    ("salary", Dtype::Float, Stype::Numerical),
                    ("hired_at", Dtype::Time, Stype::Timestamp),
                ],
            )],
            links: vec![link(&[
                ("employees", "employee_id"),
                ("employees", "manager_id"),
            ])],
        },
        "schema-7" => QuerySchema {
            tables: vec![
                table(
                    "members",
                    "member_id",
                    None,
                    &[
                        ("member_id", Dtype::Int, Stype::Id),
                        ("age", Dtype::Int, Stype::Numerical),
                    ],
                ),
                table(
                    "likes",
                    "like_id",
                    None,
                    &[
                        ("like_id", Dtype::Int, Stype::Id),
                        ("member_id", Dtype::Int, Stype::Id),
                        ("item_id", Dtype::Int, Stype::Id),
                    ],
                ),
                table(
                    "items",
                    "item_id",
                    None,
                    &[
                        ("item_id", Dtype::Int, Stype::Id),
                        ("name", Dtype::String, Stype::Text),
                    ],
                ),
            ],
            links: vec![
                link(&[("likes", "member_id"), ("members", "member_id")]),
                link(&[("items", "item_id"), ("likes", "item_id")]),
            ],
        },
        _ => panic!("unknown test schema {name}"),
    }
}
