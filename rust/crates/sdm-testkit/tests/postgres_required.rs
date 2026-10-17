// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Postgres service a Postgres job runs against is there (H26; T2.1a),
//! the Rust side of `contract/tests/test_postgres_required.py`.
//!
//! A job that runs the core on Postgres sets `SDM_REQUIRE_POSTGRES` and
//! names its server in `SDM_POSTGRES_TEST_DSN` (a `postgresql://` URI or a
//! `key=value` conninfo). There, a missing DSN or a server that does not
//! answer as Postgres fails by name: the test never stands down. It asks
//! with the protocol's own SSLRequest, which every Postgres answers with
//! one byte, `S` or `N`, before any authentication, so it needs no driver.
//! Where `SDM_REQUIRE_POSTGRES` is not set, nothing is required.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// The host and port a DSN names, libpq's defaults filling the gaps.
fn host_and_port(dsn: &str) -> (String, u16) {
    let (mut host, mut port) = (String::from("localhost"), 5432_u16);
    if let Some(rest) = dsn
        .strip_prefix("postgresql://")
        .or_else(|| dsn.strip_prefix("postgres://"))
    {
        let authority = rest.split(['/', '?']).next().unwrap_or_default();
        let address = authority.rsplit('@').next().unwrap_or_default();
        let (name, number) = match address.rsplit_once(':') {
            Some((name, number)) => (name, Some(number)),
            None => (address, None),
        };
        if !name.is_empty() {
            host = name.to_owned();
        }
        if let Some(number) = number {
            port = number.parse().expect("the DSN's port is a number");
        }
    } else {
        for pair in dsn.split_whitespace() {
            match pair.split_once('=') {
                Some(("host", value)) => host = value.to_owned(),
                Some(("port", value)) => {
                    port = value.parse().expect("the DSN's port is a number");
                }
                _ => {}
            }
        }
    }
    (host, port)
}

/// Whether a Postgres server answers at the address, within the timeout.
fn answers_as_postgres(host: &str, port: u16) -> Result<(), String> {
    let timeout = Duration::from_secs(10);
    let addresses = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("{host}:{port} does not resolve: {error}"))?;
    let mut last = format!("{host}:{port} resolves to no address");
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(mut stream) => {
                stream
                    .set_read_timeout(Some(timeout))
                    .map_err(|e| e.to_string())?;
                // SSLRequest: length 8, code 80877103.
                let mut request = 8_i32.to_be_bytes().to_vec();
                request.extend_from_slice(&80_877_103_i32.to_be_bytes());
                stream.write_all(&request).map_err(|e| e.to_string())?;
                let mut reply = [0_u8; 1];
                stream
                    .read_exact(&mut reply)
                    .map_err(|e| format!("{address}: no answer: {e}"))?;
                return if matches!(reply[0], b'S' | b'N') {
                    Ok(())
                } else {
                    Err(format!(
                        "{address}: answered {:?}, not as Postgres",
                        reply[0]
                    ))
                };
            }
            Err(error) => last = format!("{address}: {error}"),
        }
    }
    Err(last)
}

#[test]
fn the_postgres_service_is_reachable_where_required() {
    if std::env::var_os("SDM_REQUIRE_POSTGRES").is_none_or(|v| v.is_empty()) {
        return;
    }
    let dsn = std::env::var("SDM_POSTGRES_TEST_DSN")
        .ok()
        .filter(|dsn| !dsn.is_empty())
        .expect("SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not");
    let (host, port) = host_and_port(&dsn);
    if let Err(problem) = answers_as_postgres(&host, port) {
        panic!(
            "SDM_REQUIRE_POSTGRES is set and the Postgres service at \
             SDM_POSTGRES_TEST_DSN is unreachable: {problem}"
        );
    }
}

#[test]
fn a_dsn_names_its_host_and_port() {
    assert_eq!(
        host_and_port("postgresql://user@db.invalid:6543/sdm_test"),
        ("db.invalid".to_owned(), 6543)
    );
    assert_eq!(
        host_and_port("postgresql://localhost/sdm_test?sslmode=disable"),
        ("localhost".to_owned(), 5432)
    );
    assert_eq!(
        host_and_port("host=10.0.0.1 port=5433 dbname=sdm_test"),
        ("10.0.0.1".to_owned(), 5433)
    );
    assert_eq!(
        host_and_port("dbname=sdm_test"),
        ("localhost".to_owned(), 5432)
    );
}

#[test]
fn a_closed_port_is_unreachable() {
    // Seeded: a port nothing listens on fails, so the check above can.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    assert!(answers_as_postgres("127.0.0.1", port).is_err());
}
