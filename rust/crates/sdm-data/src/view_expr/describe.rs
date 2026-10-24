// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What `DESCRIBE TABLE EXTENDED` says of a metric view (T3.5, M1,
//! old-behaviour rows 21 to 24), read from its rows as the old client reads
//! them (`DB/metric_view.py:1389-1475`): the columns first, up to a blank
//! row, then sections headed `# Name` with an empty second cell. Only the
//! first two cells are read, since a SQL warehouse gives four and a Spark
//! session three, and an empty cell may be `''` or null. Each section is
//! read on its own, so a parameter named like a detail, `Type` say, cannot
//! stand in for it.
//!
//! The statement itself is the Databricks dialect's, written with the rest
//! of its catalog statements once that dialect lands (T3.4).

/// What the description says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewDescription {
    /// The YAML definition (`View Text`).
    pub definition: String,
    /// Each dimension's type, by name, in order: the column rows whose
    /// type does not end in ` measure`.
    pub dtypes: Vec<(String, String)>,
    /// Each parameter's type, by name, in order (`# Parameters`).
    pub parameters: Vec<(String, String)>,
    /// The view's catalog and schema (`Catalog`, `Database`), if given.
    pub catalog: Option<String>,
    pub schema: Option<String>,
}

/// Why the relation described is no metric view the conversion reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DescribeRefusal {
    /// Its `Type` is not `METRIC_VIEW`: what it is, if it says.
    NotAMetricView { got: Option<String> },
    /// Its `View Text` is empty or missing.
    NoDefinition,
}

/// The rows of a section, by the section's lower-case header without its
/// `#`; the columns under the empty header.
fn sections(rows: &[Vec<Option<String>>]) -> Vec<(String, Vec<(String, String)>)> {
    let mut sections: Vec<(String, Vec<(String, String)>)> = vec![(String::new(), Vec::new())];
    let mut current: Option<usize> = Some(0);
    for row in rows {
        let name = row
            .first()
            .and_then(Option::as_deref)
            .map_or(String::new(), |cell| super::stripped(cell).to_owned());
        let value = row
            .get(1)
            .and_then(Option::as_deref)
            .unwrap_or("")
            .to_owned();
        if name.starts_with('#') && super::stripped(&value).is_empty() {
            let header = super::stripped(name.trim_start_matches('#')).to_lowercase();
            // Spark heads the rows of some sections with `# col_name`.
            if header != "col_name" {
                current = Some(match sections.iter().position(|(h, _)| *h == header) {
                    Some(at) => at,
                    None => {
                        sections.push((header, Vec::new()));
                        sections.len() - 1
                    }
                });
            }
            continue;
        }
        if name.is_empty() {
            // A blank row ends the columns; rows up to the next header
            // belong to no section.
            if current == Some(0) {
                current = None;
            }
            continue;
        }
        if let Some(at) = current {
            sections[at].1.push((name, value));
        }
    }
    sections
}

/// Pairs into a mapping where a later name replaces an earlier one's
/// value in its place, as Python's `dict` does.
fn by_name(pairs: impl IntoIterator<Item = (String, String)>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (name, value) in pairs {
        match out.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => out.push((name, value)),
        }
    }
    out
}

/// Reads a metric view's description from the rows of `DESCRIBE TABLE
/// EXTENDED`, each its cells as text or null (`read_metric_view_definition`
/// after its statement).
pub fn describe_view(rows: &[Vec<Option<String>>]) -> Result<ViewDescription, DescribeRefusal> {
    let sections = sections(rows);
    let section = |header: &str| {
        sections
            .iter()
            .find(|(h, _)| h == header)
            .map_or(&[][..], |(_, rows)| rows.as_slice())
    };
    let dtypes = by_name(
        section("")
            .iter()
            .filter(|(_, data_type)| !data_type.ends_with(" measure"))
            .cloned(),
    );
    let parameters = by_name(section("parameters").iter().cloned());
    // The first value of each detail wins.
    let detail = |name: &str| {
        section("detailed table information")
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, value)| value.clone())
    };
    let kind = detail("Type");
    if kind.as_deref() != Some("METRIC_VIEW") {
        return Err(DescribeRefusal::NotAMetricView { got: kind });
    }
    let definition = detail("View Text")
        .filter(|text| !text.is_empty())
        .ok_or(DescribeRefusal::NoDefinition)?;
    Ok(ViewDescription {
        definition,
        dtypes,
        parameters,
        catalog: detail("Catalog"),
        schema: detail("Database"),
    })
}
