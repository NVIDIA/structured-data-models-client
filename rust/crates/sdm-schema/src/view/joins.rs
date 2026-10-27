// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which of a metric view's joins the conversion keeps (T3.5's M2), in the
//! old client's order (`_add_joins`, `DB/metric_view.py:825-909`): parents
//! before children, a kept join's nested joins visited after it, a skipped
//! join's not. A join is skipped, with the joins nested in it, if its name
//! is taken (`source`, or a join kept before it), if its source is no plain
//! table name, if its cardinality is unsupported, if it joins on several
//! columns, or if its condition is no single-column equality; the source and
//! the condition are the data layer's to judge, through the two judges the
//! caller hands in, each asked at its point in that order.

use super::metric::SOURCE_ALIAS;
use super::spec::{Cardinality, JoinKeys, RefusalReason, Source, TableItem, ViewRefusal, ViewSpec};
use super::yaml::NodeId;

/// Why a join is skipped; `R` is a judge's reason.
#[derive(Clone, Debug, PartialEq)]
pub enum SkipReason<R> {
    /// Its name is `source` or a join's kept before it.
    NameInUse,
    /// Its cardinality is neither `many_to_one` nor `one_to_many`.
    Cardinality(NodeId),
    /// It joins on more than one column.
    UsingSeveral,
    /// A judge refused its source or its condition.
    Judged(R),
}

/// A join skipped, with the joins nested in it, which go with it.
#[derive(Clone, Debug, PartialEq)]
pub struct SkippedJoin<R> {
    pub alias: String,
    pub reason: SkipReason<R>,
    pub nested: Vec<String>,
}

/// A join entry the conversion cannot read, in its place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinMessage {
    /// No mapping.
    NotAMapping(NodeId),
    /// A mapping with no text name.
    Nameless(NodeId),
}

/// What the walk decided.
#[derive(Clone, Debug, PartialEq)]
pub struct JoinWalk<R> {
    /// The kept joins, by their place in [`ViewSpec::tables`], in order.
    pub kept: Vec<usize>,
    pub skipped: Vec<SkippedJoin<R>>,
    /// The messages of the entries it cannot read, in order.
    pub messages: Vec<JoinMessage>,
    /// Every alias met, kept or skipped, nested names included, each once,
    /// in the order met.
    pub aliases: Vec<String>,
    /// For each alias skipped or nested in a skipped join, the skipped join
    /// it goes with, by its place in `skipped`; the first such kept.
    pub skipped_with: Vec<(String, usize)>,
}

/// Walks the joins of `spec`. `source` judges a join's `source` (absent,
/// or a node), `condition` its `on` (absent, or a node), each refusing with
/// its reason. A semantic view has no joins: its walk keeps and skips
/// none. A cardinality that is a list or a mapping refuses the view where
/// the old client failed on it.
pub fn walk<R>(
    spec: &mut ViewSpec,
    mut source: impl FnMut(&ViewSpec, Option<NodeId>) -> Result<(), R>,
    mut condition: impl FnMut(&ViewSpec, Option<NodeId>) -> Result<(), R>,
) -> Result<JoinWalk<R>, ViewRefusal> {
    let mut walk = JoinWalk {
        kept: Vec::new(),
        skipped: Vec::new(),
        messages: Vec::new(),
        aliases: Vec::new(),
        skipped_with: Vec::new(),
    };
    // The source table counts as kept, so its joins are visited.
    let mut visited = vec![false; spec.tables.len()];
    if let Some(first) = visited.first_mut() {
        *first = true;
    }
    let mut kept_names: Vec<String> = Vec::new();
    for at in 1..spec.tables.len() {
        let parent = match &spec.tables[at] {
            TableItem::NotAMapping { parent, .. } | TableItem::Nameless { parent, .. } => *parent,
            TableItem::Table(table) => table.parent.unwrap_or(0),
            TableItem::Declared { .. } => continue,
        };
        if !visited.get(parent).copied().unwrap_or(false) {
            continue;
        }
        let table = match spec.tables[at].clone() {
            TableItem::NotAMapping { cfg, .. } => {
                walk.messages.push(JoinMessage::NotAMapping(cfg));
                continue;
            }
            TableItem::Nameless { cfg, .. } => {
                walk.messages.push(JoinMessage::Nameless(cfg));
                continue;
            }
            TableItem::Declared { .. } => continue,
            TableItem::Table(table) => table,
        };
        // Only a metric view's join has keys and a source name.
        let (Some(join), Source::Name(node)) = (table.join.clone(), table.source.clone()) else {
            continue;
        };
        let judged = if table.alias == SOURCE_ALIAS || kept_names.contains(&table.alias) {
            Err(SkipReason::NameInUse)
        } else if let Err(reason) = source(spec, node) {
            Err(SkipReason::Judged(reason))
        } else {
            match join.cardinality {
                Cardinality::Unhashable => {
                    return Err(spec.refusal(RefusalReason::Unhashable("cardinality".into())));
                }
                Cardinality::Unsupported(node) => Err(SkipReason::Cardinality(node)),
                Cardinality::ManyToOne | Cardinality::OneToMany => match join.keys {
                    JoinKeys::UsingSeveral => Err(SkipReason::UsingSeveral),
                    JoinKeys::Using(_) => Ok(()),
                    JoinKeys::On(on) => condition(spec, on).map_err(SkipReason::Judged),
                },
            }
        };
        match judged {
            Ok(()) => {
                visited[at] = true;
                kept_names.push(table.alias.clone());
                walk.kept.push(at);
                add_alias(&mut walk.aliases, &table.alias);
            }
            Err(reason) => {
                let nested = spec.nested_names(table.cfg)?;
                let index = walk.skipped.len();
                for name in std::iter::once(&table.alias).chain(&nested) {
                    if !walk.skipped_with.iter().any(|(n, _)| n == name) {
                        walk.skipped_with.push((name.clone(), index));
                    }
                }
                add_alias(&mut walk.aliases, &table.alias);
                for name in &nested {
                    add_alias(&mut walk.aliases, name);
                }
                walk.skipped.push(SkippedJoin {
                    alias: table.alias.clone(),
                    reason,
                    nested,
                });
            }
        }
    }
    Ok(walk)
}

fn add_alias(aliases: &mut Vec<String>, alias: &str) {
    if !aliases.iter().any(|a| a == alias) {
        aliases.push(alias.to_owned());
    }
}
