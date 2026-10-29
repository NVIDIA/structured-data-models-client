// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless native builtin value storage shared by response products.
//! This postorder value codec carries data, never native actions or owners.

use sdm_contract::scalar::native_value::{
    NativeObjectKey, NativeTreeMember, NativeTreeNode, NativeValue,
};
use sdm_contract::scalar::unicode::UnicodeText;
use sdm_contract::{CoreError, Sentence};
use sdm_wire::json::{Value, is_integer_token};

fn protocol(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// Exact generated-decoder data, with no decimal display-limit observation.
pub fn native(value: &Value) -> Result<NativeValue, CoreError> {
    Ok(match value {
        Value::Null => NativeValue::Null,
        Value::Bool(value) => NativeValue::Boolean { value: *value },
        Value::Number(digits) if is_integer_token(digits) => NativeValue::Integer {
            digits: sdm_wire::response::integer_text(digits)
                .ok_or_else(|| protocol("decoded native integer is unreadable"))?,
        },
        Value::Number(token) => NativeValue::Float {
            bits: token
                .parse::<f64>()
                .map_err(|_| protocol("decoded native number is unreadable"))?
                .to_be_bytes(),
        },
        Value::Text(text) => NativeValue::Unicode {
            points: text.points().collect(),
        },
        Value::Array(_) | Value::Object(_) => native_tree(value)?,
    })
}

/// Flatten actual decoded containers without doubling their JSON nesting.
/// This is a postorder value tree, not a parser or a native action protocol.
fn native_tree(value: &Value) -> Result<NativeValue, CoreError> {
    enum Visit<'a> {
        Value(&'a Value),
        Array(usize),
        Object(&'a [(String, Value)]),
    }
    let mut pending = vec![Visit::Value(value)];
    let mut roots = Vec::new();
    let mut nodes = Vec::new();
    while let Some(visit) = pending.pop() {
        let node =
            match visit {
                Visit::Value(Value::Array(items)) => {
                    pending.push(Visit::Array(items.len()));
                    pending.extend(items.iter().rev().map(Visit::Value));
                    continue;
                }
                Visit::Value(Value::Object(members)) => {
                    pending.push(Visit::Object(members));
                    pending.extend(members.iter().rev().map(|(_, value)| Visit::Value(value)));
                    continue;
                }
                Visit::Value(value) => NativeTreeNode::Scalar {
                    value: native(value)?,
                },
                Visit::Array(length) => NativeTreeNode::Array {
                    children: roots.split_off(roots.len().checked_sub(length).ok_or_else(
                        || protocol("decoded native array has inconsistent child data"),
                    )?),
                },
                Visit::Object(members) => {
                    let children =
                        roots.split_off(roots.len().checked_sub(members.len()).ok_or_else(
                            || protocol("decoded native object has inconsistent member data"),
                        )?);
                    NativeTreeNode::Object {
                        members: members
                            .iter()
                            .zip(children)
                            .map(|((key, _), value)| NativeTreeMember {
                                key: NativeObjectKey::Unicode {
                                    points: key.chars().map(u32::from).collect(),
                                },
                                value,
                            })
                            .collect(),
                    }
                }
            };
        let ordinal = u32::try_from(nodes.len())
            .map_err(|_| protocol("decoded native tree exceeds local node ordinals"))?;
        nodes.push(node);
        roots.push(ordinal);
    }
    Ok(NativeValue::Tree { nodes })
}

pub(crate) fn text(points: &[u32]) -> Result<UnicodeText, CoreError> {
    UnicodeText::from_points(points.to_vec())
        .map_err(|_| protocol("native response text has an invalid Unicode point"))
}

pub fn logical(value: &NativeValue) -> Result<Value, CoreError> {
    Ok(match value {
        NativeValue::Null => Value::Null,
        NativeValue::Boolean { value } => Value::Bool(*value),
        NativeValue::Integer { digits } => {
            if sdm_wire::response::integer_text(digits).as_deref() != Some(digits.as_str()) {
                return Err(protocol("native response integer is not canonical"));
            }
            Value::Number(digits.clone())
        }
        NativeValue::Float { bits } => {
            let value = f64::from_be_bytes(*bits);
            Value::Number(if value.is_nan() {
                "NaN".into()
            } else if value == f64::INFINITY {
                "Infinity".into()
            } else if value == f64::NEG_INFINITY {
                "-Infinity".into()
            } else {
                // A native float must remain a float, including -0.0.
                let token = value.to_string();
                if token.contains('.') || token.contains('e') || token.contains('E') {
                    token
                } else {
                    format!("{token}.0")
                }
            })
        }
        NativeValue::Unicode { points } => Value::Text(text(points)?),
        NativeValue::Array { items } => {
            Value::Array(items.iter().map(logical).collect::<Result<_, _>>()?)
        }
        NativeValue::Object { members } => Value::Object(
            members
                .iter()
                .map(|member| {
                    let NativeObjectKey::Unicode { points } = &member.key;
                    let key = text(points)?;
                    Ok((
                        key.utf8()
                            .ok_or_else(|| {
                                protocol("decoded response member has exceptional text")
                            })?
                            .to_owned(),
                        logical(&member.value)?,
                    ))
                })
                .collect::<Result<_, CoreError>>()?,
        ),
        NativeValue::Tree { nodes } => logical_tree(nodes)?,
    })
}

fn logical_tree(nodes: &[NativeTreeNode]) -> Result<Value, CoreError> {
    if nodes
        .last()
        .is_none_or(|node| matches!(node, NativeTreeNode::Scalar { .. }))
    {
        return Err(protocol(
            "native container tree lacks its final container root",
        ));
    }
    let mut values: Vec<Option<Value>> = Vec::with_capacity(nodes.len());
    let mut depths = Vec::with_capacity(nodes.len());
    let mut starts = Vec::with_capacity(nodes.len());
    for (ordinal, node) in nodes.iter().enumerate() {
        let children = match node {
            NativeTreeNode::Scalar { value } => {
                if matches!(
                    value,
                    NativeValue::Array { .. }
                        | NativeValue::Object { .. }
                        | NativeValue::Tree { .. }
                ) {
                    return Err(protocol(
                        "native tree scalar contains another container owner",
                    ));
                }
                values.push(Some(logical(value)?));
                depths.push(0usize);
                starts.push(ordinal);
                continue;
            }
            NativeTreeNode::Array { children } => children.clone(),
            NativeTreeNode::Object { members } => {
                members.iter().map(|member| member.value).collect()
            }
        };
        if children.windows(2).any(|pair| pair[0] >= pair[1])
            || children.iter().any(|child| *child as usize >= ordinal)
        {
            return Err(protocol(
                "native tree children are not an ordered earlier population",
            ));
        }
        let start = children
            .first()
            .map_or(ordinal, |child| starts[*child as usize]);
        let mut next = start;
        for child in &children {
            if starts[*child as usize] != next {
                return Err(protocol(
                    "native tree interleaves separate child populations",
                ));
            }
            next = *child as usize + 1;
        }
        if next != ordinal {
            return Err(protocol(
                "native tree omits an intervening child population",
            ));
        }
        let depth = 1 + children
            .iter()
            .map(|child| depths[*child as usize])
            .max()
            .unwrap_or(0);
        if depth > sdm_wire::json::MAX_DEPTH {
            return Err(protocol(
                "native tree exceeds the existing logical value depth",
            ));
        }
        let items = children
            .iter()
            .map(|child| {
                values[*child as usize].take().ok_or_else(|| {
                    protocol("native tree reuses a consumed child or mixes container owners")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = match node {
            NativeTreeNode::Array { .. } => Value::Array(items),
            NativeTreeNode::Object { members } => Value::Object(
                members
                    .iter()
                    .zip(items)
                    .map(|(member, value)| {
                        let NativeObjectKey::Unicode { points } = &member.key;
                        let key = text(points)?;
                        let key = key.utf8().ok_or_else(|| {
                            protocol("decoded native member has exceptional text")
                        })?;
                        Ok((key.to_owned(), value))
                    })
                    .collect::<Result<_, CoreError>>()?,
            ),
            NativeTreeNode::Scalar { .. } => unreachable!("scalar handled before child reads"),
        };
        values.push(Some(value));
        depths.push(depth);
        starts.push(start);
    }
    let root = values
        .pop()
        .flatten()
        .ok_or_else(|| protocol("native tree has no final root"))?;
    if values.iter().any(Option::is_some) {
        return Err(protocol("native tree contains unreachable node owners"));
    }
    Ok(root)
}
