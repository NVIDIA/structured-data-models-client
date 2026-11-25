// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keyword token spelling and multiword token boundaries.
use sdm_pql::lexer::{Kind, tokenize};

#[test]
fn keyword_spelling_and_multiword_tokens_are_deliberate() {
    use Kind::*;
    for (words, kind) in [
        ("predict PREDICT", Predict),
        ("count COUNT", Aggr),
        ("null NULL", Null),
        ("weeks WEEKS", TimeUnit),
        ("true True TRUE false False FALSE", Bool),
        ("IN in", RelOp),
        ("Predict Count Null Days tRUE Like", Id),
    ] {
        for word in words.split(' ') {
            let tokens = tokenize(&word.chars().collect::<Vec<_>>()).unwrap();
            assert_eq!(tokens.len(), 1, "{word}");
            assert_eq!(tokens[0].kind, kind, "{word}");
        }
    }
    for (text, expected) in [
        ("IS NOT", vec![RelOp]),
        ("is not", vec![RelOp]),
        ("IS  NOT", vec![RelOp, Not]),
        ("IS not", vec![RelOp, Not]),
        ("IS\tNOT", vec![RelOp, Not]),
        ("IS\nNOT", vec![RelOp, Not]),
    ] {
        let tokens = tokenize(&text.chars().collect::<Vec<_>>()).unwrap();
        assert_eq!(
            tokens.iter().map(|t| t.kind).collect::<Vec<_>>(),
            expected,
            "{text:?}"
        );
    }
}
