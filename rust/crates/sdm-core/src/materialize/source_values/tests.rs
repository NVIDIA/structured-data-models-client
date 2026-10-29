// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn native_context_diagnostic_is_scoped_to_complete_empty_context_sentences() {
    for sentence in [
        Sentence::ContextTooRestrictive,
        Sentence::NoEntityAtAnchor,
        Sentence::NoEntityAtAnchorLater,
        Sentence::NoEntityPassed,
        Sentence::NoEntityPassedSeveral,
        Sentence::AnchorTooLate,
        Sentence::AnchorTooLateSeveral,
    ] {
        let raw = CoreError::new(sentence, "context is empty")
            .with_field("condition", Detail::text("users.age > 22"));
        let original = raw.clone();
        let selected = native_context_diagnostic(raw);
        assert_eq!(selected.sentence, original.sentence);
        assert_eq!(selected.code, original.code);
        assert_eq!(selected.fields["condition"], original.fields["condition"]);
        assert_eq!(
            selected.fields["native_key_reason"],
            Detail::text("task_validation")
        );
        assert_eq!(selected.public_details.as_deref().unwrap().len(), 0);
        assert!(original.public_details.is_none());
        assert!(!original.fields.contains_key("native_key_reason"));
    }
    let error = CoreError::new(Sentence::InvalidMessage, "invalid native input");
    let original = error.clone();
    assert_eq!(native_context_diagnostic(error), original);
}
