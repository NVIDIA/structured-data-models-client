// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-local reuse of exactly identical relational contexts.
use crate::run::Asker;
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence, run::Endpoint};
use sdm_request::{frame_session::subset, predict};
use sdm_wire::json::Value;

pub(crate) struct Session {
    create: Value,
    current: Option<String>,
    known: Vec<String>,
    recreated: bool,
    disabled: bool,
}

fn same_context(left: &Value, right: &Value) -> bool {
    // Compare borrowed wire values: no extra copy of a potentially large context.
    // Object-order differences only disable this optimization.
    if ["model", "task", "schema", "context", "output", "inference"]
        .iter()
        .any(|field| left.get(field) != right.get(field))
    {
        return false;
    }
    match (left.get("metadata"), right.get("metadata")) {
        (Some(Value::Object(left)), Some(Value::Object(right))) => left
            .iter()
            .filter(|(key, _)| key != "request_id")
            .eq(right.iter().filter(|(key, _)| key != "request_id")),
        (left, right) => left == right,
    }
}

impl Session {
    pub(crate) fn eligible(bodies: &[Value], enabled: bool) -> Option<Self> {
        if !enabled
            || bodies.len() < 2
            || bodies[0].get("model").and_then(Value::as_text) != Some(predict::RELATIONAL)
        {
            return None;
        }
        if bodies[0]
            .get("task")
            .and_then(|task| task.get("kind"))
            .and_then(Value::as_text)
            == Some("forecasting")
        {
            return None;
        }
        if bodies[1..]
            .iter()
            .any(|body| !same_context(&bodies[0], body))
        {
            return None;
        }
        Some(Self {
            create: subset(
                &bodies[0],
                &["model", "task", "schema", "context", "metadata"],
            ),
            current: None,
            known: vec![],
            recreated: false,
            disabled: false,
        })
    }

    async fn open(&mut self, asker: &Asker) -> Result<bool, CoreError> {
        let answer =
            match crate::frame_session::call(asker, Endpoint::CreateSession, self.create.clone())
                .await
            {
                Ok(answer) => answer,
                Err(error)
                    if [404, 405, 501]
                        .iter()
                        .any(|status| has_status(&error, *status)) =>
                {
                    self.disabled = true;
                    return Ok(false);
                }
                Err(error) => return Err(error),
            };
        let id = crate::frame_session::session_id(&answer)?.to_owned();
        self.known.push(id.clone());
        self.current = Some(id);
        Ok(true)
    }

    /// The session the batches are scored against, once it is open.
    pub(crate) fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// None requests ordinary stateless fallback, never another retry layer.
    pub(crate) async fn score(
        &mut self,
        asker: &Asker,
        body: &Value,
    ) -> Result<Option<Value>, CoreError> {
        if self.disabled || (self.current.is_none() && !self.open(asker).await?) {
            return Ok(None);
        }
        let scoring = subset(body, &["predict", "output", "inference", "metadata"]);
        let endpoint = Endpoint::SessionPredict {
            session_id: self.current.clone().unwrap(),
        };
        match crate::frame_session::call(asker, endpoint, scoring.clone()).await {
            Ok(answer) => Ok(Some(answer)),
            Err(error) if has_status(&error, 404) && !self.recreated => {
                self.recreated = true;
                self.current = None;
                if !self.open(asker).await? {
                    return Ok(None);
                }
                crate::frame_session::call(
                    asker,
                    Endpoint::SessionPredict {
                        session_id: self.current.clone().unwrap(),
                    },
                    scoring,
                )
                .await
                .map(Some)
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) async fn close(self, asker: &Asker) {
        for session_id in self.known {
            if let Err(error) = crate::frame_session::call(
                asker,
                Endpoint::DeleteSession { session_id },
                Value::Object(vec![]),
            )
            .await
                && !has_status(&error, 404)
            {
                asker.warn(CoreWarning::new(
                    Sentence::SessionCleanup,
                    "temporary session cleanup failed",
                ));
            }
        }
    }
}

fn has_status(error: &CoreError, status: u16) -> bool {
    error.fields.get("status") == Some(&Detail::Scalar(Scalar::UInt16(status)))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn body() -> Value {
        sdm_wire::json::read(br#"{"model":"kumo-relational","task":{"kind":"regression"},"schema":{"destinations":[1,2]},"context":{"rows":[1,2]},"predict":{"rows":[3]},"inference":{"seed":42},"output":["prediction"],"metadata":{"request_id":"a"}}"#).unwrap()
    }
    #[test]
    fn exact_reuse_ignores_only_prediction_and_correlation() {
        let original = body();
        let mut next = original.clone();
        next.set("predict", Value::Null);
        next.set(
            "metadata",
            sdm_wire::json::read(br#"{"request_id":"b"}"#).unwrap(),
        );
        assert!(Session::eligible(&[original.clone(), next.clone()], true).is_some());
        assert!(Session::eligible(&[original.clone(), next.clone()], false).is_none());
        for field in ["task", "schema", "context", "inference", "output"] {
            let mut changed = next.clone();
            changed.set(field, Value::Null);
            assert!(
                Session::eligible(&[original.clone(), changed], true).is_none(),
                "{field}"
            );
        }
        next.set(
            "metadata",
            sdm_wire::json::read(br#"{"request_id":"b","other":1}"#).unwrap(),
        );
        assert!(Session::eligible(&[original, next], true).is_none());
    }
}
