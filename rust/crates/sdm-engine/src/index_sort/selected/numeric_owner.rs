// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reached native numeric observations around one retained ordering future.
use super::{invalid, numeric_objects as numeric, object_sort};
use numeric::{Number, Owner};
use sdm_contract::CoreError;
use sdm_contract::numeric_owner::*;
use sdm_contract::tabular::{
    NumericComparisonProfile as Policy, TabularIndexScalar as Atom, TabularSortProfile,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

/// Attachments have been decoded by the composition boundary, never the engine.
pub enum Observation {
    Control(NumericOwnerOutcome),
    Captured {
        owners: Vec<NumericOwnerFact>,
        positions: Vec<usize>,
        missing: Vec<usize>,
    },
}
pub trait Observer: Send {
    fn observe(
        &mut self,
        request: NumericOwnerRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Observation, CoreError>> + Send + '_>>;
}
enum Failure {
    Protocol(CoreError),
    Native(ExceptionId, NumericExceptionClass),
}
impl From<CoreError> for Failure {
    fn from(value: CoreError) -> Self {
        Self::Protocol(value)
    }
}
struct OrderingRun<O> {
    observer: O,
    owners: BTreeMap<OwnerId, Atom>,
    input: OrderNumericOwners,
}
/// Own the captured copy while lending one comparison to the shared sorter.
struct SortComparator<'a, O> {
    run: &'a mut OrderingRun<O>,
    owners: Vec<NumericOwnerFact>,
}
impl<O: Observer> object_sort::Comparator<Failure> for SortComparator<'_, O> {
    async fn less(&mut self, left: usize, right: usize) -> Result<bool, Failure> {
        if self
            .run
            .compare(&self.owners[left], &self.owners[right])
            .await?
            == Some(Ordering::Less)
        {
            return Ok(true);
        }
        let _ = self
            .run
            .compare(&self.owners[left], &self.owners[right])
            .await?
            == Some(Ordering::Greater);
        Ok(false)
    }
}

impl<O: Observer> OrderingRun<O> {
    async fn ask(&mut self, request: NumericOwnerRequest) -> Result<Observation, Failure> {
        let result = self.observer.observe(request).await?;
        if let Observation::Control(NumericOwnerOutcome::Raised { exception, class }) = result {
            if exception.0 == 0 {
                return Err(invalid().into());
            }
            return Err(Failure::Native(exception, class));
        }
        Ok(result)
    }
    fn retain(&mut self, owner: OwnerId, fact: Atom) -> Result<NumericOwnerFact, Failure> {
        if owner.0 == 0 {
            return Err(invalid().into());
        }
        numeric::observe(&fact, true)?;
        if let Some(previous) = self.owners.get(&owner) {
            if previous != &fact {
                return Err(invalid().into());
            }
        } else {
            self.owners.insert(owner, fact.clone());
        }
        Ok(NumericOwnerFact { owner, fact })
    }
    async fn load(&mut self, position: usize) -> Result<NumericOwnerFact, Failure> {
        match self
            .ask(NumericOwnerRequest::LoadProbeOwner { position })
            .await?
        {
            Observation::Control(NumericOwnerOutcome::ProbeOwner { owner, fact }) => {
                self.retain(owner, fact)
            }
            _ => Err(invalid().into()),
        }
    }
    async fn policy(&mut self) -> Result<Policy, Failure> {
        let Observation::Control(NumericOwnerOutcome::ComparisonEnvironment { policy }) = self
            .ask(NumericOwnerRequest::ObserveComparisonEnvironment)
            .await?
        else {
            return Err(invalid().into());
        };
        let valid = match self.input.source_rules {
            NumericSourceRules::CommonTypeOnly => policy == Policy::CommonType,
            NumericSourceRules::MutableFixedIntegerPairs => matches!(
                policy,
                Policy::ExactIntegerPairs | Policy::WeakFixedIntegerPairs
            ),
            NumericSourceRules::MutableExactUnboundedIntegers => matches!(
                policy,
                Policy::LegacyExactUnboundedIntegers | Policy::WeakScalars
            ),
            NumericSourceRules::FixedWeakScalars => policy == Policy::WeakScalars,
        };
        if !valid {
            return Err(invalid().into());
        }
        Ok(policy)
    }
    async fn convert(
        &mut self,
        value: &NumericOwnerFact,
        destination: NumericDestination,
    ) -> Result<Atom, Failure> {
        let key = numeric::observe(&value.fact, true)?;
        let overflow_possible = if let NumericDestination::BinaryFloat { width } = destination {
            numeric::as_double(key).is_ok_and(|v| {
                v.is_finite()
                    && match width {
                        2 => half::f16::from_f64(v).is_infinite(),
                        4 => (v as f32).is_infinite(),
                        _ => false,
                    }
            })
        } else {
            false
        };
        let Observation::Control(NumericOwnerOutcome::Converted { bits }) = self
            .ask(NumericOwnerRequest::ConvertOwner {
                owner: value.owner,
                destination,
                overflow_possible,
            })
            .await?
        else {
            return Err(invalid().into());
        };
        let (width, atom) = match destination {
            NumericDestination::BinaryFloat { width } => {
                (width, Atom::FixedFloating { width, bits })
            }
            NumericDestination::SignedInteger { width } => {
                let shift = 64 - u32::from(width) * 8;
                (
                    width,
                    Atom::FixedInteger {
                        signed: true,
                        width,
                        value: (((bits << shift) as i64) >> shift).to_string(),
                    },
                )
            }
            NumericDestination::UnsignedInteger { width } => (
                width,
                Atom::FixedInteger {
                    signed: false,
                    width,
                    value: bits.to_string(),
                },
            ),
        };
        if width < 8 && bits >> (u32::from(width) * 8) != 0 {
            return Err(invalid().into());
        }
        numeric::observe(&atom, true)?;
        Ok(atom)
    }
    async fn compare(
        &mut self,
        left: &NumericOwnerFact,
        right: &NumericOwnerFact,
    ) -> Result<Option<Ordering>, Failure> {
        let policy = self.policy().await?;
        let a = numeric::observe(&left.fact, true)?;
        let b = numeric::observe(&right.fact, true)?;
        let weak = matches!(policy, Policy::WeakScalars | Policy::WeakFixedIntegerPairs);
        let mut left_atom = left.fact.clone();
        let mut right_atom = right.fact.clone();
        // A reached source conversion acts on the native owner, not its serialized value.
        if weak {
            for (fixed, other, retained, output) in
                [(a, b, right, &mut right_atom), (b, a, left, &mut left_atom)]
            {
                let destination = match (fixed.owner, other.owner) {
                    (Owner::Floating { width }, Owner::UnboundedInteger) => {
                        Some(NumericDestination::BinaryFloat { width })
                    }
                    (Owner::Floating { width }, Owner::Binary64) if width < 8 => {
                        Some(NumericDestination::BinaryFloat { width })
                    }
                    (Owner::Integer { signed, width }, Owner::UnboundedInteger)
                        if policy == Policy::WeakFixedIntegerPairs =>
                    {
                        let Number::Integer(text) = other.number else {
                            unreachable!()
                        };
                        if signed && width == 8 && text.parse::<i64>().is_ok() {
                            None
                        } else if signed {
                            Some(NumericDestination::SignedInteger { width })
                        } else {
                            Some(NumericDestination::UnsignedInteger { width })
                        }
                    }
                    _ => None,
                };
                if let Some(destination) = destination {
                    *output = self.convert(retained, destination).await?;
                }
            }
        }
        let effective = match policy {
            Policy::WeakFixedIntegerPairs => Policy::WeakScalars,
            Policy::LegacyExactUnboundedIntegers => Policy::ExactIntegerPairs,
            other => other,
        };
        numeric::compare(
            numeric::observe(&left_atom, true)?,
            numeric::observe(&right_atom, true)?,
            effective,
        )
        .map_err(|_| Failure::Protocol(invalid()))
    }
}
impl<O: Observer> OrderingRun<O> {
    async fn monotonic(&mut self) -> Result<bool, Failure> {
        if self.input.length == 0 {
            return Ok(true);
        }
        let mut previous = self.load(0).await?;
        let self_equal = if let Atom::NativeMissing { owner } = previous.fact {
            owner == sdm_contract::tabular::NativeMissingOwner::None
        } else {
            self.compare(&previous, &previous).await? == Some(Ordering::Equal)
        };
        if !self_equal {
            return Ok(false);
        }
        if self.input.length == 1 {
            return Ok(true);
        }
        let (mut increasing, mut decreasing) = (true, true);
        for position in 1..self.input.length {
            let current = self.load(position).await?;
            if self.compare(&current, &previous).await? == Some(Ordering::Less) {
                increasing = false;
            } else if self.compare(&current, &previous).await? == Some(Ordering::Greater) {
                decreasing = false;
            } else if self.compare(&current, &previous).await? != Some(Ordering::Equal) {
                increasing = false;
                decreasing = false;
            }
            if !increasing && !decreasing {
                break;
            }
            previous = current;
        }
        Ok(increasing)
    }
    async fn run(&mut self) -> Result<OrderedNumericOwners, Failure> {
        let monotonic = match self.monotonic().await {
            Ok(value) => value,
            Err(Failure::Native(_, class))
                if matches!(
                    class,
                    NumericExceptionClass::TypeError | NumericExceptionClass::ValueError
                ) || class == NumericExceptionClass::DecimalInvalidOperation
                    && self.input.probe_exception_policy
                        == ProbeExceptionPolicy::TypeValueAndDecimalInvalid =>
            {
                false
            }
            Err(error) => return Err(error),
        };
        if monotonic {
            return Ok(OrderedNumericOwners::Sorted {
                positions: (0..self.input.length).collect(),
                already_ordered: true,
            });
        }
        let Observation::Captured {
            owners,
            positions,
            missing,
        } = self.ask(NumericOwnerRequest::CaptureSortOwners).await?
        else {
            return Err(invalid().into());
        };
        if owners.len() != positions.len()
            || positions.len().checked_add(missing.len()) != Some(self.input.length)
            || positions.windows(2).any(|pair| pair[0] >= pair[1])
            || missing.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid().into());
        }
        let mut seen = vec![false; self.input.length];
        for &position in positions.iter().chain(&missing) {
            let Some(slot) = seen.get_mut(position) else {
                return Err(invalid().into());
            };
            if *slot {
                return Err(invalid().into());
            }
            *slot = true;
        }
        for owner in &owners {
            let key = numeric::observe(&owner.fact, false)?;
            if matches!(key.number,Number::Floating(value) if value.is_nan()) {
                return Err(invalid().into());
            }
            self.retain(owner.owner, owner.fact.clone())?;
        }
        let permutation =
            object_sort::argsort_async(owners.len(), SortComparator { run: self, owners }).await?;
        let mut result: Vec<_> = permutation.into_iter().map(|i| positions[i]).collect();
        result.extend(missing);
        Ok(OrderedNumericOwners::Sorted {
            positions: result,
            already_ordered: false,
        })
    }
}
pub async fn order<O: Observer>(
    input: OrderNumericOwners,
    observer: O,
) -> Result<OrderedNumericOwners, CoreError> {
    if input.sort_profile != TabularSortProfile::Scalar {
        return Err(invalid());
    }
    let mut state = OrderingRun {
        observer,
        owners: BTreeMap::new(),
        input,
    };
    match state.run().await {
        Ok(value) => Ok(value),
        Err(Failure::Protocol(error)) => Err(error),
        Err(Failure::Native(exception, _)) => Ok(OrderedNumericOwners::Propagate { exception }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::task::{Context, Poll, Waker};
    struct Script(VecDeque<(NumericOwnerRequest, Observation)>);
    impl Observer for Script {
        fn observe(
            &mut self,
            request: NumericOwnerRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Observation, CoreError>> + Send + '_>> {
            Box::pin(async move {
                let (expected, result) = self.0.pop_front().expect("unexpected extra observation");
                assert_eq!(request, expected);
                Ok(result)
            })
        }
    }
    fn input(length: usize) -> OrderNumericOwners {
        OrderNumericOwners {
            length,
            sort_profile: TabularSortProfile::Scalar,
            source_rules: NumericSourceRules::FixedWeakScalars,
            probe_exception_policy: ProbeExceptionPolicy::TypeAndValue,
        }
    }
    fn run(input: OrderNumericOwners, script: Script) -> Result<OrderedNumericOwners, CoreError> {
        let future = order(input, script);
        let mut future = std::pin::pin!(future);
        let Poll::Ready(result) = future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        else {
            panic!("script observations are immediately ready")
        };
        result
    }
    #[test]
    fn reused_owner_identity_cannot_change_immutable_facts() {
        let first = Observation::Control(NumericOwnerOutcome::ProbeOwner {
            owner: OwnerId(1),
            fact: Atom::Integer { value: "2".into() },
        });
        let second = Observation::Control(NumericOwnerOutcome::ProbeOwner {
            owner: OwnerId(1),
            fact: Atom::Integer { value: "1".into() },
        });
        assert!(
            run(
                input(2),
                Script(VecDeque::from([
                    (NumericOwnerRequest::LoadProbeOwner { position: 0 }, first),
                    (
                        NumericOwnerRequest::ObserveComparisonEnvironment,
                        Observation::Control(NumericOwnerOutcome::ComparisonEnvironment {
                            policy: Policy::WeakScalars
                        })
                    ),
                    (NumericOwnerRequest::LoadProbeOwner { position: 1 }, second)
                ]))
            )
            .is_err()
        );
    }
    #[test]
    fn retained_native_exception_is_propagated_without_capture_or_restart() {
        let exception = ExceptionId(7);
        let script = Script(VecDeque::from([(
            NumericOwnerRequest::LoadProbeOwner { position: 0 },
            Observation::Control(NumericOwnerOutcome::Raised {
                exception,
                class: NumericExceptionClass::Other,
            }),
        )]));
        assert_eq!(
            run(input(1), script).unwrap(),
            OrderedNumericOwners::Propagate { exception }
        );
    }
    #[test]
    fn leading_nan_stops_probe_before_loading_the_next_owner() {
        for fact in [
            Atom::Floating {
                bits: f64::NAN.to_bits(),
            },
            Atom::FixedFloating {
                width: 4,
                bits: u64::from(f32::NAN.to_bits()),
            },
        ] {
            let script = Script(VecDeque::from([
                (
                    NumericOwnerRequest::LoadProbeOwner { position: 0 },
                    Observation::Control(NumericOwnerOutcome::ProbeOwner {
                        owner: OwnerId(1),
                        fact,
                    }),
                ),
                (
                    NumericOwnerRequest::ObserveComparisonEnvironment,
                    Observation::Control(NumericOwnerOutcome::ComparisonEnvironment {
                        policy: Policy::WeakScalars,
                    }),
                ),
                (
                    NumericOwnerRequest::CaptureSortOwners,
                    Observation::Captured {
                        owners: vec![],
                        positions: vec![],
                        missing: vec![0, 1],
                    },
                ),
            ]));
            assert_eq!(
                run(input(2), script).unwrap(),
                OrderedNumericOwners::Sorted {
                    positions: vec![0, 1],
                    already_ordered: false
                }
            );
        }
    }
    #[test]
    fn empty_input_performs_no_native_observations() {
        assert_eq!(
            run(input(0), Script(VecDeque::new())).unwrap(),
            OrderedNumericOwners::Sorted {
                positions: vec![],
                already_ordered: true
            }
        );
    }
}
