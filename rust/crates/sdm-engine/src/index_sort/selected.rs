// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selected-subset ordering. Values stay separate from original row identities.

use std::cell::Cell;
use std::cmp::Ordering;
use sdm_contract::tabular::{OrderedSampledTabularRows as ResultOrder, TabularIndexFacts as Facts, TabularIndexScalar as Atom, TabularIndexValues as Values, TabularOrderError, TabularSortProfile as Profile};
use sdm_contract::{CoreError, Sentence};
use super::{SimdArgsortProfile, SimdKey, simd_argsort};
use super::scalar::scalar_argsort;

fn invalid() -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, "sampled index facts are inconsistent")
}

fn sorted(positions: Vec<usize>, already_ordered: bool) -> ResultOrder { ResultOrder::Sorted { positions, already_ordered } }

fn scalar<T: PartialOrd>(values: &[T], profile: Profile) -> Vec<usize> {
    if profile == Profile::Stable {
        let mut positions: Vec<_> = (0..values.len()).collect();
        positions.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap_or(Ordering::Equal));
        positions
    } else { scalar_argsort(values) }
}

#[derive(Clone, Copy)]
struct MissingLast<T>(T);
impl<T: PartialOrd> PartialEq for MissingLast<T> {
    fn eq(&self, other:&Self)->bool {self.partial_cmp(other)==Some(Ordering::Equal)}
}
impl<T: PartialOrd> PartialOrd for MissingLast<T> {
    fn partial_cmp(&self, other:&Self)->Option<Ordering> {
        self.0.partial_cmp(&other.0).or_else(||Some(self.0.partial_cmp(&self.0).is_none().cmp(&other.0.partial_cmp(&other.0).is_none())))
    }
}

fn numeric<T: SimdKey>(values: &[T], profile: Profile) -> Vec<usize> {
    if values.iter().any(|v|v.partial_cmp(v).is_none()) {
        let ordered:Vec<_>=values.iter().copied().map(MissingLast).collect();
        if matches!(profile,Profile::Avx2Gnu14|Profile::Avx512Gnu14) {
            let mut positions:Vec<_>=(0..values.len()).collect();
            super::gnu::sort(&ordered,&mut positions);
            return positions;
        }
        return scalar(&ordered,profile);
    }
    match profile {
        Profile::Scalar => scalar(values, profile),
        Profile::Stable => {
            let mut positions:Vec<_>=(0..values.len()).collect();
            positions.sort_by(|a,b| match values[*a].partial_cmp(&values[*b]) {
                Some(order)=>order,
                None=>values[*a].partial_cmp(&values[*a]).is_none().cmp(&values[*b].partial_cmp(&values[*b]).is_none()),
            });
            return Ok(sorted(positions, monotonic));
        },
        Profile::Avx2Gnu14 => simd_argsort(values, SimdArgsortProfile::Avx2Gnu14),
        Profile::Avx512Gnu14 => simd_argsort(values, SimdArgsortProfile::Avx512Gnu14),
    }
}

fn order<T: PartialOrd + Clone>(values: &[T], missing: &[bool], sorter: impl FnOnce(&[T]) -> Vec<usize>) -> Result<(Vec<usize>, bool), CoreError> {
    if values.len() != missing.len() { return Err(invalid()); }
    if !missing.iter().any(|v| *v) && values.windows(2).all(|w| w[0] <= w[1]) {
        return Ok(((0..values.len()).collect(), true));
    }
    let present: Vec<_> = missing.iter().enumerate().filter_map(|(i,m)| (!m).then_some(i)).collect();
    let keys: Vec<_> = present.iter().map(|i| values[*i].clone()).collect();
    let mut positions: Vec<_> = sorter(&keys).into_iter().map(|i| present[i]).collect();
    positions.extend(missing.iter().enumerate().filter_map(|(i,m)| m.then_some(i)));
    Ok((positions, false))
}

fn integer(value: &str) -> Result<(&str, bool), CoreError> {
    let (digits, negative) = value.strip_prefix('-').map_or((value, false), |v| (v,true));
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit())
        || digits.len()>1 && digits.starts_with('0') || negative && digits == "0" { return Err(invalid()); }
    Ok((digits,negative))
}
fn compare_integer(a: &str, b: &str) -> Ordering {
    let (a,na)=integer(a).expect("validated integer");
    let (b,nb)=integer(b).expect("validated integer");
    if na != nb { return if na { Ordering::Less } else { Ordering::Greater }; }
    let result=a.len().cmp(&b.len()).then_with(|| a.cmp(b));
    if na { result.reverse() } else { result }
}
fn integer_float(a: &str, b: f64) -> Ordering {
    if b == f64::INFINITY { return Ordering::Less; }
    if b == f64::NEG_INFINITY { return Ordering::Greater; }
    let whole=if b.trunc()==0.0 { "0".into() } else { format!("{:.0}",b.trunc()) };
    compare_integer(a,&whole).then_with(|| 0.0_f64.partial_cmp(&b.fract()).expect("nonmissing number"))
}
fn atom_type(a: &Atom) -> u8 { if matches!(a, Atom::Text{..}) { 1 } else { 0 } }
fn compare_atom(a: &Atom,b: &Atom) -> Option<Ordering> {
    if atom_type(a)!=atom_type(b) { return None; }
    match (a,b) {
        (Atom::Text{points:a},Atom::Text{points:b}) => Some(a.cmp(b)),
        (Atom::Floating{bits:a},Atom::Floating{bits:b}) => f64::from_bits(*a).partial_cmp(&f64::from_bits(*b)),
        (Atom::Floating{bits},other) => compare_atom(other,&Atom::Floating{bits:*bits}).map(Ordering::reverse),
        (other,Atom::Floating{bits}) => Some(integer_float(&integer_atom(other),f64::from_bits(*bits))),
        _ => Some(compare_integer(&integer_atom(a),&integer_atom(b))),
    }
}
fn integer_atom(a: &Atom) -> String {
    match a { Atom::Integer{value}=>value.clone(),Atom::Boolean{value}=>if *value {"1"}else{"0"}.into(), _=>unreachable!("numeric integer atom") }
}
#[derive(Clone)]
struct Object<'a> { value: &'a Atom, position: usize, failure: &'a Cell<Option<(usize,usize)>> }
impl PartialEq for Object<'_> { fn eq(&self, other:&Self)->bool { compare_atom(self.value,other.value)==Some(Ordering::Equal) } }
impl PartialOrd for Object<'_> {
    fn partial_cmp(&self,other:&Self)->Option<Ordering> {
        let result=compare_atom(self.value,other.value);
        if result.is_none() && self.failure.get().is_none() {self.failure.set(Some((self.position,other.position)));}
        result
    }
}

fn multi(levels:&[Facts], codes:&[Vec<i64>], missing:&[bool]) -> Result<ResultOrder,CoreError> {
    if levels.len()!=codes.len() || codes.iter().any(|c|c.len()!=missing.len()) {return Err(invalid());}
    let mut ranks=Vec::new();
    let mut incomparable_levels = Vec::new();
    for (level, code) in levels.iter().zip(codes) {
        let n=level.missing.len();
        let permutation=match order_sampled(level)? {
            ResultOrder::Sorted{positions,..}=>{incomparable_levels.push(false);positions},
            ResultOrder::Refused{..}=>{incomparable_levels.push(true);(0..n).collect()},
        };
        let mut rank=vec![0;n];
        for (at,original) in permutation.into_iter().enumerate(){rank[original]=at;}
        let mut row_ranks=Vec::with_capacity(code.len());
        for c in code { row_ranks.push(if *c == -1 {usize::MAX} else { *rank.get(usize::try_from(*c).map_err(|_|invalid())?).ok_or_else(invalid)? }); }
        ranks.push(row_ranks);
    }
    let mut positions:Vec<_>=(0..missing.len()).collect();
    positions.sort_by(|a,b| ranks.iter().map(|level|level[*a].cmp(&level[*b])).find(|v|*v!=Ordering::Equal).unwrap_or(Ordering::Equal));
    let already_ordered = !codes.iter().flatten().any(|c| *c == -1) && (1..missing.len()).all(|row| {
        for (level_no, level) in levels.iter().enumerate() {
            let left = codes[level_no][row - 1] as usize;
            let right = codes[level_no][row] as usize;
            if left == right {continue;}
            let compared = if incomparable_levels[level_no] {
                match &level.values {Values::Object{values}=>compare_atom(&values[left], &values[right]), _=>None}
            } else {Some(ranks[level_no][row - 1].cmp(&ranks[level_no][row]))};
            if compared != Some(Ordering::Equal) {return compared == Some(Ordering::Less);}
        }
        true
    });
    Ok(sorted(positions, already_ordered))
}

pub fn order_sampled(facts:&Facts)->Result<ResultOrder,CoreError> {
    let mask=&facts.missing; let profile=facts.profile;
    let result=match &facts.values {
        Values::Signed{values,width} | Values::Categorical{codes:values,width} => {
            match width {
                1|2 => order(values,mask,|v|scalar(v,profile))?,
                4 => {let v:Vec<_>=values.iter().map(|v|i32::try_from(*v).map_err(|_|invalid())).collect::<Result<_,_>>()?;order(&v,mask,|v|numeric(v,profile))?},
                8 => order(values,mask,|v|numeric(v,profile))?,
                _ => return Err(invalid()),
            }
        }
        Values::Unsigned{values,width} => match width {
            1|2 => order(values,mask,|v|scalar(v,profile))?,
            4 => {let v:Vec<_>=values.iter().map(|v|u32::try_from(*v).map_err(|_|invalid())).collect::<Result<_,_>>()?;order(&v,mask,|v|numeric(v,profile))?},
            8 => order(values,mask,|v|numeric(v,profile))?, _=>return Err(invalid()),
        },
        Values::Floating{bits,width} => {
            // Half storage is carried as its exact float32 value by the host.
            if *width==2 || *width==4 {
                let v:Vec<_>=bits.iter().map(|b|u32::try_from(*b).map(f32::from_bits).map_err(|_|invalid())).collect::<Result<_,_>>()?;
                order(&v,mask,|v|if *width==2 {scalar(&v.iter().copied().map(MissingLast).collect::<Vec<_>>(),profile)} else {numeric(v,profile)})?
            } else if *width==8 {
                let v:Vec<_>=bits.iter().map(|b|f64::from_bits(*b)).collect();
                order(&v,mask,|v|numeric(v,profile))?
            } else {return Err(invalid());}
        }
        Values::Temporal{values}=>order(values,mask,|v|scalar(v,profile))?,
        Values::Unicode{values}=>{
            if values.iter().flatten().any(|point|*point>0x10ffff){return Err(invalid());}
            order(values,mask,|v|scalar(v,profile))?
        },
        Values::Multi{levels,codes}=>return multi(levels,codes,mask),
        Values::Object{values}=>{
            if values.len()!=mask.len(){return Err(invalid());}
            for (v,m) in values.iter().zip(mask) {if !m {match v {
                Atom::Integer{value}=>{integer(value)?;},
                Atom::Floating{bits} if f64::from_bits(*bits).is_nan()=>return Err(invalid()),
                Atom::Text{points} if points.iter().any(|p|*p>0x10ffff)=>return Err(invalid()),_=>{},
            }}}
            let failed=Cell::new(None);
            let v:Vec<_>=values.iter().enumerate().map(|(position,value)|Object{value,position,failure:&failed}).collect();
            // Monotonic probing suppresses incomparable errors in the native index.
            let monotonic=!mask.iter().any(|v|*v)&&v.windows(2).all(|w|w[0]<=w[1]);
            failed.set(None);
            let positions=if monotonic {(0..v.len()).collect()} else {
                let present:Vec<_>=mask.iter().enumerate().filter_map(|(i,m)|(!m).then_some(i)).collect();
                let keys:Vec<_>=present.iter().map(|i|v[*i].clone()).collect();
                let mut positions:Vec<_>=scalar(&keys,profile).into_iter().map(|i|present[i]).collect();
                positions.extend(mask.iter().enumerate().filter_map(|(i,m)|m.then_some(i)));positions
            };
            if let Some((left,right))=failed.get(){return Ok(ResultOrder::Refused{reason:TabularOrderError::Incomparable,left,right});}
            return Ok(sorted(positions, monotonic));
        },
    };
    Ok(sorted(result.0, result.1))
}
