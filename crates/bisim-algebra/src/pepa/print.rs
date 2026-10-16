//! Pretty-printing in the syntax accepted by [`parse_pepa`](super::parse_pepa).
//!
//! Printing then parsing a model gives back the same AST; nested operands are
//! parenthesised so that the tree shape survives the round trip.

use std::fmt;

use super::ast::{Action, ActionSet, ModelTerm, PepaModel, RateExpr, Seq};

impl<A: Action> fmt::Display for PepaModel<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (name, value) in &self.params {
            writeln!(f, "{name} = {value};")?;
        }
        for (name, def) in &self.seqs {
            if let Some(def) = def {
                writeln!(f, "{name} = {};", self.show_seq(def))?;
            }
        }
        writeln!(f, "{};", self.show_term(&self.system))
    }
}

impl<A: Action> PepaModel<A> {
    pub(crate) fn show_seq(&self, s: &Seq<A>) -> String {
        match s {
            Seq::Prefix(a, r, next) => {
                let next_text = self.show_seq(next);
                let next_text = match &**next {
                    Seq::Choice(xs) if xs.len() > 1 => format!("({next_text})"),
                    _ => next_text,
                };
                format!("({a}, {}).{next_text}", self.show_rate(r))
            }
            Seq::Choice(xs) if xs.is_empty() => "0".to_owned(),
            Seq::Choice(xs) => xs
                .iter()
                .map(|x| match x {
                    Seq::Choice(ys) if ys.len() > 1 => format!("({})", self.show_seq(x)),
                    _ => self.show_seq(x),
                })
                .collect::<Vec<_>>()
                .join(" + "),
            Seq::Const(id) => self.seq_name(*id).to_owned(),
        }
    }

    pub(crate) fn show_rate(&self, r: &RateExpr) -> String {
        match r {
            RateExpr::Lit(v) => format!("{v}"),
            RateExpr::Param(id) => self.params[id.0 as usize].0.clone(),
            RateExpr::Count(n) => format!("#{n}"),
            RateExpr::Passive(w) if **w == RateExpr::Lit(1.0) => "infty".to_owned(),
            RateExpr::Passive(w) => format!("{} * infty", self.rate_operand(w)),
            RateExpr::Add(a, b) => self.binary(a, "+", b),
            RateExpr::Sub(a, b) => self.binary(a, "-", b),
            RateExpr::Mul(a, b) => self.binary(a, "*", b),
            RateExpr::Div(a, b) => self.binary(a, "/", b),
        }
    }

    fn binary(&self, a: &RateExpr, op: &str, b: &RateExpr) -> String {
        format!("{} {op} {}", self.rate_operand(a), self.rate_operand(b))
    }

    fn rate_operand(&self, r: &RateExpr) -> String {
        match r {
            RateExpr::Lit(_) | RateExpr::Param(_) | RateExpr::Count(_) => self.show_rate(r),
            _ => format!("({})", self.show_rate(r)),
        }
    }

    pub(crate) fn show_term(&self, t: &ModelTerm<A>) -> String {
        match t {
            ModelTerm::Coop(l, set, r) => {
                let right = match &**r {
                    ModelTerm::Coop(..) => format!("({})", self.show_term(r)),
                    _ => self.show_term(r),
                };
                let op = if set.is_empty() {
                    "||".to_owned()
                } else {
                    format!("<{}>", show_set(set))
                };
                format!("{} {op} {right}", self.show_term(l))
            }
            ModelTerm::Hide(x, set) => {
                let inner = match &**x {
                    ModelTerm::Coop(..) => format!("({})", self.show_term(x)),
                    _ => self.show_term(x),
                };
                format!("{inner} / {{{}}}", show_set(set))
            }
            ModelTerm::Component(id) => self.seq_name(*id).to_owned(),
            ModelTerm::Array(id, n) => format!("{}[{n}]", self.seq_name(*id)),
        }
    }
}

fn show_set<A: Action>(set: &ActionSet<A>) -> String {
    set.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
