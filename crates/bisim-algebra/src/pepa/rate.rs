use std::fmt;

use super::ast::RateExpr;

/// A state-dependent (functional) rate expression over species counts.
///
/// Parameters are folded to literals when the prefix is compiled; what
/// remains is an expression over `#{derivative}` counts, evaluated per
/// global state during derivation. Prints (and re-parses) in the same
/// `#Name` syntax, e.g. `0.3 * #I`.
#[derive(Debug, Clone, PartialEq)]
pub enum FunExpr {
    Lit(f64),
    Count(String),
    Add(Box<FunExpr>, Box<FunExpr>),
    Sub(Box<FunExpr>, Box<FunExpr>),
    Mul(Box<FunExpr>, Box<FunExpr>),
    Div(Box<FunExpr>, Box<FunExpr>),
}

impl FunExpr {
    /// Evaluates the expression; `None` on division by zero or overflow.
    /// `counts` maps a derivative name to its current number of copies.
    pub(crate) fn eval(&self, counts: &dyn Fn(&str) -> f64) -> Option<f64> {
        let v = match self {
            FunExpr::Lit(v) => *v,
            FunExpr::Count(n) => counts(n),
            FunExpr::Add(a, b) => a.eval(counts)? + b.eval(counts)?,
            FunExpr::Sub(a, b) => a.eval(counts)? - b.eval(counts)?,
            FunExpr::Mul(a, b) => a.eval(counts)? * b.eval(counts)?,
            FunExpr::Div(a, b) => a.eval(counts)? / b.eval(counts)?,
        };
        v.is_finite().then_some(v)
    }

    fn operand(&self) -> String {
        match self {
            lit @ (FunExpr::Lit(_) | FunExpr::Count(_)) => lit.to_string(),
            other => format!("({other})"),
        }
    }
}

impl fmt::Display for FunExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunExpr::Lit(v) => write!(f, "{v}"),
            FunExpr::Count(n) => write!(f, "#{n}"),
            FunExpr::Add(a, b) => write!(f, "{} + {}", a.operand(), b.operand()),
            FunExpr::Sub(a, b) => write!(f, "{} - {}", a.operand(), b.operand()),
            FunExpr::Mul(a, b) => write!(f, "{} * {}", a.operand(), b.operand()),
            FunExpr::Div(a, b) => write!(f, "{} / {}", a.operand(), b.operand()),
        }
    }
}

/// An evaluated PEPA rate.
#[derive(Debug, Clone, PartialEq)]
pub enum Rate {
    /// Exponential rate in s⁻¹.
    Active(f64),
    /// Passive rate `w · ⊤`.
    Passive(f64),
    /// State-dependent rate over species counts, evaluated per global
    /// state during derivation (Bio-PEPA-style functional rate). Never
    /// passive: counts under `infty` are rejected when parsing.
    Functional(FunExpr),
}

/// Mixing active and passive rates for one action in one component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MixedRates;

impl Rate {
    pub fn is_passive(self: &Rate) -> bool {
        matches!(self, Rate::Passive(_))
    }

    /// Whether this rate depends on species counts.
    pub fn is_functional(&self) -> bool {
        matches!(self, Rate::Functional(_))
    }

    pub(crate) fn value(&self) -> f64 {
        match self {
            Rate::Active(r) | Rate::Passive(r) => *r,
            Rate::Functional(_) => {
                unreachable!("functional rates have no static value; concretize per state first")
            }
        }
    }

    pub(crate) fn scale(self, k: f64) -> Rate {
        match self {
            Rate::Active(r) => Rate::Active(r * k),
            Rate::Passive(w) => Rate::Passive(w * k),
            Rate::Functional(f) => {
                Rate::Functional(FunExpr::Mul(Box::new(f), Box::new(FunExpr::Lit(k))))
            }
        }
    }

    /// Sum used for apparent rates. Functional rates add expression-wise;
    /// mixing anything functional with anything passive is an error.
    pub(crate) fn add(self, other: Rate) -> Result<Rate, MixedRates> {
        match (self, other) {
            (Rate::Active(a), Rate::Active(b)) => Ok(Rate::Active(a + b)),
            (Rate::Passive(a), Rate::Passive(b)) => Ok(Rate::Passive(a + b)),
            (Rate::Active(a), Rate::Functional(f)) | (Rate::Functional(f), Rate::Active(a)) => Ok(
                Rate::Functional(FunExpr::Add(Box::new(FunExpr::Lit(a)), Box::new(f))),
            ),
            (Rate::Functional(f), Rate::Functional(g)) => {
                Ok(Rate::Functional(FunExpr::Add(Box::new(f), Box::new(g))))
            }
            _ => Err(MixedRates),
        }
    }

    /// `min` with `⊤` larger than every active rate: `min(r, w⊤) = r`.
    ///
    /// Functional rates must be concretized per state before cooperation;
    /// reaching this with one is a caller bug.
    pub(crate) fn min(self, other: Rate) -> Rate {
        match (self, other) {
            (Rate::Active(a), Rate::Active(b)) => Rate::Active(a.min(b)),
            (Rate::Active(a), Rate::Passive(_)) | (Rate::Passive(_), Rate::Active(a)) => {
                Rate::Active(a)
            }
            (Rate::Passive(a), Rate::Passive(b)) => Rate::Passive(a.min(b)),
            _ => unreachable!("functional rates must be concretized before cooperation"),
        }
    }

    /// Rate of a synchronised activity (Hillston's cooperation rule):
    /// `(r₁ / r_α(P)) · (r₂ / r_α(Q)) · min(r_α(P), r_α(Q))`.
    ///
    /// All arguments must already be concrete (see [`Rate::min`]).
    pub(crate) fn cooperate(r1: Rate, apparent1: Rate, r2: Rate, apparent2: Rate) -> Rate {
        if r1.is_functional()
            || r2.is_functional()
            || apparent1.is_functional()
            || apparent2.is_functional()
        {
            unreachable!("functional rates must be concretized before cooperation");
        }
        let share = (r1.value() / apparent1.value()) * (r2.value() / apparent2.value());
        apparent1.min(apparent2).scale(share)
    }
}

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rate::Active(r) => write!(f, "{r}"),
            Rate::Passive(w) if *w == 1.0 => write!(f, "infty"),
            Rate::Passive(w) => write!(f, "{w} * infty"),
            Rate::Functional(e) => write!(f, "{e}"),
        }
    }
}

impl RateExpr {
    /// Whether the expression refers to any species count.
    pub(crate) fn is_functional(&self) -> bool {
        match self {
            RateExpr::Count(_) => true,
            RateExpr::Lit(_) | RateExpr::Param(_) => false,
            RateExpr::Passive(w) => w.is_functional(),
            RateExpr::Add(a, b)
            | RateExpr::Sub(a, b)
            | RateExpr::Mul(a, b)
            | RateExpr::Div(a, b) => a.is_functional() || b.is_functional(),
        }
    }

    /// Collects every species name (`#Name`) in the expression.
    pub(crate) fn counts(&self, out: &mut Vec<String>) {
        match self {
            RateExpr::Count(n) => {
                if !out.contains(n) {
                    out.push(n.clone());
                }
            }
            RateExpr::Lit(_) | RateExpr::Param(_) => {}
            RateExpr::Passive(w) => w.counts(out),
            RateExpr::Add(a, b)
            | RateExpr::Sub(a, b)
            | RateExpr::Mul(a, b)
            | RateExpr::Div(a, b) => {
                a.counts(out);
                b.counts(out);
            }
        }
    }

    /// Folds parameters to literals, keeping counts symbolic.
    /// `None` for passive expressions (counts under `infty` are rejected).
    fn to_functional(&self, params: &[(String, f64)]) -> Option<FunExpr> {
        match self {
            RateExpr::Lit(v) => Some(FunExpr::Lit(*v)),
            RateExpr::Param(id) => Some(FunExpr::Lit(params.get(id.0 as usize)?.1)),
            RateExpr::Count(n) => Some(FunExpr::Count(n.clone())),
            RateExpr::Passive(_) => None,
            RateExpr::Add(a, b) => Some(FunExpr::Add(
                Box::new(a.to_functional(params)?),
                Box::new(b.to_functional(params)?),
            )),
            RateExpr::Sub(a, b) => Some(FunExpr::Sub(
                Box::new(a.to_functional(params)?),
                Box::new(b.to_functional(params)?),
            )),
            RateExpr::Mul(a, b) => Some(FunExpr::Mul(
                Box::new(a.to_functional(params)?),
                Box::new(b.to_functional(params)?),
            )),
            RateExpr::Div(a, b) => Some(FunExpr::Div(
                Box::new(a.to_functional(params)?),
                Box::new(b.to_functional(params)?),
            )),
        }
    }

    /// Evaluates the expression; `None` if it is not a valid rate (passive
    /// below the top level, division by zero, …). Expressions over species
    /// counts become [`Rate::Functional`], evaluated per state at derivation.
    pub(crate) fn eval(&self, params: &[(String, f64)]) -> Option<Rate> {
        match self {
            RateExpr::Passive(w) => {
                if w.is_functional() {
                    return None;
                }
                Some(Rate::Passive(w.eval_active(params)?))
            }
            other if other.is_functional() => Some(Rate::Functional(other.to_functional(params)?)),
            other => Some(Rate::Active(other.eval_active(params)?)),
        }
    }

    fn eval_active(&self, params: &[(String, f64)]) -> Option<f64> {
        let v = match self {
            RateExpr::Lit(v) => *v,
            RateExpr::Param(id) => params.get(id.0 as usize)?.1,
            RateExpr::Passive(_) | RateExpr::Count(_) => return None,
            RateExpr::Add(a, b) => a.eval_active(params)? + b.eval_active(params)?,
            RateExpr::Sub(a, b) => a.eval_active(params)? - b.eval_active(params)?,
            RateExpr::Mul(a, b) => a.eval_active(params)? * b.eval_active(params)?,
            RateExpr::Div(a, b) => a.eval_active(params)? / b.eval_active(params)?,
        };
        v.is_finite().then_some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pepa::ParamId;

    #[test]
    fn cooperation_rule() {
        // Active P with apparent rate 2 meets passive Q: P sets the pace.
        let r = Rate::cooperate(
            Rate::Active(2.0),
            Rate::Active(2.0),
            Rate::Passive(1.0),
            Rate::Passive(2.0),
        );
        assert_eq!(r, Rate::Active(1.0));
        // Both active: bounded by the slower partner.
        let r = Rate::cooperate(
            Rate::Active(3.0),
            Rate::Active(3.0),
            Rate::Active(1.0),
            Rate::Active(1.0),
        );
        assert_eq!(r, Rate::Active(1.0));
        assert!(Rate::Active(1.0).add(Rate::Passive(1.0)).is_err());
    }

    #[test]
    fn functional_expression_folds_params_and_evaluates() {
        let params = vec![("beta".to_owned(), 0.3)];
        let e = RateExpr::Mul(
            Box::new(RateExpr::Param(ParamId(0))),
            Box::new(RateExpr::Count("I".into())),
        );
        let r = e.eval(&params).unwrap();
        let Rate::Functional(f) = r else {
            panic!("expected functional rate");
        };
        assert_eq!(f.to_string(), "0.3 * #I");
        assert_eq!(f.eval(&|n| if n == "I" { 4.0 } else { 0.0 }), Some(1.2));
        assert!(RateExpr::Passive(Box::new(e)).eval(&params).is_none());
    }
}
