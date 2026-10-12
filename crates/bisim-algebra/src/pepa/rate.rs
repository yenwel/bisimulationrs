use std::fmt;

use super::ast::RateExpr;

/// An evaluated PEPA rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rate {
    /// Exponential rate in s⁻¹.
    Active(f64),
    /// Passive rate `w · ⊤`.
    Passive(f64),
}

/// Mixing active and passive rates for one action in one component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MixedRates;

impl Rate {
    pub fn is_passive(self) -> bool {
        matches!(self, Rate::Passive(_))
    }

    pub(crate) fn value(self) -> f64 {
        match self {
            Rate::Active(r) | Rate::Passive(r) => r,
        }
    }

    pub(crate) fn scale(self, k: f64) -> Rate {
        match self {
            Rate::Active(r) => Rate::Active(r * k),
            Rate::Passive(w) => Rate::Passive(w * k),
        }
    }

    /// Sum used for apparent rates.
    pub(crate) fn add(self, other: Rate) -> Result<Rate, MixedRates> {
        match (self, other) {
            (Rate::Active(a), Rate::Active(b)) => Ok(Rate::Active(a + b)),
            (Rate::Passive(a), Rate::Passive(b)) => Ok(Rate::Passive(a + b)),
            _ => Err(MixedRates),
        }
    }

    /// `min` with `⊤` larger than every active rate: `min(r, w⊤) = r`.
    pub(crate) fn min(self, other: Rate) -> Rate {
        match (self, other) {
            (Rate::Active(a), Rate::Active(b)) => Rate::Active(a.min(b)),
            (Rate::Active(a), Rate::Passive(_)) | (Rate::Passive(_), Rate::Active(a)) => {
                Rate::Active(a)
            }
            (Rate::Passive(a), Rate::Passive(b)) => Rate::Passive(a.min(b)),
        }
    }

    /// Rate of a synchronised activity (Hillston's cooperation rule):
    /// `(r₁ / r_α(P)) · (r₂ / r_α(Q)) · min(r_α(P), r_α(Q))`.
    pub(crate) fn cooperate(r1: Rate, apparent1: Rate, r2: Rate, apparent2: Rate) -> Rate {
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
        }
    }
}

impl RateExpr {
    /// Evaluates the expression; `None` if it is not a valid rate (passive
    /// below the top level, division by zero, …).
    pub(crate) fn eval(&self, params: &[(String, f64)]) -> Option<Rate> {
        match self {
            RateExpr::Passive(w) => Some(Rate::Passive(w.eval_active(params)?)),
            other => Some(Rate::Active(other.eval_active(params)?)),
        }
    }

    fn eval_active(&self, params: &[(String, f64)]) -> Option<f64> {
        let v = match self {
            RateExpr::Lit(v) => *v,
            RateExpr::Param(id) => params.get(id.0 as usize)?.1,
            RateExpr::Passive(_) => return None,
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
}
