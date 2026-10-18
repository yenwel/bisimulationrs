//! Resolved PRISM AST: names replaced by indices, constants folded.

use std::fmt;

use crate::pepa::ParseError;

/// Errors raised when parsing, resolving or deriving a PRISM model.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PrismError {
    Parse(ParseError),
    /// Two constants, formulas, modules or variables share a name.
    Duplicate(String),
    /// Reference to an unknown constant, formula, variable or module.
    Unknown(String),
    /// `formula`s defined in terms of each other.
    CyclicFormula(String),
    /// An expression had the wrong type (e.g. a non-boolean guard).
    Type(String),
    /// Division by zero, integer overflow or a non-finite intermediate.
    Arithmetic(String),
    /// An update writes a value outside the variable's bounds.
    OutOfBounds {
        var: String,
        value: i64,
    },
    /// The probabilities of one DTMC command do not sum to 1.
    ProbSum {
        command: String,
        sum: f64,
    },
    /// Two DTMC commands are enabled at once (that is nondeterminism:
    /// write an `mdp` model, which this importer does not support).
    Nondeterministic(String),
    /// A language feature outside the supported subset.
    Unsupported(String),
    /// The reachable state space exceeds the derivation limit.
    StateSpaceTooLarge {
        limit: usize,
    },
}

impl fmt::Display for PrismError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrismError::Parse(e) => write!(f, "{e}"),
            PrismError::Duplicate(n) => write!(f, "`{n}` is defined more than once"),
            PrismError::Unknown(n) => write!(f, "unknown identifier `{n}`"),
            PrismError::CyclicFormula(n) => {
                write!(f, "formula `{n}` is defined in terms of itself")
            }
            PrismError::Type(msg) => write!(f, "type error: {msg}"),
            PrismError::Arithmetic(msg) => write!(f, "arithmetic error: {msg}"),
            PrismError::OutOfBounds { var, value } => write!(
                f,
                "update writes {value} to `{var}`, outside its declared bounds"
            ),
            PrismError::ProbSum { command, sum } => {
                write!(f, "probabilities of {command} sum to {sum}, not 1")
            }
            PrismError::Nondeterministic(ctx) => write!(
                f,
                "more than one DTMC command enabled in {ctx}; this is nondeterminism (mdp models are not supported)"
            ),
            PrismError::Unsupported(msg) => write!(f, "not supported: {msg}"),
            PrismError::StateSpaceTooLarge { limit } => {
                write!(f, "state space exceeds the limit of {limit} states")
            }
        }
    }
}

impl std::error::Error for PrismError {}

impl From<ParseError> for PrismError {
    fn from(e: ParseError) -> Self {
        PrismError::Parse(e)
    }
}

/// `dtmc` (probabilities) or `ctmc` (rates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Dtmc,
    Ctmc,
}

/// Integer range or boolean variable.
#[derive(Debug, Clone, PartialEq)]
pub enum VarType {
    Int { lo: i64, hi: i64 },
    Bool,
}

/// A module variable with its initial valuation.
#[derive(Debug, Clone, PartialEq)]
pub struct Var {
    pub name: String,
    pub typ: VarType,
    pub init: i64,
}

/// One probabilistic/rate-weighted alternative of a command.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// Probability (DTMC) or rate (CTMC); `1` for empty (`true`) updates.
    pub weight: Expr,
    /// Simultaneous assignments as `(global variable index, expression)`.
    pub assigns: Vec<Assign>,
}

/// One `(variable := expression)` assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    pub var: usize,
    pub expr: Expr,
}

/// A guarded command `[label] guard -> updates`.
#[derive(Debug, Clone, PartialEq)]
pub struct Command {
    /// `None` for unlabelled (`[]`) commands, which always interleave.
    pub label: Option<String>,
    pub guard: Expr,
    pub updates: Vec<Update>,
}

/// A module: local variables and commands over global variable indices.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub name: String,
    pub vars: Vec<usize>,
    pub commands: Vec<Command>,
}

/// A fully resolved PRISM model: variables are indexed `0..n`, constants
/// and formulas are folded away. Build it with [`parse_prism`](super::parse_prism).
#[derive(Debug, Clone, PartialEq)]
pub struct PrismModel {
    pub kind: ModelKind,
    pub modules: Vec<Module>,
    pub vars: Vec<Var>,
    /// Reward structures (`rewards...endrewards` blocks).
    pub rewards: Vec<RewardBlock>,
    /// Module composition from the `system` block (default: full sync).
    pub sync: SyncConfig,
}
impl PrismModel {
    /// Global index of a variable.
    pub fn var_index(&self, name: &str) -> Option<usize> {
        self.vars.iter().position(|v| v.name == name)
    }

    /// Initial valuation, in global variable order.
    pub fn initial(&self) -> Vec<i64> {
        self.vars.iter().map(|v| v.init).collect()
    }

    fn kinds(&self) -> Vec<VarKind> {
        self.vars
            .iter()
            .map(|v| match v.typ {
                VarType::Bool => VarKind::Bool,
                VarType::Int { .. } => VarKind::Int,
            })
            .collect()
    }

    /// Evaluates a query predicate in a valuation.
    pub fn eval_pred(&self, pred: &Expr, state: &[i64]) -> Result<bool, PrismError> {
        eval(pred, state, &self.kinds())
            .and_then(Value::as_bool)
            .ok_or_else(|| PrismError::Type("query predicate is not boolean".to_owned()))
    }

    /// Evaluates a numeric expression (reward value, weight) in a valuation.
    pub fn eval_num(&self, expr: &Expr, state: &[i64]) -> Result<f64, PrismError> {
        eval(expr, state, &self.kinds())
            .and_then(Value::as_num)
            .ok_or_else(|| PrismError::Type("expected a number".to_owned()))
    }
}

/// Expressions with variables as global indices and constants folded.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Var(usize),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Min(Vec<Expr>),
    Max(Vec<Expr>),
    Floor(Box<Expr>),
    Ceil(Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
}

/// A runtime value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Value {
    /// Numeric view: integers widen to floats.
    pub fn as_num(self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(i as f64),
            Value::Float(x) => Some(x),
            Value::Bool(_) => None,
        }
    }

    pub fn as_bool(self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }
}

/// Types of the variables a state vector is evaluated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    Int,
    Bool,
}

/// Evaluates an expression in `state`; `None` on division by zero,
/// integer overflow, non-finite floats or type mismatch.
pub fn eval(expr: &Expr, state: &[i64], kinds: &[VarKind]) -> Option<Value> {
    match expr {
        Expr::Int(i) => Some(Value::Int(*i)),
        Expr::Float(x) => Some(Value::Float(*x)),
        Expr::Bool(b) => Some(Value::Bool(*b)),
        Expr::Var(i) => Some(match kinds[*i] {
            VarKind::Bool => Value::Bool(state[*i] != 0),
            VarKind::Int => Value::Int(state[*i]),
        }),
        Expr::Neg(e) => match eval(e, state, kinds)? {
            Value::Int(i) => Some(Value::Int(i.checked_neg()?)),
            Value::Float(x) => x.is_finite().then_some(Value::Float(-x)),
            Value::Bool(_) => None,
        },
        Expr::Not(e) => Some(Value::Bool(!eval(e, state, kinds)?.as_bool()?)),
        Expr::Add(a, b) => num2(
            eval(a, state, kinds)?,
            eval(b, state, kinds)?,
            |x, y| x.checked_add(y),
            |x, y| x + y,
        ),
        Expr::Sub(a, b) => num2(
            eval(a, state, kinds)?,
            eval(b, state, kinds)?,
            |x, y| x.checked_sub(y),
            |x, y| x - y,
        ),
        Expr::Mul(a, b) => num2(
            eval(a, state, kinds)?,
            eval(b, state, kinds)?,
            |x, y| x.checked_mul(y),
            |x, y| x * y,
        ),
        Expr::Div(a, b) => {
            // PRISM and JANI `/` is always real division (`1/2` is 0.5).
            let (x, y) = (eval(a, state, kinds)?, eval(b, state, kinds)?);
            let v = x.as_num()? / y.as_num()?;
            v.is_finite().then_some(Value::Float(v))
        }
        Expr::Mod(a, b) => {
            let (x, y) = (eval(a, state, kinds)?, eval(b, state, kinds)?);
            match (x, y) {
                (Value::Int(x), Value::Int(y)) => Some(Value::Int(x.checked_rem(y)?)),
                _ => {
                    let (x, y) = (x.as_num()?, y.as_num()?);
                    let v = x % y;
                    v.is_finite().then_some(Value::Float(v))
                }
            }
        }
        Expr::Pow(a, b) => {
            let (x, y) = (eval(a, state, kinds)?, eval(b, state, kinds)?);
            // As in PRISM, an integer to a non-negative integer power is
            // an integer.
            if let (Value::Int(base), Value::Int(exp)) = (x, y) {
                if let Some(p) = u32::try_from(exp).ok().and_then(|e| base.checked_pow(e)) {
                    return Some(Value::Int(p));
                }
            }
            let v = x.as_num()?.powf(y.as_num()?);
            v.is_finite().then_some(Value::Float(v))
        }
        Expr::Min(es) | Expr::Max(es) => {
            let is_min = matches!(expr, Expr::Min(_));
            let mut ints = Vec::with_capacity(es.len());
            let mut floats = Vec::with_capacity(es.len());
            for e in es {
                match eval(e, state, kinds)? {
                    Value::Int(i) => ints.push(i),
                    Value::Float(x) => floats.push(x),
                    Value::Bool(_) => return None,
                }
            }
            if floats.is_empty() {
                let v = if is_min {
                    *ints.iter().min()?
                } else {
                    *ints.iter().max()?
                };
                Some(Value::Int(v))
            } else {
                let mut best = if is_min {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
                for i in ints {
                    let x = i as f64;
                    best = if is_min { best.min(x) } else { best.max(x) };
                }
                for x in floats {
                    best = if is_min { best.min(x) } else { best.max(x) };
                }
                best.is_finite().then_some(Value::Float(best))
            }
        }
        Expr::Floor(e) => {
            let v = eval(e, state, kinds)?.as_num()?;
            let w = v.floor();
            (w.is_finite() && w >= i64::MIN as f64 && w <= i64::MAX as f64)
                .then_some(Value::Int(w as i64))
        }
        Expr::Ceil(e) => {
            let v = eval(e, state, kinds)?.as_num()?;
            let w = v.ceil();
            (w.is_finite() && w >= i64::MIN as f64 && w <= i64::MAX as f64)
                .then_some(Value::Int(w as i64))
        }
        Expr::Eq(a, b) => cmp(eval(a, state, kinds)?, eval(b, state, kinds)?, |o| {
            o == std::cmp::Ordering::Equal
        }),
        Expr::Ne(a, b) => cmp(eval(a, state, kinds)?, eval(b, state, kinds)?, |o| {
            o != std::cmp::Ordering::Equal
        }),
        Expr::Lt(a, b) => cmp(eval(a, state, kinds)?, eval(b, state, kinds)?, |o| {
            o == std::cmp::Ordering::Less
        }),
        Expr::Le(a, b) => cmp(eval(a, state, kinds)?, eval(b, state, kinds)?, |o| {
            o != std::cmp::Ordering::Greater
        }),
        Expr::Gt(a, b) => cmp(eval(a, state, kinds)?, eval(b, state, kinds)?, |o| {
            o == std::cmp::Ordering::Greater
        }),
        Expr::Ge(a, b) => cmp(eval(a, state, kinds)?, eval(b, state, kinds)?, |o| {
            o != std::cmp::Ordering::Less
        }),
        Expr::And(a, b) => Some(Value::Bool(
            eval(a, state, kinds)?.as_bool()? && eval(b, state, kinds)?.as_bool()?,
        )),
        Expr::Or(a, b) => Some(Value::Bool(
            eval(a, state, kinds)?.as_bool()? || eval(b, state, kinds)?.as_bool()?,
        )),
        Expr::Cond(c, a, b) => {
            if eval(c, state, kinds)?.as_bool()? {
                eval(a, state, kinds)
            } else {
                eval(b, state, kinds)
            }
        }
    }
}

fn num2(
    x: Value,
    y: Value,
    int: impl Fn(i64, i64) -> Option<i64>,
    float: impl Fn(f64, f64) -> f64,
) -> Option<Value> {
    match (x, y) {
        (Value::Int(x), Value::Int(y)) => Some(Value::Int(int(x, y)?)),
        (x, y) => {
            let v = float(x.as_num()?, y.as_num()?);
            v.is_finite().then_some(Value::Float(v))
        }
    }
}

fn cmp(x: Value, y: Value, keep: impl Fn(std::cmp::Ordering) -> bool) -> Option<Value> {
    match (x, y) {
        (Value::Bool(a), Value::Bool(b)) => {
            let o = a.cmp(&b);
            Some(Value::Bool(keep(o)))
        }
        (x, y) => {
            let (x, y) = (x.as_num()?, y.as_num()?);
            // Not `total_cmp`, which orders -0.0 before 0.0; values are
            // finite here, so `partial_cmp` always answers.
            let o = x.partial_cmp(&y)?;
            Some(Value::Bool(keep(o)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: i64) -> Vec<i64> {
        vec![i]
    }

    #[test]
    fn arithmetic_and_logic() {
        let kinds = [VarKind::Int];
        let e = Expr::Add(Box::new(Expr::Var(0)), Box::new(Expr::Int(1)));
        assert_eq!(eval(&e, &v(41), &kinds), Some(Value::Int(42)));
        // `/` is real division, also on integers; mixed arithmetic widens.
        let d = Expr::Div(Box::new(Expr::Int(7)), Box::new(Expr::Int(2)));
        assert_eq!(eval(&d, &[], &[]), Some(Value::Float(3.5)));
        assert_eq!(
            eval(
                &Expr::Div(Box::new(Expr::Int(7)), Box::new(Expr::Int(0))),
                &[],
                &[]
            ),
            None
        );
        let g = Expr::And(
            Box::new(Expr::Lt(Box::new(Expr::Var(0)), Box::new(Expr::Int(10)))),
            Box::new(Expr::Not(Box::new(Expr::Bool(false)))),
        );
        assert_eq!(eval(&g, &v(3), &kinds), Some(Value::Bool(true)));
        assert_eq!(eval(&g, &v(30), &kinds), Some(Value::Bool(false)));
        let m = Expr::Min(vec![Expr::Var(0), Expr::Int(5)]);
        assert_eq!(eval(&m, &v(9), &kinds), Some(Value::Int(5)));
        let c = Expr::Cond(
            Box::new(Expr::Bool(true)),
            Box::new(Expr::Int(1)),
            Box::new(Expr::Int(2)),
        );
        assert_eq!(eval(&c, &[], &[]), Some(Value::Int(1)));
    }
}

/// A comparison against a probability threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cmp {
    Ge,
    Gt,
    Le,
    Lt,
}

impl Cmp {
    pub fn holds(self, prob: f64, bound: f64) -> bool {
        match self {
            Cmp::Ge => prob >= bound,
            Cmp::Gt => prob > bound,
            Cmp::Le => prob <= bound,
            Cmp::Lt => prob < bound,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Cmp::Ge => ">=",
            Cmp::Gt => ">",
            Cmp::Le => "<=",
            Cmp::Lt => "<",
        }
    }
}

/// `F target` (eventually) or `F<=t target` (time/steps-bounded).
#[derive(Debug, Clone, PartialEq)]
pub enum Until {
    Eventually,
    Bounded(f64),
}

/// `P=? [F pred]`, `P>=0.9 [F<=10 pred]`: reachability from the initial
/// state. On DTMCs the bound counts steps (and must be integral); on
/// CTMCs it counts seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbQuery {
    /// `None` for quantitative `P=?`.
    pub threshold: Option<(Cmp, f64)>,
    pub until: Until,
    pub pred: Expr,
}

/// `S=? [pred]`, `S<0.01 [pred]`: long-run probability of `pred`.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyQuery {
    /// `None` for quantitative `S=?`.
    pub threshold: Option<(Cmp, f64)>,
    pub pred: Expr,
}

/// A CSL/PCTL-style query. Parse it with
/// [`parse_query`](super::parse_query).
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    Prob(ProbQuery),
    Steady(SteadyQuery),
}

/// One item of a `rewards` block: a state reward (`guard : value`,
/// accumulating `value` per time unit in `guard` states) or a transition
/// reward (`[label] guard : value`, accumulating `value` per firing).
#[derive(Debug, Clone, PartialEq)]
pub enum RewardItem {
    State {
        guard: Expr,
        value: Expr,
    },
    Trans {
        label: String,
        guard: Expr,
        value: Expr,
    },
}

/// A named reward structure (`rewards "name" ... endrewards`).
#[derive(Debug, Clone, PartialEq)]
pub struct RewardBlock {
    pub name: String,
    pub items: Vec<RewardItem>,
}

/// Module composition from a `system` block: which labels synchronise.
///
/// `None` (no block, or `||`) synchronises on every shared label;
/// `Some(set)` (from `|[a, b]|`) synchronises only on those labels and
/// interleaves everything else. Nested compositions are rejected: use one
/// top-level parallel composition.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SyncConfig {
    pub restricted: Option<std::collections::BTreeSet<String>>,
}

impl SyncConfig {
    /// Whether a labelled command synchronises (`None` labels always
    /// interleave).
    pub fn synchronises(&self, label: Option<&str>) -> bool {
        match (label, &self.restricted) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(l), Some(set)) => set.contains(l),
        }
    }
}
