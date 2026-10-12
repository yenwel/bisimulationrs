use std::ops::Add;

use super::ast::{Action, ModelTerm, ParamId, PepaModel, RateExpr, Seq, SeqId};
use super::validate::PepaError;

/// Typed builder for [`PepaModel`]s.
///
/// ```
/// use bisim_algebra::pepa::{act, passive, PepaModel};
///
/// let mut m = PepaModel::builder();
/// let r = m.param("r", 2.0); // s⁻¹
/// let (idle, busy, server) = (m.seq("Idle"), m.seq("Busy"), m.seq("Server"));
/// m.define(idle, act("work", r).then(busy));
/// m.define(busy, act("rest", 1.0).then(idle) + act("fail", 0.01).then(idle));
/// m.define(server, act("work", passive()).then(server));
/// m.system(idle.times(3).coop(["work"], server));
/// let model = m.build().unwrap();
/// assert_eq!(model.derive().unwrap().num_states(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct PepaBuilder<A> {
    params: Vec<(String, f64)>,
    seqs: Vec<(String, Option<Seq<A>>)>,
    system: Option<ModelTerm<A>>,
}

impl<A: Action> PepaModel<A> {
    pub fn builder() -> PepaBuilder<A> {
        PepaBuilder {
            params: Vec::new(),
            seqs: Vec::new(),
            system: None,
        }
    }
}

impl<A: Action> PepaBuilder<A> {
    /// Declares (or updates) a rate parameter in s⁻¹.
    pub fn param(&mut self, name: &str, value: f64) -> ParamId {
        if let Some(i) = self.params.iter().position(|(n, _)| n == name) {
            self.params[i].1 = value;
            return ParamId(i as u32);
        }
        self.params.push((name.to_owned(), value));
        ParamId(self.params.len() as u32 - 1)
    }

    /// Declares a sequential component name; returns the existing id if known.
    pub fn seq(&mut self, name: &str) -> SeqId {
        if let Some(i) = self.seqs.iter().position(|(n, _)| n == name) {
            return SeqId(i as u32);
        }
        self.seqs.push((name.to_owned(), None));
        SeqId(self.seqs.len() as u32 - 1)
    }

    /// Sets the body of a sequential component.
    pub fn define(&mut self, id: SeqId, body: Seq<A>) -> &mut Self {
        self.seqs[id.0 as usize].1 = Some(body);
        self
    }

    /// Sets the system equation.
    pub fn system(&mut self, term: impl Into<ModelTerm<A>>) -> &mut Self {
        self.system = Some(term.into());
        self
    }

    /// Validates and returns the model.
    pub fn build(&self) -> Result<PepaModel<A>, PepaError> {
        let model = PepaModel {
            params: self.params.clone(),
            seqs: self.seqs.clone(),
            system: self.system.clone().ok_or(PepaError::MissingSystem)?,
        };
        model.validate()?;
        Ok(model)
    }
}

/// The first half of a prefix `(α, r).`; finish it with [`PrefixStart::then`].
#[derive(Debug, Clone)]
pub struct PrefixStart<A> {
    action: A,
    rate: RateExpr,
}

/// Starts a prefix `(action, rate).…`.
pub fn act<A>(action: A, rate: impl Into<RateExpr>) -> PrefixStart<A> {
    PrefixStart {
        action,
        rate: rate.into(),
    }
}

impl<A> PrefixStart<A> {
    /// `(α, r).next`.
    pub fn then(self, next: impl Into<Seq<A>>) -> Seq<A> {
        Seq::Prefix(self.action, self.rate, Box::new(next.into()))
    }
}

/// The passive rate `⊤` (`infty`).
pub fn passive() -> RateExpr {
    passive_weighted(1.0)
}

/// The weighted passive rate `w · ⊤`.
pub fn passive_weighted(weight: f64) -> RateExpr {
    RateExpr::Passive(Box::new(RateExpr::Lit(weight)))
}

/// The deadlocked process `0`, e.g. an absorbing state.
pub fn stop<A>() -> Seq<A> {
    Seq::Choice(Vec::new())
}

/// `S + S`: choice, flattened so `a + b + c` is one three-way choice.
impl<A> Add for Seq<A> {
    type Output = Seq<A>;

    fn add(self, rhs: Seq<A>) -> Seq<A> {
        let mut items = match self {
            Seq::Choice(xs) => xs,
            other => vec![other],
        };
        match rhs {
            Seq::Choice(xs) => items.extend(xs),
            other => items.push(other),
        }
        if items.len() == 1 {
            items.pop().expect("one item")
        } else {
            Seq::Choice(items)
        }
    }
}
