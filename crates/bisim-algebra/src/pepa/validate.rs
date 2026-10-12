use std::collections::BTreeSet;
use std::fmt;

use super::ast::{Action, ModelTerm, PepaModel, RateExpr, Seq, SeqId};
use super::parse::ParseError;

/// Errors raised when building, parsing, validating or deriving a model.
#[derive(Debug, Clone, PartialEq)]
pub enum PepaError {
    Parse(ParseError),
    MissingSystem,
    UndefinedComponent(String),
    UndefinedParameter(String),
    /// `P = Q + …` cycles through constants without passing a prefix.
    UnguardedRecursion(String),
    /// A rate is not a positive finite number (or `infty` is misplaced).
    InvalidRate {
        component: String,
        action: String,
    },
    /// `P[0]` or a non-positive passive weight.
    EmptyArray(String),
    /// A passive action that no active partner synchronises with.
    UnsynchronisedPassive(String),
    /// One component offers the same action both actively and passively.
    MixedRates {
        action: String,
    },
    StateSpaceTooLarge {
        limit: usize,
    },
}

impl fmt::Display for PepaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PepaError::Parse(e) => write!(f, "{e}"),
            PepaError::MissingSystem => write!(f, "model has no system equation"),
            PepaError::UndefinedComponent(n) => {
                write!(f, "component `{n}` is used but never defined")
            }
            PepaError::UndefinedParameter(n) => {
                write!(f, "rate parameter `{n}` is used but never defined")
            }
            PepaError::UnguardedRecursion(n) => {
                write!(
                    f,
                    "component `{n}` recurses without passing through a prefix"
                )
            }
            PepaError::InvalidRate { component, action } => write!(
                f,
                "rate of `{action}` in `{component}` is not a positive finite number or a passive rate"
            ),
            PepaError::EmptyArray(n) => write!(f, "array `{n}[0]` has no members"),
            PepaError::UnsynchronisedPassive(a) => write!(
                f,
                "passive action `{a}` is not synchronised with an active partner"
            ),
            PepaError::MixedRates { action } => write!(
                f,
                "action `{action}` is offered both actively and passively by one component"
            ),
            PepaError::StateSpaceTooLarge { limit } => {
                write!(f, "state space exceeds the limit of {limit} states")
            }
        }
    }
}

impl std::error::Error for PepaError {}

impl From<ParseError> for PepaError {
    fn from(e: ParseError) -> Self {
        PepaError::Parse(e)
    }
}

impl<A: Action> PepaModel<A> {
    /// Static well-formedness checks shared by every front end.
    ///
    /// - every used component is defined and recursion is guarded;
    /// - every rate is a positive finite number or `w · infty` with `w > 0`;
    /// - arrays are non-empty;
    /// - every passive action is synchronised by an active partner.
    pub fn validate(&self) -> Result<(), PepaError> {
        for (i, (name, def)) in self.seqs.iter().enumerate() {
            let Some(def) = def else {
                if self.is_used(SeqId(i as u32)) {
                    return Err(PepaError::UndefinedComponent(name.clone()));
                }
                continue;
            };
            self.check_rates(name, def)?;
            self.check_guarded(SeqId(i as u32))?;
        }
        self.check_term(&self.system)?;
        if let Some(a) = self.open_passive(&self.system).into_iter().next() {
            return Err(PepaError::UnsynchronisedPassive(a.to_string()));
        }
        Ok(())
    }

    /// Non-fatal issues, e.g. cooperation over actions a side never performs.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_warnings(&self.system, &mut out);
        out
    }

    fn collect_warnings(&self, term: &ModelTerm<A>, out: &mut Vec<String>) {
        match term {
            ModelTerm::Coop(l, set, r) => {
                let (al, ar) = (self.term_alphabet(l), self.term_alphabet(r));
                for a in set {
                    if !al.contains(a) || !ar.contains(a) {
                        out.push(format!(
                            "cooperation action `{a}` is not performed by both sides; it will be blocked"
                        ));
                    }
                }
                self.collect_warnings(l, out);
                self.collect_warnings(r, out);
            }
            ModelTerm::Hide(x, _) => self.collect_warnings(x, out),
            ModelTerm::Component(_) | ModelTerm::Array(..) => {}
        }
    }

    fn is_used(&self, id: SeqId) -> bool {
        fn in_seq<A>(s: &Seq<A>, id: SeqId) -> bool {
            match s {
                Seq::Prefix(_, _, k) => in_seq(k, id),
                Seq::Choice(xs) => xs.iter().any(|x| in_seq(x, id)),
                Seq::Const(c) => *c == id,
            }
        }
        fn in_term<A>(t: &ModelTerm<A>, id: SeqId) -> bool {
            match t {
                ModelTerm::Coop(l, _, r) => in_term(l, id) || in_term(r, id),
                ModelTerm::Hide(x, _) => in_term(x, id),
                ModelTerm::Component(c) | ModelTerm::Array(c, _) => *c == id,
            }
        }
        in_term(&self.system, id)
            || self
                .seqs
                .iter()
                .any(|(_, d)| d.as_ref().is_some_and(|d| in_seq(d, id)))
    }

    fn check_rates(&self, name: &str, s: &Seq<A>) -> Result<(), PepaError> {
        match s {
            Seq::Prefix(a, r, k) => {
                let ok = match r.eval(&self.params) {
                    Some(rate) => rate.value() > 0.0,
                    None => false,
                } && !matches!(r, RateExpr::Passive(w) if contains_passive(w));
                if !ok {
                    return Err(PepaError::InvalidRate {
                        component: name.to_owned(),
                        action: a.to_string(),
                    });
                }
                self.check_rates(name, k)
            }
            Seq::Choice(xs) => xs.iter().try_for_each(|x| self.check_rates(name, x)),
            Seq::Const(_) => Ok(()),
        }
    }

    /// Detects a cycle of constants reachable without passing a prefix.
    fn check_guarded(&self, start: SeqId) -> Result<(), PepaError> {
        fn unguarded<A>(s: &Seq<A>, out: &mut Vec<SeqId>) {
            match s {
                Seq::Prefix(..) => {}
                Seq::Choice(xs) => xs.iter().for_each(|x| unguarded(x, out)),
                Seq::Const(c) => out.push(*c),
            }
        }
        let mut stack = vec![start];
        let mut seen = BTreeSet::new();
        while let Some(id) = stack.pop() {
            let Some(def) = &self.seqs[id.0 as usize].1 else {
                return Err(PepaError::UndefinedComponent(self.seq_name(id).to_owned()));
            };
            let mut next = Vec::new();
            unguarded(def, &mut next);
            for n in next {
                if n == start {
                    return Err(PepaError::UnguardedRecursion(
                        self.seq_name(start).to_owned(),
                    ));
                }
                if seen.insert(n) {
                    stack.push(n);
                }
            }
        }
        Ok(())
    }

    fn check_term(&self, t: &ModelTerm<A>) -> Result<(), PepaError> {
        match t {
            ModelTerm::Coop(l, _, r) => {
                self.check_term(l)?;
                self.check_term(r)
            }
            ModelTerm::Hide(x, _) => self.check_term(x),
            ModelTerm::Component(id) => self.require_defined(*id),
            ModelTerm::Array(id, n) => {
                self.require_defined(*id)?;
                if *n == 0 {
                    return Err(PepaError::EmptyArray(self.seq_name(*id).to_owned()));
                }
                Ok(())
            }
        }
    }

    fn require_defined(&self, id: SeqId) -> Result<(), PepaError> {
        match self.seqs.get(id.0 as usize) {
            Some((_, Some(_))) => Ok(()),
            Some((name, None)) => Err(PepaError::UndefinedComponent(name.clone())),
            None => Err(PepaError::UndefinedComponent(format!("#{}", id.0))),
        }
    }

    /// Every action a sequential component can ever perform, with whether it
    /// is ever passive.
    pub(crate) fn seq_alphabet(&self, start: SeqId) -> (BTreeSet<A>, BTreeSet<A>) {
        fn walk<A: Action>(
            s: &Seq<A>,
            all: &mut BTreeSet<A>,
            passive: &mut BTreeSet<A>,
            todo: &mut Vec<SeqId>,
        ) {
            match s {
                Seq::Prefix(a, r, k) => {
                    all.insert(a.clone());
                    if matches!(r, RateExpr::Passive(_)) {
                        passive.insert(a.clone());
                    }
                    walk(k, all, passive, todo);
                }
                Seq::Choice(xs) => xs.iter().for_each(|x| walk(x, all, passive, todo)),
                Seq::Const(c) => todo.push(*c),
            }
        }
        let (mut all, mut passive) = (BTreeSet::new(), BTreeSet::new());
        let mut todo = vec![start];
        let mut seen = BTreeSet::new();
        while let Some(id) = todo.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Some(def) = &self.seqs[id.0 as usize].1 {
                walk(def, &mut all, &mut passive, &mut todo);
            }
        }
        (all, passive)
    }

    fn term_alphabet(&self, t: &ModelTerm<A>) -> BTreeSet<A> {
        match t {
            ModelTerm::Coop(l, _, r) => {
                let mut a = self.term_alphabet(l);
                a.extend(self.term_alphabet(r));
                a
            }
            ModelTerm::Hide(x, set) => self.term_alphabet(x).difference(set).cloned().collect(),
            ModelTerm::Component(id) | ModelTerm::Array(id, _) => self.seq_alphabet(*id).0,
        }
    }

    /// Passive actions a term can perform that are not yet synchronised with
    /// an active partner (a static over-approximation).
    fn open_passive(&self, t: &ModelTerm<A>) -> BTreeSet<A> {
        match t {
            ModelTerm::Component(id) | ModelTerm::Array(id, _) => self.seq_alphabet(*id).1,
            // Hidden passive actions become passive τ: still unsynchronised.
            ModelTerm::Hide(x, _) => self.open_passive(x),
            ModelTerm::Coop(l, set, r) => {
                let (pl, pr) = (self.open_passive(l), self.open_passive(r));
                let (al, ar) = (self.term_alphabet(l), self.term_alphabet(r));
                let mut open = BTreeSet::new();
                for a in pl.iter().chain(&pr) {
                    let still_open = if !set.contains(a) {
                        true
                    } else if !(al.contains(a) && ar.contains(a)) {
                        // Blocked: the partner never performs it (see `warnings`).
                        false
                    } else {
                        // Synchronised; it stays passive only if both sides are.
                        pl.contains(a) && pr.contains(a)
                    };
                    if still_open {
                        open.insert(a.clone());
                    }
                }
                open
            }
        }
    }
}

fn contains_passive(r: &RateExpr) -> bool {
    match r {
        RateExpr::Passive(_) => true,
        RateExpr::Lit(_) | RateExpr::Param(_) => false,
        RateExpr::Add(a, b) | RateExpr::Sub(a, b) | RateExpr::Mul(a, b) | RateExpr::Div(a, b) => {
            contains_passive(a) || contains_passive(b)
        }
    }
}
