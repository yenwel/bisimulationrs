use std::collections::BTreeSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// Action types usable in PEPA models: a Rust enum, `&'static str`, or
/// [`Sym`] for models read from text.
pub trait Action: Clone + Eq + Ord + Hash + Debug + Display {}

impl<T: Clone + Eq + Ord + Hash + Debug + Display> Action for T {}

/// Action names of models read from text.
pub type Sym = String;

/// Cooperation or hiding set.
pub type ActionSet<A> = BTreeSet<A>;

/// Index of a sequential component definition (`P = …`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(pub u32);

/// Index of a rate parameter (`r = 0.5;`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamId(pub u32);

/// A rate expression in s⁻¹, evaluated against the model's parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum RateExpr {
    Lit(f64),
    Param(ParamId),
    /// Species count `#{name}`: the number of copies currently in local
    /// derivative `name`. This makes the rate *functional* (Bio-PEPA-style):
    /// it is evaluated per global state during derivation (see
    /// [`Rate::Functional`](super::rate::Rate)). Parameters and passive
    /// weights cannot contain counts.
    Count(String),
    /// Passive rate `w · ⊤`; the weight sets the split between alternatives.
    Passive(Box<RateExpr>),
    Add(Box<RateExpr>, Box<RateExpr>),
    Sub(Box<RateExpr>, Box<RateExpr>),
    Mul(Box<RateExpr>, Box<RateExpr>),
    Div(Box<RateExpr>, Box<RateExpr>),
}

impl From<f64> for RateExpr {
    fn from(value: f64) -> Self {
        RateExpr::Lit(value)
    }
}

impl From<ParamId> for RateExpr {
    fn from(id: ParamId) -> Self {
        RateExpr::Param(id)
    }
}

/// Sequential component: `(α, r).S`, `S + S`, a constant, or `0` (empty choice).
#[derive(Debug, Clone, PartialEq)]
pub enum Seq<A> {
    Prefix(A, RateExpr, Box<Seq<A>>),
    /// `Choice(vec![])` is the deadlocked process `0` (e.g. an absorbing state).
    Choice(Vec<Seq<A>>),
    Const(SeqId),
}

impl<A> From<SeqId> for Seq<A> {
    fn from(id: SeqId) -> Self {
        Seq::Const(id)
    }
}

/// Model component built from sequential components.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelTerm<A> {
    /// `P ⋈_L Q`; an empty set is pure interleaving `P || Q`.
    Coop(Box<ModelTerm<A>>, ActionSet<A>, Box<ModelTerm<A>>),
    /// `P / L`: actions in `L` become the silent action τ.
    Hide(Box<ModelTerm<A>>, ActionSet<A>),
    Component(SeqId),
    /// `P[n]`: `n` interleaved copies of `P`, represented by counts per
    /// local state (the numerical vector form used by grouped PEPA).
    Array(SeqId, u32),
}

impl<A> From<SeqId> for ModelTerm<A> {
    fn from(id: SeqId) -> Self {
        ModelTerm::Component(id)
    }
}

impl<A: Action> ModelTerm<A> {
    /// `self ⋈_L other`.
    pub fn coop(self, set: impl IntoIterator<Item = A>, other: impl Into<ModelTerm<A>>) -> Self {
        ModelTerm::Coop(
            Box::new(self),
            set.into_iter().collect(),
            Box::new(other.into()),
        )
    }

    /// `self || other` (cooperation over the empty set).
    pub fn par(self, other: impl Into<ModelTerm<A>>) -> Self {
        self.coop([], other)
    }

    /// `self / L`.
    pub fn hide(self, set: impl IntoIterator<Item = A>) -> Self {
        ModelTerm::Hide(Box::new(self), set.into_iter().collect())
    }
}

impl SeqId {
    /// The component as a model term.
    pub fn comp<A>(self) -> ModelTerm<A> {
        ModelTerm::Component(self)
    }

    /// `P[n]`: `n` copies of this component.
    pub fn times<A>(self, n: u32) -> ModelTerm<A> {
        ModelTerm::Array(self, n)
    }

    /// `P ⋈_L other`.
    pub fn coop<A: Action>(
        self,
        set: impl IntoIterator<Item = A>,
        other: impl Into<ModelTerm<A>>,
    ) -> ModelTerm<A> {
        self.comp().coop(set, other)
    }
}

/// A complete PEPA model: parameters, sequential definitions, system equation.
#[derive(Debug, Clone, PartialEq)]
pub struct PepaModel<A> {
    pub(crate) params: Vec<(String, f64)>,
    pub(crate) seqs: Vec<(String, Option<Seq<A>>)>,
    pub(crate) system: ModelTerm<A>,
}

impl<A: Action> PepaModel<A> {
    pub fn params(&self) -> impl Iterator<Item = (ParamId, &str, f64)> {
        self.params
            .iter()
            .enumerate()
            .map(|(i, (name, v))| (ParamId(i as u32), name.as_str(), *v))
    }

    pub fn param_id(&self, name: &str) -> Option<ParamId> {
        self.params
            .iter()
            .position(|(n, _)| n == name)
            .map(|i| ParamId(i as u32))
    }

    /// Re-instantiates the model with a different parameter value, e.g. a rate
    /// estimated from observed data.
    pub fn with_param(mut self, id: ParamId, value: f64) -> Self {
        self.params[id.0 as usize].1 = value;
        self
    }

    pub fn seq_name(&self, id: SeqId) -> &str {
        &self.seqs[id.0 as usize].0
    }

    pub fn seq_id(&self, name: &str) -> Option<SeqId> {
        self.seqs
            .iter()
            .position(|(n, _)| n == name)
            .map(|i| SeqId(i as u32))
    }

    pub fn definition(&self, id: SeqId) -> Option<&Seq<A>> {
        self.seqs[id.0 as usize].1.as_ref()
    }

    pub fn system(&self) -> &ModelTerm<A> {
        &self.system
    }

    /// The same model with every array `P[n]` replaced by a single `P`.
    /// For components that evolve independently, one copy determines the
    /// behaviour of the whole array, so this is how large populations are
    /// compared.
    pub fn single_copies(&self) -> Self {
        fn term<A: Clone>(t: &ModelTerm<A>) -> ModelTerm<A> {
            match t {
                ModelTerm::Coop(l, set, r) => {
                    ModelTerm::Coop(Box::new(term(l)), set.clone(), Box::new(term(r)))
                }
                ModelTerm::Hide(x, set) => ModelTerm::Hide(Box::new(term(x)), set.clone()),
                ModelTerm::Component(id) | ModelTerm::Array(id, _) => ModelTerm::Component(*id),
            }
        }
        PepaModel {
            params: self.params.clone(),
            seqs: self.seqs.clone(),
            system: term(&self.system),
        }
    }

    /// Maps action names, e.g. from parsed [`Sym`]s into a Rust enum.
    pub fn map_actions<B: Action>(&self, mut f: impl FnMut(&A) -> B) -> PepaModel<B> {
        fn seq<A, B>(s: &Seq<A>, f: &mut impl FnMut(&A) -> B) -> Seq<B> {
            match s {
                Seq::Prefix(a, r, k) => Seq::Prefix(f(a), r.clone(), Box::new(seq(k, f))),
                Seq::Choice(xs) => Seq::Choice(xs.iter().map(|x| seq(x, f)).collect()),
                Seq::Const(id) => Seq::Const(*id),
            }
        }
        fn term<A, B: Action>(t: &ModelTerm<A>, f: &mut impl FnMut(&A) -> B) -> ModelTerm<B> {
            match t {
                ModelTerm::Coop(l, set, r) => ModelTerm::Coop(
                    Box::new(term(l, f)),
                    set.iter().map(&mut *f).collect(),
                    Box::new(term(r, f)),
                ),
                ModelTerm::Hide(x, set) => {
                    ModelTerm::Hide(Box::new(term(x, f)), set.iter().map(&mut *f).collect())
                }
                ModelTerm::Component(id) => ModelTerm::Component(*id),
                ModelTerm::Array(id, n) => ModelTerm::Array(*id, *n),
            }
        }
        PepaModel {
            params: self.params.clone(),
            seqs: self
                .seqs
                .iter()
                .map(|(n, d)| (n.clone(), d.as_ref().map(|s| seq(s, &mut f))))
                .collect(),
            system: term(&self.system, &mut f),
        }
    }
}
