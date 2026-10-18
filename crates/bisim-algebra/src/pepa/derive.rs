//! Structural operational semantics: PEPA model → rate-labelled multi-LTS.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use bisim_core::{Real, StateId, Wlts, WltsBuilder};

use super::ast::{Action, ActionSet, ModelTerm, PepaModel, Seq, SeqId};
use super::rate::Rate;
use super::validate::PepaError;

/// Transition label: a visible action or the silent action τ (from hiding).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Act<A> {
    Tau,
    Act(A),
}

impl<A: fmt::Display> fmt::Display for Act<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Act::Tau => write!(f, "tau"),
            Act::Act(a) => write!(f, "{a}"),
        }
    }
}

/// Limits for state-space exploration.
#[derive(Debug, Clone, Copy)]
pub struct DeriveOptions {
    pub max_states: usize,
}

impl Default for DeriveOptions {
    fn default() -> Self {
        Self {
            max_states: 1_000_000,
        }
    }
}

/// The derived state space: a labelled CTMC as a [`Wlts`] with rates (s⁻¹)
/// as weights. Parallel activities are summed, so the weight of
/// `s -a-> t` is the total rate of `a` from `s` to `t`.
#[derive(Debug, Clone)]
pub struct DerivedModel<A> {
    pub lts: Wlts<Act<A>, Real>,
    states: Vec<Vec<u32>>,
    names: Vec<String>,
    layout: Layout,
    local: Local<A>,
    shape: Shape<A>,
}

impl<A: Action> DerivedModel<A> {
    pub fn num_states(&self) -> usize {
        self.lts.num_states()
    }

    /// Human-readable state, e.g. `(Healthy, Vet)` or `{Healthy: 48, Sick: 2}`.
    pub fn state_name(&self, s: StateId) -> String {
        let v = &self.states[s.index()];
        self.layout.describe(v, &self.names)
    }

    /// Names of the components in the system equation, left to right, e.g.
    /// `["Healthy[3]", "Vet"]`. Indices are the `leaf` arguments below.
    pub fn leaves(&self) -> &[String] {
        &self.layout.leaf_names
    }

    /// The raw state vector (local derivative or counts per leaf slot).
    pub fn state_vector(&self, s: StateId) -> &[u32] {
        &self.states[s.index()]
    }

    /// Initial counts as floats, in slot order (see [`Self::fluid_slots`]).
    pub fn initial_counts(&self) -> Vec<f64> {
        self.states[0].iter().map(|&c| c as f64).collect()
    }

    /// `(leaf name, derivative name)` per slot, in slot order. Single
    /// components contribute one slot each (marked as such); see
    /// [`Self::fluid_drift`], which only supports arrays.
    pub fn fluid_slots(&self) -> Vec<(String, String)> {
        slots_of(&self.layout, &self.names)
    }

    /// Mean-field drift `dx/dt` at continuous counts `x` (slot order as in
    /// [`Self::fluid_slots`]): for every array move, its rate (functional
    /// rates evaluated at `x`) times its stoichiometry, summed. This is the
    /// Bio-PEPA fluid semantics restricted to array populations; single
    /// components are an error, since one copy has no mean field.
    pub fn fluid_drift(&self, x: &[f64]) -> Result<Vec<f64>, PepaError> {
        let engine = Engine {
            local: &self.local,
            layout: &self.layout,
            names: &self.names,
        };
        engine.drift(&self.shape, x)
    }

    /// For a component (or array) at the given leaf position (left to right
    /// in the system equation), the number of copies in each local state.
    pub fn local_counts(&self, s: StateId, leaf: usize) -> Vec<(String, u32)> {
        let v = &self.states[s.index()];
        match &self.layout.leaves[leaf] {
            Leaf::Single { offset } => vec![(self.names[v[*offset] as usize].clone(), 1)],
            Leaf::Array { offset, nodes, .. } => nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (self.names[*n as usize].clone(), v[offset + i]))
                .collect(),
        }
    }
}

/// Local derivative graph of all sequential components. Node ids index every
/// syntactic `Seq` subterm; constants are resolved to their definition, so
/// `(a, r).P` and `P`'s own body lead to the same derivative.
#[derive(Debug, Clone)]
struct Local<A> {
    nodes: Vec<Node<A>>,
    roots: Vec<u32>,
    /// Constants defined as another constant (`P = Q;`): their name and the
    /// derivative they stand for, which is named after the target.
    aliases: Vec<(String, u32)>,
}

#[derive(Debug, Clone)]
enum Node<A> {
    Prefix(A, Rate, u32),
    Choice(Vec<u32>),
    Const(SeqId),
}

impl<A: Action> Local<A> {
    fn compile(model: &PepaModel<A>) -> Result<(Self, Vec<String>), PepaError> {
        let mut local = Local {
            nodes: Vec::new(),
            roots: vec![u32::MAX; model.seqs.len()],
            aliases: Vec::new(),
        };
        let mut names = Vec::new();
        for (i, (name, def)) in model.seqs.iter().enumerate() {
            if let Some(def) = def {
                let root = local.add(model, name, def, &mut names)?;
                local.roots[i] = root;
                names[root as usize] = name.clone();
            }
        }
        for (i, (name, def)) in model.seqs.iter().enumerate() {
            if def.is_some() {
                let target = local.canon(local.roots[i]);
                if target != local.roots[i] {
                    local.aliases.push((name.clone(), target));
                }
            }
        }
        Ok((local, names))
    }

    fn add(
        &mut self,
        model: &PepaModel<A>,
        owner: &str,
        s: &Seq<A>,
        names: &mut Vec<String>,
    ) -> Result<u32, PepaError> {
        let node = match s {
            Seq::Prefix(a, r, next) => {
                let rate = r
                    .eval(&model.params)
                    .ok_or_else(|| PepaError::InvalidRate {
                        component: owner.to_owned(),
                        action: a.to_string(),
                    })?;
                let next = self.add(model, owner, next, names)?;
                Node::Prefix(a.clone(), rate, next)
            }
            Seq::Choice(xs) => {
                let mut ids = Vec::with_capacity(xs.len());
                for x in xs {
                    ids.push(self.add(model, owner, x, names)?);
                }
                Node::Choice(ids)
            }
            Seq::Const(id) => Node::Const(*id),
        };
        self.nodes.push(node);
        names.push(model.show_seq(s));
        Ok(self.nodes.len() as u32 - 1)
    }

    /// The canonical derivative for a node: constants resolved to bodies.
    fn canon(&self, mut n: u32) -> u32 {
        while let Node::Const(id) = &self.nodes[n as usize] {
            n = self.roots[id.0 as usize];
        }
        n
    }

    fn root(&self, id: SeqId) -> u32 {
        self.canon(self.roots[id.0 as usize])
    }

    /// One-step local transitions `(action, rate, canonical target)`.
    fn transitions(&self, n: u32, out: &mut Vec<(A, Rate, u32)>) {
        match &self.nodes[n as usize] {
            Node::Prefix(a, r, next) => out.push((a.clone(), r.clone(), self.canon(*next))),
            Node::Choice(xs) => xs.iter().for_each(|x| self.transitions(*x, out)),
            Node::Const(id) => self.transitions(self.root(*id), out),
        }
    }

    /// All canonical derivatives reachable from `start`.
    fn reachable(&self, start: u32) -> Vec<u32> {
        let mut seen = vec![start];
        let mut i = 0;
        let mut buf = Vec::new();
        while i < seen.len() {
            buf.clear();
            self.transitions(seen[i], &mut buf);
            for (_, _, t) in &buf {
                if !seen.contains(t) {
                    seen.push(*t);
                }
            }
            i += 1;
        }
        seen
    }
}

/// Where each leaf of the system equation lives in the flat state vector.
#[derive(Debug, Clone)]
enum Leaf {
    /// One component: the slot holds its local derivative.
    Single { offset: usize },
    /// `P[n]`: one slot per reachable local derivative, holding a count.
    Array {
        offset: usize,
        nodes: Vec<u32>,
        index: HashMap<u32, usize>,
    },
}

#[derive(Debug, Clone)]
enum Shape<A> {
    Leaf(usize),
    Coop(Box<Shape<A>>, ActionSet<A>, Box<Shape<A>>),
    Hide(Box<Shape<A>>, ActionSet<A>),
}

#[derive(Debug, Clone)]
struct Layout {
    leaves: Vec<Leaf>,
    /// Component name of every leaf, e.g. `Vet` or `Healthy[3]`.
    leaf_names: Vec<String>,
}

impl Layout {
    fn describe(&self, v: &[u32], names: &[String]) -> String {
        let parts: Vec<String> = self
            .leaves
            .iter()
            .map(|leaf| match leaf {
                Leaf::Single { offset } => names[v[*offset] as usize].clone(),
                Leaf::Array { offset, nodes, .. } => {
                    let counts: Vec<String> = nodes
                        .iter()
                        .enumerate()
                        .map(|(i, n)| format!("{}: {}", names[*n as usize], v[offset + i]))
                        .collect();
                    format!("{{{}}}", counts.join(", "))
                }
            })
            .collect();
        if parts.len() == 1 {
            parts.into_iter().next().expect("one part")
        } else {
            format!("({})", parts.join(", "))
        }
    }
}

/// A transition of a subterm: label, rate, and slot updates applied in order.
type Move<A> = (Act<A>, Rate, Vec<(usize, u32)>);

struct Engine<'a, A> {
    local: &'a Local<A>,
    layout: &'a Layout,
    names: &'a [String],
}

impl<A: Action> Engine<'_, A> {
    /// Whether derivative `node` is the one called `name`, also under the
    /// name of an alias for it.
    fn is_named(&self, node: u32, name: &str) -> bool {
        self.names[node as usize] == name
            || self
                .local
                .aliases
                .iter()
                .any(|(alias, target)| *target == node && alias == name)
    }

    /// Number of copies currently in local derivative `name`, over all
    /// leaves of the system equation.
    fn count(&self, v: &[u32], name: &str) -> f64 {
        let mut n = 0u64;
        for leaf in &self.layout.leaves {
            match leaf {
                Leaf::Single { offset } => {
                    if self.is_named(v[*offset], name) {
                        n += 1;
                    }
                }
                Leaf::Array { offset, nodes, .. } => {
                    for (slot, node) in nodes.iter().enumerate() {
                        if self.is_named(*node, name) {
                            n += u64::from(v[offset + slot]);
                        }
                    }
                }
            }
        }
        n as f64
    }

    /// Evaluates a (possibly functional) rate in the current global state.
    /// Zero evaluates to a zero rate (dropped downstream like any zero
    /// weight); negative or non-finite values are a model error.
    fn concretize(&self, rate: &Rate, v: &[u32], label: &Act<A>) -> Result<Rate, PepaError> {
        match rate {
            Rate::Functional(f) => {
                let value = f.eval(&|n| self.count(v, n)).unwrap_or(f64::NAN);
                if !value.is_finite() || value < 0.0 {
                    return Err(PepaError::InvalidFunctionalRate {
                        action: label.to_string(),
                        value,
                    });
                }
                Ok(Rate::Active(value))
            }
            other => Ok(other.clone()),
        }
    }

    fn moves(&self, shape: &Shape<A>, v: &[u32]) -> Result<Vec<Move<A>>, PepaError> {
        match shape {
            Shape::Leaf(i) => {
                let mut out = Vec::new();
                let mut buf = Vec::new();
                match &self.layout.leaves[*i] {
                    Leaf::Single { offset } => {
                        self.local.transitions(v[*offset], &mut buf);
                        for (a, r, t) in buf {
                            let label = Act::Act(a);
                            let r = self.concretize(&r, v, &label)?;
                            out.push((label, r, vec![(*offset, t)]));
                        }
                    }
                    Leaf::Array {
                        offset,
                        nodes,
                        index,
                    } => {
                        for (slot, &node) in nodes.iter().enumerate() {
                            let count = v[offset + slot];
                            if count == 0 {
                                continue;
                            }
                            buf.clear();
                            self.local.transitions(node, &mut buf);
                            for (a, r, t) in buf.drain(..) {
                                let label = Act::Act(a);
                                let r = self.concretize(&r, v, &label)?;
                                let to = offset + index[&t];
                                // Two updates: leave `node`, enter `t` (equal for self-loops).
                                let updates = vec![
                                    (offset + slot, count - 1),
                                    (to, v[to] + 1 - u32::from(to == offset + slot)),
                                ];
                                out.push((label, r.scale(f64::from(count)), updates));
                            }
                        }
                    }
                }
                Ok(out)
            }
            Shape::Hide(x, set) => Ok(self
                .moves(x, v)?
                .into_iter()
                .map(|(l, r, u)| match l {
                    Act::Act(a) if set.contains(&a) => (Act::Tau, r, u),
                    other => (other, r, u),
                })
                .collect()),
            Shape::Coop(l, set, r) => {
                let left = self.moves(l, v)?;
                let right = self.moves(r, v)?;
                let mut out = Vec::new();
                let shared = |m: &Move<A>| matches!(&m.0, Act::Act(a) if set.contains(a));
                out.extend(left.iter().filter(|m| !shared(m)).cloned());
                out.extend(right.iter().filter(|m| !shared(m)).cloned());
                for a in set {
                    let ls: Vec<&Move<A>> =
                        left.iter().filter(|m| m.0 == Act::Act(a.clone())).collect();
                    let rs: Vec<&Move<A>> = right
                        .iter()
                        .filter(|m| m.0 == Act::Act(a.clone()))
                        .collect();
                    if ls.is_empty() || rs.is_empty() {
                        continue;
                    }
                    let apparent = |ms: &[&Move<A>]| {
                        ms.iter()
                            .skip(1)
                            .try_fold(ms[0].1.clone(), |acc, m| acc.add(m.1.clone()))
                            .map_err(|_| PepaError::MixedRates {
                                action: a.to_string(),
                            })
                    };
                    let (ra_l, ra_r) = (apparent(&ls)?, apparent(&rs)?);
                    for ml in &ls {
                        for mr in &rs {
                            let rate = Rate::cooperate(
                                ml.1.clone(),
                                ra_l.clone(),
                                mr.1.clone(),
                                ra_r.clone(),
                            );
                            let mut updates = ml.2.clone();
                            updates.extend(mr.2.iter().copied());
                            out.push((Act::Act(a.clone()), rate, updates));
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    /// Continuous counts for the fluid approximation: like [`Self::count`]
    /// but over a float vector.
    fn count_float(&self, x: &[f64], name: &str) -> f64 {
        let mut n = 0.0;
        for leaf in &self.layout.leaves {
            match leaf {
                // Singles never reach here (fluid rejects them first); the
                // bounds check keeps fractional vectors from panicking.
                Leaf::Single { offset } => {
                    let id = x[*offset];
                    if id >= 0.0 && id < self.names.len() as f64 && self.is_named(id as u32, name) {
                        n += 1.0;
                    }
                }
                Leaf::Array { offset, nodes, .. } => {
                    for (slot, node) in nodes.iter().enumerate() {
                        if self.is_named(*node, name) {
                            n += x[offset + slot];
                        }
                    }
                }
            }
        }
        n
    }

    /// Like [`Self::concretize`] but over continuous counts.
    fn concretize_float(&self, rate: &Rate, x: &[f64], label: &Act<A>) -> Result<Rate, PepaError> {
        match rate {
            Rate::Functional(f) => {
                let value = f.eval(&|n| self.count_float(x, n)).unwrap_or(f64::NAN);
                if !value.is_finite() || value < 0.0 {
                    return Err(PepaError::InvalidFunctionalRate {
                        action: label.to_string(),
                        value,
                    });
                }
                Ok(Rate::Active(value))
            }
            other => Ok(other.clone()),
        }
    }

    /// Fluid moves: the float mirror of [`Self::moves`]. Rates are
    /// concretized at `x` (functional rates read continuous counts) and
    /// updates are stoichiometry deltas `(slot, delta)` instead of absolute
    /// assignments. Single components are rejected (one copy has no mean
    /// field); self-loops contribute nothing and are skipped.
    fn dmove(&self, shape: &Shape<A>, x: &[f64]) -> Result<Vec<DMove<A>>, PepaError> {
        match shape {
            Shape::Leaf(i) => {
                let mut out = Vec::new();
                let mut buf = Vec::new();
                match &self.layout.leaves[*i] {
                    Leaf::Single { .. } => {
                        return Err(PepaError::FluidRequiresArrays {
                            component: self.layout.leaf_names[*i].clone(),
                        });
                    }
                    Leaf::Array {
                        offset,
                        nodes,
                        index,
                    } => {
                        for (slot, &node) in nodes.iter().enumerate() {
                            let count = x[offset + slot];
                            if count <= 0.0 {
                                continue;
                            }
                            buf.clear();
                            self.local.transitions(node, &mut buf);
                            for (a, r, t) in buf.drain(..) {
                                let label = Act::Act(a);
                                let r = self.concretize_float(&r, x, &label)?;
                                let to = offset + index[&t];
                                // A self-loop has no net flow (its updates
                                // cancel in `drift`) but still counts towards
                                // cooperation, so it must be kept.
                                let updates = vec![(offset + slot, -1.0), (to, 1.0)];
                                out.push((label, r.scale(count), updates));
                            }
                        }
                    }
                }
                Ok(out)
            }
            Shape::Hide(inner, set) => Ok(self
                .dmove(inner, x)?
                .into_iter()
                .map(|(l, r, u)| match l {
                    Act::Act(a) if set.contains(&a) => (Act::Tau, r, u),
                    other => (other, r, u),
                })
                .collect()),
            Shape::Coop(l, set, r) => {
                let left = self.dmove(l, x)?;
                let right = self.dmove(r, x)?;
                let mut out = Vec::new();
                let shared = |m: &DMove<A>| matches!(&m.0, Act::Act(a) if set.contains(a));
                out.extend(left.iter().filter(|m| !shared(m)).cloned());
                out.extend(right.iter().filter(|m| !shared(m)).cloned());
                for a in set {
                    let ls: Vec<&DMove<A>> =
                        left.iter().filter(|m| m.0 == Act::Act(a.clone())).collect();
                    let rs: Vec<&DMove<A>> = right
                        .iter()
                        .filter(|m| m.0 == Act::Act(a.clone()))
                        .collect();
                    if ls.is_empty() || rs.is_empty() {
                        continue;
                    }
                    let apparent = |ms: &[&DMove<A>]| {
                        ms.iter()
                            .skip(1)
                            .try_fold(ms[0].1.clone(), |acc, m| acc.add(m.1.clone()))
                            .map_err(|_| PepaError::MixedRates {
                                action: a.to_string(),
                            })
                    };
                    let (ra_l, ra_r) = (apparent(&ls)?, apparent(&rs)?);
                    for ml in &ls {
                        for mr in &rs {
                            let rate = Rate::cooperate(
                                ml.1.clone(),
                                ra_l.clone(),
                                mr.1.clone(),
                                ra_r.clone(),
                            );
                            let mut updates = ml.2.clone();
                            updates.extend(mr.2.iter().copied());
                            out.push((Act::Act(a.clone()), rate, updates));
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    /// Mean-field drift `dx/dt` at `x`: every move's rate times its
    /// stoichiometry, summed per slot. Passive rates reaching the top
    /// level are the same modelling error as in SOS derivation.
    fn drift(&self, shape: &Shape<A>, x: &[f64]) -> Result<Vec<f64>, PepaError> {
        let mut dx = vec![0.0; x.len()];
        for (label, rate, updates) in self.dmove(shape, x)? {
            let Rate::Active(r) = rate else {
                return Err(PepaError::UnsynchronisedPassive(label.to_string()));
            };
            for (slot, delta) in updates {
                dx[slot] += r * delta;
            }
        }
        Ok(dx)
    }
}

/// One fluid move: label, concretized rate, stoichiometry deltas.
type DMove<A> = (Act<A>, Rate, Vec<(usize, f64)>);

/// A model compiled for analysis without enumerating its state space:
/// local derivative graphs, slot layout and cooperation shape. This is
/// all the fluid (mean-field) approximation needs, so large populations
/// stay cheap.
#[derive(Debug, Clone)]
pub struct FluidModel<A> {
    names: Vec<String>,
    layout: Layout,
    local: Local<A>,
    shape: Shape<A>,
    init: Vec<u32>,
}

impl<A: Action> FluidModel<A> {
    /// Initial counts as floats, in slot order (see [`Self::fluid_slots`]).
    pub fn initial_counts(&self) -> Vec<f64> {
        self.init.iter().map(|&c| c as f64).collect()
    }

    /// `(leaf name, derivative name)` per slot, in slot order.
    pub fn fluid_slots(&self) -> Vec<(String, String)> {
        slots_of(&self.layout, &self.names)
    }

    /// Names of the components in the system equation, left to right.
    pub fn leaves(&self) -> &[String] {
        &self.layout.leaf_names
    }

    /// Mean-field drift `dx/dt` at continuous counts `x`; see
    /// [`DerivedModel::fluid_drift`].
    pub fn fluid_drift(&self, x: &[f64]) -> Result<Vec<f64>, PepaError> {
        let engine = Engine {
            local: &self.local,
            layout: &self.layout,
            names: &self.names,
        };
        engine.drift(&self.shape, x)
    }
}

/// `(leaf name, derivative name)` per slot, in slot order.
fn slots_of(layout: &Layout, names: &[String]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (leaf, l) in layout.leaves.iter().enumerate() {
        let leaf_name = layout.leaf_names[leaf].clone();
        match l {
            Leaf::Single { .. } => out.push((leaf_name, "(single)".to_owned())),
            Leaf::Array { nodes, .. } => {
                for n in nodes {
                    out.push((leaf_name.clone(), names[*n as usize].clone()));
                }
            }
        }
    }
    out
}

/// Everything derivation needs before state-space enumeration.
struct Prepared<A> {
    local: Local<A>,
    names: Vec<String>,
    layout: Layout,
    shape: Shape<A>,
    init: Vec<u32>,
}

impl<A: Action> PepaModel<A> {
    /// Validates, compiles locals, resolves species names and builds the
    /// cooperation shape: the shared front half of [`Self::derive_with`]
    /// and [`Self::fluid_model`].
    fn prepare(&self) -> Result<Prepared<A>, PepaError> {
        self.validate()?;
        let (local, names) = Local::compile(self)?;
        // Functional rates refer to derivatives by name; resolve them now
        // that every derivative has a compiled name.
        let mut needed = Vec::new();
        for (_, def) in &self.seqs {
            if let Some(def) = def {
                collect_rate_counts(def, &mut needed);
            }
        }
        if !needed.is_empty() {
            let available: Vec<String> = self
                .seqs
                .iter()
                .filter_map(|(n, d)| d.as_ref().map(|_| n.clone()))
                .collect();
            for name in &needed {
                if !names.iter().any(|n| n == name) {
                    return Err(PepaError::UndefinedSpecies {
                        name: name.clone(),
                        available: available.clone(),
                    });
                }
            }
        }
        let mut leaves = Vec::new();
        let mut init = Vec::new();
        let shape = build_shape(&self.system, &local, &mut leaves, &mut init);
        let mut leaf_names = Vec::new();
        self.leaf_names(&self.system, &mut leaf_names);
        let layout = Layout { leaves, leaf_names };
        Ok(Prepared {
            local,
            names,
            layout,
            shape,
            init,
        })
    }

    /// Compiles the model for fluid (mean-field) analysis without
    /// enumerating its state space.
    pub fn fluid_model(&self) -> Result<FluidModel<A>, PepaError> {
        let prepared = self.prepare()?;
        Ok(FluidModel {
            names: prepared.names,
            layout: prepared.layout,
            local: prepared.local,
            shape: prepared.shape,
            init: prepared.init,
        })
    }

    /// Derives the state space with default limits. See [`Self::derive_with`].
    pub fn derive(&self) -> Result<DerivedModel<A>, PepaError> {
        self.derive_with(DeriveOptions::default())
    }

    /// Explores all reachable states by structural operational semantics.
    ///
    /// Arrays `P[n]` are kept in aggregated (counting) form, which is exact
    /// for strong equivalence: the result is lumpably equivalent to expanding
    /// the `n` copies. Passive activities left at the top level are an error.
    pub fn derive_with(&self, options: DeriveOptions) -> Result<DerivedModel<A>, PepaError> {
        let prepared = self.prepare()?;
        let Prepared {
            local,
            names,
            layout,
            shape,
            init,
        } = prepared;
        let engine = Engine {
            local: &local,
            layout: &layout,
            names: &names,
        };

        let mut ids: HashMap<Vec<u32>, StateId> = HashMap::new();
        let mut states = vec![init.clone()];
        ids.insert(init, StateId(0));
        let mut b = WltsBuilder::new();
        b.set_initial(StateId(0));
        let mut queue = VecDeque::from([StateId(0)]);
        while let Some(s) = queue.pop_front() {
            let v = states[s.index()].clone();
            for (label, rate, updates) in engine.moves(&shape, &v)? {
                let Rate::Active(r) = rate else {
                    return Err(PepaError::UnsynchronisedPassive(label.to_string()));
                };
                // Rates are checked when defined, but sums and products of
                // them can still overflow.
                if !r.is_finite() {
                    return Err(PepaError::InvalidRate {
                        component: "the system equation".to_owned(),
                        action: label.to_string(),
                    });
                }
                // A zero rate is no transition: its target must not be
                // discovered either (it may be unreachable).
                if r == 0.0 {
                    continue;
                }
                let mut next = v.clone();
                for (slot, value) in updates {
                    next[slot] = value;
                }
                let t = match ids.get(&next) {
                    Some(&t) => t,
                    None => {
                        if states.len() >= options.max_states {
                            return Err(PepaError::StateSpaceTooLarge {
                                limit: options.max_states,
                            });
                        }
                        let t = StateId(states.len() as u32);
                        ids.insert(next.clone(), t);
                        states.push(next);
                        queue.push_back(t);
                        t
                    }
                };
                b.add_transition(s, label, t, Real(r));
            }
        }
        Ok(DerivedModel {
            lts: b.build(),
            states,
            names,
            layout,
            local,
            shape,
        })
    }
}

/// A component of the system equation, as seen by observation tools.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafInfo {
    /// As written in the system equation, e.g. `Plant[24]` or `Vet`.
    pub name: String,
    /// The sequential component, e.g. `Plant`.
    pub component: String,
    /// Number of copies (1 for a single component).
    pub copies: u32,
    /// No cooperation constrains this leaf: its copies evolve independently
    /// of each other and of the rest of the model, so one copy's chain
    /// (and a binomial count) describes it exactly.
    pub independent: bool,
}

/// One transition of a component's local derivative graph.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalTransition<A> {
    pub source: usize,
    pub target: usize,
    pub action: A,
    pub rate: Rate,
}

/// The local derivative graph of one sequential component, as used by the
/// visualiser: derivatives are named by their definition or their term.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalGraph<A> {
    pub component: String,
    pub states: Vec<String>,
    pub transitions: Vec<LocalTransition<A>>,
}

impl<A: Action> PepaModel<A> {
    fn leaf_names(&self, t: &ModelTerm<A>, out: &mut Vec<String>) {
        match t {
            ModelTerm::Coop(l, _, r) => {
                self.leaf_names(l, out);
                self.leaf_names(r, out);
            }
            ModelTerm::Hide(x, _) => self.leaf_names(x, out),
            ModelTerm::Component(id) => out.push(self.seq_name(*id).to_owned()),
            ModelTerm::Array(id, n) => out.push(format!("{}[{n}]", self.seq_name(*id))),
        }
    }

    /// The components of the system equation, left to right (same order as
    /// [`DerivedModel::leaves`]).
    pub fn leaf_info(&self) -> Vec<LeafInfo> {
        fn walk<A: Action>(
            model: &PepaModel<A>,
            t: &ModelTerm<A>,
            synced: &[&ActionSet<A>],
            out: &mut Vec<LeafInfo>,
        ) {
            match t {
                ModelTerm::Coop(l, set, r) => {
                    let mut inner = synced.to_vec();
                    inner.push(set);
                    walk(model, l, &inner, out);
                    walk(model, r, &inner, out);
                }
                ModelTerm::Hide(x, _) => walk(model, x, synced, out),
                ModelTerm::Component(id) | ModelTerm::Array(id, _) => {
                    let alphabet = model.seq_alphabet(*id).0;
                    let independent = synced
                        .iter()
                        .all(|set| set.iter().all(|a| !alphabet.contains(a)));
                    let (name, copies) = match t {
                        ModelTerm::Array(_, n) => (format!("{}[{n}]", model.seq_name(*id)), *n),
                        _ => (model.seq_name(*id).to_owned(), 1),
                    };
                    out.push(LeafInfo {
                        name,
                        component: model.seq_name(*id).to_owned(),
                        copies,
                        independent,
                    });
                }
            }
        }
        let mut out = Vec::new();
        walk(self, &self.system, &[], &mut out);
        out
    }

    /// Local derivative graphs of the sequential components that occur in the
    /// system equation (each component type once).
    pub fn component_graphs(&self) -> Result<Vec<LocalGraph<A>>, PepaError> {
        self.validate()?;
        let (local, names) = Local::compile(self)?;
        let mut ids = Vec::new();
        fn collect<A>(t: &ModelTerm<A>, ids: &mut Vec<SeqId>) {
            match t {
                ModelTerm::Coop(l, _, r) => {
                    collect(l, ids);
                    collect(r, ids);
                }
                ModelTerm::Hide(x, _) => collect(x, ids),
                ModelTerm::Component(id) | ModelTerm::Array(id, _) => {
                    if !ids.contains(id) {
                        ids.push(*id);
                    }
                }
            }
        }
        collect(&self.system, &mut ids);
        let mut graphs = Vec::new();
        let mut buf = Vec::new();
        for id in ids {
            let nodes = local.reachable(local.root(id));
            let index: HashMap<u32, usize> =
                nodes.iter().enumerate().map(|(i, n)| (*n, i)).collect();
            let mut transitions = Vec::new();
            for (i, &n) in nodes.iter().enumerate() {
                buf.clear();
                local.transitions(n, &mut buf);
                for (action, rate, t) in buf.drain(..) {
                    transitions.push(LocalTransition {
                        source: i,
                        target: index[&t],
                        action,
                        rate,
                    });
                }
            }
            graphs.push(LocalGraph {
                component: self.seq_name(id).to_owned(),
                states: nodes.iter().map(|n| names[*n as usize].clone()).collect(),
                transitions,
            });
        }
        Ok(graphs)
    }
}

/// Every species name (`#Name`) used by functional rates in a component.
fn collect_rate_counts<A>(s: &Seq<A>, out: &mut Vec<String>) {
    match s {
        Seq::Prefix(_, r, k) => {
            r.counts(out);
            collect_rate_counts(k, out);
        }
        Seq::Choice(xs) => xs.iter().for_each(|x| collect_rate_counts(x, out)),
        Seq::Const(_) => {}
    }
}

fn build_shape<A: Action>(
    term: &ModelTerm<A>,
    local: &Local<A>,
    leaves: &mut Vec<Leaf>,
    init: &mut Vec<u32>,
) -> Shape<A> {
    match term {
        ModelTerm::Coop(l, set, r) => Shape::Coop(
            Box::new(build_shape(l, local, leaves, init)),
            set.clone(),
            Box::new(build_shape(r, local, leaves, init)),
        ),
        ModelTerm::Hide(x, set) => {
            Shape::Hide(Box::new(build_shape(x, local, leaves, init)), set.clone())
        }
        ModelTerm::Component(id) => {
            leaves.push(Leaf::Single { offset: init.len() });
            init.push(local.root(*id));
            Shape::Leaf(leaves.len() - 1)
        }
        ModelTerm::Array(id, n) => {
            let root = local.root(*id);
            let nodes = local.reachable(root);
            let index = nodes.iter().enumerate().map(|(i, n)| (*n, i)).collect();
            let offset = init.len();
            init.extend(nodes.iter().map(|&node| if node == root { *n } else { 0 }));
            leaves.push(Leaf::Array {
                offset,
                nodes,
                index,
            });
            Shape::Leaf(leaves.len() - 1)
        }
    }
}
