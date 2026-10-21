//! Shared evaluation of a model's behaviour over time.
//!
//! Components that no cooperation constrains evolve independently, so one
//! copy's small Markov chain describes each copy exactly and counts are
//! binomial. Only when some component interacts is the full state space
//! derived. This keeps a bed of 50 plants as cheap as a single plant.

use bisim_algebra::pepa::{DeriveOptions, DerivedModel, LeafInfo, PepaModel, Rate, Sym};
use bisim_core::{Real, StateId, WltsBuilder};
use bisim_stochastic::{CountDistribution, Ctmc};

use crate::{Diagnostic, Options, Series, diagnostic};

/// One copy of an independent component.
struct LocalChain {
    derivatives: Vec<String>,
    ctmc: Ctmc,
}

pub(crate) struct Evaluator {
    leaves: Vec<LeafInfo>,
    local: Vec<Option<LocalChain>>,
    full: Option<(DerivedModel<Sym>, Ctmc)>,
}

/// Probabilities at a list of times, per leaf (independent) or per full state.
pub(crate) struct Snapshots {
    /// `local[leaf][time][derivative]` for independent leaves.
    local: Vec<Option<Vec<Vec<f64>>>>,
    /// `full[time][state]` when the full chain was needed.
    full: Option<Vec<Vec<f64>>>,
}

impl Snapshots {
    /// Whether every probability is a finite number.
    pub(crate) fn is_finite(&self) -> bool {
        let finite = |rows: &Vec<Vec<f64>>| rows.iter().flatten().all(|p| p.is_finite());
        self.local.iter().flatten().all(finite) && self.full.as_ref().is_none_or(finite)
    }
}

/// Where an evaluation has got to: the distributions at `time`, so that a
/// later time is reached by advancing from here instead of from 0.
pub(crate) struct Cursor {
    time: f64,
    local: Vec<Option<Vec<f64>>>,
    full: Option<Vec<f64>>,
}

/// Distributions at increasing times, advancing from one time to the next.
fn advance(ctmc: &Ctmc, times: &[f64]) -> Vec<Vec<f64>> {
    let mut order: Vec<usize> = (0..times.len()).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
    let mut out = vec![Vec::new(); times.len()];
    let (mut now, mut p) = (0.0, ctmc.initial_distribution());
    for i in order {
        p = ctmc.transient(&p, times[i] - now, 1e-10);
        now = times[i];
        out[i] = p.clone();
    }
    out
}

impl Evaluator {
    pub(crate) fn new(model: &PepaModel<Sym>, options: Options) -> Result<Self, Vec<Diagnostic>> {
        let leaves = model.leaf_info();
        let graphs = model.component_graphs().map_err(|e| vec![diagnostic(&e)])?;
        let local: Vec<Option<LocalChain>> = leaves
            .iter()
            .map(|leaf| {
                if !leaf.independent {
                    return None;
                }
                let g = graphs.iter().find(|g| g.component == leaf.component)?;
                // One copy's chain only describes the leaf when its rates
                // are static: passive rates need a partner, functional
                // rates (#Name) read the global state and couple all copies.
                if g.transitions
                    .iter()
                    .any(|t| t.rate.is_passive() || matches!(t.rate, Rate::Functional(_)))
                {
                    return None;
                }
                let mut b = WltsBuilder::new();
                b.ensure_states(g.states.len());
                b.set_initial(StateId(0));
                for t in &g.transitions {
                    let Rate::Active(r) = t.rate else {
                        unreachable!()
                    };
                    b.add_transition(
                        StateId(t.source as u32),
                        t.action.clone(),
                        StateId(t.target as u32),
                        Real(r),
                    );
                }
                Some(LocalChain {
                    derivatives: g.states.clone(),
                    ctmc: Ctmc::from_wlts(&b.build()),
                })
            })
            .collect();
        let full = if local.iter().all(Option::is_some) {
            None
        } else {
            let derived = model
                .derive_with(DeriveOptions {
                    max_states: options.max_states,
                })
                .map_err(|e| vec![diagnostic(&e)])?;
            let ctmc = Ctmc::from_wlts(&derived.lts);
            Some((derived, ctmc))
        };
        Ok(Self {
            leaves,
            local,
            full,
        })
    }

    pub(crate) fn leaves(&self) -> &[LeafInfo] {
        &self.leaves
    }

    /// Whether this leaf is analysed from one copy's chain (binomial counts).
    pub(crate) fn is_independent(&self, leaf: usize) -> bool {
        self.local[leaf].is_some()
    }

    pub(crate) fn snapshots(&self, times: &[f64]) -> Snapshots {
        Snapshots {
            local: self
                .local
                .iter()
                .map(|c| c.as_ref().map(|c| advance(&c.ctmc, times)))
                .collect(),
            full: self.full.as_ref().map(|(_, ctmc)| advance(ctmc, times)),
        }
    }

    /// The distributions at time 0.
    pub(crate) fn cursor(&self) -> Cursor {
        Cursor {
            time: 0.0,
            local: self
                .local
                .iter()
                .map(|c| c.as_ref().map(|c| c.ctmc.initial_distribution()))
                .collect(),
            full: self
                .full
                .as_ref()
                .map(|(_, ctmc)| ctmc.initial_distribution()),
        }
    }

    /// Snapshots at the single time `t`. Advances `cursor` when `t` is not
    /// earlier than it (the same steps as [`snapshots`](Self::snapshots)
    /// over sorted times); an earlier `t` is computed from 0 and leaves the
    /// cursor where it is.
    pub(crate) fn snapshot_at(&self, cursor: &mut Cursor, t: f64) -> Snapshots {
        if t < cursor.time {
            return self.snapshots(&[t]);
        }
        let dt = t - cursor.time;
        for (c, p) in self.local.iter().zip(&mut cursor.local) {
            if let (Some(c), Some(p)) = (c, p) {
                *p = c.ctmc.transient(p, dt, 1e-10);
            }
        }
        if let (Some((_, ctmc)), Some(p)) = (&self.full, &mut cursor.full) {
            *p = ctmc.transient(p, dt, 1e-10);
        }
        cursor.time = t;
        Snapshots {
            local: cursor
                .local
                .iter()
                .map(|p| p.clone().map(|p| vec![p]))
                .collect(),
            full: cursor.full.clone().map(|p| vec![p]),
        }
    }

    /// Local derivative names of a leaf, in a stable order.
    pub(crate) fn derivatives(&self, leaf: usize) -> Vec<String> {
        if let Some(c) = &self.local[leaf] {
            return c.derivatives.clone();
        }
        let (model, _) = self.full.as_ref().expect("full model for dependent leaves");
        let mut names: Vec<String> = Vec::new();
        for s in model.lts.states() {
            for (d, _) in model.local_counts(s, leaf) {
                if !names.contains(&d) {
                    names.push(d);
                }
            }
        }
        names
    }

    /// Distribution of the number of copies of `leaf` whose derivative is
    /// one of `matched`, at time index `time`.
    pub(crate) fn count_distribution(
        &self,
        snaps: &Snapshots,
        time: usize,
        leaf: usize,
        matched: &[String],
    ) -> CountDistribution {
        let copies = self.leaves[leaf].copies;
        if let (Some(c), Some(probs)) = (&self.local[leaf], &snaps.local[leaf]) {
            let p: f64 = c
                .derivatives
                .iter()
                .zip(&probs[time])
                .filter(|(d, _)| matched.contains(d))
                .map(|(_, p)| p)
                .sum();
            return CountDistribution::binomial(copies, p);
        }
        let (model, _) = self.full.as_ref().expect("full model for dependent leaves");
        let probs = &snaps.full.as_ref().expect("full snapshots")[time];
        let mut pmf = vec![0.0; copies as usize + 1];
        for s in model.lts.states() {
            let k: u32 = model
                .local_counts(s, leaf)
                .into_iter()
                .filter(|(d, _)| matched.contains(d))
                .map(|(_, n)| n)
                .sum();
            pmf[k as usize] += probs[s.index()];
        }
        CountDistribution::from_pmf(pmf)
    }

    /// Expected number of copies of each leaf in each local derivative.
    pub(crate) fn occupancy(&self, snaps: &Snapshots, num_times: usize) -> Vec<Series> {
        let mut series = Vec::new();
        for (leaf, info) in self.leaves.iter().enumerate() {
            for d in self.derivatives(leaf) {
                let values = (0..num_times)
                    .map(|t| {
                        if let (Some(c), Some(probs)) = (&self.local[leaf], &snaps.local[leaf]) {
                            let i = c.derivatives.iter().position(|x| *x == d).expect("known");
                            return f64::from(info.copies) * probs[t][i];
                        }
                        let (model, _) = self.full.as_ref().expect("full model");
                        let probs = &snaps.full.as_ref().expect("full snapshots")[t];
                        model
                            .lts
                            .states()
                            .map(|s| {
                                let n: u32 = model
                                    .local_counts(s, leaf)
                                    .into_iter()
                                    .filter(|(x, _)| *x == d)
                                    .map(|(_, n)| n)
                                    .sum();
                                f64::from(n) * probs[s.index()]
                            })
                            .sum()
                    })
                    .collect();
                series.push(Series {
                    component: info.name.clone(),
                    derivative: d,
                    values,
                });
            }
        }
        series
    }
}
