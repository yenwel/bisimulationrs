//! Explicit-state derivation: breadth-first enumeration of the reachable
//! valuations into a rate/probability-labelled [`Wlts`].
//!
//! Unlabelled commands interleave; commands sharing a label synchronise
//! CSP-style: every module offering the label takes one enabled labelled
//! command, rates multiply (CTMC, as in PRISM), updates apply to disjoint
//! variables. Synchronisation is unavailable for DTMC (normalisation is
//! subtle there) and rejected with an explicit error.

use std::collections::{BTreeSet, HashMap, VecDeque};

use bisim_core::{Real, StateId, Wlts, WltsBuilder};

use crate::pepa::{Act, DeriveOptions};

use super::ast::{
    Command, Expr, ModelKind, Module, PrismError, PrismModel, Update, Value, Var, VarKind, VarType,
    eval,
};

/// A variable of the derived state space.
#[derive(Debug, Clone, PartialEq)]
pub struct VarInfo {
    pub name: String,
    pub lo: i64,
    pub hi: i64,
}

/// A concrete transition: label, probability/rate, successor valuation.
type Transition = (Act<String>, f64, Vec<i64>);

/// The derived state space: a labelled DTMC/CTMC as a [`Wlts`] with
/// probabilities/rates as weights, plus the valuations behind each state.
#[derive(Debug, Clone)]
pub struct PrismDerived {
    pub lts: Wlts<Act<String>, Real>,
    pub states: Vec<Vec<i64>>,
    pub vars: Vec<VarInfo>,
}

impl PrismDerived {
    pub fn num_states(&self) -> usize {
        self.lts.num_states()
    }

    /// Human-readable valuation, e.g. `(s = 9, i = 1)`.
    pub fn state_name(&self, s: StateId) -> String {
        let parts: Vec<String> = self
            .vars
            .iter()
            .zip(&self.states[s.index()])
            .map(|(v, x)| format!("{} = {x}", v.name))
            .collect();
        format!("({})", parts.join(", "))
    }

    /// Value of `var` in state `s`.
    pub fn value(&self, s: StateId, var: &str) -> Option<i64> {
        self.vars
            .iter()
            .position(|v| v.name == var)
            .map(|i| self.states[s.index()][i])
    }
}

impl PrismModel {
    /// Derives the state space with default limits. See [`Self::derive_with`].
    pub fn derive(&self) -> Result<PrismDerived, PrismError> {
        self.derive_with(DeriveOptions::default())
    }

    /// Enumerates all valuations reachable from the initial one.
    pub fn derive_with(&self, options: DeriveOptions) -> Result<PrismDerived, PrismError> {
        let kinds: Vec<VarKind> = self
            .vars
            .iter()
            .map(|v| match v.typ {
                VarType::Bool => VarKind::Bool,
                VarType::Int { .. } => VarKind::Int,
            })
            .collect();
        let init = self.initial();
        for (v, x) in self.vars.iter().zip(&init) {
            check_bounds(v.name.as_str(), &v.typ, *x)?;
        }

        let mut ids: HashMap<Vec<i64>, StateId> = HashMap::new();
        let mut states = vec![init.clone()];
        ids.insert(init, StateId(0));
        let mut b = WltsBuilder::new();
        b.set_initial(StateId(0));
        let mut queue = VecDeque::from([StateId(0)]);
        while let Some(s) = queue.pop_front() {
            let st = states[s.index()].clone();
            for (label, weight, next) in self.transitions_from(&st, &kinds)? {
                let t = match ids.get(&next) {
                    Some(&t) => t,
                    None => {
                        if states.len() >= options.max_states {
                            return Err(PrismError::StateSpaceTooLarge {
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
                b.add_transition(s, label, t, Real(weight));
            }
        }
        Ok(PrismDerived {
            lts: b.build(),
            states,
            vars: self
                .vars
                .iter()
                .map(|v| match &v.typ {
                    VarType::Bool => VarInfo {
                        name: v.name.clone(),
                        lo: 0,
                        hi: 1,
                    },
                    VarType::Int { lo, hi } => VarInfo {
                        name: v.name.clone(),
                        lo: *lo,
                        hi: *hi,
                    },
                })
                .collect(),
        })
    }

    /// All concrete `(label, weight, successor)` triples out of `state`.
    fn transitions_from(
        &self,
        state: &[i64],
        kinds: &[VarKind],
    ) -> Result<Vec<Transition>, PrismError> {
        if self.modules.len() == 1 {
            return self.fire_module(&self.modules[0], state, kinds);
        }
        if self.kind == ModelKind::Dtmc {
            return Err(PrismError::Unsupported(
                "multi-module dtmc models are not supported; use one module or ctmc".to_owned(),
            ));
        }
        let mut out = Vec::new();
        // Commands whose labels do not synchronise interleave on their own
        // (unlabelled ones always; others only under `|[...]|` restriction).
        for m in &self.modules {
            for (ci, cmd) in m.commands.iter().enumerate() {
                if self.sync.synchronises(cmd.label.as_deref()) {
                    continue;
                }
                if !guard_true(&m.name, ci, &cmd.guard, state, kinds)? {
                    continue;
                }
                for u in &cmd.updates {
                    let w = weight(&m.name, ci, &u.weight, state, kinds)?;
                    if w == 0.0 {
                        continue;
                    }
                    let label = match &cmd.label {
                        Some(l) => Act::Act(l.clone()),
                        None => Act::Tau,
                    };
                    out.push((label, w, apply(state, &self.vars, u, kinds)?));
                }
            }
        }
        // Synchronising labels combine CSP-style over all modules
        // offering them (restricted to the `system` set, if any).
        let mut labels = BTreeSet::new();
        for m in &self.modules {
            for cmd in &m.commands {
                if let Some(l) = &cmd.label {
                    if self.sync.synchronises(Some(l)) {
                        labels.insert(l.clone());
                    }
                }
            }
        }
        for label in labels {
            let mut per_module: Vec<Vec<(&Command, &Update)>> = Vec::new();
            for m in &self.modules {
                if !m
                    .commands
                    .iter()
                    .any(|c| c.label.as_deref() == Some(&label))
                {
                    continue;
                }
                let mut choices = Vec::new();
                for cmd in m
                    .commands
                    .iter()
                    .filter(|c| c.label.as_deref() == Some(&label))
                {
                    if !guard_true(&m.name, 0, &cmd.guard, state, kinds)? {
                        continue;
                    }
                    for u in &cmd.updates {
                        choices.push((cmd, u));
                    }
                }
                if choices.is_empty() {
                    // One partner cannot move: the synchronisation is blocked.
                    per_module.clear();
                    break;
                }
                per_module.push(choices);
            }
            for combo in product(&per_module) {
                let mut rate = 1.0;
                let mut next = state.to_vec();
                for (_cmd, u) in combo {
                    let w = eval_to_num(&u.weight, state, kinds).ok_or_else(|| {
                        PrismError::Arithmetic(format!(
                            "weight of [{label}] is not finite in this state"
                        ))
                    })?;
                    if w < 0.0 || !w.is_finite() {
                        return Err(PrismError::Arithmetic(format!(
                            "weight {w} of [{label}] is not a non-negative rate"
                        )));
                    }
                    rate *= w;
                    apply_into(&mut next, &self.vars, u, state, kinds)?;
                }
                if rate == 0.0 {
                    continue;
                }
                out.push((Act::Act(label.clone()), rate, next));
            }
        }
        Ok(out)
    }

    /// Fires one module on its own (single-module models, any kind).
    fn fire_module(
        &self,
        m: &Module,
        state: &[i64],
        kinds: &[VarKind],
    ) -> Result<Vec<Transition>, PrismError> {
        let mut enabled: Vec<(usize, &Command)> = Vec::new();
        for (ci, cmd) in m.commands.iter().enumerate() {
            if guard_true(&m.name, ci, &cmd.guard, state, kinds)? {
                enabled.push((ci, cmd));
            }
        }
        if self.kind == ModelKind::Dtmc && enabled.len() > 1 {
            return Err(PrismError::Nondeterministic(format!("module `{}`", m.name)));
        }
        let mut out = Vec::new();
        for (ci, cmd) in enabled {
            let label = match &cmd.label {
                Some(l) => Act::Act(l.clone()),
                None => Act::Tau,
            };
            if self.kind == ModelKind::Dtmc {
                let sum: f64 = cmd
                    .updates
                    .iter()
                    .map(|u| {
                        eval_to_num(&u.weight, state, kinds).ok_or_else(|| {
                            PrismError::Type(format!(
                                "weight of command #{} in `{}` is not a number",
                                ci + 1,
                                m.name
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .sum();
                if (sum - 1.0).abs() > 1e-9 {
                    return Err(PrismError::ProbSum {
                        command: describe(&m.name, ci, &cmd.label),
                        sum,
                    });
                }
            }
            for u in &cmd.updates {
                let w = weight(&m.name, ci, &u.weight, state, kinds)?;
                if w == 0.0 {
                    continue;
                }
                out.push((label.clone(), w, apply(state, &self.vars, u, kinds)?));
            }
        }
        Ok(out)
    }
}

fn describe(module: &str, ci: usize, label: &Option<String>) -> String {
    match label {
        Some(l) => format!("command [{l}] #{} in `{module}`", ci + 1),
        None => format!("command #{} in `{module}`", ci + 1),
    }
}

fn guard_true(
    module: &str,
    ci: usize,
    guard: &Expr,
    state: &[i64],
    kinds: &[VarKind],
) -> Result<bool, PrismError> {
    eval(guard, state, kinds)
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            PrismError::Type(format!(
                "guard of command #{} in `{module}` is not boolean",
                ci + 1
            ))
        })
}

fn eval_to_num(expr: &Expr, state: &[i64], kinds: &[VarKind]) -> Option<f64> {
    eval(expr, state, kinds)?.as_num()
}

/// Weight of one update: a probability (DTMC) or a rate (CTMC).
/// Zero weights are skipped downstream; negative or non-finite ones error.
fn weight(
    module: &str,
    ci: usize,
    expr: &Expr,
    state: &[i64],
    kinds: &[VarKind],
) -> Result<f64, PrismError> {
    let w = eval_to_num(expr, state, kinds).ok_or_else(|| {
        PrismError::Type(format!(
            "weight of command #{} in `{module}` is not a number",
            ci + 1
        ))
    })?;
    if !w.is_finite() || w < 0.0 {
        return Err(PrismError::Arithmetic(format!(
            "weight {w} of command #{} in `{module}` is not a non-negative number",
            ci + 1
        )));
    }
    Ok(w)
}

fn check_bounds(name: &str, typ: &VarType, value: i64) -> Result<(), PrismError> {
    let ok = match typ {
        VarType::Bool => value == 0 || value == 1,
        VarType::Int { lo, hi } => value >= *lo && value <= *hi,
    };
    if ok {
        Ok(())
    } else {
        Err(PrismError::OutOfBounds {
            var: name.to_owned(),
            value,
        })
    }
}

/// Applies one update's assignments simultaneously (all right-hand sides
/// read the pre-update state).
fn apply(
    state: &[i64],
    vars: &[Var],
    update: &Update,
    kinds: &[VarKind],
) -> Result<Vec<i64>, PrismError> {
    let mut next = state.to_vec();
    apply_into(&mut next, vars, update, state, kinds)?;
    Ok(next)
}

fn apply_into(
    next: &mut [i64],
    vars: &[Var],
    update: &Update,
    state: &[i64],
    kinds: &[VarKind],
) -> Result<(), PrismError> {
    for a in &update.assigns {
        let v = eval(&a.expr, state, kinds).ok_or_else(|| {
            PrismError::Arithmetic(format!(
                "assignment to `{}` is not finite",
                vars[a.var].name
            ))
        })?;
        let value = match (&vars[a.var].typ, v) {
            (VarType::Bool, Value::Bool(b)) => i64::from(b),
            (VarType::Bool, Value::Int(0)) => 0,
            (VarType::Bool, Value::Int(1)) => 1,
            (VarType::Int { .. }, Value::Int(i)) => i,
            _ => {
                return Err(PrismError::Type(format!(
                    "cannot assign `{v:?}` to `{}`",
                    vars[a.var].name
                )));
            }
        };
        check_bounds(&vars[a.var].name, &vars[a.var].typ, value)?;
        next[a.var] = value;
    }
    Ok(())
}

/// Cartesian product of per-module choices (one combination per row).
fn product<T: Clone>(sets: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut out: Vec<Vec<T>> = vec![Vec::new()];
    for set in sets {
        let mut next = Vec::new();
        for prefix in &out {
            for item in set {
                let mut row = prefix.clone();
                row.push(item.clone());
                next.push(row);
            }
        }
        out = next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prism::parse_prism;

    fn rates(lts: &bisim_core::Wlts<Act<String>, Real>, s: StateId) -> Vec<(String, f64, StateId)> {
        lts.successors(s)
            .iter()
            .map(|e| (lts.label(e.label).to_string(), e.weight.value(), e.target))
            .collect()
    }

    #[test]
    fn two_state_ctmc_matches_closed_form() {
        let d = parse_prism(
            "ctmc
             module m
               s : [0..1] init 0;
               [] s=0 -> 1.5 : (s'=1);
               [] s=1 -> 2.5 : (s'=0);
             endmodule",
        )
        .unwrap()
        .derive()
        .unwrap();
        assert_eq!(d.num_states(), 2);
        assert_eq!(d.state_name(StateId(0)), "(s = 0)");
        let r0 = rates(&d.lts, StateId(0));
        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].0, "tau");
        assert!((r0[0].1 - 1.5).abs() < 1e-12);
        assert!((rates(&d.lts, StateId(1))[0].1 - 2.5).abs() < 1e-12);
    }

    #[test]
    fn dtmc_coin_and_probability_checks() {
        let d = parse_prism(
            "dtmc
             module coin
               s : [0..2] init 0;
               [] s=0 -> 0.3 : (s'=1) + 0.7 : (s'=2);
               [done] s=1 | s=2 -> 1 : (s'=0);
             endmodule",
        )
        .unwrap()
        .derive()
        .unwrap();
        assert_eq!(d.num_states(), 3);
        let mut r0 = rates(&d.lts, StateId(0));
        r0.sort_by(|a, b| a.1.total_cmp(&b.1));
        assert_eq!(r0[0].0, "tau");
        assert!((r0[0].1 - 0.3).abs() < 1e-12);
        assert!((r0[1].1 - 0.7).abs() < 1e-12);

        // Probabilities must sum to 1 …
        let e = parse_prism(
            "dtmc module m x : [0..1] init 0; [] x=0 -> 0.3 : (x'=1) + 0.6 : (x'=0); endmodule",
        )
        .unwrap()
        .derive()
        .unwrap_err();
        assert!(matches!(e, PrismError::ProbSum { .. }));
        // … and overlapping DTMC guards are nondeterminism, not probability.
        let e = parse_prism(
            "dtmc module m x : [0..1] init 0; [] x=0 -> 1 : (x'=1); [] x=0 -> 1 : (x'=0); endmodule",
        )
        .unwrap()
        .derive()
        .unwrap_err();
        assert!(matches!(e, PrismError::Nondeterministic(_)));
    }

    #[test]
    fn synchronised_rates_multiply() {
        let d = parse_prism(
            "ctmc
             module a
               x : [0..1] init 0;
               [tick] x=0 -> 2.0 : (x'=1);
               [tick] x=1 -> 3.0 : (x'=0);
             endmodule
             module b
               y : [0..1] init 0;
               [tick] y=0 -> 5.0 : (y'=1);
               [tick] y=1 -> 7.0 : (y'=0);
               [] y=1 -> 11.0 : true;
             endmodule",
        )
        .unwrap()
        .derive()
        .unwrap();
        // (0,0) -tick,2*5=10-> (1,1); (1,1) -tick,3*7=21-> (0,0),
        // plus the interleaved self-loop on (1,1) at 11.
        assert_eq!(d.num_states(), 2);
        let r0 = rates(&d.lts, StateId(0));
        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].0, "tick");
        assert!((r0[0].1 - 10.0).abs() < 1e-12);
        let mut r1 = rates(&d.lts, StateId(1));
        r1.sort_by(|a, b| a.1.total_cmp(&b.1));
        assert_eq!(r1.len(), 2);
        assert!((r1[0].1 - 11.0).abs() < 1e-12);
        assert!((r1[1].1 - 21.0).abs() < 1e-12);

        // Multi-module DTMC is rejected explicitly.
        let e = parse_prism(
            "dtmc module a x : [0..1] init 0; [t] x=0 -> 1 : (x'=1); endmodule
             module b y : [0..1] init 0; [t] y=0 -> 1 : (y'=1); endmodule",
        )
        .unwrap()
        .derive()
        .unwrap_err();
        assert!(matches!(e, PrismError::Unsupported(_)));
    }

    #[test]
    fn out_of_bounds_updates_error() {
        let e = parse_prism("ctmc module m x : [0..1] init 1; [] x=1 -> 1 : (x'=x+1); endmodule")
            .unwrap()
            .derive()
            .unwrap_err();
        assert!(matches!(e, PrismError::OutOfBounds { .. }));
    }

    const TWO_TICK: &str = "ctmc
        module a x : [0..1] init 0; [tick] x=0 -> 2.0 : (x'=1); endmodule
        module b y : [0..1] init 0; [tick] y=0 -> 3.0 : (y'=1); endmodule";

    #[test]
    fn system_block_restricts_synchronisation() {
        // Default: full sync on the shared label (single joint transition).
        let d = parse_prism(TWO_TICK).unwrap().derive().unwrap();
        let init = d.lts.initial().unwrap();
        let r: Vec<(String, f64)> = rates(&d.lts, init)
            .into_iter()
            .map(|(l, w, _)| (l, w))
            .collect();
        assert_eq!(r, vec![("tick".to_owned(), 6.0)]);
        // `|[other]|`: `tick` is not synchronised, both interleave.
        let d = parse_prism(&format!("{TWO_TICK} system a |[other]| b endsystem"))
            .unwrap()
            .derive()
            .unwrap();
        let mut r: Vec<(String, f64)> = rates(&d.lts, d.lts.initial().unwrap())
            .into_iter()
            .map(|(l, w, _)| (l, w))
            .collect();
        r.sort_by(|a, b| a.1.total_cmp(&b.1));
        assert_eq!(r, vec![("tick".to_owned(), 2.0), ("tick".to_owned(), 3.0)]);
        // `||` is the default: joint synchronisation, 2 states.
        let d = parse_prism(&format!("{TWO_TICK} system a || b endsystem"))
            .unwrap()
            .derive()
            .unwrap();
        assert_eq!(d.num_states(), 2);
    }

    #[test]
    fn system_block_validates_modules() {
        let e = parse_prism(&format!("{TWO_TICK} system a || c endsystem")).unwrap_err();
        assert!(matches!(e, PrismError::Unknown(_)));
        let e = parse_prism(&format!("{TWO_TICK} system a || a endsystem")).unwrap_err();
        assert!(matches!(e, PrismError::Duplicate(_)));
        let e = parse_prism(&format!("{TWO_TICK} system a[b=c] || b endsystem")).unwrap_err();
        assert!(e.to_string().contains("renaming"));
    }
}
