# SIR epidemic: one model, three languages

Three encodings of an SIR outbreak (9 susceptibles, 1 infected at start),
all deriving the same kind of rate-labelled system in this repo:

| file | language | infection pressure | states |
|---|---|---|---|
| `sir.pepa` | classic PEPA cooperation | `#S * beta` (susceptibles set the pace against a passive carrier) | 55 counting states |
| `sir_fun.pepa` | PEPA + functional rate `beta * #I` | `#S * beta * #I` (true mass action) | 110 counting states |
| `sir.prism` | PRISM guarded commands, rate `beta*s*i` | `#S * beta * #I` (true mass action) | 65 `(s, i)` valuations |

All three agree initially (9 susceptibles, 1 infected: pressure
`9 * 0.3 * 1 = 2.7` per day); the cooperation and mass-action versions
diverge once the infected count moves away from 1 — and the functional
and PRISM chains agree *exactly* (same expected infected count over 30
days; see `crates/bisim-stochastic/tests/sir.rs`). That gap is exactly
why Bio-PEPA added functional rates: PEPA cooperation cannot express a
rate that scales with *both* populations.

```rust
use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::algebra::prism::parse_prism;
use bisimulationrs::refine::minimise;
use bisimulationrs::stochastic::{Ctmc, gillespie};

// Any of the three: derive, lump, solve, or sample.
let model = parse_pepa(include_str!("sir_fun.pepa"))?;
let derived = model.derive()?;
let lumped = minimise(&derived.lts);       // ordinary lumpability
let ctmc = Ctmc::from_wlts(&lumped);
let p = ctmc.transient(&ctmc.initial_distribution(), 30.0, 1e-9);
let traj = gillespie(&derived.lts, derived.lts.initial().unwrap(), 30.0, 10_000, 42);

let prism = parse_prism(include_str!("sir.prism"))?;
let pderived = prism.derive()?;
```

## Queries, rewards and fluid lines

The same SIR model answers CSL-style queries, all cross-checked in
`crates/bisim-stochastic/tests/sir.rs` and `crates/bisim-api/tests/api.rs`:

```rust
// P=? [F<=t (i>0)]: infection still present at day t (CTMC, uniformisation).
let q = query_prism(&prism_text, "P=? [F<=5 i>0]", Options::default())?;
// S=? [s=0]: long-run extinction probability (steady state).
let s = query_prism(&prism_text, "S=? [s=0]", Options::default())?;
// Mean-field ODEs without enumerating 110 states (or 1000s of them):
let f = fluid(&fun_text, &[30.0], 0.05, Options::default())?;
```

```python
prism = bs.PrismModel.from_file("examples/sir/sir.prism")
prism.query("P>=0.9 [F<=5 i>0]")
sir.fluid_over_time([0, 30], dt=0.05)   # same table shape as over_time()
```

## The PEPA version (`sir.pepa`)

Nine susceptibles `S` are exposed to one chronic `Carrier`. Infection is
cooperation on `infect`: susceptibles offer it actively at `beta`, the
carrier passively (`infty`), so each infection attempt runs at the
susceptibles' pace; the newly infected `SI` offers no `infect` itself
and just recovers. Recovery interleaves on its own.

The static roles are the point, not an accident: one array cannot mix
active and passive offers of one action (its apparent rate would be
undefined, hence `MixedRates`), so susceptibles must become a silent
`SI` instead of a shared `I`. The total pressure is therefore
`#S * beta`, independent of how many are infected — and the designated
infector can never change roles. Both rigidities vanish below.

## The functional version (`sir_fun.pepa`)

Bio-PEPA's answer, now native syntax here:

```pepa
S = (infect, beta * #I).I;
```

`#I` is the current number of copies in derivative `I`, evaluated per
global state during derivation. No cooperation needed; 9 susceptibles
with 2 infected give `9 * beta * 2`. Parameters fold to numbers
(`0.3 * #I`), passive weights cannot contain counts, and unknown names
are rejected with the available derivatives listed.

## The PRISM version (`sir.prism`)

The same mass action as a guarded command over integer variables:

```prism
[infect] s > 0 & i > 0 -> beta*s*i : (s'=s-1) & (i'=i+1);
```

Supported subset: `dtmc`/`ctmc`, typed `const`s, `formula` macros,
`min/max/floor/ceil/pow`, ternary, modules with bounded variables,
probabilistic/rate-weighted updates, CSP synchronisation on shared
labels (CTMC rates multiply). Not supported, with explicit errors:
`mdp`, rewards, `system`/`init` blocks, multi-module DTMC.
Analyse it with `bisim_api::analyse_prism` (or `{"op":"analyse_prism"}`
over JSON) — same lumped graphs and steady states as PEPA models.

## Where this repo sits

| question | this repo | modern tool |
|---|---|---|
| small exact CTMC | uniformisation (`ctmc.rs`) — standard | PRISM explicit / Storm sparse |
| large / infinite | independent-component trick + SSA sampling | fluid ODEs, mean-field, SMC |
| exact equivalence | partition refinement + HML witness | Storm bisimulation, mCRL2, CoPaR `O(m log n)` |
| noisy rates | `(ε,δ)` quasi-bisimulation (`approx.rs`) | 2024–25 approx. bisimulation, bisimulation distances |
| language | PEPA (+ functional rates), PRISM subset import | Bio-PEPA compartments/stoichiometry, Modest data/types, CSL/PCTL |

Still future work: full Bio-PEPA compartments, PRISM rewards and
`system` composition, CSL/PCTL checking, symbolic (MTBDD) engines.
