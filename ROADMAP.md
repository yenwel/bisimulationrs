# Roadmap

Three tracks:
1. making every answer **verifiable**;
2. a **causal** layer;
3. the remaining **engineering** work;
4. **scaling** past state-space explosion without giving up sound verdicts.

References are in the [README](README.md#references). Dates are dd/mm/yyyy.

## 1. Fully verifiable

**Goal:** every answer the library gives can be trusted without trusting the
implementation. Either it comes with a certificate that a small, proved checker accepts,
or the code that produced it is itself proved.

### Where we are (23/10/2026)

| layer | status |
|---|---|
| Theory | Lean 4 + Mathlib, no `sorry` (`lean/`): weighted bisimulation is a congruence for parallel composition; PEPA apparent-rate cooperation preserves strong equivalence; lumpings survive contexts; rate interventions can break lumpings |
| "Differs" answers | carry a Hennessy–Milner formula, re-checked by `Formula::holds` in the tests |
| "Same" answers | no certificate yet |
| Algorithms | property and differential tests against independent oracles (naive refinement, game vs refinement, incremental vs replay, distance 0 vs lumping, (ε, δ)₀,₀ vs strong bisimulation), closed forms, mutation fuzzing of the parsers |
| Numerics | `f64`; truncation bounds for uniformisation and interval iteration; certified error bound for the distance; no rounding-error analysis |
| Rust ↔ proofs | no link: the Lean model is a simplification of `bisim-core` |

### Level 1: certificates for every verdict

Make each answer checkable by a short independent checker, in the spirit of the
fixed-point certificates for MDPs (Chatterjee et al., TACAS 2025), whose checker is
proved in Isabelle/HOL.

| answer | certificate | check |
|---|---|---|
| bisimilar / lumpable | the partition | each block has equal class sums for every label (one pass over the transitions) |
| simulation holds | the relation | every move answered inside the relation |
| differs | the HML formula (exists) | evaluate on both systems |
| prefix "holds so far" / final | the explored game positions and refuting moves | replay the refutation counters on the certificate only |
| distance `d ± e` | the coupling and the values `V` | Bellman residual `‖TV − V‖∞ / (1 − γ) ≤ e`, transport plans with correct marginals |
| (ε, δ)-bisimilar | the relation plus one flow per related pair | label and rate conditions, flow value `≥ 1 − ε`, conservation |
| reachability / transient bounds | lower and upper value vectors (interval iteration) | one Bellman step each way |

Deliverables:
- `--certificate` output in the API (a JSON `certificate` field);
- a `bisim-check` crate holding only the checkers, small and dependency-free.

### Level 2: a checker proved in Lean

- Formalise the checkers of Level 1 in Lean over the existing `WLTS` model (`lean/Bisim`).
  Prove *soundness*: checker accepts ⇒ the claimed relation holds. Bisimulation and HML
  semantics are in reach with the current Mathlib setup.
- Run the Lean checkers as executables in CI, against certificates the Rust code emits
  for every test and example model. This gives verified checking of real answers without
  verifying the Rust.
- Formalise PEPA syntax and SOS in Lean (prefix, choice, constants, hiding, cooperation)
  and prove the congruence results for the full language, not only the cooperation rule.

### Level 3: the Rust code itself

- **Aeneas** (Ho & Protzenko, ICFP 2022) translates safe Rust to Lean. The workspace
  already forbids `unsafe` outside the wasm boundary, which suits it. First targets:
  1. the Level 1 checkers (`bisim-check`), so the checker that runs is the one proved;
  2. `refine_fast` (worklist partition refinement). A search found no mechanised
     Paige–Tarjan-style refinement proof, so this is new work;
  3. the incremental game (`live.rs`): the invariant that refutations only grow under
     the prefix discipline, and equality with the batch game.
- **Kani** (bounded model checking of Rust): harnesses for absence of panics and
  overflow in the parsers, the JSON boundary and the wasm allocator. It is cheap and
  runs in CI.
- **Verus** as an alternative where Aeneas does not fit (interior mutability, such as
  the `OnceLock` partitions).

### Level 4: numbers you can trust

- An exact mode with rational arithmetic for small models (lumping, distances, (ε, δ)),
  used as an oracle in tests and on request.
- Interval arithmetic for transient and steady-state results: certified enclosures
  instead of `f64` estimates. Uniformisation truncation is already bounded; add rounding.
- A link to the Isabelle/HOL Markov-chain library (Hölzl, JAR 2017) or a Lean
  counterpart for the semantics of CTMC results.

### Supporting work

- CI that runs `lake build`, the Lean checkers on emitted certificates, Kani, the Python
  and R suites, and the browser smoke test of PEPA Studio.
- Differential tests against PRISM/Storm on shared models (open item in REVIEW.md).
- `cargo-fuzz` targets for the parsers and the JSON entry point (open item in
  REVIEW.md).

## 2. A causal layer

**Status:** model-based what-if and Shapley attribution exist (`what_if`, `attribute`).
Lean proves that interventions expressible as contexts commute with lumping, and that
rate interventions need not. Which causal layer comes next is **Jan's choice, still
open**. The plan below does event causality first, then inference; either can go first.

### 2a. Event causality (true concurrency)

Which observed event made which possible, and comparing systems up to that.

Status (23/10/2026): steps 1, 2, 3 and 5 are done, untimed and for models without
functional rates:
- `PepaModel::causal_net`;
- `bisim_refine::hp_bisimilar`, with a witness formula carrying causes;
- JSON op `causal`;
- example `causal_farm`.

The finite game groups past steps by the tokens that depend on them. That is our own
construction, now proved in Lean to have the same winner as the definition
(`lean/Bisim/Causal.lean`). Not proved: that the Rust search computes that game, and its
witnesses. Step 4 (rates) is open.

1. Give PEPA a causal semantics: unfold models into event structures or occurrence nets,
   where events are causally ordered, concurrent, or in conflict.
2. History-preserving bisimulation on the finite safe fragment (DEXPTIME-complete:
   Jategaonkar & Meyer, TCS 1996). It is decidable there
   (Vogler, ICALP 1991; Montanari & Pistore, STACS 1997). Use the coalgebraic treatment
   of Bruni, Montanari & Sammartino (2015) as the design reference.
3. Canonical test pairs where interleaving bisimilarity and history-preserving
   bisimilarity differ, e.g. `a | b` vs `a.b + b.a`, as the regression suite.
4. A stochastic or timed version. **No literature found**, so this is our own design: it
   must be stated as such, and proved (Lean) before it is relied on.
5. Explanations: when two systems differ causally, give the event pair whose ordering
   differs, analogous to the HML witness.

### 2b. Causal inference on the models

What would have happened under a different plan, identified from farm data where
possible.

1. **Interventions:** the ones that are contexts (replace or add a component) are safe
   to lumping, as proved. Rate interventions need the full model. Make the API say which
   kind an intervention is.
2. **Rate-scaling interventions** on event intensities, matching our parameter
   interventions. Rytgaard & van der Laan (arXiv:2510.16798, 2025) give a causal
   framework and estimators for exactly this kind of intervention in time-to-event data.
3. **Counterfactuals** for Markov-process models via structural causal models (Ness,
   Paneri, Vitek, NeurIPS 2019). For example: what would this bed have produced had it
   been a guild.
4. **Identification from observational records** in continuous time (Ying,
   arXiv:2206.12525) and before/after comparisons with a counterfactual baseline
   (CausalImpact, Brodersen et al. 2015). Report when an effect is *not* identifiable.
5. **Formal side:** relate interventions to string-diagram surgery (Jacobs, Kissinger,
   Zanasi, FoSSaCS 2019) and Markov categories (Fritz). Prove the intervention results
   for the full PEPA semantics in Lean (extends Level 2).

## 3. Engineering

Done (most recent first):

- [x] Compaction of settled prefix positions (`PrefixChecker::compact`, `compact_and_minimise`); verdicts still equal replay.
- [x] Behavioural distances (`bisim-metric`), JSON ops, sessions, Studio panel, Python and R wrappers.
- [x] Incremental prefix games, verdict streams (`streaming`), online rates, observation sessions, warm-started fits, response cache in PEPA Studio.
- [x] Model-based what-if and Shapley attribution.
- [x] Worklist refinement backend (`refine_fast`), on-the-fly game with budgets, simulation preorder on the quotient.
- [x] PEPA (builder, parser, `pepa!`, validator, SOS, functional rates, fluid limits); PRISM/JANI subset import with queries and rewards.
- [x] CTMC transient and steady state, reachability with interval iteration, Gillespie SSA with event logs and rate learning.
- [x] Observations against a plan (dated counts, exact ranges, on-track verdict, fitting).
- [x] PEPA Studio (WebAssembly), Python and R bindings; benchmark suite.

Open:

- [ ] Full counting-based Paige–Tarjan `O(m log n)` backend; symbolic (MTBDD) engine.
- [ ] Up-to techniques (up-to context/congruence) in the game.
- [ ] Full Modest import; CSL `G`/`U`; nested `system` composition; renaming synchronisation; fluid limits for single components.
- [ ] Bounded memory for cyclic streams that never settle: compaction only helps once positions settle.
- [ ] Confidence-aware verdicts (`Unknown(p)`) for models fitted from data.
- [ ] Open items listed in [REVIEW.md](REVIEW.md) ("Not done").

## 4. Scaling

**Goal:** handle models whose state space is too large to enumerate, without weakening
what a verdict means. Every route below says which answers stay **sound** (true of the
model whenever given) and which become estimates. An estimate is never reported as
`Holds` or `Violated`: it is `Unknown`, `Unknown(p)` or a number with its error.

### Where we are (23/10/2026)

| route | saves | sound verdicts | status |
|---|---|---|---|
| Compositional minimisation | the product of the components' sizes, down to the product of their quotients | all: same, differs, quantities | proved (Lean); not wired into derivation |
| Symbolic engine (MTBDD) | memory for structured state spaces | all, when exact | open (section 3) |
| Fluid and mean-field limits | the whole state space, for large populations | none; approximations with limit guarantees | done for PEPA (`fluid_model`), no error bound |
| Statistical model checking | the whole state space | none; confidence-bounded estimates | SSA exists, no stopping rules |
| Learned bisimulation metric (deep RL) | the space of state pairs | none; estimates | done (`learn_metric`): within about 0.01 of the exact distance on the beds where both were computed |
| RL as a witness finder | search effort for refutations | **differs** only, after re-checking the witness | not started |
| Quantum algorithms | at most quadratic search | not applicable today | exploratory only |

### 4a. Compositional minimisation (first)

Minimise each component, or each sub-term of the system equation, before composing,
and compose the quotients. Lean already proves this sound: weighted bisimulation is a
congruence for parallel composition (`IsBisim.par`), PEPA cooperation preserves strong
equivalence (`IsBisim.pepaPar`), and lumpings survive contexts (`lump_in_context`).

- **Sound:** all verdicts and all quantities computed on the composed quotient, for
  strong equivalence. The quotient is exact, not an approximation.
- Not covered by the proofs: hiding, and composing more than two components other than by
  iterating the theorem (Level 2 above extends this).
- Use the ordering heuristics of "smart reduction" (Crouzen & Lang, FMOODS/FORTE 2011):
  pick the composition order that keeps the intermediate products small.
- Emit the partitions as Level 1 certificates, per component.

### 4b. Symbolic engine (MTBDD)

Represent transition systems and partitions as multi-terminal decision diagrams
(Hermanns, Kwiatkowska, Norman, Parker & Siegle, JLAP 2003) and refine partitions
symbolically (Wimmer, Derisavi & Hermanns, *Sigref*, 2006–2010).

- **Sound:** all verdicts, if refinement runs to its fixed point. Memory, not
  correctness, is what is at stake.
- Gains depend on structure (symmetry, regular composition); an unstructured model can
  get larger. Report the diagram size, and fall back to the explicit engine.
- Check: the explicit partition from `refine_fast` agrees on all test models
  (differential test), and symbolic partitions export as Level 1 certificates.

### 4c. Fluid and mean-field limits

Already available for PEPA (`fluid_model`, `fluid_drift`): one ODE per local
derivative, after Hillston (QEST 2005). As populations grow, the scaled CTMC converges to
the ODE on finite time horizons (Kurtz's theorem).

- **Sound:** no verdict. The ODE is a limit, so its answers are estimates for any finite
  population, and say nothing about equivalence of individual states.
- Next: report an error estimate alongside the trajectory, for example a
  second-order (linear noise) correction or a comparison against SSA runs, and state the
  population size and horizon the estimate is for.
- Fluid limits for single components remain open (section 3).
- Equivalences at the fluid level (lumping the ODE itself, as in the constrained lumping
  of Leguizamon-Robayo et al., 2024) are a separate notion and must be named as such.

### 4d. Statistical model checking

Answer quantitative questions by simulation, with explicit confidence: Gillespie SSA
already exists. Add sequential stopping rules (Younes & Simmons, CAV 2002) and
Chernoff–Hoeffding bounds.

- **Sound:** no `Holds`/`Violated`. Answers are `Unknown(p)` or an interval with its
  confidence level, which fits the planned confidence-aware verdicts (section 3).
- A simulated trace that violates a safety property is a genuine counterexample, and can
  be reported as `Violated` with the trace as its witness.

### 4e. Reinforcement learning as a witness finder

Learned state representations approximate bisimulation metrics (Ferns, Panangaden &
Precup, UAI 2004; Gelada et al., *DeepMDP*, ICML 2019; Zhang et al., *Deep Bisimulation
for Control*, ICLR 2021). Use learning only to **guide search**, never to decide:

- train a Spoiler policy that picks moves likely to refute equivalence in the game, or
  that proposes candidate distinguishing formulas;
- every witness it finds is re-checked exactly (`Formula::holds`, `CausalFormula::holds`),
  as all "differs" answers already are.
- **Sound:** **differs**, after the re-check. **Not sound:** "same". A learned partition
  or a failed search proves nothing, and checking a learned partition as a certificate
  needs the full state space again. When the search finds nothing, answer `Unknown`.
- A learned abstraction can still order the exact search (which pairs to explore first).
  That changes speed, not verdicts.
- Causal RL (interventions as actions) relates to section 2b, not to scaling.

### 4e′. Learned bisimulation metric (done, 24/10/2026)

`bisim_metric::learn_metric` trains a small neural network so that ℓ1 distances between
state embeddings approximate the bisimulation distance. The objective follows Deep
Bisimulation for Control (Zhang et al., ICLR 2021). Since the model is known, it uses
exact transport between the successor distributions, which DBC replaces by a learned
Gaussian model.
- **Measured:** 0.120 vs 0.122 and 0.129 vs 0.126 (exact) on the permaculture beds. On
  beds with 1.7 million state pairs it gave 0.143 in 36 s, against an exact 0.134 that took
  8.4 minutes.
- **Training length:** the number of steps needed grows like 1/(1 − γ), and too short a
  training overestimated by 2.5×. The default step count follows that rule.
- **Sound:** nothing; estimates only, never `Holds` or `Violated`.
- **Open:** a sample-only variant (model-free), and features for PEPA models
  generated automatically.

### 4f. Quantum algorithms (exploratory)

Recorded so the trade-off is explicit, not planned as a backend.

- Search (Grover): a quadratic speedup to find a reachable bad state or a refuting
  move. On `2ⁿ` states that is still `2ⁿᐟ²`: exponential.
- Partition refinement is P-complete and no exponential quantum speedup for it is known.
  History-preserving bisimulation is DEXPTIME-complete (section 2a), far beyond what
  quantum computers are believed to solve efficiently.
- Linear systems (HHL, Harrow, Hassidim & Lloyd, PRL 2009) for steady states: the
  speedup needs sparse, well-conditioned systems, quantum input, and readout of
  expectation values only. Classical "dequantized" algorithms (Tang, STOC 2019) have
  removed several claimed speedups of this kind.
- Hardware: no fault-tolerant devices of useful size exist yet.
- **Sound:** a witness found by quantum search would be re-checked classically like any
  other, so "differs" would stay sound. Nothing else.
- Revisit if a problem the library needs is shown to have a super-polynomial quantum
  speedup that survives dequantization.

### Order

1. 4a (compositional minimisation): proved already, exact, largest gain for PEPA.
2. 4b (MTBDD), with differential tests against the explicit engine.
3. 4c and 4d: error estimates and stopping rules, reported as estimates.
4. 4e as an optional witness finder behind a feature flag.
5. 4f stays a note.
