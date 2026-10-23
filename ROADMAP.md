# Roadmap

Three tracks:
1. making every answer **verifiable**;
2. a **causal** layer;
3. the remaining **engineering** work.

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
