# Library review: bisimulationrs 0.2.0

Review date: 17/10/2026. Scope: all eight workspace crates (~12k lines of
Rust excluding tests) plus the wasm, Python and R bindings.

Baseline at review time: `cargo fmt --all --check`, `cargo clippy --workspace
--all-targets -- -D warnings` and `cargo test --workspace` all pass.

The architecture is sound: a semiring-generic weighted LTS, a reference
partition refinement plus a faster worklist one, Hennessy–Milner witnesses,
and one JSON API shared by wasm/Python/R. The main problem is that several
paths **return wrong answers silently**, and the tests are too small to catch
them.

Unless marked *(from reading)*, every finding below was reproduced by running
it in a scratch crate outside the repo; the quoted outputs are verbatim.

Suggested order: section 1 (items 1, 2, 8, 9 first), then the crash guards
in section 2, then the `refine_fast` and split-tree work in section 3.

## Status

Fixed on 18/10/2026, each with regression tests that failed before the fix:
items 1 (game solver rewritten: counter-based, no recursion, no retracted
assumptions), 2 (relative key grid, exact-zero only, order-independent sums,
invalid weights rejected), 3 (PRISM blocked synchronisation), 4 (fluid
self-loops in cooperation), 5 (0/0 in cooperation), 6 (PRISM/JANI real
division) and 7 (zero-rate moves create no states); also 11 (unbounded
reachability: prob-1 precomputation plus interval iteration), 12 (steady
state: bottom SCCs, exact GTH up to 400 states, Gauss–Seidel with residual
stop beyond, weighted from the initial state) and 13 (uniformisation stops
on a tail bound instead of a mass sum). From section 2: dates (4-digit
years 1583–9999, digits only), infinite time bounds and step bounds above
1e8 rejected, wasm `alloc`/`call` use a boxed slice, and PEPA/PRISM parsers
stop at nesting depth 200. Workspace tests, clippy, fmt and the wasm build
pass. Section 3 (done, 18/10/2026): `refine_fast` hashes only dirty
states, `bisimilar_states`, `compare` and API `compare` run it first and
build witnesses only on a difference, and `refine` stores its rounds as an
O(n) split tree (`Refinement.history` is gone; `up_to` rebuilds a round on
demand). Release timings on the n-state `a`-chain: `refine_fast`
1.05 s → 3.3 ms at n = 10,000 and 0.55 s at n = 1,600,000;
`bisimilar_states(x, x)` 13.6 s → 5 ms at n = 10,000. `refine` itself is
still O(n·rounds) time, and `simulation_preorder` (O(n²) memory) and
`Formula::holds` are untouched. Items 8, 9, 10 and 14 are fixed too: JANI unknown edge locations are
errors, `initial-locations`, `{"exp": …}`, array `variables`, expression
bounds, `null` and positional sync entries, string `result` and
parameter-free functions are accepted, `base` is checked, and nested
operator errors are no longer masked; PEPA parameters must be defined
before use and names only once; `#Name` sees through aliases; PRISM modules
cannot write each other's variables; `integrate` is adaptive (step
doubling, `dt` is the largest step) with the fixed-step scheme kept as
`integrate_fixed`. Not covered from item 8: JANI `bool`/`real` bounded
types beyond the `base` check. Rest of section 2 (done): PRISM formula expansion is bounded (200,000
nested references, then an error); API `compare` skips the quadratic
simulation checks above 2,000 states (new `VerdictView.checked` field;
Markovian and untimed bisimilarity still run); JSON `max_states` is
clamped to 1,000,000 and fluid requests to 5e6 steps; species counts are
summed in `u64` and array sizes are limited to 1e9; the wasm boundary
rejects invalid UTF-8 (a panic there still traps the module: wasm32
aborts on panic, so `catch_unwind` cannot help); parsers bound the tree
depth, so `1 + 1 + … + 1` with 200,000 terms is an error, not a stack
overflow. Also done (18/10/2026): `simulation_preorder` now works on the quotient by
bisimilarity (the table is per class, not per state); `Formula::holds`
caches per (subformula, state) and `Formula::satisfying` evaluates a whole
system; `try_bisimilar`/`try_compare` return an error instead of
panicking; `ParseError` implements `std::error::Error`; `Verdict`,
`PepaError`, `PrismError` and `MatrixError` are `#[non_exhaustive]`; JSON
requests reject unknown fields, accept and report `"version": 1`, and
`forecast` validates `level`; `TimeUnit` accepts the same aliases in JSON as
elsewhere; rustdoc is warning-free; `LICENSE-MIT` added; `Cargo.lock` is no
longer ignored; the dead `rate_sys` helper is gone.

Section 5: `PrefixChecker` (in `bisim-refine`) compares two systems observed
piece by piece, with `complete_left/right(state)` to say that all of a
state's transitions have been seen. `Violated` is final (an observed move
without an answer from a complete state), `Holds` means everything involved
is complete, and the new `Verdict::HoldsSoFar` means no difference yet. A
property test reveals random system pairs in random order and checks that no
violation is ever retracted and that the final verdicts equal `compare`.
Update (21/10/2026): it is now incremental. The three games are updated in place per event (`bisim-refine/src/live.rs`), transitions from complete states are rejected, and the old replay is kept as `replay_verdict`, the reference the incremental answer is tested against after every event.
Testing: mutation fuzzing of all three parsers (90,000 inputs, no panics
found), larger random property tests, stiff and reducible chains with closed
forms. Five runnable examples are in `crates/bisimulationrs/examples/`.

Small fixes (done): PRISM `-0.0 = 0`, integer `pow`, DTMC deadlock self-loop;
negative PEPA parameters print as `0 - x`; `Ctmc::from_wlts` merges parallel
transitions by sorting; observations reject non-finite probabilities; the
fit's likelihood ratio is labelled an approximation. Still open from the
small findings: `with_param` and computed parameters, hidden passive actions
in validation, `learn_rates` horizon, `local_counts` allocation, missing
line/column in a few PEPA errors and macro error positions.

Not done: differential testing against PRISM or Storm (not available
here); `cargo-fuzz` targets (needs nightly); an `LICENSE-APACHE` file (the
manifest says `MIT OR Apache-2.0`; the official text must be added by the
owner); error codes in diagnostics; solver tolerances and convergence flags
in query results; the `moves`/`dmove` and lexer duplication; Python and R
tests were not re-run after the JSON changes (they should be unaffected: the
new `version` key is additive, but `VerdictView` gained `checked`); formula
and witness recursion still use the call stack, so a witness thousands of
steps deep can overflow it.

## 1. Wrong answers

### Bisimulation core

1. **`game_bisimilar` can say `Holds` for non-bisimilar systems.**
   [`game.rs:134`](crates/bisim-refine/src/game.rs) —
   `self.memo.insert((x, y), outcome.clone());` stores `Dup` results that only
   held under the cycle assumption (`return Outcome::Dup;` when the pair is on
   the stack). When that assumed pair later turns out `Spoil`, the cached
   `Dup` stays. On a 16-state example:

   ```
   partition refinement: holds=false "Violated ⟨e⟩⟨c⟩⟨d⟩tt"
   on-the-fly game     : holds=true "Holds"
   ```

   Random fuzzing found more cases at n=8 and n=14. The property test only
   uses `let n = 2 + (round % 5) as u32;` (2–6 states).
   *Fix:* treat `Dup` as tentative until the root of its strongly connected
   component finishes (Tarjan-style); discard on `Spoil`. Raise property-test
   sizes to ~20 states.

   Related *(from reading)*: depth-capped `Dup` results are memoised too, so a
   pair first reached deep (and capped) is reused when reached shallowly, and
   `HoldsUpTo { depth }` can be claimed for systems that differ within that
   depth.

2. **`Real` signature keys use an absolute 1e-9 grid.**
   [`semiring.rs:119`](crates/bisim-core/src/semiring.rs) —
   `RealKey((self.0 * Self::SCALE).round() as i64)`:
   - a rate of 4e-10 s⁻¹ is dropped: `num_transitions=0`;
   - every rate above ~9.2e9 has the same key (the cast saturates):
     `Real(1e10).key()==Real(2e10).key(): true`;
   - NaN counts as zero: `Real(NaN).is_zero()=true`; negatives are accepted;
   - summation order matters: two states with the same rates
     `727511, 115478.57142857143, 172873.99999999997` into equivalent targets,
     listed in a different order, are reported **not bisimilar**:
     `Violated ⟨go⟩=1015863.571428572 tt`.

   *Fix:* sort summands before adding (equal multisets → bit-identical sums),
   drop only exact zeros in `build`, key on a relative grid, and reject
   non-finite and negative weights.

### Front-ends (`bisim-algebra`)

3. **PRISM: a blocked synchronisation emits a fake rate-1 self-loop.**
   [`prism/derive.rs:211`](crates/bisim-algebra/src/prism/derive.rs) —
   `per_module.clear(); break;`, then `product(&[])` yields one empty
   combination. The existing `TWO_TICK` model gets `-tick:1->` self-loops.
   *Fix:* keep a `blocked` flag and `continue` to the next label.
4. **Fluid drift skips self-loops before cooperation.**
   [`pepa/derive.rs:498`](crates/bisim-algebra/src/pepa/derive.rs) —
   `continue; // self-loop: no net flow`. For
   `S[10] <infect> Env[1]` the drift is `[0.0, 0.0, 0.0]`; expected dS = −2.
   Root cause: `moves` and `dmove` are copy-pasted and diverged.
5. **Fluid drift returns NaN for a zero functional rate inside cooperation.**
   [`pepa/rate.rs:146`](crates/bisim-algebra/src/pepa/rate.rs) —
   `let share = (r1.value() / apparent1.value()) * ...` is 0/0. State-space
   derivation only escapes because NaN currently keys as zero (item 2).
6. **PRISM/JANI `/` truncates integers.**
   [`prism/ast.rs:284`](crates/bisim-algebra/src/prism/ast.rs) —
   `(Value::Int(x), Value::Int(y)) => Some(Value::Int(x.checked_div(y)?))`.
   `[] s=0 -> 1/2 : (s'=1)` produces no transition.
7. **PEPA keeps unreachable states for zero-rate moves.**
   [`pepa/derive.rs:741-763`](crates/bisim-algebra/src/pepa/derive.rs) adds
   the target before the weight is checked. `S[3]` with `beta * #I` reports
   4 states, of which 1 is reachable.
8. **JANI:** a misspelled edge `location` silently drops the location guard
   ([`jani.rs:403`](crates/bisim-algebra/src/jani.rs)); several
   spec-conformant shapes are rejected (`initial-locations`, `{"exp": …}`,
   array `variables`, expression `upper-bound`, `null` in sync vectors).
9. **PEPA parameters:** `a = b + 1; b = 5;` gives `a = 1.0`; duplicate
   parameter and process definitions are accepted
   ([`pepa/parse.rs:496-499`](crates/bisim-algebra/src/pepa/parse.rs)).
10. **`#Name` on an alias always counts 0**, and a module can write another
    module's variables in PRISM.

### Numerics (`bisim-stochastic`)

11. **Unbounded reachability stops when one step changes little.**
    [`reach.rs:126`](crates/bisim-stochastic/src/reach.rs) —
    `if diff < tolerance {`. With an escape rate ε = 1e-6, `P=? [F s=2]`
    returns **0.6321**; the true value is 1. ε = 1e-8 returns 0.00995.
    *Fix:* precompute probability-1 states from the graph, then solve to a
    residual tolerance or use interval iteration; report non-convergence.
12. **Steady state is power iteration from a uniform start.**
    [`ctmc.rs:163`](crates/bisim-stochastic/src/ctmc.rs) —
    `let mut p = vec![1.0 / n as f64; n];`.
    - Stiff chain (0↔1 at 1000 s⁻¹, 1→2 at 0.001, 2→1 at 0.01): **0.0863**
      against an exact **0.0476**.
    - Reducible chain (s0→s1 at 1, s0→s2 at 3): `S=? [s=1]` gives **0.4167**
      against **0.25** from the initial state.

    *Fix:* bottom-SCC decomposition, Gauss–Seidel or a direct solver,
    residual stopping, a `converged` flag.
13. **Uniformisation stalls at its 10,000-term cap per slice** when
    `1 − ε/slices` rounds to 1.0. t = 3.0e6 s takes 0.19 s;
    t = 3.00000555e6 s takes 28.3 s. Compute Fox–Glynn weights once and reuse
    them.
14. **Fluid RK4 at the default dt = 0.01 goes negative** for fast rates:
    P = −20.9 at t = 0.5 s for rate 1000 (exact ≈ 0.01). Use an adaptive
    integrator or a stability check.

## 2. Crashes and hangs reachable from wasm, Python and R

- `game_bisimilar` recurses once per pair: two identical 50,000-state chains
  with the **default** budget →
  `thread 'main' has overflowed its stack … fatal runtime error: stack overflow, aborting`.
- The parsers have no depth limit: ~10,000 nested parentheses (~20 KB) overflow
  in release. PRISM formulas are inlined exponentially (22 levels: 4.9 s).
- API `compare` runs `simulation_preorder`, which allocates
  `vec![u32::MAX; n * n]`. At the default `max_states: 20_000` per model the
  union has 40,000 states → ~6.4 GB, beyond wasm32's 4 GB. *(from reading)*
- `P=? [F<=1e400 …]` hangs: the bound parses to +∞ and passes `if *t < 0.0`.
  DTMC `F<=k` is linear in k (k = 1e12 ≈ 5 h).
- Dates: `01/01/99999999999999999` overflows the day computation
  ([`observe.rs:41`](crates/bisim-api/src/observe.rs)); `15/05/26` is
  accepted as the year 26.
- wasm `alloc` uses `Vec::with_capacity(len)` then
  `Vec::from_raw_parts(ptr, len, len)`; capacity equal to `len` is not
  guaranteed. Use `Box<[u8]>` (`into_raw`/`from_raw`).
- No `catch_unwind` at the wasm boundary; `max_states` from JSON is
  unbounded.

## 3. Scalability of the bisimulation algorithms

On an `a`-chain of n states:

| n | `refine` (keeps history) | `refine_fast` | `bisimilar_states(x, x)` |
|---|---|---|---|
| 2,000 | 296 ms, 16 MB | 45 ms | 546 ms |
| 5,000 | 1.96 s, 100 MB | 301 ms | 3.35 s |
| 10,000 | 7.48 s, 400 MB | 1.05 s | 13.6 s |

- **`refine` stores a full partition per round** — O(n²) time and memory.
  `bisimilar_states`, `compare` and API `compare` use it even when the answer
  is "holds". *Fix:* `refine_fast` first, build witnesses only on violation;
  store the history as a split tree (parent block + creation round), O(n).
- **`refine_fast` is also quadratic.** It rehashes the whole block for one
  dirty state ([`fast.rs:118`](crates/bisim-refine/src/fast.rs)). After a
  block is processed, all its clean states share one signature, so only dirty
  states need hashing and comparing against it.
- `simulation_preorder`: O(n²) memory, O(n²·m) per round → Ranzato–Tapparo or
  HHK.
- `Formula::holds` re-evaluates subformulas per successor (exponential in
  depth) → bottom-up satisfaction sets. *(from reading)*
- `witness` recursion depth equals the separation depth.

## 4. API and engineering

- Panics: `bisimilar`/`compare` panic without an initial state (return
  `Result`); `Refinement.history` is `pub` (empty history → `partition()`
  panics); public fields on `PrismModel`/`DerivedModel` allow states that
  panic later.
- Errors: `ParseError` does not implement `std::error::Error`; API errors are
  `Vec<Diagnostic>` with line 0 for non-parse errors; add `#[non_exhaustive]`
  to `Verdict` and the error/option types.
- JSON protocol: unknown fields are ignored (`"DT":5.0` silently uses 0.01) —
  add `#[serde(deny_unknown_fields)]`; add a protocol version; expose solver
  tolerances and convergence status.
- Duplication: `moves`/`dmove`, `concretize`/`concretize_float` and the two
  lexers/parsers are near copies; item 4 is the cost.
- Repo hygiene: no root `LICENSE-MIT`/`LICENSE-APACHE` despite
  `license = "MIT OR Apache-2.0"`; `Cargo.lock` is gitignored although the
  workspace ships binaries; `cargo doc` has unresolved links
  (`Ctmc::reach_bounded`, `Ctmc::embedded`) and links to private `fast`/`game`;
  dead test helper `rate_sys` in
  [`approx.rs:167`](crates/bisim-refine/src/approx.rs).

## 5. Direction

- The README's headline goal (streaming verdicts over `stream_a`/`stream_b`)
  only exists in deprecated `legacy`, whose own doc says it "tests
  transition-set inclusion on state names, which is not simulation". A fixed
  game solver is the natural core for a real incremental checker: lazily
  discovered pairs, invalidated when new transitions arrive.
- Testing that would have caught most of section 1: larger random property
  tests, `cargo-fuzz` on the parsers, stiff and reducible CTMCs with
  closed-form answers, differential runs against PRISM/Storm on `examples/`.

## Literature check (19/10/2026)

References were checked against web search results (titles, authors, venues);
the full texts were not read. Verified: Brodersen et al., CausalImpact
(Annals of Applied Statistics 9(1), 2015, arXiv:1506.00356); Girard and
Pappas, "Approximation metrics for discrete and continuous systems" (IEEE TAC
2007); Zhang et al., "Learning Invariant Representations for Reinforcement
Learning without Reconstruction" (ICLR 2021, arXiv:2006.10742; "deep
bisimulation for control" is the method's informal name and was not found
verbatim); Ostrow et al., dynamical similarity analysis (NeurIPS 2023,
arXiv:2306.10168); Bruni, Montanari and Sammartino, Acta Informatica 52
(2015, doi:10.1007/s00236-014-0213-y; the arXiv preprint 1507.06462 is
titled "A coalgebraic semantics for causality in Petri nets"); Spork et al.,
"A Spectrum of Approximate Probabilistic Bisimulations" (CONCUR 2024,
arXiv:2407.07584) and "Approximate Probabilistic Bisimulation for
Continuous-Time Markov Chains" (CAV 2025, arXiv:2505.15587); Shapley (1953).

Corrected: the Studio tutorial cited "Dissimilarity assertions for
probabilistic model checking" (Spork, Baier, Katoen, 2024), which could not be
found; Baier and Katoen "ch. 10, 24" (the book has ten chapters; bisimulation
is ch. 7, probabilistic systems ch. 10); Larsen's "Modal specifications" is
1990, not 1991; "Monotonicity in performance evaluation" (Jensen, 1993) could
not be found and is replaced by Jensen's 1953 paper. The `(ε, δ)` check in
`bisim-refine::approx` is described as being in the spirit of the CAV 2025
paper but not its definition: there ε is additive on transition probabilities
and δ multiplicative on exit rates, with proved reachability bounds; here ε is
on class-sum rates and no such bound is claimed.
Not checked: Kurtz, Hillston (1996), Sangiorgi (2012) and the older
references in the README list.

## Event causality: status (20/10/2026, updated 23/10/2026)

Update 23/10/2026: implemented, untimed:
- PEPA gets a causal semantics as a safe Petri net (`PepaModel::causal_net`). Its
  interleaving is tested to be bisimilar to the derivation.
- History-preserving bisimulation is decided on a finite game whose positions group
  past steps by the tokens that depend on them (`bisim_refine::hp_bisimilar`). This
  grouping is our own construction. It agrees with the definition, played with full
  histories, on 1,205 random acyclic net pairs, 212 of which differ only in causality.
  Proved in Lean on a model of the game (`lean/Bisim/Causal.lean`, `hpBisimilar_iff`):
  the grouped game and the definition have the same winner. The search with refutation
  counters and the witnesses are not part of the proof.
- Violations carry formulas with causes, re-checked on both nets.
- Functional rates are refused, and rates are not part of the causal comparison.

The original note follows.

Not implemented; left as planned (Tutorial section 12). What a search found
(abstract level, full texts not read): history-preserving bisimilarity is
decidable for finite safe Petri nets (Vogler, ICALP 1991) and finite n-safe
nets (Montanari and Pistore, STACS 1997); decidability of finer variants such
as the reversible and hereditary ones is open or negative, and the
Jurdziński–Nielsen–Srba undecidability result I recalled was not found and is
unverified. For fully concurrent and causal-net bisimilarity on bounded nets,
Cesco and Gorrieri (LMCS 2023) extend the ordered-marking technique. No source
was found that combines causal bisimulation with rates (PEPA, CTMC
semantics), so a stochastic version would be new research. Smaller options
considered: precedence checks ("no fruiting without an earlier flowering in
the same plant") on plan, fitted model and event logs, and an untimed
history-preserving module through a PEPA-to-net translation.

## Mathematical review: cutting edge or dead end? (13/11/2026)

Question from Jan: is what is built here at the cutting edge of mathematics and analytics,
or a dead end? Scope: the mathematics and its fit to the stated purpose, not code quality
(sections 1–4 cover that).

How this was checked:
- Literature: web search on 13/11/2026, at abstract level. Full texts were not read.
- Repository: quotes are verbatim, with file and line as of commit `1088a8e`. Lines marked
  *(annotation)* are the reviewer's own.

### Verdict

- **Mathematics:** not at the cutting edge. The library proves no new theorem of
  substance. It integrates known results with care, some of them from 2024–2025.
- **Analytics for farms:** the headline framing, judging a running farm against a plan
  *by bisimulation*, has a mathematical mismatch that more engineering will not fix
  (Problems 1–3 below). In that framing it is a dead end.
- **Software and verification:** not a dead end. Comparing two models, the explanations,
  the implementations of recent distances with measured numbers, and the Lean-checked
  certificates are real, and rarely found together.

### What is new, component by component

| component | origin of the mathematics | status here |
|---|---|---|
| exact and Markovian bisimulation, lumping, partition refinement | 1980s–1996 (Park, Milner, Paige–Tarjan, Larsen–Skou, Hillston) | implementation. README.md:152–153: "Exact bisimulation is classic (Park, Milner; Paige–Tarjan `O(m log n)`) and remains infrastructure." |
| optimal-transport distance, distance from samples, (ε, δ)-bisimulation | Calo et al. (NeurIPS 2024 and 2025); Spork et al. (CAV 2025) | faithful implementations of other people's recent results, with measured accuracy. Useful to that community; not new mathematics |
| learned distance | Zhang et al. (ICLR 2021) | known method; an estimate only |
| learned challenger | nearest prior work in the README: Desharnais, Laviolette, Zhioua (FSTTCS 2006) | a negative result. README.md:116: "misses 7 of 21 differences" |
| incremental prefix game | counter-based on-the-fly games, plus the observation that refutations only grow when edges come only from open states | sound engineering with no complexity result. The observation is probably folklore; this was not checked |
| history-preserving bisimulation by grouping, proved in Lean (`hpBisimilar_iff`) | REVIEW.md:320: "grouping is our own construction" | **the one candidate original result.** It was not compared in detail with Vogler (1991) or Montanari and Pistore (1997) |
| Lean: congruence for parallel composition, PEPA cooperation, lumping in context | textbook theorems (Hillston 1996) | mechanised for a simplified model. One search found no proof-assistant development of PEPA's strong equivalence, so the cooperation proof may be a first as a formalisation. One search does not prove absence |
| Lean certificates | certifying algorithms; Chatterjee et al. (TACAS 2025) for MDPs | a standard technique, on four small models. Prior certified lumping checkers were not searched for, so no novelty is claimed |
| dynamical similarity analysis | Ostrow et al. (NeurIPS 2023) | implementation; not a bisimulation |

### Problem 1: a run does not show branching

Bisimulation differs from trace equivalence only in the choices that were *not* taken.
The repository's own example shows it.
crates/bisimulationrs/examples/compare_systems.rs:3–4: "`a.(b + c) + a.b` and `a.(b + c)`
simulate each other but are not bisimilar".

*(annotation)* Both systems have the same runs: `a`, `a b`, `a c`. So every run of one is
a run of the other, and no log of runs, however long, tells them apart.

There are three ways to see more than runs. Each needs something a farm log does not give:
- **An experimenter** who offers actions and sees which are refused. That is active
  testing, not a passive log. Van Glabbeek's linear time – branching time spectrum
  classifies equivalences by such testing scenarios (known from the search summary; the
  paper was not read).
- **Probabilities and many runs.** These identify the distribution over runs, and still
  not the branching. *(annotation)* `a.(½ b + ½ c)` and `½ a.b + ½ a.c` give the same
  distribution over runs, `a b` and `a c` with probability ½ each, and are not
  probabilistically bisimilar.
- **Observed states.** The object is then a Markov chain whose state is known at each
  observation, and what the data identify are its transition rates. This is the farm
  case: dated counts per local state. Bisimulation's role here is lumping, which says
  which states may be merged without changing the law of the counts. "Is the system the
  plan?" becomes a statistical test on rates.

So on farm data, the extra resolving power that bisimulation has over simpler notions is
not what is being used.

The prefix checker does not escape this. It needs every state marked complete once all its
moves are known, so it is told the transition structure state by state. That is a lazy
comparison of two models. It is useful, but it is not inference from a run.

Process mining chose the trace level explicitly. Stochastic conformance checking compares
an event log and a stochastic model as distributions over traces, with the earth mover's
distance (Leemans, van der Aalst, Brockhoff, Polyvyanyy, Information Systems 102, 2021).

### Problem 2: the question is statistical, and most guarantees are missing

"Is the running system the plan?", asked of finite data, is a hypothesis test.

- **One answer has a controlled error rate.** `check_observations` tests dated counts
  against the plan. crates/bisim-api/src/observe.rs:5–7: "The model gives the exact
  distribution of that count on that date, hence an expected value, a range that holds
  with the chosen probability (95 % by default) and a p-value. The overall verdict adjusts
  for the number of counts checked (Bonferroni)". This is the part that works at farm
  scale. It is a goodness-of-fit test of a continuous-time Markov chain: classical
  statistics, not coinduction. Established tools cover it, including hidden Markov models
  for misclassified states and piecewise-constant time-varying intensities (Jackson,
  *Multi-State Models for Panel Data: The msm Package for R*, Journal of Statistical
  Software 38(8), 2011).
- **The answers that use bisimulation on data have none.**
  - The distance from samples needs independent draws.
    crates/bisim-metric/src/samples.rs:21–24: "The guarantee (Theorem 1) assumes each pair
    is an independent draw from the discounted occupancy (restart after a geometric time;
    see [`occupancy_sample`]); consecutive pairs of one long stream are the paper's
    practical variant, outside the guarantee." A farm log is one long stream.
  - It also needs far more data than a farm has. README.md:109: "needs very many
    transitions (≈10⁶ for ±0.07 on 4 states)". A season gives hundreds to thousands of
    observations *(annotation: an estimate, not measured)*.
  - Exact bisimilarity is discontinuous in the rates. Any perturbation breaks it, so a
    model fitted from data almost surely "differs" from the plan. The distances repair
    this for two *models*. For data they need a fitted model first, and then the
    uncertainty of the fit is the whole question. `OnlineRates` has credible intervals
    for the rates (crates/bisim-stochastic/src/simulate.rs:352), but they do not reach
    the verdicts. ROADMAP.md:176 lists that as open: "Confidence-aware verdicts
    (`Unknown(p)`) for models fitted from data."
  - The (ε, δ) guarantee is weak where it is needed. README.md:108: "the bound is loose
    and grows with the horizon".
- **The rigorous theory exists and is not used.** Identity testing of Markov chains asks
  whether a single observed trajectory comes from a reference chain or from one far from
  it. Wolfer and Kontorovich (*Minimax Testing of Identity to a Reference Ergodic Markov
  Chain*, AISTATS 2020) give sample-complexity bounds that they report as nearly tight,
  and that depend on the reference chain only. *(annotation)* Their setting is discrete
  time; a continuous-time version would also have to test the holding times.

### Problem 3: the model class

- A CTMC has exponential holding times and constant rates. Growth stages of plants and
  animals last about a typical time, and rates follow seasons and weather.
  - Stage durations can be repaired with chains of sub-stages (phase-type distributions),
    at the cost of more states.
  - Time-varying rates are not supported. A grep of `crates/*/src` for "inhomogeneous",
    "time-varying", "seasonal" and "piecewise" finds nothing.
- Mainstream crop analytics works with continuous states and noisy, partial observations:
  process-based crop models with data assimilation (ensemble Kalman and particle
  filters). A review: *Progress and perspectives in data assimilation algorithms for
  remote sensing and crop growth model* (Science of Remote Sensing, December 2024). This
  library does not touch that area.

### The motivating argument

README.md:5–6: "Classical monitoring judges a crop, a herd or a company by its final
yield. When the harvest is continuous, as in permaculture, there is no final yield to wait
for."

The observation is right. The step from it to coinduction does not follow. Never-ending
behaviour is also handled by long-run averages: stationary distributions and yield per
unit time, which `bisim-stochastic` already computes. Coinduction is needed to *define and
prove equality of the infinite behaviours of models*. It is not needed to judge a
continuous harvest against a target.

### Where it is alive

- **Model against model**, both fully known: lumping and minimisation, comparing variants
  of a plan, distances with the state pairs that drive them, causal comparison with
  witnesses. This is the classical, sound use.
- **Checked answers:** certificates verified by the Lean kernel. Certified model checking
  is an active area (Chatterjee et al., TACAS 2025).
- **Behavioural metrics** are an active research area, mostly in reinforcement learning.
  A June 2025 survey of state representation learning (arXiv:2506.17518) covers
  bisimulation metrics (known from the search summary; not read). The library follows
  that work closely.
- **Smart-contract equivalence** (ROADMAP §5c): both sides are known transition systems,
  so Problem 1 does not arise. Of the new directions this is the cleanest fit.
- **Agent monitoring** (ROADMAP §5b) needs a caveat: a run is again a trace. What can be
  checked live is trace-level safety, as in runtime verification. Branching time helps
  only when two specifications, or two extracted models, are compared.
- **Teaching:** PEPA Studio and its tutorial.

### What would put it at the edge

Pick one, and prove or measure something nobody has.

1. **Lump, then test.** A finite-sample test of "the running system is within ε of the
   plan", run on the lumped chain. The sample complexity of identity testing grows with
   the number of states. Lumping n identical components with k local states reduces kⁿ
   states to C(n + k − 1, k − 1). A theorem with an experiment would connect bisimulation
   to statistics and give the farm framing a sound basis. Prior work on this combination
   was not searched for.
2. **Incremental Markovian lumping** with complexity bounds. The search in October 2026
   found Saha (FSTTCS 2007) for plain bisimulation, and nothing recent for the Markovian
   case.
3. **A mechanised proof of worklist partition refinement.** ROADMAP.md:72–73: "A search
   found no mechanised Paige–Tarjan-style refinement proof, so this is new work".
4. **The grouping construction for history-preserving bisimulation.** Compare it with
   Vogler and with Montanari and Pistore, with someone who knows that literature, and
   write it up if it is new. The Lean proof exists already.
5. **Causality with rates.** REVIEW.md:339: "a stochastic version would be new research".

### Three ideas that might be new (conjectures, 13/11/2026)

Asked by Jan: can anything new be imagined mathematically? Each idea below answers one of
the three problems above. None is proved here. Ideas 1 and 3 were built and tested on
20/11/2026; their results follow their descriptions. Searches on 13/11/2026 found the
ingredients of each idea and not the combination. That is weak evidence of novelty, not
proof. Authors, titles and venues in this subsection were checked against Crossref.

#### Idea 1: bet against the plan (answers Problem 2)

- **Setting:** the plan is a CTMC `P`. The real system is any CTMC `Q`, on any state
  space, seen through an observation that bisimulation respects (counts, labels). The
  null hypothesis `H0` is "`Q` is Markovian bisimilar to `P`". That is a very large set of
  systems.
- **Step 1 (known):** every `Q` in `H0` gives the observed process the *same law* as `P`
  does, because the block process of a lumpable chain is its quotient chain (Buchholz,
  *Exact and ordinary lumpability in finite Markov chains*, J. Appl. Probab. 31(1), 1994).
  On what is observed, `H0` is a single distribution.
- **Step 2 (known):** take any alternative `A`. The likelihood ratio `L` of the observed
  path, `A` against `P`, is a non-negative martingale under every member of `H0`. By
  Ville's inequality, the probability that `L` ever reaches `1/α` is at most `α`.
  - "Violated at level α" is then valid at any stopping time and never has to be
    withdrawn. That is the statistical counterpart of the prefix discipline, where a
    final refutation stays final.
  - A move the plan cannot make gives `L = ∞`. The logical verdict `Violated` is the
    special case of infinite evidence.
- **Conjecture 1a (sufficiency):** among all tests that are valid for every member of
  `H0`, nothing is lost by looking only at the block path. The block-level likelihood
  ratio is the growth-optimal bet against `H0`. *(annotation: proof idea)* For any `A`,
  build a member of `H0` that copies `A`'s behaviour inside the blocks. The reverse
  information projection of `A` onto `H0` then has exactly the block-level divergence.
- **Conjecture 1b (detection time):** the expected time until `L` reaches `1/α` is about
  `log(1/α) / D`, where `D` is the relative entropy rate of the real block process against
  the plan's quotient.
  - When the real system has the plan's structure with other rates, `D` has a closed form.
    Per unit time in a block it is the sum, over the transitions with plan rate `r` and
    real rate `r'`, of `r' log(r'/r) − r' + r`.
  - *(annotation: computed here for a known alternative, not measured)* At α = 0.05 this
    is about 2,600 events when the rates are 5 % off, 680 at 10 %, 190 at 20 % and 40 at
    50 %. Compare README.md:109: "≈10⁶ for ±0.07 on 4 states".
- **Open part:** when the real system is not lumpable over the plan's blocks, its block
  process is a hidden Markov process, and `D` has no closed form in general *(from
  memory)*. A fixed-point bound on `D`, computable like the existing distance, would be
  new. For aggregating a single chain, Geiger, Petrov, Kubin and Koeppl (*Optimal
  Kullback–Leibler Aggregation via Information Bottleneck*, IEEE TAC 60(4), 2015) have an
  upper bound that is tight under lumpability.
- **Known nearby:**
  - anytime-valid inference (Ramdas, Grünwald, Vovk, Shafer, *Game-Theoretic Statistics
    and Safe Anytime-Valid Inference*, Statistical Science 38(4), 2023);
  - a chi-squared test of lumpability (Jernigan and Baran, Statistics & Probability
    Letters 64, 2003);
  - identity testing (Wolfer and Kontorovich, above).

  The search found nothing that joins e-processes with lumpability or bisimulation.
- **First step:** `OnlineRates` already keeps counts and dwell times per transition with
  Gamma priors, which is what a mixture likelihood ratio needs. Build the monitor, check
  1a and 1b by simulation, then prove them.

**Built and tested (20/11/2026).** The monitor is `bisim_stochastic::PlanEvidence`
(crates/bisim-stochastic/src/evidence.rs), with tests in
crates/bisim-stochastic/tests/evidence.rs and a study in
`cargo run --release -p bisimulationrs --example bet_against_plan`.

- **The bet:** a mixture likelihood ratio against the plan's quotient, with Gamma priors
  centred on the plan's rates. It combines two bets:
  - one factor per activity (label): half the stake on exactly one activity being off,
    half on all of them;
  - one rate per transition of the quotient.

  `Pooling::Both` averages the two bets, and it is never worse than the better one by
  more than `ln 2` nats.
- **Explanations:** it reports which activity is off and by what factor (`labels()`), the
  statistical counterpart of a distinguishing formula.
- **Test model:** all numbers below are from a simulated herd of 4 animals:
  - rates per day: sicken 0.02, heal 0.1, die 0.01, replace 0.05;
  - observed as counts (healthy, sick, dead) plus every event.

  The output below is verbatim from the study; the reviewer's own lines are marked
  *(annotation)*.

**Validity holds.**
```
1. False alarms within 3000 events, 2000 runs (Ville: at most 5 % and 1 %)
   system (bisimilar to the plan)      states   at 95 %   at 99 %
   the plan itself                         15      4.8 %      0.7 %
   each animal on its own                  81      4.3 %      0.8 %
   random split, 3 copies per state        45      4.7 %      0.8 %
```
*(annotation)* A separate run of 20,000 histories of the plan gave, for each bet:

| bet | false alarms at 95 % | at 99 % |
|---|---|---|
| per label | 4.70 % | 0.96 % |
| per transition | 4.52 % | 0.89 % |
| both | 4.59 % | 0.87 % |

The bound is nearly tight. False alarms come early: half before the 9th event, three
quarters before the 26th.

**Conjecture 1b holds for an oracle, and with a regret term for the real bet.**
```
2. Events until "off plan" at 95 %, all animals heal at a factor of the plan (300 runs)
   factor  nats/1000     Wald  plug-in |   oracle per label per trans     both | both med
     1.05      0.512     5847    20290 |     6089     16552    179063    17880 |    16910
     1.10      1.936     1547     5010 |     1608      4146     32328     4547 |     4372
     1.20      6.958      431     1300 |      439      1068      5529     1173 |     1165
     1.50     32.787       91      253 |       92       190       448      208 |      188
     0.90      2.465     1216     3880 |     1209      3008     26683     3345 |     3007
```
- **Oracle:** an oracle that knows the factor matches Wald's `ln(1/α) / D` to within 4 %
  in every row.
- **The per-label bet:** it does not know the factor, and needs 2.1–2.8 times the
  oracle's events. That is the price of the unknown factor (about `½ ln n` nats per
  activity).
- **The per-transition bet:** about 30 times the oracle's events at 5 %, because it has
  one unknown rate per transition.
- **The plug-in prediction** (expected counts in the bet, which includes the regret) is
  10–22 % above the measured mean. A first passage comes before the mean path crosses.
- **Correction to the annotation above:** the figure of "about 2,600 events when the rates
  are 5 % off" counted healing events only. In this herd about 44 % of events are
  healings, so the oracle needs 6,089 events and the bet 17,880.
- *(annotation, computed and not simulated)* The divergence per event does not depend on
  the herd size, and the regret grows only logarithmically. A herd of 100 animals with
  these rates makes about 3.4 events per day. Healing that is 5 % off would then take
  about 14 years to call, 10 % about 3.7 years, 20 % about a year and 50 % about two
  months.

**Conjecture 1a is false as stated, and true for lumpable alternatives.**
- **The divergences:** write `f(λ, q) = λ ln(λ/q) − λ + q`. Let `λ_A(s, a, B')` be the
  alternative's rate from state `s` under label `a` into observed block `B'`, and
  `q(B, a, B')` the plan's. There are three growth rates:
  - `D_M = Σ_B π(B) Σ f(λ̄, q)`, with `λ̄` the occupancy-weighted average rate: what the
    bet reaches (`Divergences::per_transition`);
  - `D_obs`: the relative entropy rate of what is observed;
  - `D_full = Σ_s π(s) Σ f(λ_A(s, a, B'), q)`: the rate for a tester that also sees the
    internal states (`Divergences::full`).
- **Why `D_M ≤ D_obs ≤ D_full`** *(annotation: proof sketch)*:
  - `D_obs = D(A_obs ‖ M̄) + D_M`, where `M̄` is the Markov chain with the averaged rates.
    So `D_M ≤ D_obs`.
  - Lumpability pins every member of `H0` to the plan's class sums *in every internal
    state*. So the per-state likelihood ratio is a martingale under all of `H0` at once,
    and the internal tester can bet at rate `D_full`.
  - Data processing gives `D_obs ≤ D_full`.
  - When the alternative is lumpable on the observation, all three are equal.
- **Measured:**
```
3. Alternatives that average out: divergence rates (nats per 1000 events)
   alternative                        per label per trans      full      found  full, Wald
   hidden sickness phases                 0.000     0.000   239.952      3.3 %          12
                                      found after 171 events on average (median 5); plug-in inf
   2 heal at 0.18/day, 2 at the rest      0.000     0.581    43.238      6.3 %          69
                                      found after 861 events on average (median 17); plug-in 201723
   (slow rate 0.0635/day; found = called off plan at 95 % within 20000 events, 300 runs.
   full, Wald = events a tester that saw every animal's state would need.)

   One animal with hidden phases, nats per 1000 days: per transition 0.0000 ≤ observed 1.7981 ≤ full 10.3455
```
  - **Hidden sickness phases:** they are tuned so that healing per sick day averages to
    the plan's. The bet finds them no more often than it raises false alarms.
  - **Two fast and two slow healers:** they are tuned the same way. The bet barely sees
    them. A tester that saw every animal would need about 69 events.
  - **One animal:** `D_obs` comes from numerical integration over one sick period (the
    observed process is semi-Markov there). The three rates are strictly ordered.

**What is new, and what is not** *(annotation)*:
- **Classical ingredients:** likelihood-ratio martingales, Ville's inequality and Wald's
  approximation.
- **Possibly new:** using lumpability to turn the composite hypothesis into one law; the
  per-state pinning that makes internal-state tests valid for all of `H0`; and the
  ordering `D_M ≤ D_obs ≤ D_full`. These were not checked against the literature beyond
  the search of 13/11/2026.
- **Open:**
  - a bet that reaches `D_obs`, which needs hidden-state alternatives such as phase-type
    sick periods (Idea 3);
  - a computable bound on `D_obs`;
  - a proof of the plug-in prediction;
  - the JSON API, Studio and Python/R wrappers.

#### Idea 2: the bisimulation game as experiment design (answers Problem 1)

- A passive log shows runs only. An experimenter who *acts* sees more. For reactive
  probabilistic systems, bisimilarity is characterised by testing (Larsen and Skou,
  *Bisimulation through probabilistic testing*, Information and Computation 94(1), 1991;
  the statement is from memory). On a farm the actions are interventions: treat or not,
  prune or not, move the herd.
- **Idea:** read Spoiler as a gambler. A challenge is an intervention, and a
  distinguishing formula is a bet on its outcome. Game-theoretic probability (Shafer and
  Vovk, 2001) already treats testing as a betting game between Skeptic and Reality.
- **Conjecture 2a:** for finite reactive systems with observed outcomes, Skeptic has an
  intervention-and-betting strategy with positive expected growth exactly when the two
  systems are not bisimilar. *(annotation)* One direction should follow from Larsen and
  Skou.
- **Conjecture 2b:** the best achievable growth rate is a new behavioural divergence. It
  solves a Bellman equation over pairs of states: choose the intervention whose two
  outcome laws differ most now, plus the expected value afterwards. It is zero exactly
  for bisimilar states, and its optimal policy is the most informative experiment.
- **What it would give:** a ranked answer to "what should we try next on the farm to tell
  plan from reality fastest?", with a detection-time guarantee. It would also join three
  tracks of this library: the game, the statistics, and the interventions of ROADMAP §2b.
- **Known nearby:** the search found no link between betting games and bisimulation
  games. The nearest hit is a BSc thesis (Nakagawa) that treats Markov chains in
  game-theoretic probability and mentions bisimulation. It was not read.
- **Risk:** hidden state. If the state after an intervention is not observed, the Bellman
  equation runs over beliefs and may be intractable.

#### Idea 3: causal equivalence as robustness to durations (answers Problem 3)

- With exponential durations, `a ∥ b` and `a.b + b.a` give the same Markov chain: when
  `a` finishes, the remaining time of `b` is exponential again. With any other duration
  law they differ, because in `a ∥ b` the activity `b` has been running all along. This
  is the difference between two workers and one farmer in the `causal_farm` example.
- **Conjecture 3:** two PEPA models stay Markovian bisimilar under *every* replacement of
  the exponential durations by phase-type durations (the same replacement for the same
  action) exactly when they are equivalent in a rate-aware causal sense.
  - Rate-aware history-preserving bisimilarity should be sufficient.
  - The exact characterisation is probably an ST-type bisimulation, which is coarser
    *(from memory, not verified)*.
- **What it would give:** a "same" verdict that does not depend on the exponential
  assumption, which is the assumption Problem 3 attacks. The library already decides the
  untimed causal comparison, with a Lean proof of its game.
- **Known nearby:** ST semantics is the standard way to give process algebras general
  durations:
  - Bravetti, Bernardo, Gorrieri, *Towards performance evaluation with general
    distributions in process algebras* (CONCUR 1998);
  - Bravetti and Gorrieri, *The theory of interactive generalized semi-Markov processes*
    (Theoretical Computer Science 282(1), 2002);
  - van Glabbeek and Goltz, *Refinement of actions and equivalence notions for concurrent
    systems* (Acta Informatica 37(4–5), 2001), for the untimed case.

  The search did not show whether the "exactly when" above is already a theorem for PEPA.
  REVIEW.md:339 found no source combining causal bisimulation with rates.
- **First step:** test it. Expand each activity into Erlang stages, lump with
  `refine_fast`, and compare with `hp_bisimilar` on random nets. A counterexample is as
  valuable as support.

**Tested (20/11/2026).** The code is `bisim_refine::durations`
(crates/bisim-refine/src/durations.rs). The search is
`cargo run --release -p bisimulationrs --example durations`, and a smaller regression is
crates/bisim-refine/tests/durations.rs.

- **Setting.** A model is a safe net whose transitions carry rates. Durations follow the
  race policy with enabling memory: every enabled transition runs its own clock, the
  first clock to finish fires, and a clock is kept while its transition stays enabled.
  This is a stochastic net, not PEPA. `PepaModel::causal_net` leaves rates out, and the
  apparent-rate rule ties the rates of nested cooperations to the global state.
- **`erlang_expansion`** replaces each clock of rate `r` by `k` stages of rate `k·r`, the
  same mean, with stage changes as internal moves.
- **The conjecture, sharpened.** History-preserving bisimilarity is the wrong candidate.
  The right one is *enabling-preserving bisimilarity* (van Glabbeek, Höfner and Wang,
  *Enabling Preserving Bisimulation Equivalence*, CONCUR 2021, LIPIcs,
  doi:10.4230/LIPIcs.CONCUR.2021.33, arXiv:2108.00142). Its related states also pair
  their enabled transitions, and the pairing of transitions that stay enabled is kept.
  The rated version (`rated_ep_bisimilar`) requires two more things: the pairing is one
  to one, and paired transitions have equal labels and rates.
- **Conjecture 3′:** rated ep-bisimilar ⇔ Markovian bisimilar after every Erlang (or
  phase-type) expansion.
  - "⇒" holds by construction: match the clocks through the pairing.
  - "⇐" is open.
- **The search.** Random tiny nets were grouped into classes that are Markovian bisimilar
  with exponential clocks, and pairs inside a class were compared. The nets have up to 3
  components, 3 places each, and 4 or 7 transitions, with labels a/b and rates 1/2.
  Output (verbatim, abridged to the counts):
```
2427 distinct nets in 593 classes
1. Random pairs within a class: 16172 pairs Markovian bisimilar with exponential clocks
   EP 13334   E2 13334   E3 13334   HP 16088
   EP but not E2 or E3 (would refute EP ⇒ E): 0
   E2 or E3 but not EP (would refute the conjecture): 0
   E2 and E3 disagree: 0
   HP but not EP: 2754 (1436 of them through split clocks)   EP but not HP: 0
   not E2, yet no observable difference in up to 6 events: 0

2. Nets against their sequentialisation: 2176 pairs Markovian bisimilar with exponential clocks
   EP  1544   E2  1544   E3  1544   HP  1544
   (no disagreement of any kind)

1177 distinct nets in 78 classes
3. One label, one rate: pairs within a class: 11504 pairs Markovian bisimilar with exponential clocks
   EP 11280   E2 11280   E3 11280   HP 11280
   (no disagreement of any kind)

4377 distinct nets in 2052 classes
4. Up to 7 transitions: pairs within a class: 18779 pairs Markovian bisimilar with exponential clocks
   EP 14677   E2 14677   E3 14677   HP 18569
   EP but not E2 or E3 (would refute EP ⇒ E): 0
   E2 or E3 but not EP (would refute the conjecture): 0
   E2 and E3 disagree: 0
   HP but not EP: 3892 (1805 of them through split clocks)   EP but not HP: 0
   not E2, yet no observable difference in up to 6 events: 0
```
- **What it shows** *(annotation)*:
  - Conjecture 3′ held on all 48,631 pairs, and two Erlang stages were always enough.
  - Every pair that Erlang-2 separates is also separated by what is *observed*: the
    probability of some sequence of up to 6 events by time 0.5, 1, 2 or 5, with stages
    hidden. So for these nets the stronger comparison, with stages visible, loses
    nothing that matters.
- **Idea 3 as first stated is wrong in one respect.** History-preserving bisimilarity
  ignores rates, so it accepts pairs whose clocks are split differently. One example is a
  single `a` clock of rate 2 against two of rate 1, which match only for exponential
  clocks. A rate-aware version would have to match transitions one to one: rated EP.
  On these samples rated EP always implied HP. Whether it does in general was not
  checked.
- **What it would give:** a "same" verdict that survives any duration law with the same
  means (race policy, enabling memory), decided by a game on the net instead of by a
  numerical comparison.
- **Open:**
  - a proof of "⇐", or a counterexample on larger or unsafe nets;
  - other memory policies (age memory, preselection);
  - durations whose shape differs between transitions with the same rate;
  - weak versions for hidden actions.
- **Literature** *(from a search, abstracts only)*: no source was found that
  characterises Markovian bisimilarity under general durations by ep-bisimilarity.
  Bravetti and Gorrieri's ST-based semantics (above) is the closest.

**Proof attempt (21/11/2026).** The converse is proved in one case and stays open in the
other.

*Theorem (type-unambiguous systems).* Suppose no reachable state enables two transitions
with the same label and rate. If the Erlang-2 expansions of two states are Markovian
bisimilar, the states are rated ep-bisimilar. With the easy direction, the following are
then equivalent:
- rated ep-bisimilar;
- Markovian bisimilar after the Erlang-2 expansion;
- Markovian bisimilar after every phase-type expansion that depends only on label and rate.

So two stages decide robustness to every duration law with the same means.

*Proof sketch.* Write `(m, S)` for a state of the expansion, where `S` is the set of clocks
that completed their first stage.
1. **Totals.** Class sums determine every class-invariant total. In particular,
   bisimilar states have the same visible rate per label: the rates of the clocks in `S`
   with that label.
2. **Matching first stages.** If `(m₁, S₁) ~ (m₂, S₂)`, every internal step adding `t` is
   matched by one adding some `t'`, into the same class. The visible rates then grow by
   the same amount, so `t` and `t'` have the same label and rate.
3. **Lifting.** Without ambiguous types, step 2 pins `t'`: it is the unique partner of `t`.
   By induction, `(m₁, ∅) ~ (m₂, ∅)` implies `(m₁, S) ~ (m₂, β(S))` for every `S`, where
   `β` pairs the clocks of equal type.
4. **Firing.** From `(m₁, {t})`, the only visible step fires `t`. So the states after `t`
   and after `β(t)` fire are bisimilar again.
5. **Clocks that keep running.** From `(m₁, {t, s})`, fire `t`. The visible rate for
   `s`'s label afterwards is `s`'s rate if `s` kept running, else 0. If the matching step
   fires `β(s)` instead, the two share a label but not a rate, and neither can have kept
   the other running. Either way `s` keeps running exactly when `β(s)` does.
6. **Closing the argument.** "`(m₁, ∅) ~ (m₂, ∅)` and `β` pairs equal types" is then a
   rated ep-bisimulation. The new pairing agrees with the old one on the clocks that keep
   running, because the types are unambiguous.

*Lean.* `lean/Bisim/Durations.lean`, `epBisimilar_of_erlang2`. It uses an abstract clock
system: states, enabled clocks, `next` and `kept`. Any Markovian bisimulation of the
expansion will do, not only the largest. It is checked only in GitHub CI, on the scratch
branch `ci/duration-proof`. The easy direction and the link from nets to clock systems
are not formalised.

*The general case.* It is open, and the obstruction is concrete. Step 3 needs "phase
reset": if `(m₁, S₁) ~ (m₂, S₂)`, then `(m₁, ∅) ~ (m₂, ∅)`. That fails once types repeat.
*(annotation: worked by hand)* Take two conflicting clocks `a`, rate 1, leading to the same
state, against one clock `a`, rate 2:
- with both first stages completed, each side fires `a` at total rate 2 into the same
  state, so they are bisimilar;
- fresh, the first side can complete one stage and show `a` at rate 1, which the second
  cannot, so they are not.

So the pairing can no longer be read off stage by stage. Any proof must use the fact that
the two systems start out bisimilar while still fresh.

*More evidence for the general case.* A fifth search used one label and one rate, up to 4
components and 8 transitions: 15,489 more pairs, all in agreement. Across the five
searches, 64,120 pairs were compared:
- 40,012 have unambiguous types, a case now proved;
- 24,108 do not; they are evidence only, and none disagreed.

### Recommendation

- Reword the README's claim. The library compares *models* exactly, and data against a
  model *statistically*. Say which answers come with a guarantee.
- Stop adding comparison methods until one bisimulation-based answer on data has a
  calibrated error.
- Choose one direction. Idea 1 is the sharper form of candidate 1: it serves the original
  purpose and can start from existing code. Candidate 4 is the closest to done.
- *Update 20/11/2026:* Idea 1 is built (`PlanEvidence`). It is the first answer on data
  with a calibrated error. Small deviations need many events, and alternatives that
  average out stay invisible to it.
- *Update 20/11/2026:* Idea 3 is tested (`durations`). Rated enabling-preserving
  bisimilarity agreed with Erlang expansions on all 48,631 random pairs. That is the
  best-supported candidate for a new theorem here: a proof of the converse would give
  "same" verdicts that do not depend on exponential durations.
- *Update 21/11/2026:* the converse is proved, and checked in Lean, for systems where no
  state enables two transitions of the same label and rate. Two Erlang stages decide
  robustness to every duration law with the same means. Populations of identical
  individuals fall outside it, and the general case is open.

### Sources (searched 13/11/2026)

- Wolfer & Kontorovich, AISTATS 2020: https://proceedings.mlr.press/v108/wolfer20a.html
- Leemans, van der Aalst, Brockhoff, Polyvyanyy, *Stochastic process mining: Earth movers' stochastic conformance*: https://findanexpert.unimelb.edu.au/scholarlywork/1495382-stochastic-process-mining--earth-movers%E2%80%99-stochastic-conformance
- van Glabbeek, *The linear time – branching time spectrum*: https://cgi.cse.unsw.edu.au/~rvg/3152/spectrum1.pdf
- Jackson, the `msm` package: https://www.jstatsoft.org/v38/i08
- Crop model data assimilation review: https://doaj.org/article/a17f9d5f3f314d098d82ccbcbf8c9373
- *A Survey of State Representation Learning for Deep Reinforcement Learning*: https://arxiv.org/abs/2506.17518
- Hillston, strong equivalence is a congruence for PEPA (book chapter): https://www.cambridge.org/core/books/compositional-approach-to-performance-modelling/strong-equivalence/7558D5A5858625697EA5439F61CB3789
- Ramdas, Grünwald, Vovk, Shafer, Statistical Science 38(4), 2023, doi:10.1214/23-sts894: https://arxiv.org/abs/2210.01948
- Shafer & Vovk, game-theoretic probability (talk page): https://www.microsoft.com/en-us/research/?p=182333
- Jernigan & Baran, *Testing lumpability in Markov chains*: https://ideas.repec.org/a/eee/stapro/v64y2003i1p17-23.html
- Geiger & Temmel, *Lumpings of Markov chains, entropy rate preservation, and higher-order lumpability*: https://arxiv.org/abs/1212.4375
- Geiger, Petrov, Kubin, Koeppl, doi:10.1109/tac.2014.2364971
- Buchholz, doi:10.2307/3215235
- Larsen & Skou, doi:10.1016/0890-5401(91)90030-6
- Bravetti, Bernardo, Gorrieri, doi:10.1007/bfb0055638; Bravetti & Gorrieri, doi:10.1016/s0304-3975(01)00043-3
- van Glabbeek & Goltz, doi:10.1007/s002360000041
- Nakagawa, BSc thesis (not read): https://group-mmm.org/~snakagawa/BScthesis.pdf
- van Glabbeek, Höfner, Wang, *Enabling Preserving Bisimulation Equivalence* (CONCUR 2021): https://arxiv.org/abs/2108.00142
