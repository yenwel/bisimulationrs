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

## Event causality: status (20/10/2026)

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
