# Lean proofs

Machine-checked statements about the bisimulation theory used in this library.
Lean 4.35.0, Mathlib v4.35.0 (pinned in `lakefile.toml`).

```
cd lean && lake build
```

| File | Statement |
|---|---|
| `Bisim/Basic.lean` | `IsBisim.par`: weighted bisimulation is a congruence for parallel composition (synchronised labels multiply weights, other labels interleave), over any commutative semiring. `lump_in_context`: a lumping of a component remains a lumping of the composite for every environment, so interventions expressible as contexts commute with quotienting. |
| `Bisim/Pepa.lean` | PEPA cooperation with apparent-rate `min` over `ℝ≥0`: `appRate_eq` (bisimilar states have equal apparent rate), `pepaPar_sync_weight` (the synchronised weight is exactly `(r₁/ra₁)(r₂/ra₂)·min(ra₁,ra₂)` when both apparent rates are non-zero), and `IsBisim.pepaPar` (strong equivalence is a congruence for it). |
| `Bisim/Intervention.lean` | A rate-parameter intervention does not: a three-state model is lumpable at `θ = 1` and not at `θ = 2`. |
| `Bisim/Causal.lean` | History-preserving bisimulation on nets (`bisim-refine/src/causal.rs`). `causes_eq_iff`: the groups `(L, R)` determine exactly which matched steps have equal cause sets. `matched_groups`: `matched` maps the groups of a position to the groups after the step. `hpBisimilar_iff`: the definition (full histories, equal cause sets) and the library's game on `(marking₁, marking₂, groups)` have the same winner; `hp_iff_group` says so at every position whose tokens depend only on steps already taken (`HPos.Fresh`, true of every reachable one). |

## What is not covered

* PEPA is modelled as a weighted LTS with the cooperation rule above. The syntax (prefix,
  choice, constants, hiding) and its SOS derivation are not formalised, nor is the
  cooperation of more than two components beyond iterating the theorem.
* States are finite (`Fintype`), labels arbitrary.
* Nothing here concerns do-calculus or causal graphs. `Causal.lean` covers event-level
  causality in nets only.
* The model is a simplification of `bisim-core`; the Rust code is not extracted or verified.

### `Causal.lean`: what is and is not covered

Covered:

* Nets with labelled transitions, `pre`/`post` place sets and an initial marking, over any
  place, transition and label types (the two nets may differ in all three).
* The definition, as in the `oracle` of `tests/causal.rs` but also for cyclic nets: histories
  map tokens to the set of matched step numbers they depend on, `causesOf` and `advance`
  follow `causes_of` and `advance`, and Duplicator wins iff some relation closed under
  matching (same label, equal cause sets) contains the position.
* The grouped game with `matched` stated as in Rust: `None` when some group is met on one
  side only, else update every group, drop empty ones, add `(post₁, post₂)` if non-empty.
  Groups are a `Finset`, as the `BTreeSet` in Rust. Over finitely many places there are
  finitely many positions (`Finite (GPos P₁ P₂)`).
* `hpBisimilar_iff`: the two games have the same winner from the initial markings.

Not covered:

* Firing a step that would put a second token on a place: here such a step is simply not
  possible (`Net.Fires`), in both games alike; `SafeNet::fire` returns `Contact` instead. On
  safe nets the two agree.
* Winning is stated as the greatest fixed point (a bisimulation relation exists). That the
  on-the-fly search with refutation counters in `hp_bisimilar` computes this fixed point, the
  `max_positions` budget and `Unknown`, and the `CausalFormula` witnesses are not proved.
* Places are `Finset`s, not sorted `Vec<u32>`; the Rust code is not extracted or verified.
