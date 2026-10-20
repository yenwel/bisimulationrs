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

## What is not covered

* PEPA is modelled as a weighted LTS with the cooperation rule above. The syntax (prefix,
  choice, constants, hiding) and its SOS derivation are not formalised, nor is the
  cooperation of more than two components beyond iterating the theorem.
* States are finite (`Fintype`), labels arbitrary.
* Nothing here concerns do-calculus, causal graphs or event-level causality.
* The model is a simplification of `bisim-core`; the Rust code is not extracted or verified.
