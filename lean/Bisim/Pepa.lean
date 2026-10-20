import Bisim.Basic
import Mathlib.Data.NNReal.Defs
import Mathlib.Algebra.Order.Field.Basic

/-!
# PEPA cooperation: strong equivalence is a congruence

PEPA's cooperation `P ⋈_L Q` gives a shared action `a ∈ L` the rate

  `(r₁ / ra(P,a)) * (r₂ / ra(Q,a)) * min (ra(P,a), ra(Q,a))`

where `ra` is the apparent rate: the total rate of `a` out of the component. Here the weights
are in `ℝ≥0`, where `x / 0 = 0`; that is harmless because `ra = 0` forces every `r = 0`.

The bisimulation of `Bisim.Basic` (equal total rate into every class, for every label) is
PEPA's strong equivalence in this finite setting. We show it is a congruence for this
cooperation (Hillston 1996 states this for PEPA; this file checks it for the model below).
-/

open Finset

namespace Bisim

variable {S₁ S₂ L : Type*} [Fintype S₁] [Fintype S₂] [DecidableEq S₁] [DecidableEq S₂]

/-- Apparent rate of label `a` in state `p`. -/
def appRate (M : WLTS S₁ L NNReal) (p : S₁) (a : L) : NNReal := ∑ p', M.w p a p'

/-- Bisimilar states have the same apparent rate: it is the sum of the class sums. -/
theorem appRate_eq {E : S₁ → S₁ → Prop} [DecidableRel E] {M : WLTS S₁ L NNReal}
    (h : IsBisim E M) {p q : S₁} (hpq : E p q) (a : L) :
    appRate M p a = appRate M q a := by
  classical
  let s : Setoid S₁ := ⟨E, h.equiv⟩
  -- total = sum over classes of the class sums
  have key : ∀ r : S₁, appRate M r a = ∑ c : Quotient s, classSum E M r a c.out := by
    intro r
    unfold appRate classSum
    rw [← Finset.sum_fiberwise Finset.univ (fun p' => (Quotient.mk s p')) (fun p' => M.w r a p')]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [← Finset.sum_filter]
    refine Finset.sum_congr ?_ fun _ _ => rfl
    ext p'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · intro hp'
      have : Quotient.mk s c.out = c := Quotient.out_eq c
      rw [← this] at hp'
      exact h.equiv.symm (Quotient.exact hp')
    · intro hp'
      have : Quotient.mk s c.out = c := Quotient.out_eq c
      rw [← this]
      exact Quotient.sound (h.equiv.symm hp')
  rw [key p, key q]
  exact Finset.sum_congr rfl fun c _ => h.step hpq a c.out

/-- PEPA's synchronisation factor: `r₁ r₂ · factor = (r₁/ra₁)(r₂/ra₂) · min(ra₁, ra₂)`. -/
noncomputable def pepaFactor (M₁ : WLTS S₁ L NNReal) (M₂ : WLTS S₂ L NNReal)
    (p : S₁) (q : S₂) (a : L) : NNReal :=
  min (appRate M₁ p a) (appRate M₂ q a) / (appRate M₁ p a * appRate M₂ q a)

/-- PEPA cooperation over the actions satisfying `sync`. -/
noncomputable def pepaPar (sync : L → Prop) [DecidablePred sync]
    (M₁ : WLTS S₁ L NNReal) (M₂ : WLTS S₂ L NNReal) : WLTS (S₁ × S₂) L NNReal :=
  parWith sync (pepaFactor M₁ M₂) M₁ M₂

/-- Faithfulness: when both apparent rates are non-zero the synchronised weight is exactly
PEPA's `(r₁/ra₁) * (r₂/ra₂) * min ra₁ ra₂`. -/
theorem pepaPar_sync_weight {sync : L → Prop} [DecidablePred sync]
    (M₁ : WLTS S₁ L NNReal) (M₂ : WLTS S₂ L NNReal) (p : S₁) (q : S₂) (a : L)
    (p' : S₁) (q' : S₂) (hs : sync a)
    (h₁ : appRate M₁ p a ≠ 0) (h₂ : appRate M₂ q a ≠ 0) :
    (pepaPar sync M₁ M₂).w (p, q) a (p', q')
      = M₁.w p a p' / appRate M₁ p a * (M₂.w q a q' / appRate M₂ q a)
        * min (appRate M₁ p a) (appRate M₂ q a) := by
  simp only [pepaPar, parWith, hs, ite_true, pepaFactor]
  simp only [div_eq_mul_inv, mul_inv]
  ring

/-- **Hillston-style congruence for PEPA cooperation.** If `E₁`, `E₂` are strong equivalences
(bisimulations) on the components, their product is one on the cooperation. -/
theorem IsBisim.pepaPar (sync : L → Prop) [DecidablePred sync]
    {E₁ : S₁ → S₁ → Prop} {E₂ : S₂ → S₂ → Prop} [DecidableRel E₁] [DecidableRel E₂]
    {M₁ : WLTS S₁ L NNReal} {M₂ : WLTS S₂ L NNReal}
    (h₁ : IsBisim E₁ M₁) (h₂ : IsBisim E₂ M₂) :
    IsBisim (prodRel E₁ E₂) (Bisim.pepaPar sync M₁ M₂) :=
  IsBisim.parWith sync _ h₁ h₂ (by
    intro p p' q q' a hp hq
    simp only [pepaFactor, appRate_eq h₁ hp a, appRate_eq h₂ hq a])

/-- A lumping of a component survives cooperation with any environment. -/
theorem pepa_lump_in_context (sync : L → Prop) [DecidablePred sync]
    {E₁ : S₁ → S₁ → Prop} [DecidableRel E₁] {M₁ : WLTS S₁ L NNReal}
    (h : IsBisim E₁ M₁) (M₂ : WLTS S₂ L NNReal) :
    IsBisim (prodRel E₁ (· = ·)) (Bisim.pepaPar sync M₁ M₂) :=
  h.pepaPar sync (isBisim_eq M₂)

end Bisim

#print axioms Bisim.appRate_eq
#print axioms Bisim.pepaPar_sync_weight
#print axioms Bisim.IsBisim.pepaPar
