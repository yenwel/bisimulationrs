import Bisim.Basic

/-!
# A rate-parameter intervention need not preserve lumpability

Three states `0, 1, 2` with one label. State `0` reaches `2` at rate `θ`, state `1` reaches `2`
at rate `1`. At `θ = 1` the partition `{0,1},{2}` is a lumping. After `do(θ = 2)` it is not.

So a *rate* intervention is not a context in the sense of `Bisim.lump_in_context`: it can
distinguish states that the quotient had merged, and the quotient cannot express it.
-/

namespace Bisim

/-- The model with parameter `θ`. -/
def paramModel (θ : ℕ) : WLTS (Fin 3) Unit ℕ where
  w := fun p _ p' => if p' = 2 then (if p = 0 then θ else if p = 1 then 1 else 0) else 0

/-- Merge states `0` and `1`. -/
def merge01 (x y : Fin 3) : Prop := x = y ∨ (x.val < 2 ∧ y.val < 2)

instance : DecidableRel merge01 := fun _ _ => inferInstanceAs (Decidable (_ ∨ _))

theorem merge01_equiv : Equivalence merge01 := by
  have symm' : ∀ x y : Fin 3, merge01 x y → merge01 y x := by decide
  have trans' : ∀ x y z : Fin 3, merge01 x y → merge01 y z → merge01 x z := by decide
  exact ⟨fun _ => Or.inl rfl, fun {x y} h => symm' x y h, fun {x y z} h h' => trans' x y z h h'⟩

theorem lumpable_at_one : IsBisim merge01 (paramModel 1) where
  equiv := merge01_equiv
  step := by
    unfold classSum
    decide +revert

theorem not_lumpable_at_two : ¬ IsBisim merge01 (paramModel 2) := by
  intro h
  have := h.step (p := 0) (q := 1) (Or.inr (by decide)) () 2
  revert this
  decide

end Bisim

#print axioms Bisim.lumpable_at_one
#print axioms Bisim.not_lumpable_at_two
