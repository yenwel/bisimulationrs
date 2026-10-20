import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.BigOperators

/-!
# Weighted bisimulation is a congruence for parallel composition

A deliberately small model of the setting in `bisim-core`:

* a finite weighted LTS over a commutative semiring `R`, weight `0` meaning "no transition";
* a bisimulation is an equivalence `E` whose related states have equal total weight into
  every `E`-class, for every label (Markovian lumping when `R = ℝ≥0`);
* parallel composition: labels satisfying `sync` fire jointly with the product of the weights,
  all other labels interleave.

Not modelled: PEPA's apparent-rate cooperation (`min` of apparent rates). That is Hillston's
congruence theorem and is a separate proof.
-/

open Finset

namespace Bisim

variable {S₁ S₂ L R : Type*} [Fintype S₁] [Fintype S₂] [DecidableEq S₁] [DecidableEq S₂]
  [CommSemiring R]

/-- A finite weighted LTS. -/
structure WLTS (S L R : Type*) where
  w : S → L → S → R

/-- Total weight from `p` under label `a` into the `E`-class of `t`. -/
def classSum (E : S₁ → S₁ → Prop) [DecidableRel E] (M : WLTS S₁ L R)
    (p : S₁) (a : L) (t : S₁) : R :=
  ∑ p', if E t p' then M.w p a p' else 0

/-- `E` is a (weighted, strong) bisimulation on `M`. -/
structure IsBisim (E : S₁ → S₁ → Prop) [DecidableRel E] (M : WLTS S₁ L R) : Prop where
  equiv : Equivalence E
  step : ∀ {p q}, E p q → ∀ a t, classSum E M p a t = classSum E M q a t

/-- Parallel composition with a state-dependent synchronisation factor `c`:
labels satisfying `sync` fire jointly with weight `c p q a * (w₁ * w₂)`, other labels
interleave. `c = 1` is the plain product; PEPA's apparent-rate rule is another choice of `c`. -/
def parWith (sync : L → Prop) [DecidablePred sync] (c : S₁ → S₂ → L → R)
    (M₁ : WLTS S₁ L R) (M₂ : WLTS S₂ L R) : WLTS (S₁ × S₂) L R where
  w := fun pq a pq' =>
    if sync a then c pq.1 pq.2 a * (M₁.w pq.1 a pq'.1 * M₂.w pq.2 a pq'.2)
    else (if pq.2 = pq'.2 then M₁.w pq.1 a pq'.1 else 0)
       + (if pq.1 = pq'.1 then M₂.w pq.2 a pq'.2 else 0)

/-- Parallel composition: synchronise on `sync` labels (product of weights), interleave others. -/
def par (sync : L → Prop) [DecidablePred sync] (M₁ : WLTS S₁ L R) (M₂ : WLTS S₂ L R) :
    WLTS (S₁ × S₂) L R :=
  parWith sync (fun _ _ _ => 1) M₁ M₂

/-- Product of two relations. -/
def prodRel (E₁ : S₁ → S₁ → Prop) (E₂ : S₂ → S₂ → Prop) : S₁ × S₂ → S₁ × S₂ → Prop :=
  fun x y => E₁ x.1 y.1 ∧ E₂ x.2 y.2

instance (E₁ : S₁ → S₁ → Prop) (E₂ : S₂ → S₂ → Prop) [DecidableRel E₁] [DecidableRel E₂] :
    DecidableRel (prodRel E₁ E₂) := fun _ _ => inferInstanceAs (Decidable (_ ∧ _))

/-- The class sum of a composite state splits along the two components. -/
theorem classSum_parWith (sync : L → Prop) [DecidablePred sync] (c : S₁ → S₂ → L → R)
    (E₁ : S₁ → S₁ → Prop) (E₂ : S₂ → S₂ → Prop) [DecidableRel E₁] [DecidableRel E₂]
    (M₁ : WLTS S₁ L R) (M₂ : WLTS S₂ L R) (p : S₁) (q : S₂) (a : L) (t₁ : S₁) (t₂ : S₂) :
    classSum (prodRel E₁ E₂) (parWith sync c M₁ M₂) (p, q) a (t₁, t₂)
      = if sync a then c p q a * (classSum E₁ M₁ p a t₁ * classSum E₂ M₂ q a t₂)
        else (if E₂ t₂ q then classSum E₁ M₁ p a t₁ else 0)
           + (if E₁ t₁ p then classSum E₂ M₂ q a t₂ else 0) := by
  classical
  unfold classSum
  rw [Fintype.sum_prod_type]
  by_cases h : sync a
  · simp only [parWith, h, ite_true, prodRel]
    rw [Finset.sum_mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun p' _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun q' _ => ?_
    by_cases h1 : E₁ t₁ p' <;> by_cases h2 : E₂ t₂ q' <;> simp [h1, h2]
  · simp only [parWith, h, ite_false, prodRel]
    have split : ∀ p' : S₁, (∑ q' : S₂, if E₁ t₁ p' ∧ E₂ t₂ q' then
          ((if q = q' then M₁.w p a p' else 0) + (if p = p' then M₂.w q a q' else 0)) else 0)
        = (if E₁ t₁ p' ∧ E₂ t₂ q then M₁.w p a p' else 0)
          + (∑ q' : S₂, if E₁ t₁ p' ∧ E₂ t₂ q' then
              (if p = p' then M₂.w q a q' else 0) else 0) := by
      intro p'
      have h1 : (∑ q' : S₂, if E₁ t₁ p' ∧ E₂ t₂ q' then
            (if q = q' then M₁.w p a p' else 0) else 0)
          = if E₁ t₁ p' ∧ E₂ t₂ q then M₁.w p a p' else 0 := by
        rw [Finset.sum_eq_single q]
        · simp
        · intro b _ hb
          simp [Ne.symm hb]
        · intro hq
          exact absurd (Finset.mem_univ q) hq
      rw [← h1, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun q' _ => ?_
      by_cases hc : E₁ t₁ p' ∧ E₂ t₂ q' <;> simp [hc]
    simp only [split, Finset.sum_add_distrib]
    congr 1
    · by_cases h2 : E₂ t₂ q
      · simp [h2]
      · simp [h2]
    · rw [Finset.sum_eq_single p]
      · by_cases h1 : E₁ t₁ p <;> simp [h1]
      · intro b _ hb
        simp [Ne.symm hb]
      · intro hp
        exact absurd (Finset.mem_univ p) hp

/-- **Congruence, general form.** The factor `c` must respect the two relations. -/
theorem IsBisim.parWith (sync : L → Prop) [DecidablePred sync] (c : S₁ → S₂ → L → R)
    {E₁ : S₁ → S₁ → Prop} {E₂ : S₂ → S₂ → Prop} [DecidableRel E₁] [DecidableRel E₂]
    {M₁ : WLTS S₁ L R} {M₂ : WLTS S₂ L R} (h₁ : IsBisim E₁ M₁) (h₂ : IsBisim E₂ M₂)
    (hc : ∀ {p p' q q'} a, E₁ p p' → E₂ q q' → c p q a = c p' q' a) :
    IsBisim (prodRel E₁ E₂) (Bisim.parWith sync c M₁ M₂) where
  equiv :=
    { refl := fun _ => ⟨h₁.equiv.refl _, h₂.equiv.refl _⟩
      symm := fun h => ⟨h₁.equiv.symm h.1, h₂.equiv.symm h.2⟩
      trans := fun h h' => ⟨h₁.equiv.trans h.1 h'.1, h₂.equiv.trans h.2 h'.2⟩ }
  step := by
    rintro ⟨p, q⟩ ⟨p', q'⟩ ⟨hp, hq⟩ a ⟨t₁, t₂⟩
    rw [classSum_parWith, classSum_parWith, h₁.step hp a t₁, h₂.step hq a t₂, hc a hp hq]
    have e₂ : E₂ t₂ q ↔ E₂ t₂ q' :=
      ⟨fun h => h₂.equiv.trans h hq, fun h => h₂.equiv.trans h (h₂.equiv.symm hq)⟩
    have e₁ : E₁ t₁ p ↔ E₁ t₁ p' :=
      ⟨fun h => h₁.equiv.trans h hp, fun h => h₁.equiv.trans h (h₁.equiv.symm hp)⟩
    simp only [e₁, e₂]

/-- **Congruence.** If `E₁` is a bisimulation on `M₁` and `E₂` one on `M₂`, then the product
relation is a bisimulation on the parallel composition. -/
theorem IsBisim.par (sync : L → Prop) [DecidablePred sync]
    {E₁ : S₁ → S₁ → Prop} {E₂ : S₂ → S₂ → Prop} [DecidableRel E₁] [DecidableRel E₂]
    {M₁ : WLTS S₁ L R} {M₂ : WLTS S₂ L R} (h₁ : IsBisim E₁ M₁) (h₂ : IsBisim E₂ M₂) :
    IsBisim (prodRel E₁ E₂) (Bisim.par sync M₁ M₂) :=
  IsBisim.parWith sync _ h₁ h₂ (fun _ _ _ => rfl)

/-- Equality is a bisimulation on any LTS. -/
theorem isBisim_eq (M : WLTS S₂ L R) : IsBisim (· = ·) M where
  equiv := eq_equivalence
  step := by
    rintro p q rfl a t
    rfl

/-- **Interventions that are contexts commute with lumping.** Replace the environment `M₂`
(any LTS: a stub, a source, another component) and compose with a lumpable component `M₁`:
the lumping of `M₁` is still a lumping of the composite, whatever `M₂` is. -/
theorem lump_in_context (sync : L → Prop) [DecidablePred sync]
    {E₁ : S₁ → S₁ → Prop} [DecidableRel E₁] {M₁ : WLTS S₁ L R}
    (h : IsBisim E₁ M₁) (M₂ : WLTS S₂ L R) :
    IsBisim (prodRel E₁ (· = ·)) (Bisim.par sync M₁ M₂) :=
  h.par sync (isBisim_eq M₂)

end Bisim

#print axioms Bisim.IsBisim.par
#print axioms Bisim.lump_in_context
