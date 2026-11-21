import Bisim.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# "Same behaviour" that survives non-exponential durations

A *clock system* abstracts a safe net whose transitions have rates, run under the race
policy with enabling memory (`bisim-refine/src/durations.rs`): every enabled clock runs; the
first to finish fires; firing clock `t` in state `m` leads to `next m t`, and the clocks in
`kept m t` keep running (in a net: those with no input place in common with `t`).

`X2` is the Erlang-2 expansion: a state also records which enabled clocks have completed
their first stage. Every stage runs at the clock's rate; `erlang_expansion` in Rust uses twice
the rate, and scaling every rate by the same factor does not change which relations are
bisimulations.

Rated enabling-preserving bisimilarity (van Glabbeek, Höfner and Wang, CONCUR 2021, with
rates): related states pair their enabled clocks one to one, keeping labels and rates; the
clocks that stay enabled keep their partners.

**Main theorem** (`epBisimilar_of_erlang2`): if no state enables two clocks with the same
label and rate, two states whose Erlang-2 expansions are Markovian bisimilar are rated
enabling-preserving bisimilar. The converse holds for every phase-type expansion (pair the
clocks); it is not formalised here. So, for such systems, being "the same" with exponential
clocks replaced by any phase-type clocks of the same type is decided by two stages.
-/

namespace Bisim.Durations

open Finset

section Totals

variable {S L : Type*} [Fintype S] [DecidableEq S]

/-- Class sums determine the total rate weighted by any observable that is constant on
classes. -/
theorem sum_mul_eq_of_isBisim (E : S → S → Prop) [DecidableRel E] (X : WLTS S L ℝ)
    (hE : IsBisim E X) (F : S → ℝ) (hF : ∀ x y, E x y → F x = F y)
    {p q : S} (hpq : E p q) (a : L) :
    ∑ p', X.w p a p' * F p' = ∑ p', X.w q a p' * F p' := by
  have key : ∀ r : S, ∑ p', X.w r a p' * F p' =
      ∑ t, classSum E X r a t * (F t / ((univ.filter (E t)).card : ℝ)) := by
    intro r
    simp only [classSum, Finset.sum_mul]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun p' _ => ?_
    have hc : ∀ t, E t p' → univ.filter (E t) = univ.filter (E p') := by
      intro t ht
      ext x
      simp only [mem_filter, mem_univ, true_and]
      exact ⟨fun h => hE.equiv.trans (hE.equiv.symm ht) h, fun h => hE.equiv.trans ht h⟩
    have hpos : (0 : ℝ) < ((univ.filter (E p')).card : ℝ) := by
      exact_mod_cast Finset.card_pos.mpr ⟨p', by simp [hE.equiv.refl p']⟩
    have hfilter : univ.filter (fun t => E t p') = univ.filter (E p') := by
      ext x
      simp only [mem_filter, mem_univ, true_and]
      exact ⟨fun h => hE.equiv.symm h, fun h => hE.equiv.symm h⟩
    have step : ∀ t, (if E t p' then X.w r a p' else 0) * (F t / ((univ.filter (E t)).card : ℝ))
        = if E t p' then X.w r a p' * (F p' / ((univ.filter (E p')).card : ℝ)) else 0 := by
      intro t
      split_ifs with ht
      · rw [hF t p' ht, hc t ht]
      · simp
    rw [Finset.sum_congr rfl (fun t _ => step t), ← Finset.sum_filter, Finset.sum_const,
      nsmul_eq_mul, hfilter, mul_div_assoc', mul_div_assoc',
      mul_comm (((univ.filter (E p')).card : ℝ)), mul_div_assoc, div_self hpos.ne', mul_one]
  rw [key p, key q]
  exact Finset.sum_congr rfl fun t _ => by rw [hE.step hpq a t]

end Totals

/-- Clocks, firing and the clocks that keep running: a safe net with rates under the race
policy with enabling memory. -/
structure ClockSystem (M T L : Type*) where
  /-- Enabled clocks. -/
  en : M → Finset T
  label : T → L
  rate : T → ℝ
  /-- The state after clock `t` fires. -/
  next : M → T → M
  /-- The clocks that keep running when `t` fires. -/
  kept : M → T → Finset T
  rate_pos : ∀ t, 0 < rate t
  kept_sub : ∀ m t, kept m t ⊆ en m
  kept_en : ∀ m t, kept m t ⊆ en (next m t)
  not_kept : ∀ m t, t ∉ kept m t
  kept_symm : ∀ m t s, t ∈ en m → s ∈ en m → (s ∈ kept m t ↔ t ∈ kept m s)

variable {M T L : Type*} [Fintype M] [Fintype T] [DecidableEq M] [DecidableEq T]
  [DecidableEq L]

/-- The Erlang-2 expansion. A state is a state of the system and the clocks that completed
their first stage. Internal steps (`none`) complete a first stage; a clock that completed it
fires, and the clocks it keeps running keep their stage. -/
def X2 (C : ClockSystem M T L) : WLTS (M × Finset T) (Option L) ℝ where
  w p a q := match a with
    | none => ∑ t ∈ C.en p.1, if t ∉ p.2 ∧ q = (p.1, insert t p.2) then C.rate t else 0
    | some l => ∑ t ∈ p.2, if t ∈ C.en p.1 ∧ C.label t = l ∧
        q = (C.next p.1 t, p.2.filter (· ∈ C.kept p.1 t)) then C.rate t else 0

theorem X2_w_none (C : ClockSystem M T L) (p q : M × Finset T) :
    (X2 C).w p none q =
      ∑ t ∈ C.en p.1, if t ∉ p.2 ∧ q = (p.1, insert t p.2) then C.rate t else 0 := rfl

theorem X2_w_some (C : ClockSystem M T L) (p q : M × Finset T) (l : L) :
    (X2 C).w p (some l) q =
      ∑ t ∈ p.2, if t ∈ C.en p.1 ∧ C.label t = l ∧
        q = (C.next p.1 t, p.2.filter (· ∈ C.kept p.1 t)) then C.rate t else 0 := rfl

/-- Total rate of the visible steps labelled `l`: the clocks with that label that completed
their first stage. -/
def vis (C : ClockSystem M T L) (l : L) (p : M × Finset T) : ℝ :=
  ∑ t ∈ p.2, if t ∈ C.en p.1 ∧ C.label t = l then C.rate t else 0

/-- No state enables two clocks with the same label and rate. -/
def TypeUnambiguous (C : ClockSystem M T L) : Prop :=
  ∀ m, ∀ t ∈ C.en m, ∀ t' ∈ C.en m, C.label t = C.label t' → C.rate t = C.rate t' → t = t'

/-- A pairing of the enabled clocks of `m₁` with those of `m₂`: one to one, keeping labels and
rates. -/
structure Pairing (C : ClockSystem M T L) (m₁ m₂ : M) (f : T → T) : Prop where
  maps : ∀ t ∈ C.en m₁, f t ∈ C.en m₂
  surj : ∀ u ∈ C.en m₂, ∃ t ∈ C.en m₁, f t = u
  inj : ∀ t ∈ C.en m₁, ∀ t' ∈ C.en m₁, f t = f t' → t = t'
  label : ∀ t ∈ C.en m₁, C.label (f t) = C.label t
  rate : ∀ t ∈ C.en m₁, C.rate (f t) = C.rate t

/-- Rated enabling-preserving bisimulation: when a clock fires, its partner fires; the clocks
that keep running are exactly the partners of each other's, and keep their partners. -/
def IsEPBisim (C : ClockSystem M T L) (B : M → M → (T → T) → Prop) : Prop :=
  ∀ m₁ m₂ f, B m₁ m₂ f → Pairing C m₁ m₂ f ∧
    ∀ t ∈ C.en m₁,
      (∀ s ∈ C.en m₁, (s ∈ C.kept m₁ t ↔ f s ∈ C.kept m₂ (f t))) ∧
      ∃ g, (∀ s ∈ C.kept m₁ t, g s = f s) ∧ B (C.next m₁ t) (C.next m₂ (f t)) g

def EPBisimilar (C : ClockSystem M T L) (m₁ m₂ : M) : Prop :=
  ∃ B : M → M → (T → T) → Prop, IsEPBisim C B ∧ ∃ f, B m₁ m₂ f

section Expansion

variable (C : ClockSystem M T L)

theorem sum_w_none (p : M × Finset T) (G : M × Finset T → ℝ) :
    ∑ q, (X2 C).w p none q * G q =
      ∑ t ∈ C.en p.1, if t ∉ p.2 then C.rate t * G (p.1, insert t p.2) else 0 := by
  simp only [X2_w_none, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  by_cases hc : t ∉ p.2
  · rw [if_pos hc]
    simp only [hc, not_false_eq_true, true_and, ite_mul, zero_mul, Finset.sum_ite_eq',
      Finset.mem_univ, ↓reduceIte]
  · rw [if_neg hc]
    refine Finset.sum_eq_zero fun q _ => ?_
    rw [if_neg (fun h => hc h.1), zero_mul]

theorem sum_w_some (p : M × Finset T) (l : L) (G : M × Finset T → ℝ) :
    ∑ q, (X2 C).w p (some l) q * G q =
      ∑ t ∈ p.2, if t ∈ C.en p.1 ∧ C.label t = l then
        C.rate t * G (C.next p.1 t, p.2.filter (· ∈ C.kept p.1 t)) else 0 := by
  simp only [X2_w_some, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  by_cases hc : t ∈ C.en p.1 ∧ C.label t = l
  · rw [if_pos hc]
    simp only [hc.1, hc.2, eq_self_iff_true, true_and, ite_mul, zero_mul, Finset.sum_ite_eq',
      Finset.mem_univ, ↓reduceIte]
  · rw [if_neg hc]
    refine Finset.sum_eq_zero fun q _ => ?_
    rw [if_neg (fun h => hc ⟨h.1, h.2.1⟩), zero_mul]

theorem vis_insert (l : L) (m : M) {S : Finset T} {t : T} (htS : t ∉ S) :
    vis C l (m, insert t S) =
      (if t ∈ C.en m ∧ C.label t = l then C.rate t else 0) + vis C l (m, S) := by
  unfold vis
  dsimp only
  rw [Finset.sum_insert htS]

/-- After `a` fires from a state where `a` and `b` completed their first stage, the visible
rate for `b`'s label is `b`'s rate if `b` kept running, and 0 otherwise. -/
theorem vis_after_pair {m : M} {a b : T} (hab : b ≠ a) :
    vis C (C.label b) (C.next m a, ({a, b} : Finset T).filter (· ∈ C.kept m a)) =
      if b ∈ C.kept m a then C.rate b else 0 := by
  unfold vis
  dsimp only
  rw [Finset.sum_filter, Finset.sum_pair hab.symm, if_neg (C.not_kept m a), zero_add]
  by_cases hb : b ∈ C.kept m a
  · simp [hb, C.kept_en m a hb]
  · simp [hb]

end Expansion

section Bisimulation

variable {C : ClockSystem M T L} {E : M × Finset T → M × Finset T → Prop} [DecidableRel E]

theorem vis_eq (hE : IsBisim E (X2 C)) {p q : M × Finset T} (h : E p q) (l : L) :
    vis C l p = vis C l q := by
  have key := sum_mul_eq_of_isBisim E (X2 C) hE (fun _ => (1 : ℝ)) (fun _ _ _ => rfl) h (some l)
  rw [sum_w_some, sum_w_some] at key
  simpa [vis] using key

/-- A first stage completed on one side is matched by one completed on the other. -/
theorem tau_match (hE : IsBisim E (X2 C)) {p q : M × Finset T} (h : E p q) {t : T}
    (hten : t ∈ C.en p.1) (htS : t ∉ p.2) :
    ∃ u ∈ C.en q.1, u ∉ q.2 ∧ E (p.1, insert t p.2) (q.1, insert u q.2) := by
  let F : M × Finset T → ℝ := fun x => if E (p.1, insert t p.2) x then 1 else 0
  have hF : ∀ x y, E x y → F x = F y := by
    intro x y hxy
    by_cases hx : E (p.1, insert t p.2) x
    · have hy := hE.equiv.trans hx hxy
      simp only [F, hx, hy, ↓reduceIte]
    · have hy : ¬ E (p.1, insert t p.2) y :=
        fun hy => hx (hE.equiv.trans hy (hE.equiv.symm hxy))
      simp only [F, hx, hy, ↓reduceIte]
  have key := sum_mul_eq_of_isBisim E (X2 C) hE F hF h none
  rw [sum_w_none, sum_w_none] at key
  have hpos : 0 < ∑ x ∈ C.en p.1, if x ∉ p.2 then C.rate x * F (p.1, insert x p.2) else 0 := by
    apply Finset.sum_pos'
    · intro x _
      by_cases hx : x ∉ p.2
      · rw [if_pos hx]
        apply mul_nonneg (C.rate_pos x).le
        simp only [F]
        split_ifs <;> simp
      · simp [hx]
    · refine ⟨t, hten, ?_⟩
      have hFt : F (p.1, insert t p.2) = 1 := by simp only [F, hE.equiv.refl, ↓reduceIte]
      rw [if_pos htS, hFt, mul_one]
      exact C.rate_pos t
  rw [key] at hpos
  obtain ⟨u, hu, hne⟩ := Finset.exists_ne_zero_of_sum_ne_zero hpos.ne'
  by_cases hc : u ∉ q.2
  · refine ⟨u, hu, hc, ?_⟩
    by_contra hnot
    apply hne
    simp only [F, hnot, hc, not_false_eq_true, ↓reduceIte, mul_zero]
  · exact (hne (if_neg hc)).elim

/-- A firing on one side is matched by a firing with the same label on the other. -/
theorem vis_match (hE : IsBisim E (X2 C)) {p q : M × Finset T} (h : E p q) {t : T}
    (htS : t ∈ p.2) (hten : t ∈ C.en p.1) :
    ∃ u ∈ q.2, u ∈ C.en q.1 ∧ C.label u = C.label t ∧
      E (C.next p.1 t, p.2.filter (· ∈ C.kept p.1 t))
        (C.next q.1 u, q.2.filter (· ∈ C.kept q.1 u)) := by
  let F : M × Finset T → ℝ :=
    fun x => if E (C.next p.1 t, p.2.filter (· ∈ C.kept p.1 t)) x then 1 else 0
  have hF : ∀ x y, E x y → F x = F y := by
    intro x y hxy
    by_cases hx : E (C.next p.1 t, p.2.filter (· ∈ C.kept p.1 t)) x
    · have hy := hE.equiv.trans hx hxy
      simp only [F, hx, hy, ↓reduceIte]
    · have hy : ¬ E (C.next p.1 t, p.2.filter (· ∈ C.kept p.1 t)) y :=
        fun hy => hx (hE.equiv.trans hy (hE.equiv.symm hxy))
      simp only [F, hx, hy, ↓reduceIte]
  have key := sum_mul_eq_of_isBisim E (X2 C) hE F hF h (some (C.label t))
  rw [sum_w_some, sum_w_some] at key
  have hpos : 0 < ∑ x ∈ p.2, if x ∈ C.en p.1 ∧ C.label x = C.label t then
      C.rate x * F (C.next p.1 x, p.2.filter (· ∈ C.kept p.1 x)) else 0 := by
    apply Finset.sum_pos'
    · intro x _
      split_ifs
      · apply mul_nonneg (C.rate_pos x).le
        simp only [F]
        split_ifs <;> simp
      · exact le_rfl
    · refine ⟨t, htS, ?_⟩
      have hFt : F (C.next p.1 t, p.2.filter (· ∈ C.kept p.1 t)) = 1 := by
        simp only [F, hE.equiv.refl, ↓reduceIte]
      rw [if_pos ⟨hten, rfl⟩, hFt, mul_one]
      exact C.rate_pos t
  rw [key] at hpos
  obtain ⟨u, hu, hne⟩ := Finset.exists_ne_zero_of_sum_ne_zero hpos.ne'
  by_cases hc : u ∈ C.en q.1 ∧ C.label u = C.label t
  · refine ⟨u, hu, hc.1, hc.2, ?_⟩
    by_contra hnot
    apply hne
    simp only [F, hnot, hc.1, hc.2, eq_self_iff_true, and_self, ↓reduceIte, mul_zero]
  · exact (hne (if_neg hc)).elim

/-- Completing the first stage of `t` and of `t'` changes the visible rates by the same
amount, so `t` and `t'` have the same label and rate. -/
theorem same_type_of_tau (hE : IsBisim E (X2 C)) {m₁ m₂ : M} {S₁ S₂ : Finset T} {t t' : T}
    (ht : t ∈ C.en m₁) (htS : t ∉ S₁) (ht' : t' ∈ C.en m₂) (ht'S : t' ∉ S₂)
    (h0 : E (m₁, S₁) (m₂, S₂)) (h1 : E (m₁, insert t S₁) (m₂, insert t' S₂)) :
    C.label t' = C.label t ∧ C.rate t' = C.rate t := by
  have e0 := vis_eq hE h0 (C.label t)
  have e1 := vis_eq hE h1 (C.label t)
  rw [vis_insert C _ _ htS, vis_insert C _ _ ht'S, e0, if_pos ⟨ht, rfl⟩] at e1
  have e2 := add_right_cancel e1
  by_cases hl : C.label t' = C.label t
  · rw [if_pos ⟨ht', hl⟩] at e2
    exact ⟨hl, e2.symm⟩
  · rw [if_neg (fun h => hl h.2)] at e2
    exact absurd e2 (C.rate_pos t).ne'

variable (hU : TypeUnambiguous C) (hE : IsBisim E (X2 C))
include hU hE

/-- Bisimilar states with no first stage completed enable clocks of the same types, so they
can be paired. -/
theorem exists_pairing {m₁ m₂ : M} (h : E (m₁, ∅) (m₂, ∅)) : ∃ f, Pairing C m₁ m₂ f := by
  have side : ∀ {a b : M}, E (a, ∅) (b, ∅) → ∀ t ∈ C.en a,
      ∃ u ∈ C.en b, C.label u = C.label t ∧ C.rate u = C.rate t := by
    intro a b hab t ht
    obtain ⟨u, hu, huS, h1⟩ := tau_match hE hab (t := t) ht (by simp)
    dsimp only at hu huS h1
    obtain ⟨hl, hr⟩ := same_type_of_tau hE ht (by simp) hu huS hab h1
    exact ⟨u, hu, hl, hr⟩
  let f : T → T := fun t => if ht : t ∈ C.en m₁ then Classical.choose (side h t ht) else t
  have hf : ∀ t (ht : t ∈ C.en m₁),
      f t ∈ C.en m₂ ∧ C.label (f t) = C.label t ∧ C.rate (f t) = C.rate t := by
    intro t ht
    simp only [f, dif_pos ht]
    exact Classical.choose_spec (side h t ht)
  refine ⟨f, ⟨fun t ht => (hf t ht).1, ?_, ?_, fun t ht => (hf t ht).2.1,
    fun t ht => (hf t ht).2.2⟩⟩
  · intro u hu
    obtain ⟨t, ht, hl, hr⟩ := side (hE.equiv.symm h) u hu
    refine ⟨t, ht, ?_⟩
    obtain ⟨h1, h2, h3⟩ := hf t ht
    exact hU m₂ (f t) h1 u hu (by rw [h2, hl]) (by rw [h3, hr])
  · intro t ht t' ht' heq
    obtain ⟨_, h2, h3⟩ := hf t ht
    obtain ⟨_, h2', h3'⟩ := hf t' ht'
    exact hU m₁ t ht t' ht' (by rw [← h2, ← h2', heq]) (by rw [← h3, ← h3', heq])

/-- With clocks of distinct types, the pairing carries over to every set of completed first
stages. -/
theorem lift {m₁ m₂ : M} {f : T → T} (hf : Pairing C m₁ m₂ f) (h : E (m₁, ∅) (m₂, ∅)) :
    ∀ S ⊆ C.en m₁, E (m₁, S) (m₂, S.image f) := by
  intro S
  induction S using Finset.induction_on with
  | empty => intro _; simpa using h
  | @insert t S hts ih =>
    intro hsub
    have ht : t ∈ C.en m₁ := hsub (Finset.mem_insert_self t S)
    have h0 := ih (fun x hx => hsub (Finset.mem_insert_of_mem hx))
    obtain ⟨t', ht', ht'S, h1⟩ := tau_match hE h0 (t := t) ht hts
    dsimp only at ht' ht'S h1
    obtain ⟨hl, hr⟩ := same_type_of_tau hE ht hts ht' ht'S h0 h1
    have e : t' = f t := hU m₂ t' ht' (f t) (hf.maps t ht)
      (by rw [hl, hf.label t ht]) (by rw [hr, hf.rate t ht])
    rw [Finset.image_insert, ← e]
    exact h1

/-- Paired clocks lead to bisimilar states when they fire. -/
theorem fire_match {m₁ m₂ : M} {f : T → T} (hf : Pairing C m₁ m₂ f) (h : E (m₁, ∅) (m₂, ∅))
    {t : T} (ht : t ∈ C.en m₁) : E (C.next m₁ t, ∅) (C.next m₂ (f t), ∅) := by
  have h1 := lift hU hE hf h {t} (by simpa using ht)
  rw [Finset.image_singleton] at h1
  obtain ⟨u, hu, -, -, h2⟩ := vis_match hE h1 (t := t) (by simp) ht
  dsimp only at hu h2
  rw [Finset.mem_singleton] at hu
  subst hu
  simpa [Finset.filter_singleton, C.not_kept] using h2

/-- Paired clocks keep the same other clocks running. -/
theorem kept_iff {m₁ m₂ : M} {f : T → T} (hf : Pairing C m₁ m₂ f) (h : E (m₁, ∅) (m₂, ∅))
    {t s : T} (ht : t ∈ C.en m₁) (hs : s ∈ C.en m₁) (hst : s ≠ t) :
    s ∈ C.kept m₁ t ↔ f s ∈ C.kept m₂ (f t) := by
  have hsub : ({t, s} : Finset T) ⊆ C.en m₁ := by
    intro x hx
    rw [Finset.mem_insert, Finset.mem_singleton] at hx
    rcases hx with rfl | rfl <;> assumption
  have h2 := lift hU hE hf h {t, s} hsub
  rw [Finset.image_insert, Finset.image_singleton] at h2
  obtain ⟨u, hu, huen, hul, hE2⟩ := vis_match hE h2 (t := t) (by simp) ht
  dsimp only at hu huen hul hE2
  have hfst : f s ≠ f t := fun heq => hst (hf.inj s hs t ht heq)
  have hv := vis_eq hE hE2 (C.label s)
  rw [vis_after_pair C hst] at hv
  rw [Finset.mem_insert, Finset.mem_singleton] at hu
  rcases hu with rfl | rfl
  · -- The partner of `t` fired: compare whether `s` and its partner kept running.
    rw [← hf.label s hs, vis_after_pair C hfst, hf.rate s hs] at hv
    constructor
    · intro h1
      by_contra h2'
      rw [if_pos h1, if_neg h2'] at hv
      exact (C.rate_pos s).ne' hv
    · intro h2'
      by_contra h1
      rw [if_neg h1, if_pos h2'] at hv
      exact (C.rate_pos s).ne' hv.symm
  · -- The partner of `s` fired: then `s` and `t` share a label, so their rates differ, and
    -- neither can have kept the other running.
    have hls : C.label s = C.label t := by rw [← hf.label s hs]; exact hul
    have hrate : C.rate s ≠ C.rate t := fun hr => hst (hU m₁ s hs t ht hls hr)
    have hlf : C.label s = C.label (f t) := by rw [hf.label t ht, hls]
    rw [hlf, Finset.pair_comm, vis_after_pair C hfst.symm, hf.rate t ht] at hv
    have hsymm := C.kept_symm m₂ (f t) (f s) (hf.maps t ht) (hf.maps s hs)
    constructor
    · intro h1
      exfalso
      rw [if_pos h1] at hv
      by_cases h2' : f t ∈ C.kept m₂ (f s)
      · rw [if_pos h2'] at hv
        exact hrate hv
      · rw [if_neg h2'] at hv
        exact (C.rate_pos s).ne' hv
    · intro h2'
      exfalso
      rw [if_pos (hsymm.mp h2')] at hv
      by_cases h1 : s ∈ C.kept m₁ t
      · rw [if_pos h1] at hv
        exact hrate hv
      · rw [if_neg h1] at hv
        exact (C.rate_pos t).ne' hv.symm

end Bisimulation

/-- **Main theorem.** If no state enables two clocks with the same label and rate, states whose
Erlang-2 expansions are Markovian bisimilar are rated enabling-preserving bisimilar. -/
theorem epBisimilar_of_erlang2 (C : ClockSystem M T L) (hU : TypeUnambiguous C)
    (E : M × Finset T → M × Finset T → Prop) [DecidableRel E] (hE : IsBisim E (X2 C))
    {m₁ m₂ : M} (h : E (m₁, ∅) (m₂, ∅)) : EPBisimilar C m₁ m₂ := by
  refine ⟨fun a b f => E (a, ∅) (b, ∅) ∧ Pairing C a b f, ?_, ?_⟩
  · rintro a b f ⟨hab, hf⟩
    refine ⟨hf, fun t ht => ⟨fun s hs => ?_, ?_⟩⟩
    · by_cases hst : s = t
      · subst hst
        exact ⟨fun h => absurd h (C.not_kept _ _), fun h => absurd h (C.not_kept _ _)⟩
      · exact kept_iff hU hE hf hab ht hs hst
    · have hnext := fire_match hU hE hf hab ht
      obtain ⟨g, hg⟩ := exists_pairing hU hE hnext
      refine ⟨g, fun s hs => ?_, hnext, hg⟩
      have hsen : s ∈ C.en a := C.kept_sub a t hs
      have hst : s ≠ t := fun e => by subst e; exact C.not_kept _ _ hs
      have hfs : f s ∈ C.kept b (f t) := (kept_iff hU hE hf hab ht hsen hst).mp hs
      have h1 : s ∈ C.en (C.next a t) := C.kept_en a t hs
      have h2 : f s ∈ C.en (C.next b (f t)) := C.kept_en b (f t) hfs
      exact hU _ (g s) (hg.maps s h1) (f s) h2
        (by rw [hg.label s h1, hf.label s hsen]) (by rw [hg.rate s h1, hf.rate s hsen])
  · obtain ⟨f, hf⟩ := exists_pairing hU hE h
    exact ⟨f, h, hf⟩

end Bisim.Durations
