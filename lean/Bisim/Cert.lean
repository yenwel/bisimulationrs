import Bisim.Basic

/-!
# Certificates for "same behaviour" answers

When the library says that states behave the same (a lumping, or two models that are
Markovian bisimilar), it can hand over its evidence: the partition it computed, given as a
representative state for every state. This file defines a small checker for such a
certificate and proves it sound: whenever the checker accepts, "same representative" is a
bisimulation in the sense of `Bisim.IsBisim`.

The system in a certificate is a list of transitions `src —lbl, w→ tgt` on the states
`0 … n-1`, with natural-number weights: the library's floating-point rates, each scaled by
one common power of two, which is exact. Scaling every weight by the same factor does not
change which relations are bisimulations.

`Valid` is decidable, so a certificate is checked with `decide +kernel`: the Lean kernel
evaluates the checker itself, with no trust in compiled code. The certificate lists the
labels in use and the representatives of the blocks, so the checker compares each state
with its representative once per label and block.

What is checked is the partition, against the transition system the library derived. The
derivation of that system from PEPA source is not part of the check.
-/

namespace Bisim.Cert

/-- One transition `src —lbl, w→ tgt`. -/
structure Edge where
  src : ℕ
  lbl : ℕ
  tgt : ℕ
  w : ℕ

/-- Total weight of the transitions `p —a→ q` (parallel transitions add up). -/
def wSum : List Edge → ℕ → ℕ → ℕ → ℕ
  | [], _, _, _ => 0
  | e :: es, p, a, q => (if e.src = p ∧ e.lbl = a ∧ e.tgt = q then e.w else 0) + wSum es p a q

/-- The system on the states `0 … n-1` described by an edge list. -/
def toWLTS (n : ℕ) (es : List Edge) : WLTS (Fin n) ℕ ℕ where
  w p a q := wSum es p.val a q.val

/-- `xs[i]`, or `0` past the end: a certificate lists the representative of every state. -/
def lookup : List ℕ → ℕ → ℕ
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, i + 1 => lookup xs i

/-- "Same representative". -/
abbrev Rel (n : ℕ) (r : ℕ → ℕ) (x y : Fin n) : Prop := r x.val = r y.val

instance (n : ℕ) (r : ℕ → ℕ) : DecidableRel (Rel n r) := fun x y =>
  inferInstanceAs (Decidable (r x.val = r y.val))

/-- Total weight from `p` under `a` into the block with representative `k`. -/
def blockSum (r : ℕ → ℕ) : List Edge → ℕ → ℕ → ℕ → ℕ
  | [], _, _, _ => 0
  | e :: es, p, a, k =>
      (if e.src = p ∧ e.lbl = a ∧ r e.tgt = k then e.w else 0) + blockSum r es p a k

/-- What the checker verifies, given the labels in use `ls` and the representatives `ks`:
every transition stays among the states and has a listed label, every state's
representative is listed, and every state has the same total weight as its representative
into every block, for every label. -/
abbrev Valid (n : ℕ) (es : List Edge) (r : ℕ → ℕ) (ls ks : List ℕ) : Prop :=
  (∀ e ∈ es, e.tgt < n ∧ e.lbl ∈ ls) ∧
  (∀ t, t < n → r t ∈ ks) ∧
  ∀ p, p < n → ∀ a ∈ ls, ∀ k ∈ ks, blockSum r es p a k = blockSum r es (r p) a k

/-- The class sums of `Bisim.Basic`, computed from the edge list. -/
theorem classSum_eq_blockSum (n : ℕ) (es : List Edge) (r : ℕ → ℕ)
    (hes : ∀ e ∈ es, e.tgt < n) (p : Fin n) (a : ℕ) (t : Fin n) :
    classSum (Rel n r) (toWLTS n es) p a t = blockSum r es p.val a (r t.val) := by
  induction es with
  | nil => simp [classSum, toWLTS, wSum, blockSum]
  | cons e es ih =>
    have hes' : ∀ f ∈ es, f.tgt < n := fun f hf => hes f (by simp [hf])
    have he : e.tgt < n := hes e (by simp)
    -- Split off the first transition.
    have split : classSum (Rel n r) (toWLTS n (e :: es)) p a t
        = (∑ q : Fin n, if Rel n r t q then
            (if e.src = p.val ∧ e.lbl = a ∧ e.tgt = q.val then e.w else 0) else 0)
          + classSum (Rel n r) (toWLTS n es) p a t := by
      unfold classSum
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun q _ => ?_
      simp only [toWLTS, wSum]
      split_ifs <;> simp
    -- Only the target of that transition can contribute.
    have single : (∑ q : Fin n, if Rel n r t q then
            (if e.src = p.val ∧ e.lbl = a ∧ e.tgt = q.val then e.w else 0) else 0)
        = if e.src = p.val ∧ e.lbl = a ∧ r e.tgt = r t.val then e.w else 0 := by
      rw [Finset.sum_eq_single (⟨e.tgt, he⟩ : Fin n)]
      · by_cases h1 : r e.tgt = r t.val
        · simp [Rel, h1]
        · have h1' : ¬ r t.val = r e.tgt := fun h => h1 h.symm
          simp [Rel, h1, h1']
      · intro q _ hq
        have hne : e.tgt ≠ q.val := fun h => hq (Fin.ext h.symm)
        simp [hne]
      · intro h
        exact absurd (Finset.mem_univ _) h
    rw [split, single, ih hes']
    simp only [blockSum]

/-- A label no transition carries has no weight. -/
theorem blockSum_of_not_mem (r : ℕ → ℕ) (es : List Edge) (p a k : ℕ)
    (ha : ∀ e ∈ es, e.lbl ≠ a) : blockSum r es p a k = 0 := by
  induction es with
  | nil => rfl
  | cons e es ih =>
    have hne : e.lbl ≠ a := ha e (by simp)
    have rest := ih fun f hf => ha f (by simp [hf])
    simp [blockSum, rest, hne]

/-- **Soundness.** If the checker accepts, "same representative" is a bisimulation. -/
theorem sound (n : ℕ) (es : List Edge) (r : ℕ → ℕ) (ls ks : List ℕ)
    (h : Valid n es r ls ks) : IsBisim (Rel n r) (toWLTS n es) where
  equiv := ⟨fun _ => rfl, fun h => Eq.symm h, fun h₁ h₂ => Eq.trans h₁ h₂⟩
  step := by
    obtain ⟨h1, h2, h3⟩ := h
    have htgt : ∀ e ∈ es, e.tgt < n := fun e he => (h1 e he).1
    intro p q hpq a t
    rw [classSum_eq_blockSum n es r htgt, classSum_eq_blockSum n es r htgt]
    by_cases ha : a ∈ ls
    · have hk : r t.val ∈ ks := h2 t.val t.isLt
      have hpq' : r p.val = r q.val := hpq
      rw [h3 p.val p.isLt a ha _ hk, h3 q.val q.isLt a ha _ hk, hpq']
    · have hne : ∀ e ∈ es, e.lbl ≠ a := fun e he heq => ha (by rw [← heq]; exact (h1 e he).2)
      rw [blockSum_of_not_mem r es _ a _ hne, blockSum_of_not_mem r es _ a _ hne]

/-- Two states with the same representative in an accepted certificate are bisimilar. -/
theorem bisimilar_of_valid (n : ℕ) (es : List Edge) (r : ℕ → ℕ) (ls ks : List ℕ)
    (h : Valid n es r ls ks) (s t : Fin n) (hst : r s.val = r t.val) :
    ∃ (E : Fin n → Fin n → Prop) (_ : DecidableRel E), IsBisim E (toWLTS n es) ∧ E s t :=
  ⟨Rel n r, inferInstance, sound n es r ls ks h, hst⟩

end Bisim.Cert
