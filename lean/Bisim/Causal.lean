import Mathlib.Data.Finset.Union
import Mathlib.Data.Finset.Image
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Fintype.Prod

/-!
# History-preserving bisimulation: grouped histories decide the same game

The model of `crates/bisim-refine/src/causal.rs`.

* A net has labelled transitions with sets of input (`pre`) and output (`post`) places.
  A step is possible when its input places are marked and none of its output places is still
  marked once the inputs are consumed (`Net.Fires`); on a safe net this is every enabled step,
  and where it fails `SafeNet::fire` reports a `Contact` instead.
* The definition (`IsHPBisim`, the `oracle` in `tests/causal.rs`): every current token carries
  the set of matched step numbers it depends on (`History`), and matched steps must have the
  same label and equal cause sets (`causesOf`, as in `causes_of`), with histories advanced as
  in `advance`.
* The library's game (`IsGroupBisim`): positions are `(marking₁, marking₂, groups)` and a
  pair of steps is allowed and updates the groups exactly as `matched` does.

`HPos.groups` maps a full position to its groups: past step `k` gives the group of current
tokens, left and right, that depend on it. `causes_eq_iff` says the groups determine which
matched steps are allowed, `matched_groups` that `matched` commutes with the abstraction, and
`hpBisimilar_iff` that the two games have the same winner.

Duplicator wins a bisimulation game from a position iff some bisimulation relation contains
it (the greatest fixed point); both games are stated that way.
-/

namespace Bisim.Causal

/-- A labelled net over places `P` with transitions `T` (`SafeNet`). -/
structure Net (P T L : Type*) where
  label : T → L
  pre : T → Finset P
  post : T → Finset P
  initial : Finset P

variable {P P₁ P₂ T₁ T₂ L : Type*} [DecidableEq P] [DecidableEq P₁] [DecidableEq P₂]

namespace Net

variable {T : Type*}

/-- `t` can fire at `m`: its inputs are marked and it puts no second token on a place. -/
def Fires (N : Net P T L) (m : Finset P) (t : T) : Prop :=
  N.pre t ⊆ m ∧ Disjoint (N.post t) (m \ N.pre t)

/-- The marking after firing `t` at `m` (`SafeNet::fire`). -/
def fire (N : Net P T L) (m : Finset P) (t : T) : Finset P :=
  m \ N.pre t ∪ N.post t

end Net

/-! ## The definition: full histories -/

/-- Which earlier steps (numbered from 1) each current token depends on; `∅` elsewhere. -/
abbrev History (P : Type*) := P → Finset ℕ

/-- The steps that a step consuming `pre` depends on (`causes_of`). -/
def causesOf (h : History P) (pre : Finset P) : Finset ℕ :=
  pre.biUnion h

/-- The history after step number `step` consumed `pre` and produced `post` (`advance`). -/
def advance (h : History P) (pre post : Finset P) (step : ℕ) : History P := fun p =>
  if p ∈ post then insert step (causesOf h pre) else if p ∈ pre then ∅ else h p

/-- A position of the definition: both markings, both histories, and `n` matched steps. -/
structure HPos (P₁ P₂ : Type*) where
  m₁ : Finset P₁
  m₂ : Finset P₂
  h₁ : History P₁
  h₂ : History P₂
  n : ℕ

variable (N₁ : Net P₁ T₁ L) (N₂ : Net P₂ T₂ L)

/-- The position after matching `t₁` with `t₂`, as step number `n + 1`. -/
def HPos.next (c : HPos P₁ P₂) (t₁ : T₁) (t₂ : T₂) : HPos P₁ P₂ where
  m₁ := N₁.fire c.m₁ t₁
  m₂ := N₂.fire c.m₂ t₂
  h₁ := advance c.h₁ (N₁.pre t₁) (N₁.post t₁) (c.n + 1)
  h₂ := advance c.h₂ (N₂.pre t₂) (N₂.post t₂) (c.n + 1)
  n := c.n + 1

/-- `t₂` answers `t₁` (or the other way round) at `c`: same label, equal cause sets. -/
def HPMatch (c : HPos P₁ P₂) (t₁ : T₁) (t₂ : T₂) : Prop :=
  N₁.label t₁ = N₂.label t₂ ∧ causesOf c.h₁ (N₁.pre t₁) = causesOf c.h₂ (N₂.pre t₂)

/-- A history-preserving bisimulation: every step of either side has a matching answer
leading back into the relation. -/
def IsHPBisim (R : HPos P₁ P₂ → Prop) : Prop :=
  ∀ c, R c →
    (∀ t₁, N₁.Fires c.m₁ t₁ →
      ∃ t₂, N₂.Fires c.m₂ t₂ ∧ HPMatch N₁ N₂ c t₁ t₂ ∧ R (c.next N₁ N₂ t₁ t₂)) ∧
    (∀ t₂, N₂.Fires c.m₂ t₂ →
      ∃ t₁, N₁.Fires c.m₁ t₁ ∧ HPMatch N₁ N₂ c t₁ t₂ ∧ R (c.next N₁ N₂ t₁ t₂))

/-- The initial position: no steps yet, empty histories. -/
def HPos.initial : HPos P₁ P₂ := ⟨N₁.initial, N₂.initial, fun _ => ∅, fun _ => ∅, 0⟩

/-- History-preserving bisimilarity of the initial markings. -/
def HPBisimilar : Prop :=
  ∃ R, IsHPBisim N₁ N₂ R ∧ R (HPos.initial N₁ N₂)

/-! ## The library's game: groups -/

/-- Groups of past steps: which current tokens (left, right) depend on them (`Groups`). -/
abbrev Groups (P₁ P₂ : Type*) := Finset (Finset P₁ × Finset P₂)

/-- `meets`: do `a` and `b` share a place? -/
def meets (a b : Finset P) : Bool :=
  decide (a ∩ b).Nonempty

/-- A group's tokens on one side after a step consuming `pre` and producing `post`; `hit`
says whether the step depended on the group. -/
def update (set pre post : Finset P) (hit : Bool) : Finset P :=
  set \ pre ∪ if hit then post else ∅

/-- The groups after matching a left step `(pre₁, post₁)` with a right step `(pre₂, post₂)`,
or `none` when their causes do not correspond (`matched`). -/
def matched (G : Groups P₁ P₂) (pre₁ post₁ : Finset P₁) (pre₂ post₂ : Finset P₂) :
    Option (Groups P₁ P₂) :=
  if ∀ g ∈ G, meets g.1 pre₁ = meets g.2 pre₂ then
    some (((G.image fun g =>
        (update g.1 pre₁ post₁ (meets g.1 pre₁), update g.2 pre₂ post₂ (meets g.1 pre₁))).filter
          fun g => g.1.Nonempty ∨ g.2.Nonempty) ∪
      if post₁.Nonempty ∨ post₂.Nonempty then {(post₁, post₂)} else ∅)
  else none

/-- A position of the library's game: `(marking₁, marking₂, groups)`. -/
structure GPos (P₁ P₂ : Type*) where
  m₁ : Finset P₁
  m₂ : Finset P₂
  groups : Groups P₁ P₂

/-- Over finitely many places there are finitely many positions. -/
instance [Fintype P₁] [Fintype P₂] : Finite (GPos P₁ P₂) :=
  Finite.of_injective (fun a : GPos P₁ P₂ => (a.m₁, a.m₂, a.groups)) fun a b h => by
    cases a; cases b; simp_all

/-- A bisimulation of the grouped game: every step of either side has an answer with the
same label that `matched` accepts, leading back into the relation. -/
def IsGroupBisim (S : GPos P₁ P₂ → Prop) : Prop :=
  ∀ a, S a →
    (∀ t₁, N₁.Fires a.m₁ t₁ →
      ∃ t₂ G, N₂.Fires a.m₂ t₂ ∧ N₁.label t₁ = N₂.label t₂ ∧
        matched a.groups (N₁.pre t₁) (N₁.post t₁) (N₂.pre t₂) (N₂.post t₂) = some G ∧
        S ⟨N₁.fire a.m₁ t₁, N₂.fire a.m₂ t₂, G⟩) ∧
    (∀ t₂, N₂.Fires a.m₂ t₂ →
      ∃ t₁ G, N₁.Fires a.m₁ t₁ ∧ N₁.label t₁ = N₂.label t₂ ∧
        matched a.groups (N₁.pre t₁) (N₁.post t₁) (N₂.pre t₂) (N₂.post t₂) = some G ∧
        S ⟨N₁.fire a.m₁ t₁, N₂.fire a.m₂ t₂, G⟩)

/-- Bisimilarity in the grouped game, from the initial markings and no groups. -/
def GroupBisimilar : Prop :=
  ∃ S, IsGroupBisim N₁ N₂ S ∧ S ⟨N₁.initial, N₂.initial, ∅⟩

/-! ## The abstraction -/

variable {N₁ N₂}

/-- The group of step `k`: the current tokens, left and right, that depend on it. -/
def HPos.group (c : HPos P₁ P₂) (k : ℕ) : Finset P₁ × Finset P₂ :=
  (c.m₁.filter (k ∈ c.h₁ ·), c.m₂.filter (k ∈ c.h₂ ·))

/-- The groups of a position: one per past step that some current token depends on. -/
def HPos.groups (c : HPos P₁ P₂) : Groups P₁ P₂ :=
  (c.m₁.biUnion c.h₁ ∪ c.m₂.biUnion c.h₂).image c.group

/-- The grouped position of a full one. -/
def HPos.abs (c : HPos P₁ P₂) : GPos P₁ P₂ := ⟨c.m₁, c.m₂, c.groups⟩

/-- Current tokens depend only on the `n` steps taken so far, so step `n + 1` is new. -/
def HPos.Fresh (c : HPos P₁ P₂) : Prop :=
  (∀ p ∈ c.m₁, ∀ k ∈ c.h₁ p, k ≤ c.n) ∧ (∀ p ∈ c.m₂, ∀ k ∈ c.h₂ p, k ≤ c.n)

/-- The groups are the non-empty groups of steps. -/
theorem HPos.mem_groups (c : HPos P₁ P₂) (g : Finset P₁ × Finset P₂) :
    g ∈ c.groups ↔ (∃ k, c.group k = g) ∧ (g.1.Nonempty ∨ g.2.Nonempty) := by
  simp only [HPos.groups, Finset.mem_image, Finset.mem_union, Finset.mem_biUnion]
  constructor
  · rintro ⟨k, hk, rfl⟩
    refine ⟨⟨k, rfl⟩, ?_⟩
    simp only [HPos.group, Finset.filter_nonempty_iff]
    exact hk
  · rintro ⟨⟨k, rfl⟩, hne⟩
    refine ⟨k, ?_, rfl⟩
    simpa only [HPos.group, Finset.filter_nonempty_iff] using hne

/-- A step consuming `pre ⊆ m` depends on step `k` iff it consumes a token that does. -/
theorem meets_filter {m pre : Finset P} {h : History P} (hpre : pre ⊆ m) (k : ℕ) :
    meets (m.filter (k ∈ h ·)) pre = decide (k ∈ causesOf h pre) := by
  simp only [meets, causesOf, Finset.mem_biUnion, decide_eq_decide]
  constructor
  · rintro ⟨p, hp⟩
    simp only [Finset.mem_inter, Finset.mem_filter] at hp
    exact ⟨p, hp.2, hp.1.2⟩
  · rintro ⟨p, hp, hk⟩
    exact ⟨p, by simp [hpre hp, hp, hk]⟩

/-- **The groups decide which matched steps are allowed**: two enabled steps have equal
cause sets iff they agree, group by group, on whether they depend on it. -/
theorem causes_eq_iff (c : HPos P₁ P₂) {pre₁ : Finset P₁} {pre₂ : Finset P₂}
    (h₁ : pre₁ ⊆ c.m₁) (h₂ : pre₂ ⊆ c.m₂) :
    causesOf c.h₁ pre₁ = causesOf c.h₂ pre₂ ↔
      ∀ g ∈ c.groups, meets g.1 pre₁ = meets g.2 pre₂ := by
  have hk : ∀ k, meets (c.group k).1 pre₁ = meets (c.group k).2 pre₂ ↔
      (k ∈ causesOf c.h₁ pre₁ ↔ k ∈ causesOf c.h₂ pre₂) := fun k => by
    simp only [HPos.group, meets_filter h₁, meets_filter h₂, decide_eq_decide]
  rw [Finset.ext_iff]
  constructor
  · intro h g hg
    obtain ⟨⟨k, rfl⟩, -⟩ := (c.mem_groups g).1 hg
    exact (hk k).2 (h k)
  · intro h k
    refine (hk k).1 ?_
    by_cases hne : (c.group k).1.Nonempty ∨ (c.group k).2.Nonempty
    · exact h _ ((c.mem_groups _).2 ⟨⟨k, rfl⟩, hne⟩)
    · simp only [not_or, Finset.not_nonempty_iff_eq_empty] at hne
      simp [hne, meets]

/-- One side of the update: after the step, the tokens depending on an earlier step `k`. -/
theorem filter_fire {m pre post : Finset P} {h : History P} {k n : ℕ}
    (hpre : pre ⊆ m) (hdisj : Disjoint post (m \ pre)) (hk : k ≠ n + 1) :
    (m \ pre ∪ post).filter (k ∈ advance h pre post (n + 1) ·) =
      update (m.filter (k ∈ h ·)) pre post (meets (m.filter (k ∈ h ·)) pre) := by
  rw [meets_filter hpre]
  ext p
  rw [Finset.mem_filter]
  have hdis : p ∈ post → p ∉ m \ pre := fun hp => Finset.disjoint_left.1 hdisj hp
  simp only [update, advance, Finset.mem_union, Finset.mem_sdiff, Finset.mem_filter] at hdis ⊢
  by_cases hp : p ∈ post <;> by_cases hq : p ∈ pre <;> by_cases hc : k ∈ causesOf h pre <;>
    simp [hp, hq, hc, hk] at hdis ⊢
  all_goals tauto

/-- One side of the update: the tokens depending on the new step are its outputs. -/
theorem filter_fire_new {m pre post : Finset P} {h : History P} {n : ℕ}
    (hfresh : ∀ p ∈ m, ∀ k ∈ h p, k ≤ n) :
    (m \ pre ∪ post).filter (n + 1 ∈ advance h pre post (n + 1) ·) = post := by
  ext p
  rw [Finset.mem_filter]
  by_cases hp : p ∈ post
  · simp [advance, hp]
  · by_cases hq : p ∈ pre
    · simp [advance, hp, hq]
    · have : p ∈ m → n + 1 ∉ h p := fun hm h' => by have := hfresh p hm _ h'; omega
      simpa [advance, hp, hq] using this

omit [DecidableEq P₁] [DecidableEq P₂] in
/-- Before the step, no token depends on step `n + 1`. -/
theorem HPos.group_fresh {c : HPos P₁ P₂} (hc : c.Fresh) : c.group (c.n + 1) = (∅, ∅) := by
  simp only [HPos.group, Prod.mk.injEq, Finset.filter_eq_empty_iff]
  exact ⟨fun p hp h => by have := hc.1 p hp _ h; omega,
    fun p hp h => by have := hc.2 p hp _ h; omega⟩

/-- After a step numbered `n + 1`, current tokens depend only on steps up to `n + 1`. -/
theorem fresh_advance {m pre post : Finset P} {h : History P} {n : ℕ}
    (hf : ∀ p ∈ m, ∀ k ∈ h p, k ≤ n) (hpre : pre ⊆ m) :
    ∀ p ∈ m \ pre ∪ post, ∀ k ∈ advance h pre post (n + 1) p, k ≤ n + 1 := by
  intro p hp k hk
  simp only [advance] at hk
  split_ifs at hk with hpost hq
  · simp only [Finset.mem_insert, causesOf, Finset.mem_biUnion] at hk
    rcases hk with rfl | ⟨q, hq, hk⟩
    · exact le_rfl
    · exact (hf q (hpre hq) k hk).trans (Nat.le_succ _)
  · simp at hk
  · have : p ∈ m := by simp_all
    exact (hf p this k hk).trans (Nat.le_succ _)

theorem HPos.fresh_next {c : HPos P₁ P₂} (hc : c.Fresh) (t₁ : T₁) (t₂ : T₂)
    (h₁ : N₁.pre t₁ ⊆ c.m₁) (h₂ : N₂.pre t₂ ⊆ c.m₂) : (c.next N₁ N₂ t₁ t₂).Fresh :=
  ⟨fresh_advance hc.1 h₁, fresh_advance hc.2 h₂⟩

/-- `matched` accepts only steps that agree on every group. -/
theorem agree_of_matched {G G' : Groups P₁ P₂} {pre₁ post₁ : Finset P₁}
    {pre₂ post₂ : Finset P₂} (h : matched G pre₁ post₁ pre₂ post₂ = some G') :
    ∀ g ∈ G, meets g.1 pre₁ = meets g.2 pre₂ := by
  by_contra hg
  unfold matched at h
  rw [ite_eq_right_of_eq_false _ _ (eq_false hg)] at h
  cases h

/-- **`matched` commutes with the abstraction**: for allowed steps it returns exactly the
groups of the position after the step. -/
theorem matched_groups {c : HPos P₁ P₂} (hc : c.Fresh) {t₁ : T₁} {t₂ : T₂}
    (h₁ : N₁.Fires c.m₁ t₁) (h₂ : N₂.Fires c.m₂ t₂)
    (hcause : causesOf c.h₁ (N₁.pre t₁) = causesOf c.h₂ (N₂.pre t₂)) :
    matched c.groups (N₁.pre t₁) (N₁.post t₁) (N₂.pre t₂) (N₂.post t₂) =
      some (c.next N₁ N₂ t₁ t₂).groups := by
  have hagree := (causes_eq_iff c h₁.1 h₂.1).1 hcause
  -- On each side, the step depends on step `k` or not alike.
  have hk : ∀ k, meets (c.group k).2 (N₂.pre t₂) = meets (c.group k).1 (N₁.pre t₁) := by
    intro k
    simp only [HPos.group, meets_filter h₁.1, meets_filter h₂.1, hcause]
  -- Every earlier step's group is updated by `update`; the new one is `(post₁, post₂)`.
  have hold : ∀ k, k ≠ c.n + 1 → (c.next N₁ N₂ t₁ t₂).group k =
      (update (c.group k).1 (N₁.pre t₁) (N₁.post t₁) (meets (c.group k).1 (N₁.pre t₁)),
       update (c.group k).2 (N₂.pre t₂) (N₂.post t₂) (meets (c.group k).1 (N₁.pre t₁))) := by
    intro k hne
    refine Prod.ext ?_ ?_
    · convert filter_fire (h := c.h₁) h₁.1 h₁.2 hne using 1 <;> rfl
    · rw [← hk k]
      convert filter_fire (h := c.h₂) h₂.1 h₂.2 hne using 1 <;> rfl
  have hnew : (c.next N₁ N₂ t₁ t₂).group (c.n + 1) = (N₁.post t₁, N₂.post t₂) :=
    Prod.ext (filter_fire_new hc.1) (filter_fire_new hc.2)
  have hempty : ∀ (k : ℕ), c.group k = (∅, ∅) →
      (update (c.group k).1 (N₁.pre t₁) (N₁.post t₁) (meets (c.group k).1 (N₁.pre t₁)),
       update (c.group k).2 (N₂.pre t₂) (N₂.post t₂) (meets (c.group k).1 (N₁.pre t₁))) =
        (∅, ∅) := by
    intro k hk0
    simp [hk0, update, meets]
  unfold matched
  rw [ite_eq_left_of_eq_true _ _ (eq_true hagree), Option.some.injEq]
  symm
  ext g
  rw [HPos.mem_groups (c.next N₁ N₂ t₁ t₂)]
  simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_image]
  constructor
  · rintro ⟨⟨k, rfl⟩, hne⟩
    by_cases hkn : k = c.n + 1
    · subst hkn
      right
      rw [hnew] at hne ⊢
      simp [hne]
    · left
      refine ⟨⟨c.group k, ?_, (hold k hkn).symm⟩, hne⟩
      refine (c.mem_groups _).2 ⟨⟨k, rfl⟩, ?_⟩
      by_contra h0
      simp only [not_or, Finset.not_nonempty_iff_eq_empty] at h0
      have := hempty k (Prod.ext h0.1 h0.2)
      rw [hold k hkn, this] at hne
      simp at hne
  · rintro (⟨⟨g₀, hg₀, rfl⟩, hne⟩ | hg)
    · obtain ⟨⟨k, rfl⟩, hne₀⟩ := (c.mem_groups g₀).1 hg₀
      have hkn : k ≠ c.n + 1 := by
        rintro rfl
        rw [HPos.group_fresh hc] at hne₀
        simp at hne₀
      exact ⟨⟨k, hold k hkn⟩, hne⟩
    · split_ifs at hg with hpost
      · rw [Finset.mem_singleton] at hg
        subst hg
        exact ⟨⟨c.n + 1, hnew⟩, hpost⟩
      · simp at hg

/-! ## The two games have the same winner -/

/-- A history-preserving bisimulation, abstracted, is a bisimulation of the grouped game. -/
theorem IsHPBisim.groups {R : HPos P₁ P₂ → Prop} (hR : IsHPBisim N₁ N₂ R) :
    IsGroupBisim N₁ N₂ fun a => ∃ c, R c ∧ c.Fresh ∧ c.abs = a := by
  rintro _ ⟨c, hc, hf, rfl⟩
  obtain ⟨left, right⟩ := hR c hc
  constructor
  · intro t₁ h₁
    obtain ⟨t₂, h₂, ⟨hl, hcause⟩, hnext⟩ := left t₁ h₁
    exact ⟨t₂, _, h₂, hl, matched_groups hf h₁ h₂ hcause,
      _, hnext, HPos.fresh_next hf t₁ t₂ h₁.1 h₂.1, rfl⟩
  · intro t₂ h₂
    obtain ⟨t₁, h₁, ⟨hl, hcause⟩, hnext⟩ := right t₂ h₂
    exact ⟨t₁, _, h₁, hl, matched_groups hf h₁ h₂ hcause,
      _, hnext, HPos.fresh_next hf t₁ t₂ h₁.1 h₂.1, rfl⟩

/-- A bisimulation of the grouped game, read back on full positions, is a
history-preserving bisimulation. -/
theorem IsGroupBisim.hp {S : GPos P₁ P₂ → Prop} (hS : IsGroupBisim N₁ N₂ S) :
    IsHPBisim N₁ N₂ fun c => c.Fresh ∧ S c.abs := by
  rintro c ⟨hf, hc⟩
  obtain ⟨left, right⟩ := hS _ hc
  -- An answer accepted by `matched` has equal causes and leads to the abstracted position.
  have back : ∀ t₁ t₂ G, N₁.Fires c.m₁ t₁ → N₂.Fires c.m₂ t₂ → N₁.label t₁ = N₂.label t₂ →
      matched c.groups (N₁.pre t₁) (N₁.post t₁) (N₂.pre t₂) (N₂.post t₂) = some G →
      S ⟨N₁.fire c.m₁ t₁, N₂.fire c.m₂ t₂, G⟩ →
      HPMatch N₁ N₂ c t₁ t₂ ∧ (c.next N₁ N₂ t₁ t₂).Fresh ∧ S (c.next N₁ N₂ t₁ t₂).abs := by
    intro t₁ t₂ G h₁ h₂ hl hm hs
    have hcause := (causes_eq_iff c h₁.1 h₂.1).2 (agree_of_matched hm)
    have hG : G = (c.next N₁ N₂ t₁ t₂).groups := by
      rw [matched_groups hf h₁ h₂ hcause] at hm
      exact (Option.some.inj hm).symm
    subst hG
    exact ⟨⟨hl, hcause⟩, HPos.fresh_next hf t₁ t₂ h₁.1 h₂.1, hs⟩
  constructor
  · intro t₁ h₁
    obtain ⟨t₂, G, h₂, hl, hm, hs⟩ := left t₁ h₁
    obtain ⟨hmatch, hnext⟩ := back t₁ t₂ G h₁ h₂ hl hm hs
    exact ⟨t₂, h₂, hmatch, hnext⟩
  · intro t₂ h₂
    obtain ⟨t₁, G, h₁, hl, hm, hs⟩ := right t₂ h₂
    obtain ⟨hmatch, hnext⟩ := back t₁ t₂ G h₁ h₂ hl hm hs
    exact ⟨t₁, h₁, hmatch, hnext⟩

/-- **Same winner, at every position**: Duplicator wins the definition's game from a full
position iff it wins the grouped game from its groups. -/
theorem hp_iff_group {c : HPos P₁ P₂} (hc : c.Fresh) :
    (∃ R, IsHPBisim N₁ N₂ R ∧ R c) ↔ (∃ S, IsGroupBisim N₁ N₂ S ∧ S c.abs) :=
  ⟨fun ⟨_, hR, hRc⟩ => ⟨_, hR.groups, c, hRc, hc, rfl⟩,
    fun ⟨_, hS, hSc⟩ => ⟨_, hS.hp, hc, hSc⟩⟩

/-- **Same winner**: the nets are history-preserving bisimilar iff Duplicator wins the
library's grouped game from `(initial₁, initial₂, ∅)`. -/
theorem hpBisimilar_iff : HPBisimilar N₁ N₂ ↔ GroupBisimilar N₁ N₂ := by
  have hc : (HPos.initial N₁ N₂).Fresh := ⟨by simp [HPos.initial], by simp [HPos.initial]⟩
  have habs : (HPos.initial N₁ N₂).abs = ⟨N₁.initial, N₂.initial, ∅⟩ := by
    simp only [HPos.abs, HPos.groups, HPos.initial, GPos.mk.injEq, true_and,
      Finset.image_eq_empty]
    ext; simp
  unfold HPBisimilar GroupBisimilar
  rw [hp_iff_group hc, habs]

end Bisim.Causal
