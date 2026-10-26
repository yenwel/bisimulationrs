# bisimulationrs

Compare a running system with its plan, while it runs.

Classical monitoring judges a crop, a herd or a company by its final yield. When the
harvest is continuous, as in permaculture, there is no final yield to wait for. This
library compares *behaviour*:
- what the system can do next, step by step, coinductively, so cyclic and never-ending
  behaviour needs no end;
- with rates (continuous-time Markov chains, PEPA);
- with numbers for *how far apart* two systems are;
- with verdicts that update as observations arrive.

Plans and roadmap: [ROADMAP.md](ROADMAP.md). Design review and known limits:
[REVIEW.md](REVIEW.md). Machine-checked theorems: [lean/](lean/README.md).

## What it answers

| question | how | where |
|---|---|---|
| Do two systems behave the same? | bisimulation (exact, Markovian), simulation, with a Hennessy–Milner formula as witness when they differ | `bisim-refine` |
| Is the running system still on plan? | incremental comparison of growing prefixes: "holds so far" until a difference is final | `PrefixChecker`, `streaming`, JSON sessions |
| Is reality following the plan's numbers? | dated counts against the model's exact ranges, on-track verdict, rate fitting | `check_observations`, `ObserveSession` |
| How far apart are two models? | optimal-transport bisimulation distance, with the state pairs that drive it | `bisim-metric` |
| Within tolerance, and what does that guarantee? | (ε, δ)-bisimulation with reachability bounds | `bisim-metric` |
| How far apart are two *logs*? | distance estimated from observed transitions only | `bisim-metric` |
| Do two sensor series evolve alike? | dynamical similarity analysis (not a bisimulation) | `bisim-metric` |
| Same steps, but the same *dependencies* between them? | history-preserving bisimulation on the causal (Petri net) semantics of PEPA, with a witness naming the causes | `hp_bisimilar`, `causal_net`, op `causal` |
| What if we change a rate? Which change explains the gap? | model-based what-if and Shapley attribution | `what_if`, `attribute` |

## Quick start

```rust
use bisimulationrs::{pepa, refine::{bisimilar, minimise}, stochastic::Ctmc};

let herd = pepa! {
    r_disease = 0.02;
    Healthy = (disease, r_disease).Sick;
    Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
    Dead = 0;
    Vet = (healing, infty).Vet;
    Healthy[3] <healing> Vet
}
.derive()?;

let lumped = minimise(&herd.lts);                     // Markovian bisimulation quotient
let ctmc = Ctmc::from_wlts(&lumped);
let p = ctmc.transient(&ctmc.initial_distribution(), 30.0, 1e-9);
```

Rates are per time unit of the model. Dated observations take a unit (day, week, …).

The same engine, with the same answers, is available here:

| path | what | build |
|---|---|---|
| `web/` | **PEPA Studio**: editor with live checking, state space, lumping, behaviour over time, observations, comparison and distances, a replay of streaming comparison, tutorial | `sh web/build.sh`, then `python3 -m http.server -d web/dist` |
| `python/` | Python package (pandas DataFrames) | `cd python && maturin develop --release && pytest` |
| `r/` | R package (data frames, base plots) | `R CMD INSTALL r`, tests with `devtools::test("r")` |
| `lean/` | Lean 4 proofs of the theory the engine relies on | `cd lean && lake exe cache get && lake build` |

Runnable examples are listed in [examples/README.md](examples/README.md):
- comparison, minimisation and steady state;
- streaming prefixes;
- garden observations;
- biology, ecology and economics models;
- permaculture;
- distances, tolerance, inspection logs and sensor series.

## Workspace

| crate | contents |
|---|---|
| `bisim-core` | weighted LTS `Wlts<L, W>` (coalgebra `X → FS(X, W)^L`), semirings (`bool`, `u64`, `Real`), safe Petri nets, partitions, HML formulas, verdicts |
| `bisim-refine` | partition refinement with every round kept (`∼ₙ`) and a worklist backend; distinguishing formulas; simulation preorder; on-the-fly game with budgets; incremental prefix games (`PrefixChecker`) with compaction of settled positions; history-preserving bisimulation on safe nets (`hp_bisimilar`) |
| `bisim-algebra` | PEPA (builder, parser, `pepa!`, validator, SOS with apparent-rate cooperation, passive and functional rates, arrays, fluid drift, causal net semantics); PRISM `dtmc`/`ctmc` and JANI subset import |
| `bisim-pepa-macros` | `pepa!`: PEPA models checked at compile time |
| `bisim-stochastic` | CTMC transient (uniformisation) and steady state, reachability (interval iteration), Gillespie SSA (resumable), online rate learning with credible intervals, fluid RK4 |
| `bisim-metric` | behavioural distances: optimal-transport bisimulation metric (exact and Sinkhorn), (ε, δ)-bisimulation, distance from samples, DSA |
| `bisim-api` | plain serialisable API and one JSON entry point for every binding, including stateful sessions |
| `bisim-wasm` | the API as WebAssembly, for PEPA Studio |
| `bisimulationrs` | facade; `streaming` (verdict iterators and `futures_core::Stream`s); `legacy` 0.1 API (deprecated) |

## Comparing while it runs

`PrefixChecker` keeps the simulation and bisimulation games and updates only what a new
event touches:
- A transition from a state declared complete is rejected, so a violation is final.
- After every event, the answer equals solving the game from scratch: property-tested on
  700 random cases against `replay_verdict`.
- 4,000 events take 1.6 ms, against 2.0 s for replaying.
- Settled positions can be compacted. Peak memory dropped about 43% on a harvest
  workload, and not at all on a ring workload until the stream ends.

Other ways to stream:
- **Rust:** `bisimulationrs::streaming` gives the same comparison as an iterator or a
  `futures_core::Stream` of changed verdicts.
- **JSON:** sessions (`session_open` / `session_push` / `session_close`) give it to the
  browser, Python and R.
- **Stochastic side:** online too. `OnlineRates`, `gillespie_continue`, `ObserveSession`
  (each count advances from the previous date) and warm-started fits.

## How far apart, not just whether

| question | method | guarantee | caveat |
|---|---|---|---|
| how differently do two models behave, seen through a count? | bisimulation distance (`bisim_distance`, op `distance`) | exact up to a reported error bound; 0 exactly for bisimilar states | needs both models; explores reachable state pairs |
| is the system within tolerance of the plan? | (ε, δ)-bisimulation (`min_epsilon`, op `epsdelta`) | bounds time-bounded reachability (Prop. 3 of the paper) | the bound is loose and grows with the horizon |
| how far apart are two *logged* systems? | distance from samples (`SampleDistance`, op `stream_distance`, sessions) | converges as 1/√transitions | needs very many transitions (≈10⁶ for ±0.07 on 4 states) |
| how far apart are two models too large for the exact method? | learned bisimulation distance, Deep Bisimulation for Control style (`learn_metric`) | none: an estimate; within about 0.01 of the exact value where both were computed | training time grows like 1/(1 − γ); needs state features |
| do two sensor series evolve alike? | DSA (`dsa`, op `dsa`) | invariant to rotations of the sensor coordinates | compares fitted linear dynamics; not a bisimulation |

Experimental, behind the feature `learned-challenger`: a learned Spoiler for the
bisimulation game (`refine::challenger`). Its witnesses are re-checked exactly and a failed
search answers `Unknown`, never "same". On the bundled models it is slower than the exact
game and refinement and misses 7 of 21 differences (ROADMAP §4e has the numbers).

## What is verified, and how

Not everything yet. What exists today:
- **Lean 4 proofs** (`lean/`, no `sorry`, standard axioms only):
  - weighted bisimulation is a congruence for parallel composition;
  - PEPA's apparent-rate cooperation preserves strong equivalence;
  - the finite grouped game for causal comparison decides history-preserving bisimulation
    (the same winner as the definition);
  - a lumping survives any context;
  - a rate intervention can break a lumping.
- **Witnesses:** every "differs" answer carries a formula you can check by hand.
- **Differential and property tests against independent oracles:**
  - fast vs naive refinement;
  - game vs refinement;
  - incremental vs replay;
  - distance 0 vs lumping;
  - (ε, δ) at zero tolerance vs strong bisimulation;
  - history-preserving bisimulation vs its definition with full histories.
- **Closed forms:** stochastic results are checked against them.
- **Fuzzing:** the parsers are mutation-fuzzed.

The Rust code itself is not proved correct. [ROADMAP.md](ROADMAP.md) sets out how to get
there:
1. certificates for every verdict;
2. a checker verified in Lean;
3. the core algorithms proved, with the Rust linked to the proofs via Aeneas;
4. verified numerics.

## Where this crate sits

Exact bisimulation is classic (Park, Milner; Paige–Tarjan `O(m log n)`) and remains
infrastructure. For scale, reach for the dedicated tools:
- PRISM and Storm (probabilistic model checking, symbolic engines);
- Modest (import via JANI);
- generic coalgebraic partition refinement (CoPaR).

This crate's niche is the combination: comparison of running systems with plans,
quantitative answers, explanations, and the bindings to use them on farm and population
data.

## References

Grouped by topic. Entries marked † were added or re-checked by search between 19/10 and 23/10/2026,
those marked ‡ on 26/10/2026 (against Crossref and the Institut Polytechnique de Paris research
portal); the
rest were checked on 03/10/2026. Where an original link was dead, the working source is
given.

### Bisimulation, coinduction, coalgebra

- Sangiorgi, *On the origins of bisimulation and coinduction* (TOPLAS 2009): http://www.cs.unibo.it/~sangio/DOC_public/history_bis_coind.pdf
- Sangiorgi, *Introduction to Bisimulation and Coinduction* (Cambridge University Press, 2012)
- Aceto, Ingólfsdóttir, Srba, *The Algorithmics of Bisimilarity* (2011): https://staff.ru.is/luca/PAPERS/algobisimchapter.pdf
- Staton, *Relating coalgebraic notions of bisimulation* (LMCS 2011): https://arxiv.org/abs/1101.4223
- Cattani & Winskel, *Profunctors, open maps and bisimulation* (BRICS RS-04-22): https://www.brics.dk/RS/04/22/
- Sobociński, *Relational presheaves as labelled transition systems* (CMCS 2012): http://coalg.org/cmcs12/papers/00010040.pdf; *Relational presheaves, change of base and weak simulation* (JCSS 81(5), 2015), doi:10.1016/j.jcss.2014.12.007
- Miculan & Peressotti, *Weak bisimulations for labelled transition systems weighted over semirings*: https://arxiv.org/abs/1310.4106
- Latella, Massink, de Vink, *Bisimulation of labelled state-to-function transition systems coalgebraically* (LMCS 2015): https://lmcs.episciences.org/1617/pdf; *A definition scheme for quantitative bisimulation* (QAPL 2015): https://arxiv.org/abs/1509.08563
- Davoren, *Topologies, continuity and bisimulations* (RAIRO ITA 1999): http://www.numdam.org/article/ITA_1999__33_4-5_357_0.pdf
- Corfield, *Coalgebraically thinking* (n-Category Café, 2008): https://golem.ph.utexas.edu/category/2008/11/coalgebraically_thinking.html
- Lee & Varaiya, EECS20N bisimulation notes: https://ptolemy.eecs.berkeley.edu/eecs20/week4/bisimulation.html

### Algorithms

- Dorsch, Milius, Schröder, Wißmann, *Efficient Coalgebraic Partition Refinement* (CONCUR 2017): https://arxiv.org/abs/1705.08362; journal version Wißmann, Dorsch, Milius, Schröder (LMCS 16(1), 2020)
- † Jacobs & Wißmann, *Coalgebraic Partition Refinement For All Functors* (arXiv:2204.12368, 2022): https://arxiv.org/abs/2204.12368
- Crafa & Ranzato, *Probabilistic bisimulation and simulation algorithms by abstract interpretation* (ICALP 2011): http://www.math.unipd.it/~crafa/Pubblicazioni/CrafaRanzatoICALP11.pdf
- Kułakowski, *Concurrent bisimulation algorithm*: https://arxiv.org/abs/1311.7635
- † Saha, *An incremental bisimulation algorithm* (FSTTCS 2007), as described by the BisPy documentation: https://bispy-bisimulation-in-python.readthedocs.io/en/latest/algorithms/saha.html

### Stochastic process algebra and population models

- PEPA: http://www.dcs.ed.ac.uk/pepa/; Hillston, *A Compositional Approach to Performance Modelling* (1996)
- Hillston, *Fluid flow approximation of PEPA models* (QEST 2005): http://www.dcs.ed.ac.uk/pepa/fluidflow.pdf
- Hayden & Bradley, *Fluid-flow solutions in PEPA to the state space explosion problem* (PASTA 2007): https://spiral.imperial.ac.uk/handle/10044/1/5839
- Ciocchetta & Hillston, *Bio-PEPA* (TCS 2009): http://homepages.inf.ed.ac.uk/jeh/Bio-PEPA/TCS_CH.pdf
- PRISM process-algebra operators: http://www.prismmodelchecker.org/manual/ThePRISMLanguage/ProcessAlgebraOperators
- Jensen, *Markoff chains as an aid in the study of Markoff processes* (Scandinavian Actuarial Journal, 1953): uniformisation
- † Leguizamon-Robayo, Jiménez-Pastor, Tribastone, Tschaikowski, Vandin, *Approximate Constrained Lumping of Chemical Reaction Networks* (arXiv:2411.14242, 2024): https://arxiv.org/abs/2411.14242

### Quantitative and approximate comparison

- † Calo, Jonsson, Neu, Schwartz, Segovia-Aguas, *Bisimulation Metrics are Optimal Transport Distances, and Can be Computed Efficiently* (NeurIPS 2024, arXiv:2406.04056): https://arxiv.org/abs/2406.04056
- † Calo, Jonsson, Neu, Schwartz, Segovia-Aguas, *Distances for Markov chains from sample streams* (NeurIPS 2025, arXiv:2505.18005): https://arxiv.org/abs/2505.18005
- † Spork, Baier, Katoen, Piribauer, Quatmann, *A Spectrum of Approximate Probabilistic Bisimulations* (CONCUR 2024, arXiv:2407.07584): https://arxiv.org/abs/2407.07584
- † Spork, Baier, Katoen, Klüppelholz, Piribauer, *Approximate Probabilistic Bisimulation for Continuous-Time Markov Chains* (CAV 2025, LNCS 15932, arXiv:2505.15587): https://arxiv.org/abs/2505.15587
- † Zhang, McAllister, Calandra, Gal, Levine, *Learning Invariant Representations for Reinforcement Learning without Reconstruction* (Deep Bisimulation for Control, ICLR 2021, arXiv:2006.10742): https://arxiv.org/abs/2006.10742
- † Castro, Kastner, Panangaden, Rowland, *MICo: Improved representations via sampling-based state similarity for Markov decision processes* (NeurIPS 2021, arXiv:2106.08229): https://arxiv.org/abs/2106.08229
- † Ostrow, Eisen, Kozachkov, Fiete, *Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis* (NeurIPS 2023, arXiv:2306.10168): https://arxiv.org/abs/2306.10168
- ‡ Desharnais, Laviolette, Zhioua, *Testing Probabilistic Equivalence Through Reinforcement Learning* (FSTTCS 2006, LNCS 4337, pp. 236–247), doi:10.1007/11944836_23; extended under the same title in Information and Computation 227 (2013), pp. 21–57, doi:10.1016/j.ic.2013.02.002. Abstracts only: the full texts were not reachable.
- ‡ Desharnais, Laviolette, Moturu, Zhioua, *Trace Equivalence Characterization Through Reinforcement Learning* (Canadian AI 2006, LNAI 4013, pp. 371–382), doi:10.1007/11766247_32. Abstract only.
- Altschuler, Weed, Rigollet, *Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration* (NeurIPS 2017): the rounding step used by the Sinkhorn solver
- Julius, Girard, Pappas, *Approximate bisimulation for a class of stochastic hybrid systems* (ACC 2006), via the Wayback Machine; Girard & Pappas, *Approximation metrics for discrete and continuous systems* (IEEE TAC 2007)

### Causality

- Bruni, Montanari, Sammartino, *Revisiting causality, coalgebraically* (Acta Informatica 52, 2015): https://arxiv.org/abs/1507.06462
- † Vogler, *Deciding history preserving bisimilarity* (ICALP 1991); Montanari & Pistore, *Minimal transition systems for history-preserving bisimulation* (STACS 1997)
- † Jategaonkar & Meyer, *Deciding true concurrency equivalences on safe, finite nets* (TCS 154(1), 1996; ICALP 1993): history-preserving bisimulation is DEXPTIME-complete on finite safe nets
- † Ribaudo, *Stochastic Petri net semantics for stochastic process algebras* (PNPM 1995)
- † Jacobs, Kissinger, Zanasi, *Causal Inference by String Diagram Surgery* (FoSSaCS 2019, arXiv:1811.08338; MSCS 31(5), 2021): https://arxiv.org/abs/1811.08338
- † Fritz, *A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics* (arXiv:1908.07021): https://arxiv.org/abs/1908.07021
- † Huang & Valtorta, *Pearl's calculus of intervention is complete* (UAI 2006); Shpitser & Pearl, *Identification of joint interventional distributions in recursive semi-Markovian causal models* (AAAI 2006)
- † Ness, Paneri, Vitek, *Integrating Markov processes with structural causal modeling enables counterfactual inference in complex systems* (NeurIPS 2019, arXiv:1911.02175): https://arxiv.org/abs/1911.02175
- † Ying, *Causal Inference for Complex Continuous-Time Longitudinal Studies* (arXiv:2206.12525, rev. 2025): https://arxiv.org/abs/2206.12525
- † Rytgaard & van der Laan, *Causal inference for calibrated scaling interventions on time-to-event processes* (arXiv:2510.16798, 2025): https://arxiv.org/abs/2510.16798
- † Brodersen, Gallusser, Koehler, Remy, Scott, *Inferring causal impact using Bayesian structural time-series models* (Annals of Applied Statistics 9(1), 2015)
- Shapley, *A value for n-person games* (1953): the attribution method

### Verification

- † Hölzl, *Markov chains and Markov decision processes in Isabelle/HOL* (JAR 2017, arXiv:1405.7012): https://home.cit.tum.de/~hoelzl/documents/hoelzl2017mdp.pdf
- † Chatterjee, Quatmann, Schäffeler, Weininger, Winkler, Zilken, *Fixed Point Certificates for Reachability and Expected Rewards in MDPs* (TACAS 2025, arXiv:2501.11467): https://arxiv.org/abs/2501.11467
- † Ho & Protzenko, *Aeneas: Rust Verification by Functional Translation* (ICFP 2022): https://www.microsoft.com/en-us/research/publication/aeneas-rust-verification-by-functional-translation/
- † Lattuada et al., *Verus: Verifying Rust Programs using Linear Ghost Types* (OOPSLA 2023, arXiv:2303.05491): https://arxiv.org/abs/2303.05491
- † *Kani: A Model Checker for Rust* (arXiv:2607.01504, 2026): https://arxiv.org/abs/2607.01504
- Pous, up-to techniques for bisimulation, formalised in Coq: https://perso.ens-lyon.fr/damien.pous/upto
