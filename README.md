# bisimulationrs

## References

Links checked on 03/10/2026. Where the original link is dead or wrong, the working source is given.

- Aceto, Ingólfsdóttir, Srba, *The Algorithmics of Bisimilarity* (2011): https://staff.ru.is/luca/PAPERS/algobisimchapter.pdf (moved from `www.ru.is/faculty/luca/...`)
- https://doc.rust-lang.org/std/sync/mpsc/fn.channel.html
- Kułakowski, *Concurrent bisimulation algorithm*: https://arxiv.org/abs/1311.7635
- Crafa & Ranzato, *Probabilistic bisimulation and simulation algorithms by abstract interpretation* (ICALP 2011): http://www.math.unipd.it/~crafa/Pubblicazioni/CrafaRanzatoICALP11.pdf
- Dorsch, Milius, Schröder, Wißmann, *Efficient Coalgebraic Partition Refinement* (CONCUR 2017): https://arxiv.org/pdf/1705.08362.pdf
- https://docs.rs/carboxyl/0.2.1/carboxyl/ (no longer a dependency: unmaintained)
- Lee & Varaiya, EECS20N bisimulation notes: https://ptolemy.eecs.berkeley.edu/eecs20/week4/bisimulation.html
- Staton, *Relating coalgebraic notions of bisimulation* (LMCS 2011): https://arxiv.org/pdf/1101.4223.pdf
- Sangiorgi, *On the origins of bisimulation and coinduction* (TOPLAS 2009): http://www.cs.unibo.it/~sangio/DOC_public/history_bis_coind.pdf
- Bruni, Montanari, Sammartino, *Revisiting causality, coalgebraically* (Acta Informatica 52, 2015): https://arxiv.org/abs/1507.06462 (the original ACTA14 PDF link is dead)
- Sobociński, *Relational presheaves as labelled transition systems* (CMCS 2012): http://coalg.org/cmcs12/papers/00010040.pdf
- ~~https://arxiv.org/abs/1506.01170~~: this ID is an unrelated multi-agent game-theory paper; the intended reference is unknown
- Cattani & Winskel, *Profunctors, open maps and bisimulation* (BRICS RS-04-22): https://www.brics.dk/RS/04/22/ (replaces a dead CiteSeerX link)
- Sobociński, *Relational presheaves, change of base and weak simulation* (JCSS 81(5), 2015), doi:10.1016/j.jcss.2014.12.007 (the Southampton link is dead)
- Corfield, *Coalgebraically thinking* (n-Category Café, 2008): https://golem.ph.utexas.edu/category/2008/11/coalgebraically_thinking.html
- Julius, Girard, Pappas, *Approximate bisimulation for a class of stochastic hybrid systems* (ACC 2006) (the georgejpappas.org link is dead; available via the Wayback Machine)
- Miculan & Peressotti, *Weak bisimulations for labelled transition systems weighted over semirings*: https://arxiv.org/pdf/1310.4106.pdf
- Ciocchetta & Hillston, *Bio-PEPA* (TCS 2009): http://homepages.inf.ed.ac.uk/jeh/Bio-PEPA/TCS_CH.pdf
- Latella, Massink, de Vink, *Bisimulation of labelled state-to-function transition systems coalgebraically* (LMCS 2015): https://lmcs.episciences.org/1617/pdf
- Latella, Massink, de Vink, *A definition scheme for quantitative bisimulation* (QAPL 2015): https://arxiv.org/pdf/1509.08563.pdf
- Davoren, *Topologies, continuity and bisimulations* (RAIRO ITA 1999): http://www.numdam.org/article/ITA_1999__33_4-5_357_0.pdf
- http://www.prismmodelchecker.org/manual/ThePRISMLanguage/ProcessAlgebraOperators
- http://www.dcs.ed.ac.uk/pepa/
- Hillston, *Fluid flow approximation of PEPA models* (QEST 2005): http://www.dcs.ed.ac.uk/pepa/fluidflow.pdf
- Hillston, QEST 2005 talk: http://homepages.inf.ed.ac.uk/jeh/TALKS/Qest05.pdf
- Hayden & Bradley, *Fluid-flow solutions in PEPA to the state space explosion problem* (PASTA 2007): https://spiral.imperial.ac.uk/handle/10044/1/5839 (the Semantic Scholar link is empty)

## Goal

Inputs: `stream_a`, `stream_b`.

Outputs:

- `stream<bool>`: `right_simulates_left`
- `stream<bool>`: `are_bisimulate`
- `stream<bool>`: `are_bisimilar`

The long-term objective is to compare non-deterministic, potentially infinite or cyclic systems with **continuous, prefix-based feedback**.

## Workspace

| crate | contents |
|---|---|
| `bisim-core` | weighted LTS `Wlts<L, W>` (coalgebra `X → FS(X, W)^L`), semirings (`bool`, `u64`, `Real`), partitions, HML formulas, verdicts |
| `bisim-refine` | signature-based partition refinement with every round kept (`∼ₙ`), distinguishing formulas, simulation preorder, on-the-fly game with budgets, worklist fast backend, `(ε, δ)` quasi-bisimulation, `compare` / `bisimilar` / `minimise` |
| `bisim-algebra` | PEPA: AST, typed builder, `.pepa` parser and printer, validator, SOS derivation (cooperation with apparent rates, passive rates, functional `#Count` rates, hiding, arrays in counting form), fluid drift; PRISM `dtmc`/`ctmc` subset import (consts, formulas, synchronising modules, rewards, `system` blocks, queries); JANI subset import (Storm/Modest interchange) |
| `bisim-pepa-macros` | `pepa!`: PEPA models checked at compile time (functional rates included) |
| `bisim-stochastic` | CTMC from a rate-labelled system: steady state, transient analysis by uniformisation, Gillespie SSA trajectories and ensembles, event logs + MLE rate learning, fluid RK4 integration, DTMC/CTMC reachability solvers |
| `bisim-api` | plain, serialisable API (check, analyse, analyse_prism, analyse_jani, query, fluid, over time, compare, observations, forecast, fit) and one JSON entry point shared by all bindings |
| `bisim-wasm` | the API compiled to WebAssembly for the web editor |
| `bisimulationrs` | facade re-exporting everything; `legacy` keeps the 0.1 API (deprecated) |

Bindings and tools built on `bisim-api`, so they all give the same answers:

| path | what | build |
|---|---|---|
| `web/` | **PEPA Studio**: editor with live checking, state-space and component graphs, lumping, occupancy over time, model comparison | `sh web/build.sh`, then `python3 -m http.server -d web/dist` |
| `python/` | Python package `bisimulationrs` (pandas DataFrames) | `cd python && maturin develop --release && pytest` |
| `r/` | R package `bisimulationrs` (data frames, base plots) | `R CMD INSTALL r`, tests with `devtools::test("r")` |

Times are in seconds and rates in s⁻¹.

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

**Is reality following the plan?** Give dated counts (e.g. how many tomato plants were flowering on 19/06/2026) and get, per count, the expected value and a 95 % range from the model, plus an overall on-track verdict and fitted rates. See [`examples/garden/`](examples/garden/README.md).

`bisimilar` and `compare` return a `Verdict`; a violation carries a Hennessy–Milner formula that holds in one system and not in the other, e.g. `⟨disease⟩⟨healing⟩tt` or `⟨start⟩=2 tt` for a rate difference.

## Streaming-first architecture (recommended)

The listed references converge on a practical architecture:

1. **On-the-fly checking instead of batch minimization**
   - Keep an incremental relation/witness graph.
   - Update only affected pairs when a new transition arrives.
   - Emit a new verdict for every prefix.

2. **Partition refinement as a backend optimization**
   - For finite snapshots/windows, run a fast partition-refinement pass (Paige–Tarjan style ideas).
   - Use this as a periodic compaction step, not as the main online loop.

3. **Coinduction up-to for faster convergence**
   - Implement configurable “up-to” techniques (up-to context/closure/congruence).
   - This keeps proofs/checks small in practice and reduces memory pressure.

4. **Symbolic representation for state explosion**
   - Add optional symbolic backends (bitsets/decision diagrams/sparse signatures).
   - Keep the public API generic so engines can switch without API changes.

5. **Quantitative and stochastic extension path**
   - Add pseudometrics and confidence-aware verdict streams (`HoldsSoFar`, `Violated`, `Unknown(p)`), especially for data-derived models.

## Modern landscape (where this crate sits)

Exact bisimulation is classic (Park/Milner, Paige–Tarjan `O(m log n)`)
and still infrastructure, not a fad — but the scalable front-ends live
elsewhere. This crate implements the textbook baseline; the tools below
are what to reach for at scale, and the direction new code here follows:

- **Probabilistic model checkers**: PRISM (Kwiatkowska et al.,
  `prismmodelchecker.org`) and Storm (`stormchecker.org`, Dehnert et
  al., CAV 2017) — guarded-command modelling, CSL/PCTL checking,
  symbolic (MTBDD) and sparse engines, bisimulation minimisation as a
  preprocessing pass, plus a Monte-Carlo simulator engine.
- **Modest** (Hahn et al., `modestchecker.net`) — process algebra with
  data, `modes` statistical model checker for large stochastic systems.
  Import Modest models via its JANI export (`analyse_jani`,
  `PrismModel.from_jani`); the native language remains future work.
- **Bio-PEPA** (Ciocchetta & Hillston, TCS 2009,
  `homepages.inf.ed.ac.uk/jeh/Bio-PEPA/`) — species with stoichiometry
  and *functional rates* (e.g. mass action `beta*S*I`), fluid ODE
  approximation and SSA simulation for large populations. Functional
  rates (`beta * #I`) are supported here too; compartments,
  stoichiometry and fluid limits are not (yet).
- **Coalgebraic partition refinement**: efficient generic algorithms in
  `O((m+n) log n)` (Dorsch, Milius, Schröder, Wißmann, CONCUR 2017;
  Wißmann & Jacobs, 2022) and the CoPaR tool — the faster backend our
  naive Kleene iteration (`bisim-refine::refine`) is tested against but
  does not yet match.
- **Approximate bisimulation** (the answer to noisy rates): a spectrum
  of approximate probabilistic bisimulations (Spork, Baier, Katoen,
  Piribauer, Quatmann, CONCUR 2024, arXiv:2407.07584) and
  `(ε, δ)`-bisimulation for CTMCs with bounds on time-bounded reachability
  (Spork, Baier, Katoen, Klüppelholz, Piribauer, CAV 2025,
  arXiv:2505.15587). See `bisim_refine::{approx_bisimilar_states,
  approx_distance, lumping_error}` (`Verdict::Approx`), which implements a
  quasi-lumpability check in that spirit — with `ε` on class-sum rates, not
  on transition probabilities as in the paper, and without its reachability
  bounds — and `examples/sir/` for the PEPA ↔ Bio-PEPA ↔ PRISM
  correspondence.
- **Scale-out techniques**: statistical model checking, fluid/mean-field
  limits, moment closures and on-the-fly game algorithms (see the
  streaming-first architecture above).

## Near-term roadmap for this crate

- [x] Provide an incremental prefix checker API (now `legacy`).
- [x] Add PEPA/grouped-PEPA benchmarking infrastructure (now `legacy`).
- [x] Replace `HashMap<label, (src, tgt)>` with a weighted multi-LTS (`bisim-core`).
- [x] Generic partition refinement with distinguishing formulas; simulation preorder (`bisim-refine`).
- [x] PEPA definition layer (builder, parser, `pepa!`, validator) and SOS with cooperation semantics (`bisim-algebra`).
- [x] CTMC steady-state and transient analysis (`bisim-stochastic`).
- [x] Web editor and visualiser (WebAssembly), Python and R bindings.
- [x] Observations: dated counts (dd/mm/yyyy) checked against a plan with exact ranges and an adjusted on-track verdict; rate fitting; see `examples/garden/`.
- [x] Quasi-Markovian `(ε, δ)` check with behavioural distance (`Verdict::Approx`) and `lumping_error` for quasi-lumpability (`bisim-refine::approx`).
- [x] Stochastic race semantics: Gillespie SSA trajectories + Monte-Carlo ensembles (`bisim-stochastic::simulate`); SIR correspondence example PEPA ↔ Bio-PEPA ↔ PRISM (`examples/sir/`).
- [x] Bio-PEPA-style functional rates (`beta * #I`, per-state evaluation in SOS) in text, builder and `pepa!`; PRISM `dtmc`/`ctmc` subset import with CSP synchronisation (`bisim-algebra::prism`, `analyse_prism`, `analyse_prism` JSON op).
- [x] Event logs per trajectory (`Trajectory::log`: time, state, label) and MLE rate learning from labelled or unlabelled sequences (`learn_rates`, `without_labels`; `bisim-stochastic::simulate`).
- [x] On-the-fly bisimulation game over lazily discovered pairs (Stirling game with budgets → `Holds`/`HoldsUpTo`/`Unknown`/`Violated` + HML witness; `bisim-refine::game`).
- [x] Fluid ODE mean-field limits for array populations (SOS drift + RK4, no enumeration; `fluid_model`, `bisim-stochastic::integrate`, `fluid()` API + JSON op).
- [x] PRISM rewards (`rewards` blocks → steady-state expectations) and `system` composition (`||`, `|[...]|`); JANI subset import lowering to the same AST (Storm/Modest interchange; `analyse_jani`, `query_jani`); CSL/PCTL-lite queries (`P[F]`, `P[F<=t]`, `S`, thresholds; `query_prism`, `query` JSON op).
- [x] Faster backend: worklist refinement with dirty flags, cached signatures and Hopcroft move bound (`refine_fast`, differential-tested; drives `minimise` and the API analyses).
- [x] Bounded memory: windowed prefix checker with FIFO/reservoir eviction (`legacy::WindowedPrefixChecker`).
- [x] Async adapters (`ingest_stream` over `futures::Stream`, no executor needed) and `mpsc` channel ingestion (`ingest`, `try_ingest`).
- [x] Benchmark suites (`cargo bench -p bisimulationrs`): dense finite LTS (naive vs worklist), structured PEPA models, chains, cycles, stochastic traces, SSA ensembles, enumeration-free fluid, PRISM derive/transient.
- [ ] Full counting-based Paige–Tarjan `O(m log n)` backend; symbolic (MTBDD) engine.
- [ ] Full Modest language import; CSL `G`/`U`, nested system composition, renaming synchronisation; fluid for single components.

## Practical domain fit

Animal lifecycle benchmarking is a strong use case:

- cyclic behavior,
- noisy real-world dynamics,
- rare but high-impact transitions (disease/death),
- value in immediate alerts from prefix violations.

This motivates streaming verdicts and eventually probabilistic extensions.
