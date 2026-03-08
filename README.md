# bisimulationrs

## References

- https://www.ru.is/faculty/luca/PAPERS/algobisimchapter.pdf
- https://doc.rust-lang.org/std/sync/mpsc/fn.channel.html
- https://arxiv.org/pdf/1311.7635.pdf
- http://www.math.unipd.it/~crafa/Pubblicazioni/CrafaRanzatoICALP11.pdf
- https://arxiv.org/pdf/1705.08362.pdf
- https://docs.rs/carboxyl/0.2.1/carboxyl/
- https://ptolemy.eecs.berkeley.edu/eecs20/week4/bisimulation.html
- https://arxiv.org/pdf/1101.4223.pdf
- http://www.cs.unibo.it/~sangio/DOC_public/history_bis_coind.pdf
- http://www.cs.ru.nl/M.Sammartino/publications/ACTA14.pdf
- http://coalg.org/cmcs12/papers/00010040.pdf
- https://arxiv.org/abs/1506.01170
- https://arxiv.org/pdf/1101.4223.pdf
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.8413&rep=rep1&type=pdf
- https://www.southampton.ac.uk/~ps1a06/papers/jcss.pdf
- https://golem.ph.utexas.edu/category/2008/11/coalgebraically_thinking.html
- https://www.georgejpappas.org/papers/ACC06.pdf
- https://arxiv.org/pdf/1310.4106.pdf
- http://homepages.inf.ed.ac.uk/jeh/Bio-PEPA/TCS_CH.pdf
- https://lmcs.episciences.org/1617/pdf
- https://arxiv.org/pdf/1509.08563.pdf
- http://www.numdam.org/article/ITA_1999__33_4-5_357_0.pdf
- http://www.prismmodelchecker.org/manual/ThePRISMLanguage/ProcessAlgebraOperators
- http://www.dcs.ed.ac.uk/pepa/
- http://www.dcs.ed.ac.uk/pepa/fluidflow.pdf
- http://homepages.inf.ed.ac.uk/jeh/TALKS/Qest05.pdf
- https://pdfs.semanticscholar.org/d930/9b8202bd47317a3923c54345029715604060.pdf

## Goal

Inputs: `stream_a`, `stream_b`.

Outputs:

- `stream<bool>`: `right_simulates_left`
- `stream<bool>`: `are_bisimulate`
- `stream<bool>`: `are_bisimilar`

The long-term objective is to compare non-deterministic, potentially infinite or cyclic systems with **continuous, prefix-based feedback**.

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

## Near-term roadmap for this crate

- [x] Provide an incremental prefix checker API in `src/lib.rs`.
- [ ] Replace `HashMap<label, (src, tgt)>` with a multimap adjacency structure (one label can have many transitions).
- [ ] Add a proper on-the-fly game algorithm over discovered states.
- [ ] Support bounded memory with configurable eviction/windowing policies.
- [ ] Expose async adapters (`futures::Stream`) and channel-based ingestion.
- [ ] Add benchmark suites: dense finite LTS, sparse stochastic traces, cyclic workloads.

## Practical domain fit

Animal lifecycle benchmarking is a strong use case:

- cyclic behavior,
- noisy real-world dynamics,
- rare but high-impact transitions (disease/death),
- value in immediate alerts from prefix violations.

This motivates streaming verdicts and eventually probabilistic extensions.
