# Examples

Model files (PEPA, PRISM) with their own notes:

- [`garden/`](garden/README.md): tomato plants against a plan, with dated counts.
- [`sir/`](sir/README.md): the same epidemic in PEPA, Bio-PEPA style and PRISM.

Runnable Rust programs (`cargo run -p bisimulationrs --example <name>`):

| example | shows |
|---|---|
| `compare_systems` | simulation vs bisimulation, and the distinguishing formula `⟨a⟩¬⟨c⟩tt` |
| `minimise_symmetry` | lumping three identical animals (27 → 10 states) with the same transient behaviour |
| `prefix_stream` | verdicts over a growing prefix, updated per observation: "holds so far" until a difference can no longer be repaired |
| `steady_state` | long-run distribution of a chain with very different time scales |
| `prism_model` | importing a PRISM queue and checking the long-run answer against the closed form |
| `behaviour_distance` | a number for how differently a guild and a monoculture behave, the situations that drive it, and 0 for the same bed written plant by plant |
| `tolerance_plan` | a herd within (ε, δ) of its plan, the guaranteed bound on reaching cull weight next to the actual difference |
| `inspection_logs` | the distance estimated from weekly inspection logs alone, picking the plan closest to the truth |
| `causal_farm` | same Markov chain, different dependencies: two workers vs one farmer, two vets vs one, with a witness naming the causes |
| `learned_distance` | a neural (deep-RL style) bisimulation distance next to the exact one: within about 0.01 on small and medium beds, and an estimate for beds with 1.7 million state pairs |
| `emit_certificates` | the library's "same" answers for four models written as Lean certificates (`lean/Bisim/Certificates.lean`), which the Lean kernel checks in CI |
| `sensor_similarity` | dynamical similarity of sensor series: 0 for remixed sensors, positive for a bed without companions |

## Population models from biology, ecology and economics

Each directory has one `.pepa` model (open it in PEPA Studio, `web/`, or load
it from Python or R) and the Rust example
[`domain_models`](../crates/bisimulationrs/examples/domain_models.rs) solves
all four with the fluid approximation:

| directory | field | what it models | features used |
|---|---|---|---|
| [`coral/`](coral/coral.pepa) | ecology | a reef losing coral to bleaching in recurring heat waves | environment component read through `#Heat`, arrays |
| [`antibiotic/`](antibiotic/antibiotic.pepa) | biology | drug cycling; susceptible cells die only while the drug is on and resistance rises | environment-dependent per-cell rates, mutation and reversion |
| [`bass/`](bass/bass.pepa) | economics | adoption of a product by advertising and word of mouth | functional rate `p + q * #Adopter / n_total` |
| [`credit/`](credit/credit.pepa) | finance | rating migration and defaults in a portfolio of 100 firms | absorbing state, exact vs mean-field |

Permaculture, in [`permaculture/`](permaculture/guild.pepa) with the example
[`permaculture_guild`](../crates/bisimulationrs/examples/permaculture_guild.rs):
a bed of 60 plants as a monoculture against a guild (companion plantings that
repel aphids, plus 12 ladybirds that hunt where infested plants are). It uses
`#Hunting` and `#Infested` in rates, so the predators respond to prey. After
26 weeks the guild has lost about 34 plants against 49, roughly 15 saved;
the monoculture's healthy count follows `60 e^{-0.15 t}` exactly.

Checks that hold in the example output: the Bass curve matches the closed
form `F(t) = (1 − e^{−(p+q)t}) / (1 + (q/p) e^{−(p+q)t})`, and the credit
portfolio's expected defaults equal 100 times a single firm's exact
transient probability (1.06, 4.97 and 16.75 defaults at 5, 10 and 20 years).
Parameter names must start with a lowercase letter: `N = 1000;` would be read
as a component definition.
