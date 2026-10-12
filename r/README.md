# bisimulationrs for R

Analyse and compare PEPA performance models from R. The analysis runs in
Rust; results come back as data frames. Times are in seconds and rates in
events per second (any unit works if you use it consistently, e.g. days).

```r
library(bisimulationrs)

herd <- pepa_model("
  r_disease = 0.02;
  Healthy = (disease, r_disease).Sick;
  Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
  Dead = 0;
  Healthy[3]
")

summary(herd)                          # states before / after lumping
pepa_states(herd, lumped = TRUE)       # steady-state probabilities
pepa_over_time(herd, seq(0, 100, 10))  # expected animals per state over time
pepa_plot_over_time(herd, horizon = 100)

no_recovery <- pepa_model("
  Healthy = (disease, 0.02).Sick;
  Sick = (death, 0.01).Dead;
  Dead = 0;
  Healthy[3]
")
pepa_compare(herd, no_recovery)
#> ✗ b_mimics_a: ⟨disease⟩⟨healing⟩tt holds in A   (A can get sick and heal; B cannot)
```

## Install from this repository

Needs Rust (`rustc` ≥ 1.85, https://rustup.rs). The package builds against the
Rust crates in `../crates`, so install it in place:

```sh
R CMD INSTALL r
```

or from R: `devtools::install("r", build = FALSE)`. Run the tests with
`devtools::test("r")`.
