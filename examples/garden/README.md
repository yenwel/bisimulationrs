# Is my tomato bed on track?

`tomatoes.pepa` is the plan for 24 tomato plants from transplanting
(15/05/2026): about 28 days to first flowers, 14 more to fruit set and 35
more to ripe fruit, with a small daily chance of losing a plant. Rates are
per day. Each stage is split into two equal steps (`Vegetative_1`,
`Vegetative_2`, ...), which makes stage lengths more regular than a single
random delay; you count plants simply by stage (`Vegetative`, `Flowering`,
`Fruiting`, `Ripe`, `Lost`).

`observations.csv` holds the counts made during the season (dates as
dd/mm/yyyy).

## What the check says

- **Mid-June** (first 4 counts): still consistent with the plan. One count
  (12 plants without flowers on 19/06, range 2–11) is high, but with 4 checks
  that can happen by chance.
- **Mid-July** (all 8 counts): off plan. Too many plants are still without
  flowers, and only 1 ripe plant on 17/07 where 4–13 were expected.
- **Fitting `veg_days`** gives about 47 days instead of 28: the plants are
  slow to start flowering. With that value all counts are back in range.

Each count gets an exact 95 % range from the model (binomial, because the
plants grow independently). The overall verdict accounts for checking
several counts (Bonferroni), which stays valid when you count the same
plants again later.

## Python

```python
import bisimulationrs as bs

plan = bs.Model.from_file("examples/garden/tomatoes.pepa")
counts = bs.read_observations("examples/garden/observations.csv")

report = plan.check_observations(counts, start="15/05/2026", unit="day")
report.on_track, report.explanation
report.table                                   # one row per count
plan.plot_observations(counts, start="15/05/2026", horizon=100)   # needs matplotlib

fit = plan.fit(counts, "veg_days", start="15/05/2026")
fit.params                                     # planned 28 → fitted ~47
bs.compare(plan, fit.model)                    # which rate differs, as a formula
```

## R

```r
library(bisimulationrs)

plan <- pepa_read("examples/garden/tomatoes.pepa")
counts <- pepa_read_observations("examples/garden/observations.csv")

pepa_check_observations(plan, counts, start = "15/05/2026", unit = "day")
pepa_plot_observations(plan, counts, start = "15/05/2026", horizon = 100)
fit <- pepa_fit(plan, counts, "veg_days", start = "15/05/2026")
pepa_compare(plan, fit$model)
```

## In the browser

Open PEPA Studio, pick the **Tomato bed** example and go to **Observations**.
