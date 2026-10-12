# bisimulationrs for Python

Analyse and compare PEPA performance models from Python. The analysis runs in
Rust; results come back as pandas DataFrames. Times are in seconds and rates
in events per second.

```python
import bisimulationrs as bs

herd = bs.Model("""
    r_disease = 0.02;
    Healthy = (disease, r_disease).Sick;
    Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
    Dead = 0;
    Healthy[3]
""")

herd.summary()                    # states before / after lumping
herd.states(lumped=True)          # DataFrame with steady-state probabilities
herd.over_time(range(0, 101, 10)) # expected number of animals per state over time
herd.plot_over_time(horizon=100)  # needs matplotlib

no_recovery = bs.Model("""
    Healthy = (disease, 0.02).Sick;
    Sick = (death, 0.01).Dead;
    Dead = 0;
    Healthy[3]
""")
result = bs.compare(herd, no_recovery)
result.b_mimics_a                 # False
result["b_mimics_a"].formula      # '⟨disease⟩⟨healing⟩tt': A can get sick and then heal, B cannot
```

## Build from source

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop --release        # builds the Rust extension into the venv
pytest
```
