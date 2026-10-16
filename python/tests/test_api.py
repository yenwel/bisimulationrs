import math

import pandas as pd
import pytest

import bisimulationrs as bs

CYCLE = """r = 1;
P1 = (start, r).P2;
P2 = (run, r).P3;
P3 = (stop, r).P1;
"""

HERD = """r_disease = 0.02;
Healthy = (disease, r_disease).Sick;
Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
Dead = 0;
Healthy[3]"""


def p1_exact(t):
    return (1 + 2 * math.exp(-1.5 * t) * math.cos(math.sqrt(3) / 2 * t)) / 3


def test_summary_and_lumping():
    m = bs.Model(CYCLE + "P1 || P1")
    assert m.summary() == {"states": 9, "transitions": 18, "lumped_states": 6, "components": ["P1", "P1"], "note": None}
    states = m.states()
    assert list(states.columns) == ["state", "name", "initial", "lumped_state", "steady_state"]
    assert states["steady_state"].sum() == pytest.approx(1.0)
    lumped = m.states(lumped=True)
    assert len(lumped) == 6
    assert lumped["merged_states"].sum() == 9
    assert lumped["steady_state"].sum() == pytest.approx(1.0)
    assert set(m.transitions().columns) == {"source", "target", "action", "rate"}


def test_over_time_matches_closed_form():
    m = bs.Model(CYCLE + "P1[2]")
    long = m.over_time([0.0, 1.0])
    p1 = long[(long["state"] == "P1") & (long["time"] == 1.0)]["expected"].item()
    assert p1 == pytest.approx(2 * p1_exact(1.0), abs=1e-8)

    wide = m.over_time([0.0, 1.0], wide=True)
    assert list(wide.columns) == ["time", "P1[2]: P1", "P1[2]: P2", "P1[2]: P3"]
    assert wide.loc[0, "P1[2]: P1"] == pytest.approx(2.0)


def test_components_table():
    comps = bs.Model(HERD).components()
    assert list(comps) == ["Healthy"]
    healthy = comps["Healthy"]
    assert set(healthy["action"]) == {"disease", "healing", "death"}
    assert not healthy["passive"].any()


def test_errors_carry_positions():
    with pytest.raises(bs.ModelError) as err:
        bs.Model("P = (a, r).P;\nP")
    d = err.value.diagnostics[0]
    assert (d.severity, d.line, d.column) == ("error", 1, 9)
    assert "unknown rate parameter" in str(err.value)

    assert bs.check(CYCLE + "P1") == []
    warnings = bs.check("P = (a, 1).P; Q = (b, 1).Q; P <b> Q")
    assert [w.severity for w in warnings] == ["warning"]


def test_compare_explains_difference():
    no_recovery = "Healthy = (disease, 0.02).Sick; Sick = (death, 0.01).Dead; Dead = 0; Healthy[3]"
    result = bs.compare(bs.Model(HERD), no_recovery)
    assert not result.b_mimics_a
    assert result["b_mimics_a"].formula == "⟨disease⟩⟨healing⟩tt"
    assert result["b_mimics_a"].formula_holds_in == "A"
    frame = result.to_frame()
    assert isinstance(frame, pd.DataFrame) and len(frame) == 4

    same = bs.compare(CYCLE + "P1 || P1", CYCLE + "P1[2]")
    assert same.same_behaviour and same.same_branching


def test_state_space_limit():
    # Independent copies: the state space is not listed, but behaviour over time works.
    big = bs.Model(CYCLE + "P1[200]", max_states=50)
    assert big.summary()["note"]
    assert big.over_time([1.0])["expected"].sum() == pytest.approx(200)
    # Interacting copies need the full state space, which is too large here.
    coupled = bs.Model(CYCLE + "S = (start, 5).S; P1[200] <start> S", max_states=50)
    with pytest.raises(bs.ModelError, match="exceeds the limit"):
        coupled.over_time([1.0])


GARDEN = open(__file__.replace("python/tests/test_api.py", "examples/garden/tomatoes.pepa"), encoding="utf-8").read()
SEASON = __file__.replace("python/tests/test_api.py", "examples/garden/observations.csv")


def test_garden_observations_and_fit():
    plan = bs.Model(GARDEN)
    obs = bs.read_observations(SEASON)
    june = plan.check_observations(obs.head(4), start="15/05/2026", unit="day")
    assert june.on_track
    assert list(june.table["status"]) == ["on_track", "on_track", "above_plan", "on_track"]

    july = plan.check_observations(obs, start="15/05/2026", unit="days")
    assert not july and "Off plan" in july.explanation

    fit = plan.fit(obs, "veg_days", start="15/05/2026")
    assert 35 < fit.params.loc[0, "fitted"] < 55
    assert fit.check.on_track
    result = bs.compare(plan, fit.model)
    assert not result.same_behaviour and result.same_branching
    assert "one copy" in result.note
    assert result["same_behaviour"].formula.startswith("⟨grow⟩=")


def test_observations_accept_dates_and_dicts():
    from datetime import date as d

    plan = bs.Model(GARDEN)
    rows = [{"date": d(2026, 6, 5), "state": "Vegetative", "count": 17}]
    report = plan.check_observations(rows, start=d(2026, 5, 15))
    assert report.table.loc[0, "date"] == "05/06/2026"
    band = plan.forecast("Flowering", [0, 21], start="15/05/2026")
    assert list(band["date"]) == ["15/05/2026", "05/06/2026"]
    with pytest.raises(bs.ModelError, match="has no state `Blooming`"):
        plan.check_observations([{"date": "05/06/2026", "state": "Blooming", "count": 1}], start="15/05/2026")


def test_large_independent_population():
    big = bs.Model(GARDEN.replace("Tomato[24]", "Tomato[500]"), max_states=1000)
    assert big.summary()["states"] is None
    t = big.over_time([0, 50])
    assert t[t["time"] == 50]["expected"].sum() == pytest.approx(500)


def test_prism_model():
    m = bs.PrismModel(
        "ctmc module m s : [0..1] init 0;"
        " [] s=0 -> 1.5 : (s'=1); [] s=1 -> 2.5 : (s'=0); endmodule"
    )
    assert m.summary() == {"states": 2, "transitions": 2, "lumped_states": 2, "variables": ["s"], "note": None}
    states = m.states()
    assert list(states.columns) == ["state", "name", "initial", "lumped_state", "steady_state"]
    assert states["steady_state"].sum() == pytest.approx(1.0)
    assert states.loc[states["state"] == 0, "steady_state"].item() == pytest.approx(2.5 / 4.0)
    assert set(m.transitions().columns) == {"source", "target", "action", "rate"}
    with pytest.raises(bs.ModelError, match="mdp"):
        bs.PrismModel("mdp module m x : [0..1] init 0; endmodule")


def test_query_rewards_and_fluid():
    m = bs.PrismModel(
        "ctmc module m s : [0..1] init 0;"
        " [go] s=0 -> 1.5 : (s'=1); [back] s=1 -> 2.5 : (s'=0); endmodule"
        ' rewards "r" s=1 : 10; [go] s=0 : 2; endrewards'
    )
    r = m.query("P=? [F<=1 s=1]")
    assert r["probability"] == pytest.approx(1 - math.exp(-1.5), abs=1e-6)
    assert m.query("P>=0.9 [F<=5 s=1]")["holds"] is True
    assert m.query("S=? [s=0]")["probability"] == pytest.approx(2.5 / 4.0)
    rewards = m.rewards()
    assert list(rewards.columns) == ["block", "state", "transition"]
    assert rewards.loc[0, "state"] == pytest.approx(3.75)
    assert rewards.loc[0, "transition"] == pytest.approx(1.875)

    sir = bs.Model(
        "beta = 0.3; gamma = 0.1;"
        " S = (infect, beta * #I).I; I = (recover, gamma).R; R = 0;"
        " S[9] || I[1]"
    )
    long = sir.fluid_over_time([0.0, 30.0], dt=0.05)
    assert list(long.columns) == ["time", "component", "state", "expected"]
    assert long["expected"].sum() == pytest.approx(10.0 * 2, abs=1e-6)
    i0 = long[(long["state"] == "I") & (long["time"] == 0.0)]["expected"].sum()
    assert i0 == pytest.approx(1.0)

    import json as _json

    jani = _json.dumps(
        {
            "jani-version": 1,
            "name": "m",
            "type": "ctmc",
            "variables": {
                "s": {"type": {"kind": "bounded", "lower-bound": 0, "upper-bound": 1}, "initial-value": 0}
            },
            "automata": [
                {
                    "name": "m",
                    "locations": [{"name": "l"}],
                    "initial-location": "l",
                    "edges": [
                        {"location": "l", "guard": {"op": "=", "left": "s", "right": 0},
                         "rate": 1.5, "destinations": [{"location": "l", "assignments": [{"ref": "s", "value": 1}]}]},
                        {"location": "l", "guard": {"op": "=", "left": "s", "right": 1},
                         "rate": 2.5, "destinations": [{"location": "l", "assignments": [{"ref": "s", "value": 0}]}]},
                    ],
                }
            ],
            "system": {"elements": [{"automaton": "m"}]},
        }
    )
    jm = bs.PrismModel.from_jani(jani)
    assert jm.summary()["states"] == 2
    assert jm.query("S=? [s=0]")["probability"] == pytest.approx(2.5 / 4.0)
