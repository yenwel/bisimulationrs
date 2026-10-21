"""Analyse and compare PEPA models from Python.

PEPA (Performance Evaluation Process Algebra) describes systems as components
that perform actions at exponential rates. This package checks a model, builds
its state space (a continuous-time Markov chain), merges equivalent states
(Markovian bisimulation), computes how the system evolves over time, and
explains how two models differ.

Rates are "per unit of time" and any unit works if you use it consistently
(seconds, days, ...). To compare a plan with dated counts, give the start
date (dd/mm/yyyy) and the unit, e.g. ``unit="day"`` for a garden.

>>> import bisimulationrs as bs
>>> herd = bs.Model('''
...     r_disease = 0.02;
...     Healthy = (disease, r_disease).Sick;
...     Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
...     Dead = 0;
...     Healthy[3]
... ''')
>>> herd.summary()["states"]
10
>>> table = herd.over_time([0, 30, 60])          # pandas DataFrame
>>> result = bs.compare(herd, herd)
>>> result.same_behaviour
True
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from os import PathLike
from datetime import date
from typing import Iterable, Optional, Union

import pandas as pd

from . import _native

__all__ = [
    "Model",
    "ModelError",
    "Diagnostic",
    "Check",
    "Comparison",
    "ObservationReport",
    "FitResult",
    "Distance",
    "EpsDelta",
    "StreamDistance",
    "Dsa",
    "Session",
    "check",
    "compare",
    "distance",
    "epsdelta",
    "stream_distance",
    "dsa",
    "read_observations",
]

__version__ = "0.2.0"


def _call(request: dict) -> dict:
    return json.loads(_native.handle_json(json.dumps(request)))


@dataclass(frozen=True)
class Diagnostic:
    """An error or warning, with a 1-based line and column (0 when not tied to one place)."""

    severity: str
    line: int
    column: int
    message: str

    def __str__(self) -> str:
        where = f"line {self.line}, column {self.column}: " if self.line else ""
        return f"{self.severity}: {where}{self.message}"


class ModelError(ValueError):
    """Raised when a model cannot be read or analysed.

    ``diagnostics`` lists every problem found, with positions where known.
    """

    def __init__(self, diagnostics: list[Diagnostic], label: str = "model"):
        self.diagnostics = diagnostics
        lines = "\n  ".join(str(d) for d in diagnostics) or "unknown error"
        super().__init__(f"The {label} has problems:\n  {lines}")


def _diagnostics(items: list[dict]) -> list[Diagnostic]:
    return [Diagnostic(d["severity"], d["line"], d["column"], d["message"]) for d in items]


def check(source: str) -> list[Diagnostic]:
    """Checks PEPA source text without analysing it. An empty list means it is valid."""
    return _diagnostics(_call({"op": "check", "source": source})["ok"])


def check_prism(source: str) -> list[Diagnostic]:
    """Checks PRISM source text without analysing it. An empty list means it is valid."""
    return _diagnostics(_call({"op": "check_prism", "source": source})["ok"])


def check_jani(source: str) -> list[Diagnostic]:
    """Checks JANI source text without analysing it. An empty list means it is valid."""
    return _diagnostics(_call({"op": "check_jani", "source": source})["ok"])


def _source(model: Union["Model", str]) -> str:
    return model.source if isinstance(model, Model) else model


def _occupancy_frame(occupancy: list[dict], times: list[float], wide: bool) -> pd.DataFrame:
    rows = [
        {"time": t, "component": s["component"], "state": s["derivative"], "expected": v}
        for s in occupancy
        for t, v in zip(times, s["values"])
    ]
    long = pd.DataFrame(rows, columns=["time", "component", "state", "expected"])
    if not wide:
        return long.sort_values(["time", "component", "state"], kind="stable").reset_index(drop=True)
    long["column"] = long["component"] + ": " + long["state"]
    return long.pivot(index="time", columns="column", values="expected").rename_axis(columns=None).reset_index()


class Model:
    """A PEPA model.

    Create it from source text, or with :meth:`Model.from_file`. The model is
    checked immediately; problems raise :class:`ModelError`.

    ``max_states`` limits the state space so a large model fails quickly
    instead of using all memory.
    """

    def __init__(self, source: str, *, max_states: int = 200_000):
        self.source = source
        self.max_states = max_states
        problems = [d for d in check(source) if d.severity == "error"]
        if problems:
            raise ModelError(problems)
        self._analysis: Optional[dict] = None

    @classmethod
    def from_file(cls, path: Union[str, PathLike], **kwargs) -> "Model":
        """Reads a ``.pepa`` file."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read(), **kwargs)

    def __repr__(self) -> str:
        s = self.summary()
        size = "state space too large to list" if s["states"] is None else f"{s['states']} states, {s['lumped_states']} after lumping"
        return f"<PEPA model: {size}; components {', '.join(s['components'])}>"

    # -- analysis -------------------------------------------------------------

    def _analyse(self) -> dict:
        if self._analysis is None:
            response = _call({"op": "analyse", "source": self.source, "max_states": self.max_states})
            if "error" in response:
                raise ModelError(_diagnostics(response["error"]))
            self._analysis = response["ok"]
        return self._analysis

    @property
    def warnings(self) -> list[Diagnostic]:
        """Non-fatal issues, e.g. cooperation on an action one side never performs."""
        return _diagnostics(self._analyse()["warnings"])

    def summary(self) -> dict:
        """Size of the state space before and after lumping, and the components.

        For very large models the state space is not listed (``states`` is
        None and ``note`` says why); behaviour over time and observation
        checks still work for components that evolve independently.
        """
        a = self._analyse()
        listed = a["note"] is None
        return {
            "states": len(a["full"]["states"]) if listed else None,
            "transitions": len(a["full"]["transitions"]) if listed else None,
            "lumped_states": len(a["lumped"]["states"]) if listed else None,
            "components": list(a["leaves"]),
            "note": a["note"],
        }

    def states(self, lumped: bool = False) -> pd.DataFrame:
        """One row per state, with its long-run (steady-state) probability.

        With ``lumped=True``, states that behave identically (Markovian
        bisimilar) are merged; ``merged_states`` says how many each row stands for.
        """
        a = self._analyse()
        full = pd.DataFrame(a["full"]["states"])
        full["steady_state"] = a["steady_state"]
        if not lumped:
            return full.rename(columns={"id": "state", "block": "lumped_state"}).drop(columns="size")[
                ["state", "name", "initial", "lumped_state", "steady_state"]
            ]
        blocks = pd.DataFrame(a["lumped"]["states"]).rename(
            columns={"id": "state", "size": "merged_states"}
        )
        blocks["steady_state"] = full.groupby("block")["steady_state"].sum().reindex(blocks["state"]).to_numpy()
        return blocks[["state", "name", "initial", "merged_states", "steady_state"]]

    def transitions(self, lumped: bool = False) -> pd.DataFrame:
        """One row per transition: source and target state, action, and total rate (per second)."""
        graph = self._analyse()["lumped" if lumped else "full"]
        return pd.DataFrame(graph["transitions"], columns=["source", "target", "action", "rate"])

    def components(self) -> dict[str, pd.DataFrame]:
        """For each sequential component: its local transitions (from, to, action, rate, passive)."""
        out = {}
        for c in self._analyse()["components"]:
            names = c["states"]
            rows = [
                {
                    "from": names[t["source"]],
                    "to": names[t["target"]],
                    "action": t["action"],
                    "rate": t["rate"],
                    "passive": t["passive"],
                }
                for t in c["transitions"]
            ]
            out[c["component"]] = pd.DataFrame(rows, columns=["from", "to", "action", "rate", "passive"])
        return out

    def over_time(self, times: Iterable[float], wide: bool = False) -> pd.DataFrame:
        """How the system evolves from its initial state.

        For each time (seconds) and each component, the expected number of
        copies in each local state (for a single component: the probability).
        Returns a long table (time, component, state, expected); with
        ``wide=True`` one column per component/state instead.
        """
        times = [float(t) for t in times]
        response = _call(
            {"op": "transient", "source": self.source, "times": times, "max_states": self.max_states}
        )
        if "error" in response:
            raise ModelError(_diagnostics(response["error"]))
        return _occupancy_frame(response["ok"]["occupancy"], times, wide)

    def fluid_over_time(
        self, times: Iterable[float], dt: float = 0.01, wide: bool = False
    ) -> pd.DataFrame:
        """Mean-field fluid approximation of :meth:`over_time` (same table shape).

        Integrates the SOS drift with RK4 instead of solving the Markov
        chain, so large array populations stay cheap. Only array
        populations (``P[n]``) are supported.
        """
        times = [float(t) for t in times]
        response = _call(
            {
                "op": "fluid",
                "source": self.source,
                "times": times,
                "dt": float(dt),
                "max_states": self.max_states,
            }
        )
        if "error" in response:
            raise ModelError(_diagnostics(response["error"]))
        return _occupancy_frame(response["ok"]["occupancy"], times, wide)

    def plot_over_time(self, horizon: float = 100.0, points: int = 101, component: Optional[str] = None, ax=None):
        """Line chart of :meth:`over_time` (needs matplotlib). Returns the axes."""
        import matplotlib.pyplot as plt  # optional dependency

        step = horizon / (points - 1)
        table = self.over_time([i * step for i in range(points)])
        component = component or table["component"].iloc[0]
        table = table[table["component"] == component]
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.5))
        for state, rows in table.groupby("state", sort=False):
            ax.plot(rows["time"], rows["expected"], linewidth=2, label=state)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("expected copies" if component.endswith("]") else "probability")
        ax.set_title(component)
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        return ax


    # -- observations ------------------------------------------------------------

    def check_observations(
        self,
        observations,
        start: Union[str, "date"],
        unit: str = "day",
        level: float = 0.95,
    ) -> "ObservationReport":
        """Compares dated counts with what this model (the plan) predicts.

        ``observations`` is a DataFrame (or list of dicts, or a CSV path) with
        columns ``date`` (dd/mm/yyyy or a date), ``state`` and ``count``, and
        optionally ``component`` when the model has several components. A
        stage split into ``Stage_1``, ``Stage_2`` is observed as ``Stage``.
        ``start`` is the date of the model's initial state; ``unit`` is the
        time unit of the model's rates.
        """
        response = _call(
            {
                "op": "observe",
                "source": self.source,
                "timeline": _timeline(start, unit, level),
                "observations": _observations(observations),
                "max_states": self.max_states,
            }
        )
        if "error" in response:
            raise ModelError(_diagnostics(response["error"]), label="observation check")
        return ObservationReport._from(response["ok"])

    def forecast(
        self,
        state: str,
        times: Iterable[float],
        start: Union[str, "date"],
        unit: str = "day",
        component: Optional[str] = None,
        level: float = 0.95,
    ) -> pd.DataFrame:
        """Expected count of ``state`` and its range at each time since ``start``.

        Returns columns ``elapsed`` (in ``unit``), ``date``, ``expected``,
        ``low`` and ``high`` (the range holding with probability ``level``).
        """
        response = _call(
            {
                "op": "forecast",
                "source": self.source,
                "timeline": _timeline(start, unit, level),
                "component": component,
                "state": state,
                "times": [float(t) for t in times],
                "max_states": self.max_states,
            }
        )
        if "error" in response:
            raise ModelError(_diagnostics(response["error"]), label="forecast")
        r = response["ok"]
        return pd.DataFrame(
            {"elapsed": r["elapsed"], "date": r["dates"], "expected": r["expected"], "low": r["low"], "high": r["high"]}
        )

    def fit(
        self,
        observations,
        params: Union[str, Iterable[str]],
        start: Union[str, "date"],
        unit: str = "day",
        level: float = 0.95,
    ) -> "FitResult":
        """Finds values of ``params`` (rate parameters) that best explain the counts.

        Fit one parameter at a time to see which stage differs from the plan;
        ``result.model`` is the fitted model, so ``bs.compare(plan, result.model)``
        shows the difference.
        """
        params = [params] if isinstance(params, str) else list(params)
        response = _call(
            {
                "op": "fit",
                "source": self.source,
                "timeline": _timeline(start, unit, level),
                "observations": _observations(observations),
                "params": params,
                "max_states": self.max_states,
            }
        )
        if "error" in response:
            raise ModelError(_diagnostics(response["error"]), label="fit")
        r = response["ok"]
        return FitResult(
            params=pd.DataFrame(r["params"], columns=["name", "planned", "fitted", "ratio"]),
            model=Model(r["fitted_source"], max_states=self.max_states),
            check=ObservationReport._from(r["fitted_check"]),
            explanation=r["explanation"],
        )

    def plot_observations(
        self,
        observations,
        start: Union[str, "date"],
        unit: str = "day",
        horizon: Optional[float] = None,
        level: float = 0.95,
    ):
        """Plan ranges with the observed counts on top, one panel per state (needs matplotlib)."""
        import matplotlib.pyplot as plt  # optional dependency

        report = self.check_observations(observations, start, unit, level)
        table = report.table
        horizon = horizon or max(1.0, table["elapsed"].max() * 1.25)
        states = list(dict.fromkeys(table["state"]))
        fig, axes = plt.subplots(len(states), 1, figsize=(7, 2.6 * len(states)), sharex=True, squeeze=False)
        times = [horizon * i / 100 for i in range(101)]
        for ax, state in zip(axes[:, 0], states):
            rows = table[table["state"] == state]
            band = self.forecast(state, times, start, unit, component=rows["component"].iloc[0], level=level)
            ax.fill_between(band["elapsed"], band["low"], band["high"], step="mid", alpha=0.18, color="#2a78d6", linewidth=0, label=f"{level:.0%} range (plan)")
            ax.plot(band["elapsed"], band["expected"], color="#2a78d6", linewidth=2, label="expected (plan)")
            colours = rows["status"].map({"on_track": "#0a7d32", "above_plan": "#c0352f", "below_plan": "#c0352f"})
            ax.scatter(rows["elapsed"], rows["observed"], color=colours, zorder=3, s=36, label="counted")
            ax.set_ylabel(state)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(frameon=False, loc="upper right")
        axes[0, 0].set_title(report.explanation if len(report.explanation) < 90 else ("On track" if report.on_track else "Off plan"))
        axes[-1, 0].set_xlabel(f"{unit}s since {report.start}")
        fig.tight_layout()
        return fig


class PrismModel:
    """A PRISM model (``dtmc``/``ctmc`` guarded commands, see ``examples/sir/sir.prism``).

    Create it from source text, or with :meth:`PrismModel.from_file`.
    Problems raise :class:`ModelError`.

    ``max_states`` limits the state space so a large model fails quickly
    instead of using all memory.
    """

    def __init__(self, source: str, *, max_states: int = 200_000):
        self.source = source
        self.max_states = max_states
        problems = [d for d in check_prism(source) if d.severity == "error"]
        if problems:
            raise ModelError(problems)
        self._analysis: Optional[dict] = None
        self._op = "analyse_prism"

    @classmethod
    def from_jani(cls, source: str, **kwargs) -> "PrismModel":
        """Creates a model from JANI text (the interchange format of Storm and Modest)."""
        model = cls.__new__(cls)
        model.source = source
        model.max_states = kwargs.get("max_states", 200_000)
        problems = [d for d in check_jani(source) if d.severity == "error"]
        if problems:
            raise ModelError(problems)
        model._analysis = None
        model._op = "analyse_jani"
        return model

    @classmethod
    def from_file(cls, path: Union[str, PathLike], **kwargs) -> "PrismModel":
        """Reads a ``.prism`` file."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read(), **kwargs)

    def __repr__(self) -> str:
        s = self.summary()
        size = "state space too large to list" if s["states"] is None else f"{s['states']} states, {s['lumped_states']} after lumping"
        return f"<PRISM model: {size}; variables {', '.join(s['variables'])}>"

    def _analyse(self) -> dict:
        if self._analysis is None:
            response = _call({"op": self._op, "source": self.source, "max_states": self.max_states})
            if "error" in response:
                raise ModelError(_diagnostics(response["error"]))
            self._analysis = response["ok"]
        return self._analysis

    def summary(self) -> dict:
        """Size of the state space before and after lumping, and the variables."""
        a = self._analyse()
        listed = a["note"] is None
        return {
            "states": len(a["full"]["states"]) if listed else None,
            "transitions": len(a["full"]["transitions"]) if listed else None,
            "lumped_states": len(a["lumped"]["states"]) if listed else None,
            "variables": list(a["vars"]),
            "note": a["note"],
        }

    def states(self, lumped: bool = False) -> pd.DataFrame:
        """One row per valuation, with its long-run (steady-state) probability."""
        a = self._analyse()
        full = pd.DataFrame(a["full"]["states"])
        full["steady_state"] = a["steady_state"]
        if not lumped:
            return full.rename(columns={"id": "state", "block": "lumped_state"}).drop(columns="size")[
                ["state", "name", "initial", "lumped_state", "steady_state"]
            ]
        blocks = pd.DataFrame(a["lumped"]["states"]).rename(
            columns={"id": "state", "size": "merged_states"}
        )
        blocks["steady_state"] = full.groupby("block")["steady_state"].sum().reindex(blocks["state"]).to_numpy()
        return blocks[["state", "name", "initial", "merged_states", "steady_state"]]

    def transitions(self, lumped: bool = False) -> pd.DataFrame:
        """One row per transition: source and target state, action, and rate/probability."""
        graph = self._analyse()["lumped" if lumped else "full"]
        return pd.DataFrame(graph["transitions"], columns=["source", "target", "action", "rate"])

    def query(self, query: str) -> dict:
        """Answers a CSL/PCTL-style query (``P=? [F s=1]``, ``S=? [s=0]``, …).

        Returns ``probability``, ``holds`` (for threshold queries) and an
        ``explanation``. Works for PRISM and JANI models alike.
        """
        op = "query_jani" if self._op == "analyse_jani" else "query"
        response = _call(
            {
                "op": op,
                "source": self.source,
                "query": query,
                "max_states": self.max_states,
            }
        )
        if "error" in response:
            raise ModelError(_diagnostics(response["error"]))
        return response["ok"]

    def rewards(self) -> pd.DataFrame:
        """One row per ``rewards`` block: expected state and transition rewards in steady state."""
        return pd.DataFrame(
            self._analyse()["rewards"], columns=["block", "state", "transition"]
        )


def _fmt_date(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%d/%m/%Y")
    return str(value)


def _timeline(start, unit: str, level: float) -> dict:
    units = {"s": "second", "sec": "second", "second": "second", "seconds": "second", "min": "minute",
             "minute": "minute", "minutes": "minute", "h": "hour", "hour": "hour", "hours": "hour",
             "d": "day", "day": "day", "days": "day", "w": "week", "week": "week", "weeks": "week"}
    if unit not in units:
        raise ValueError(f"unknown time unit {unit!r}; use second, minute, hour, day or week")
    return {"start": _fmt_date(start), "unit": units[unit], "level": float(level)}


def read_observations(path: Union[str, PathLike]) -> pd.DataFrame:
    """Reads observations from a CSV file with columns date (dd/mm/yyyy), state, count[, component]."""
    return pd.read_csv(path, dtype={"date": str, "state": str, "component": str})


def _observations(observations) -> list[dict]:
    if isinstance(observations, (str, PathLike)):
        observations = read_observations(observations)
    if isinstance(observations, pd.DataFrame):
        missing = {"date", "state", "count"} - set(observations.columns)
        if missing:
            raise ValueError(f"observations need columns date, state and count; missing {', '.join(sorted(missing))}")
        observations = observations.to_dict("records")
    out = []
    for o in observations:
        item = {"date": _fmt_date(o["date"]), "state": str(o["state"]), "count": int(o["count"])}
        component = o.get("component")
        if component is not None and not (isinstance(component, float) and math.isnan(component)):
            item["component"] = str(component)
        out.append(item)
    return out


@dataclass(frozen=True)
class ObservationReport:
    """Dated counts compared with the plan.

    ``table`` has one row per count: expected value, the range that holds
    with probability ``level`` under the plan, a p-value and a status
    (``on_track``, ``above_plan``, ``below_plan``). ``on_track`` is the
    overall verdict, adjusted for the number of counts checked.
    """

    on_track: bool
    explanation: str
    table: pd.DataFrame
    start: str
    unit: str
    level: float
    threshold: float
    methods: tuple[str, ...]

    @classmethod
    def _from(cls, r: dict) -> "ObservationReport":
        table = pd.DataFrame(
            r["rows"],
            columns=["date", "elapsed", "component", "state", "observed", "expected", "low", "high", "p_value", "status"],
        )
        return cls(
            on_track=r["on_track"],
            explanation=r["explanation"],
            table=table,
            start=r["start"],
            unit=r["unit"],
            level=r["level"],
            threshold=r["threshold"],
            methods=tuple(r["methods"]),
        )

    def __bool__(self) -> bool:
        return self.on_track

    def __repr__(self) -> str:
        return f"<ObservationReport: {self.explanation}>"


@dataclass(frozen=True)
class FitResult:
    """Parameter values fitted to observed counts (see :meth:`Model.fit`)."""

    params: pd.DataFrame
    model: "Model"
    check: ObservationReport
    explanation: str

    def __repr__(self) -> str:
        return f"<Fit: {self.explanation}>"


@dataclass(frozen=True)
class Check:
    """One comparison between models A and B."""

    name: str
    holds: bool
    formula: Optional[str]
    formula_holds_in: Optional[str]
    explanation: str
    states_a: list[int] = field(default_factory=list)
    states_b: list[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.holds


@dataclass(frozen=True)
class Comparison:
    """How two models relate, from strongest to weakest notion of 'the same'.

    - ``same_behaviour``: same actions *and* rates (Markovian bisimulation);
    - ``same_branching``: same actions, rates ignored (bisimulation);
    - ``mutually_similar``: each can mimic the other step by step;
    - ``b_mimics_a``: B can do everything A can (simulation).

    When a check fails, ``formula`` is a short logical description of
    something one model can do and the other cannot.
    """

    checks: tuple[Check, ...]
    #: Set when the models were too large and one copy of each component was compared.
    note: Optional[str] = None

    def _get(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)

    @property
    def same_behaviour(self) -> bool:
        return self._get("same_behaviour").holds

    @property
    def same_branching(self) -> bool:
        return self._get("same_branching").holds

    @property
    def mutually_similar(self) -> bool:
        return self._get("mutually_similar").holds

    @property
    def b_mimics_a(self) -> bool:
        return self._get("b_mimics_a").holds

    def __getitem__(self, name: str) -> Check:
        return self._get(name)

    def to_frame(self) -> pd.DataFrame:
        """One row per check: holds, formula, which model satisfies it, explanation."""
        return pd.DataFrame(
            [
                {
                    "check": c.name,
                    "holds": c.holds,
                    "formula": c.formula,
                    "formula_holds_in": c.formula_holds_in,
                    "explanation": c.explanation,
                }
                for c in self.checks
            ]
        )

    def __repr__(self) -> str:
        lines = [f"  {'✓' if c.holds else '✗'} {c.name}" + (f": {c.formula} holds in {c.formula_holds_in}" if c.formula else "") for c in self.checks]
        return "<Comparison\n" + "\n".join(lines) + "\n>"


_CHECKS = [
    ("same_behaviour", "markovian_bisimilar"),
    ("same_branching", "bisimilar"),
    ("mutually_similar", "mutually_similar"),
    ("b_mimics_a", "b_simulates_a"),
]


def compare(a: Union[Model, str], b: Union[Model, str], *, max_states: int = 200_000) -> Comparison:
    """Compares the initial states of models ``a`` and ``b`` (Model objects or source text)."""
    response = _call({"op": "compare", "a": _source(a), "b": _source(b), "max_states": max_states})
    if "error" in response:
        err = response["error"]
        raise ModelError(_diagnostics(err["a"]) + _diagnostics(err["b"]), label="comparison")
    r = response["ok"]
    return Comparison(
        tuple(
            Check(
                name=name,
                holds=r[key]["holds"],
                formula=r[key]["formula"],
                formula_holds_in=r[key]["formula_holds_in"],
                explanation=r[key]["explanation"],
                states_a=r[key]["satisfied_in_a"],
                states_b=r[key]["satisfied_in_b"],
            )
            for name, key in _CHECKS
        ),
        note=r.get("note"),
    )


# -- how far apart -------------------------------------------------------------


def _atom(state: Union[str, tuple, dict], component: Optional[str] = None) -> dict:
    if isinstance(state, dict):
        return {k: str(v) for k, v in state.items() if k in ("component", "state") and v is not None}
    if isinstance(state, tuple):
        component, state = state
    atom = {"state": str(state)}
    if component is not None:
        atom["component"] = str(component)
    return atom


def _ok(response: dict, label: str):
    if "error" in response:
        raise ModelError(_diagnostics(response["error"]), label=label)
    return response["ok"]


@dataclass(frozen=True)
class Distance:
    """How far apart two models are, seen through one count (see :func:`distance`).

    ``distance`` is in [0, 1]: 0 when the count cannot tell the models apart,
    1 when it always differs by its whole range (``scale``). The true value is
    within ``error_bound`` of it. ``pairs`` lists the state pairs that add
    most to it: the gap in the count (as a share of ``scale``), the
    discounted share of time spent in the pair, and the contribution.
    """

    distance: float
    error_bound: float
    half_life: float
    scale: float
    pairs: pd.DataFrame
    explanation: str

    def __float__(self) -> float:
        return self.distance

    def __repr__(self) -> str:
        return f"<Distance {self.distance:.4f}: {self.explanation}>"


def distance(
    a: Union[Model, str],
    b: Union[Model, str],
    state: Union[str, tuple, dict],
    half_life: float,
    *,
    component: Optional[str] = None,
    max_states: int = 200_000,
) -> Distance:
    """How differently models ``a`` and ``b`` behave, seen through the count of ``state``.

    Both models run side by side in the way that makes the count look most
    alike; differences ``half_life`` time units ahead count half (a
    bisimulation distance, Calo et al., NeurIPS 2024). Give ``component``
    (or ``state=(component, state)``) when the model has several components.
    """
    r = _ok(
        _call(
            {
                "op": "distance",
                "a": _source(a),
                "b": _source(b),
                "observe": _atom(state, component),
                "half_life": float(half_life),
                "max_states": max_states,
            }
        ),
        "distance",
    )
    return Distance(
        distance=r["distance"],
        error_bound=r["error_bound"],
        half_life=r["half_life"],
        scale=r["scale"],
        pairs=pd.DataFrame(r["pairs"], columns=["a", "b", "gap", "occupancy", "contribution"]),
        explanation=r["explanation"],
    )


@dataclass(frozen=True)
class EpsDelta:
    """Whether two models are within a tolerance (see :func:`epsdelta`).

    ``min_epsilon`` is the smallest tolerance on jump probabilities at which
    the models are related (None: none is enough); ``within`` answers for the
    requested ``epsilon``. With a horizon, ``reach_bound`` bounds how much the
    probability of reaching any labelled situation by then can differ.
    """

    delta: float
    min_epsilon: Optional[float]
    within: Optional[bool]
    rate: float
    reach_bound: Optional[float]
    rate_only_bound: Optional[float]
    explanation: str

    def __repr__(self) -> str:
        return f"<EpsDelta: {self.explanation}>"


def epsdelta(
    a: Union[Model, str],
    b: Union[Model, str],
    labels: Union[str, tuple, dict, Iterable],
    delta: float,
    *,
    epsilon: Optional[float] = None,
    horizon: Optional[float] = None,
    max_states: int = 200_000,
) -> EpsDelta:
    """Are ``a`` and ``b`` within ``epsilon`` on jump probabilities and a factor ``e**delta`` on rates?

    States are labelled by the counts of ``labels`` (a state name, a
    ``(component, state)`` tuple, or a list of them); related states must
    have equal labels ((ε, δ)-bisimulation, Spork et al., CAV 2025). Without
    ``epsilon``, only the smallest one is reported.
    """
    if isinstance(labels, (str, tuple, dict)):
        labels = [labels]
    r = _ok(
        _call(
            {
                "op": "epsdelta",
                "a": _source(a),
                "b": _source(b),
                "labels": [_atom(l) for l in labels],
                "epsilon": None if epsilon is None else float(epsilon),
                "delta": float(delta),
                "horizon": None if horizon is None else float(horizon),
                "max_states": max_states,
            }
        ),
        "tolerance check",
    )
    return EpsDelta(**{k: r[k] for k in EpsDelta.__dataclass_fields__})


@dataclass(frozen=True)
class StreamDistance:
    """A distance estimated from two logs of states, without a model (see :func:`stream_distance`).

    ``matching`` maps each state of log A to the state of log B it is paired
    with. The estimate settles slowly; compare logs of equal length.
    """

    estimate: float
    steps: int
    matching: dict
    explanation: str

    def __float__(self) -> float:
        return self.estimate

    def __repr__(self) -> str:
        return f"<StreamDistance {self.estimate:.4f} after {self.steps} steps>"


def _stream(r: dict, explanation: str = "") -> StreamDistance:
    return StreamDistance(
        estimate=r["estimate"],
        steps=r["steps"],
        matching={x: y for x, y in r["matching"]},
        explanation=r.get("explanation", explanation),
    )


def stream_distance(a: Iterable[str], b: Iterable[str], values: dict, gamma: float) -> StreamDistance:
    """Estimates the distance between two logged state sequences (Calo et al., NeurIPS 2025).

    Consecutive entries of each log are transitions; ``values`` gives each
    state's observation (e.g. ``{"ok": 0, "sick": 1}``) and ``gamma`` in
    (0, 1) discounts later steps.
    """
    r = _ok(
        _call(
            {
                "op": "stream_distance",
                "a": [str(s) for s in a],
                "b": [str(s) for s in b],
                "values": {str(k): float(v) for k, v in values.items()},
                "gamma": float(gamma),
            }
        ),
        "stream distance",
    )
    return _stream(r)


@dataclass(frozen=True)
class Dsa:
    """Dynamical similarity of two series (see :func:`dsa`); ``angle`` is in radians."""

    distance: float
    angle: float
    explanation: str

    def __float__(self) -> float:
        return self.distance

    def __repr__(self) -> str:
        return f"<Dsa: {self.explanation}>"


def _series(x) -> list[list[float]]:
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [[float(v) for v in row] if isinstance(row, (list, tuple)) else [float(row)] for row in x]


def dsa(x, y, *, delays: Optional[int] = None, rank: Optional[int] = None) -> Dsa:
    """Compares how two sensor series evolve (Ostrow et al., NeurIPS 2023).

    ``x`` and ``y`` are tables with one row per time step and one column per
    channel (a DataFrame, an array, or a list of rows; a flat list is one
    channel). Levels do not matter, only the fitted dynamics. Not a bisimulation.
    """
    r = _ok(
        _call({"op": "dsa", "x": _series(x), "y": _series(y), "delays": delays, "rank": rank}),
        "series comparison",
    )
    return Dsa(r["distance"], r["angle"], r["explanation"])


class Session:
    """A comparison that is updated as observations arrive.

    Open one with :meth:`prefix`, :meth:`observe` or :meth:`stream_distance`,
    then :meth:`push` events; each push returns the updated answer without
    recomputing what came before. Close it when done (or use ``with``; it is
    also closed when garbage collected)::

        with bs.Session.observe(plan, start="15/05/2026") as s:
            s.push([{"date": "25/05/2026", "state": "Big", "count": 2}])
    """

    def __init__(self, spec: dict):
        self.kind = spec["kind"]
        self.id = _ok(_call({"op": "session_open", "session": spec}), "session")["id"]
        self.closed = False

    @classmethod
    def prefix(cls, left: str, right: str, *, max_pairs: Optional[int] = None) -> "Session":
        """Two systems observed transition by transition, from initial states ``left`` and ``right``.

        Push events ``{"left": [s, a, t]}``, ``{"right": [s, a, t]}``,
        ``{"complete_left": s}`` and ``{"complete_right": s}`` (no more
        transitions from ``s``). Each push returns ``{"changed", "verdict"}``.
        """
        spec = {"kind": "prefix", "left": left, "right": right}
        if max_pairs is not None:
            spec["max_pairs"] = int(max_pairs)
        return cls(spec)

    @classmethod
    def observe(
        cls,
        model: Union[Model, str],
        start: Union[str, "date"],
        unit: str = "day",
        level: float = 0.95,
        *,
        max_states: int = 200_000,
    ) -> "Session":
        """Dated counts against a plan, as in :meth:`Model.check_observations`; push counts as they come."""
        return cls(
            {
                "kind": "observe",
                "source": _source(model),
                "timeline": _timeline(start, unit, level),
                "max_states": max_states,
            }
        )

    @classmethod
    def stream_distance(
        cls, states_a: Iterable[str], states_b: Iterable[str], values: dict, gamma: float
    ) -> "Session":
        """The sample distance of :func:`stream_distance`, one step at a time.

        Push steps ``{"a": [from, to], "b": [from, to]}``; ``states_a`` and
        ``states_b`` list the states each log can be in.
        """
        return cls(
            {
                "kind": "stream_distance",
                "states_a": [str(s) for s in states_a],
                "states_b": [str(s) for s in states_b],
                "values": {str(k): float(v) for k, v in values.items()},
                "gamma": float(gamma),
            }
        )

    def push(self, events):
        """Adds events; returns the updated answer.

        An :class:`ObservationReport` for ``observe`` (events may be a
        DataFrame or list of dicts), a :class:`StreamDistance` for
        ``stream_distance``, and a dict for ``prefix``.
        """
        if self.kind == "observe":
            events = _observations(events)
        else:
            events = list(events)
        r = _ok(_call({"op": "session_push", "id": self.id, "events": events}), "session")
        if self.kind == "observe":
            return ObservationReport._from(r)
        if self.kind == "stream_distance":
            return _stream(r, f"Estimated from {r['steps']} observed transition pairs.")
        return r

    def close(self) -> None:
        """Frees the session; closing twice is harmless."""
        if not self.closed:
            self.closed = True
            _call({"op": "session_close", "id": self.id})

    def __del__(self) -> None:
        # Sessions are capped per process; free one that was never closed.
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Session {self.id} ({self.kind}{', closed' if self.closed else ''})>"
