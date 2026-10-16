//! JANI import: a documented subset of the JANI model interchange format
//! (Budde et al., TACAS 2017 — the exchange language of Storm and Modest)
//! lowered to [`PrismModel`](super::prism::PrismModel), hence to the same
//! derivation, bisimulation and CTMC analysis as all other front ends.
//!
//! Supported: `dtmc`/`ctmc` automata with bounded integer/boolean
//! variables, constants, parameter-free functions, locations (lowered to
//! hidden `__loc_<automaton>` variables), edges with guards, rates and
//! probabilistic destinations, and synchronisation vectors without
//! input/output enablers. A vector whose entries name different actions
//! is rejected (renaming synchronisation); object-form vectors that list
//! only some automata offering the action rename the others' copies to
//! fresh interleaving labels (behaviour-preserving).
//!
//! Not supported, with explicit errors: `mdp`/timed/probabilistic-timed
//! automata, unbounded integers, reals, clocks, arrays, transient
//! values, parameterised functions, transcendental operators,properties
//! (ignored: they are not the model).

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

use super::prism::{
    Assign, Command, Expr, ModelKind, Module, PrismError, PrismModel, Update, Var, VarType,
};
use super::prism::{Value as PValue, eval};

/// Parses a JANI model and lowers it to a [`PrismModel`].
pub fn parse_jani(src: &str) -> Result<PrismModel, PrismError> {
    let root: Value = serde_json::from_str(src).map_err(|e| {
        PrismError::Parse(crate::pepa::ParseError {
            line: 0,
            column: 0,
            message: format!("invalid JSON: {e}"),
        })
    })?;
    Lower::new(&root)?.lower()
}

fn bad(msg: impl Into<String>) -> PrismError {
    PrismError::Unsupported(msg.into())
}

fn get<'a>(v: &'a Value, obj: &str, key: &str) -> Result<&'a Value, PrismError> {
    v.get(key)
        .ok_or_else(|| PrismError::Type(format!("{obj} misses required field `{key}`")))
}

fn as_str(v: &Value, what: &str) -> Result<String, PrismError> {
    v.as_str()
        .map(str::to_owned)
        .ok_or_else(|| PrismError::Type(format!("{what} must be a string")))
}

fn as_num_lit(v: &Value, what: &str) -> Result<Expr, PrismError> {
    if let Some(i) = v.as_i64() {
        return Ok(Expr::Int(i));
    }
    if let Some(x) = v.as_f64() {
        return Ok(Expr::Float(x));
    }
    Err(PrismError::Type(format!("{what} must be a number")))
}

struct Lower<'a> {
    root: &'a Value,
    kind: ModelKind,
    /// Global variables in order.
    globals: Vec<(String, Var)>,
    global_idx: HashMap<String, usize>,
    /// Constants and parameter-free functions, evaluated eagerly.
    consts: HashMap<String, PValue>,
}

impl<'a> Lower<'a> {
    fn new(root: &'a Value) -> Result<Self, PrismError> {
        let version = get(root, "model", "jani-version")?;
        if version.as_u64() != Some(1) {
            return Err(bad("only jani-version 1 is supported"));
        }
        if root.get("features").is_some() {
            // Features are hints; the subset checks below decide.
        }
        let kind = match get(root, "model", "type")?.as_str() {
            Some("dtmc") => ModelKind::Dtmc,
            Some("ctmc") => ModelKind::Ctmc,
            Some(other) => {
                return Err(bad(format!(
                    "model type `{other}` is not supported; use dtmc or ctmc"
                )));
            }
            None => return Err(PrismError::Type("model `type` must be a string".into())),
        };
        Ok(Lower {
            root,
            kind,
            globals: Vec::new(),
            global_idx: HashMap::new(),
            consts: HashMap::new(),
        })
    }

    fn lower(mut self) -> Result<PrismModel, PrismError> {
        self.lower_consts("constants")?;
        self.lower_consts("functions")?;
        self.lower_variables()?;
        let automata = get(self.root, "model", "automata")?;
        let automata = automata
            .as_array()
            .ok_or_else(|| PrismError::Type("model `automata` must be an array".into()))?;
        if automata.is_empty() {
            return Err(PrismError::Type("the model has no automata".into()));
        }
        // First pass: register locations counts and local variables so
        // global indices exist before edges are lowered.
        let mut per_auto: Vec<AutoInfo> = Vec::new();
        for a in automata {
            per_auto.push(self.register_automaton(a)?);
        }
        let mut modules = Vec::new();
        for (a, info) in automata.iter().zip(&per_auto) {
            modules.push(self.lower_automaton(a, info)?);
        }
        let sync = self.lower_system()?;
        let vars = self.globals.into_iter().map(|(_, v)| v).collect();
        Ok(PrismModel {
            kind: self.kind,
            modules,
            vars,
            rewards: Vec::new(),
            sync,
        })
    }

    /// Constants and parameter-free functions (treated alike).
    fn lower_consts(&mut self, key: &str) -> Result<(), PrismError> {
        let Some(list) = self.root.get(key) else {
            return Ok(());
        };
        let list = list
            .as_array()
            .ok_or_else(|| PrismError::Type(format!("model `{key}` must be an array")))?;
        for c in list {
            let name = as_str(get(c, "constant", "name")?, "constant name")?;
            if self.consts.contains_key(&name) || self.global_idx.contains_key(&name) {
                return Err(PrismError::Duplicate(name));
            }
            if c.get("parameters").is_some() {
                return Err(bad(format!(
                    "parameterised function `{name}` is not supported"
                )));
            }
            let typ = get(c, "constant", "type")?;
            let expr = self.lower_expr(get(c, "constant", "value")?, &HashMap::new())?;
            let v = eval(&expr, &[], &[]).ok_or_else(|| {
                PrismError::Arithmetic(format!("constant `{name}` is not finite"))
            })?;
            match (typ, v) {
                (Value::String(s), _) if s == "int" => {
                    if !matches!(v, PValue::Int(_)) {
                        return Err(PrismError::Type(format!(
                            "constant `{name}` is not an integer"
                        )));
                    }
                }
                (Value::String(s), _) if s == "double" || s == "real" || s == "number" => {
                    if v.as_num().is_none() {
                        return Err(PrismError::Type(format!(
                            "constant `{name}` is not a number"
                        )));
                    }
                }
                (Value::String(s), _) if s == "bool" => {
                    if !matches!(v, PValue::Bool(_)) {
                        return Err(PrismError::Type(format!(
                            "constant `{name}` is not boolean"
                        )));
                    }
                }
                _ => {
                    return Err(bad(format!("type of constant `{name}` is not supported")));
                }
            }
            self.consts.insert(name, v);
        }
        Ok(())
    }

    /// Top-level variables; automata locals are handled per automaton.
    fn lower_variables(&mut self) -> Result<(), PrismError> {
        let vars = get(self.root, "model", "variables")?;
        match vars {
            Value::Object(map) => {
                for (name, def) in map {
                    self.check_name(name)?;
                    let (typ, init) = self.var_type_init(name, def)?;
                    self.add_global(name.clone(), typ, init)?;
                }
                Ok(())
            }
            Value::Array(list) => {
                // Some producers emit variables as [{name, ...}] instead.
                for def in list {
                    let name = as_str(get(def, "variable", "name")?, "variable name")?;
                    self.check_name(&name)?;
                    let (typ, init) = self.var_type_init(&name, def)?;
                    self.add_global(name, typ, init)?;
                }
                Ok(())
            }
            _ => Err(PrismError::Type(
                "model `variables` must be an object".into(),
            )),
        }
    }

    fn check_name(&self, name: &str) -> Result<(), PrismError> {
        if name.starts_with("__loc_") {
            return Err(bad(format!(
                "variable `{name}` uses the reserved `__loc_` prefix"
            )));
        }
        if self.global_idx.contains_key(name) || self.consts.contains_key(name) {
            return Err(PrismError::Duplicate(name.to_owned()));
        }
        Ok(())
    }

    fn add_global(&mut self, name: String, typ: VarType, init: i64) -> Result<usize, PrismError> {
        match &typ {
            VarType::Bool if init != 0 && init != 1 => {
                return Err(PrismError::Type(format!(
                    "initial value of `{name}` is not boolean"
                )));
            }
            VarType::Int { lo, hi } if init < *lo || init > *hi => {
                return Err(PrismError::OutOfBounds {
                    var: name,
                    value: init,
                });
            }
            _ => {}
        }
        let idx = self.globals.len();
        self.global_idx.insert(name.clone(), idx);
        self.globals.push((name.clone(), Var { name, typ, init }));
        Ok(idx)
    }

    fn var_type_init(&self, name: &str, def: &Value) -> Result<(VarType, i64), PrismError> {
        if def.get("transient-values").is_some() {
            return Err(bad(format!(
                "transient values of `{name}` are not supported"
            )));
        }
        let typ = get(def, &format!("variable `{name}`"), "type")?;
        let init_src = get(def, &format!("variable `{name}`"), "initial-value")?;
        match typ {
            Value::String(s) if s == "bool" => {
                let init = self.closed_bool(init_src)?;
                Ok((VarType::Bool, i64::from(init)))
            }
            Value::Object(_) => {
                if typ.get("kind") != Some(&Value::String("bounded".into())) {
                    return Err(bad(format!(
                        "type of `{name}` is not supported (bounded integers and booleans only)"
                    )));
                }
                let lo = get(typ, "type", "lower-bound").and_then(|v| {
                    v.as_i64().ok_or_else(|| {
                        PrismError::Type(format!("lower bound of `{name}` is not an integer"))
                    })
                })?;
                let hi = get(typ, "type", "upper-bound").and_then(|v| {
                    v.as_i64().ok_or_else(|| {
                        PrismError::Type(format!("upper bound of `{name}` is not an integer"))
                    })
                })?;
                let init = self.closed_int(init_src)?;
                Ok((VarType::Int { lo, hi }, init))
            }
            _ => Err(bad(format!(
                "type of `{name}` is not supported (bounded integers and booleans only)"
            ))),
        }
    }

    fn closed_expr(&self, v: &Value) -> Result<Expr, PrismError> {
        self.lower_expr(v, &HashMap::new())
    }

    fn closed_int(&self, v: &Value) -> Result<i64, PrismError> {
        match self.closed_value(v)? {
            PValue::Int(i) => Ok(i),
            _ => Err(PrismError::Type("expected an integer".into())),
        }
    }

    fn closed_bool(&self, v: &Value) -> Result<bool, PrismError> {
        match self.closed_value(v)? {
            PValue::Bool(b) => Ok(b),
            _ => Err(PrismError::Type("expected a boolean".into())),
        }
    }

    fn closed_value(&self, v: &Value) -> Result<PValue, PrismError> {
        let e = self.closed_expr(v)?;
        eval(&e, &[], &[]).ok_or_else(|| PrismError::Arithmetic("not finite".into()))
    }

    /// Registers an automaton's locations and local variables.
    fn register_automaton(&mut self, a: &Value) -> Result<AutoInfo, PrismError> {
        let name = as_str(get(a, "automaton", "name")?, "automaton name")?;
        if name.starts_with("__loc_") {
            return Err(bad(format!(
                "automaton `{name}` uses the reserved `__loc_` prefix"
            )));
        }
        let locs = get(a, "automaton", "locations")?
            .as_array()
            .ok_or_else(|| PrismError::Type(format!("locations of `{name}` must be an array")))?;
        if locs.is_empty() {
            return Err(PrismError::Type(format!(
                "automaton `{name}` has no locations"
            )));
        }
        let mut loc_idx = HashMap::new();
        for (i, l) in locs.iter().enumerate() {
            let lname = as_str(get(l, "location", "name")?, "location name")?;
            if loc_idx.insert(lname.clone(), i).is_some() {
                return Err(PrismError::Duplicate(format!("{name}.{lname}")));
            }
        }
        let init_loc = get(a, "automaton", "initial-location")?;
        let init_loc = if let Some(i) = init_loc.as_u64() {
            i as usize
        } else {
            let lname = as_str(init_loc, "initial location")?;
            *loc_idx
                .get(&lname)
                .ok_or_else(|| PrismError::Unknown(format!("location `{name}.{lname}`")))?
        };
        if init_loc >= locs.len() {
            return Err(PrismError::OutOfBounds {
                var: format!("{name}.location"),
                value: init_loc as i64,
            });
        }
        // Local variables, qualified as `<automaton>_<var>`.
        let mut locals = HashMap::new();
        if let Some(vars) = a.get("variables") {
            let vars = vars.as_object().ok_or_else(|| {
                PrismError::Type(format!("variables of `{name}` must be an object"))
            })?;
            for (vname, def) in vars {
                let qname = format!("{name}_{vname}");
                self.check_name(&qname)?;
                let (typ, init) = self.var_type_init(&qname, def)?;
                let idx = self.add_global(qname.clone(), typ, init)?;
                locals.insert(vname.clone(), (qname, idx));
            }
        }
        // Hidden location variable.
        let loc_var = format!("__loc_{name}");
        let loc_idx_global = self.add_global(
            loc_var,
            VarType::Int {
                lo: 0,
                hi: locs.len() as i64 - 1,
            },
            init_loc as i64,
        )?;
        Ok(AutoInfo {
            name,
            loc_idx,
            loc_var: loc_idx_global,
            locals,
        })
    }

    fn lower_automaton(&self, a: &Value, info: &AutoInfo) -> Result<Module, PrismError> {
        let edges = get(a, "automaton", "edges")?.as_array().ok_or_else(|| {
            PrismError::Type(format!("edges of `{}` must be an array", info.name))
        })?;
        let mut var_ids: Vec<usize> = info.locals.values().map(|(_, i)| *i).collect();
        var_ids.push(info.loc_var);
        let mut commands = Vec::new();
        for e in edges {
            commands.push(self.lower_edge(e, info)?);
        }
        Ok(Module {
            name: info.name.clone(),
            vars: var_ids,
            commands,
        })
    }

    fn lower_edge(&self, e: &Value, info: &AutoInfo) -> Result<Command, PrismError> {
        let loc = match e.get("location") {
            None => None,
            Some(Value::Number(n)) => n.as_u64().map(|i| i as usize),
            Some(Value::String(s)) => info.loc_idx.get(s).copied(),
            _ => return Err(PrismError::Type("edge `location` must be a name".into())),
        };
        let action = match e.get("action") {
            None => None,
            Some(v) => Some(as_str(v, "edge action")?),
        };
        let mut guard = match e.get("guard") {
            None => Expr::Bool(true),
            Some(g) => self.lower_expr(g, &info.locals)?,
        };
        if let Some(l) = loc {
            guard = Expr::And(
                Box::new(Expr::Eq(
                    Box::new(Expr::Var(info.loc_var)),
                    Box::new(Expr::Int(l as i64)),
                )),
                Box::new(guard),
            );
        }
        // Rate (CTMC) or nothing (DTMC).
        let rate = match e.get("rate") {
            None => None,
            Some(r) => Some(self.lower_expr(r, &info.locals)?),
        };
        match (self.kind, &rate) {
            (ModelKind::Ctmc, None) => {
                return Err(PrismError::Type(format!(
                    "CTMC edge of `{}` has no rate",
                    info.name
                )));
            }
            (ModelKind::Dtmc, Some(_)) => {
                return Err(PrismError::Type(format!(
                    "DTMC edge of `{}` has a rate",
                    info.name
                )));
            }
            _ => {}
        }
        let dests = get(e, "edge", "destinations")?
            .as_array()
            .ok_or_else(|| PrismError::Type("edge `destinations` must be an array".into()))?;
        if dests.is_empty() {
            return Err(PrismError::Type("edge has no destinations".into()));
        }
        let mut updates = Vec::new();
        for d in dests {
            let prob = match d.get("probability") {
                None if dests.len() == 1 => Expr::Float(1.0),
                None => {
                    return Err(PrismError::Type(
                        "destination without probability (only allowed with a single destination)"
                            .into(),
                    ));
                }
                Some(p) => self.lower_expr(p, &info.locals)?,
            };
            let weight = match (&rate, self.kind) {
                (Some(r), _) => Expr::Mul(Box::new(r.clone()), Box::new(prob)),
                (None, _) => prob,
            };
            let mut assigns = Vec::new();
            if let Some(list) = d.get("assignments") {
                let list = list
                    .as_array()
                    .ok_or_else(|| PrismError::Type("assignments must be an array".into()))?;
                for asn in list {
                    if asn.get("index").is_some() {
                        return Err(bad("indexed (array) assignments are not supported"));
                    }
                    let r = as_str(get(asn, "assignment", "ref")?, "assignment ref")?;
                    let idx = self.assign_target(&r, info)?;
                    let value = self.lower_expr(get(asn, "assignment", "value")?, &info.locals)?;
                    assigns.push(Assign {
                        var: idx,
                        expr: value,
                    });
                }
            }
            // Locations advance with the edge.
            if let Some(dloc) = d.get("location") {
                let t = match dloc {
                    Value::Number(n) => n.as_u64().map(|i| i as usize).ok_or_else(|| {
                        PrismError::Type("destination location must be a name".into())
                    })?,
                    Value::String(s) => info.loc_idx.get(s).copied().ok_or_else(|| {
                        PrismError::Unknown(format!("location `{}.{s}`", info.name))
                    })?,
                    _ => {
                        return Err(PrismError::Type(
                            "destination location must be a name".into(),
                        ));
                    }
                };
                assigns.push(Assign {
                    var: info.loc_var,
                    expr: Expr::Int(t as i64),
                });
            }
            updates.push(Update { weight, assigns });
        }
        Ok(Command {
            label: action,
            guard,
            updates,
        })
    }

    fn assign_target(&self, r: &str, info: &AutoInfo) -> Result<usize, PrismError> {
        if let Some((_, idx)) = info.locals.get(r) {
            return Ok(*idx);
        }
        self.global_idx
            .get(r)
            .copied()
            .ok_or_else(|| PrismError::Unknown(format!("assigned variable `{r}`")))
    }

    /// Expression lowering with automaton locals in scope (`(qualified, idx)`).
    fn lower_expr(
        &self,
        v: &Value,
        locals: &HashMap<String, (String, usize)>,
    ) -> Result<Expr, PrismError> {
        match v {
            Value::Bool(b) => Ok(Expr::Bool(*b)),
            Value::Number(_) => as_num_lit(v, "literal")
                .map_err(|_| PrismError::Type("literal must be a number".into())),
            Value::String(name) => {
                if let Some((_, idx)) = locals.get(name) {
                    return Ok(Expr::Var(*idx));
                }
                if let Some(idx) = self.global_idx.get(name) {
                    return Ok(Expr::Var(*idx));
                }
                if let Some(c) = self.consts.get(name) {
                    return Ok(match *c {
                        PValue::Int(i) => Expr::Int(i),
                        PValue::Float(x) => Expr::Float(x),
                        PValue::Bool(b) => Expr::Bool(b),
                    });
                }
                Err(PrismError::Unknown(name.clone()))
            }
            Value::Array(_) => Err(PrismError::Type("unexpected array in expression".into())),
            Value::Object(_) => Err(PrismError::Type("unexpected object in expression".into())),
            Value::Null => Err(PrismError::Type("unexpected null in expression".into())),
        }
        .or_else(|first| self.lower_op(v, locals).map_err(|_| first))
    }

    /// Operator objects (`{"op": ..., ...}`); tried when `lower_expr`'s
    /// direct shapes fail.
    fn lower_op(
        &self,
        v: &Value,
        locals: &HashMap<String, (String, usize)>,
    ) -> Result<Expr, PrismError> {
        let op = as_str(get(v, "expression", "op")?, "operator")?;
        let child = |key: &str| self.lower_expr(get(v, "expression", key)?, locals);
        let binary = |mk: fn(Box<Expr>, Box<Expr>) -> Expr| {
            Ok(mk(Box::new(child("left")?), Box::new(child("right")?))) as Result<Expr, PrismError>
        };
        match op.as_str() {
            "+" => binary(Expr::Add),
            "-" if v.get("left").is_some() => binary(Expr::Sub),
            "-" => Ok(Expr::Neg(Box::new(child("exp")?))),
            "*" => binary(Expr::Mul),
            "/" => binary(Expr::Div),
            "%" | "mod" => binary(Expr::Mod),
            "pow" => binary(Expr::Pow),
            "=" | "==" => binary(Expr::Eq),
            "!=" | "≠" | "/=" => binary(Expr::Ne),
            "<" => binary(Expr::Lt),
            "<=" | "≤" => binary(Expr::Le),
            ">" => binary(Expr::Gt),
            ">=" | "≥" => binary(Expr::Ge),
            "&" | "&&" | "∧" | "and" => binary(Expr::And),
            "|" | "||" | "∨" | "or" => binary(Expr::Or),
            "=>" | "⇒" | "implies" => {
                let (a, b) = (child("left")?, child("right")?);
                Ok(Expr::Or(Box::new(Expr::Not(Box::new(a))), Box::new(b)))
            }
            "<=>" | "⇔" | "iff" => binary(Expr::Eq),
            "!" | "¬" | "not" => Ok(Expr::Not(Box::new(child("exp")?))),
            "ite" => Ok(Expr::Cond(
                Box::new(child("if")?),
                Box::new(child("then")?),
                Box::new(child("else")?),
            )),
            "min" | "max" => {
                let args = if let Some(list) = v.get("exp").and_then(Value::as_array) {
                    list.iter()
                        .map(|e| self.lower_expr(e, locals))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    vec![child("left")?, child("right")?]
                };
                if args.is_empty() {
                    return Err(PrismError::Type(format!("`{op}` needs arguments")));
                }
                Ok(if op == "min" {
                    Expr::Min(args)
                } else {
                    Expr::Max(args)
                })
            }
            "floor" => Ok(Expr::Floor(Box::new(child("exp")?))),
            "ceil" => Ok(Expr::Ceil(Box::new(child("exp")?))),
            "abs" => {
                let e = child("exp")?;
                Ok(Expr::Cond(
                    Box::new(Expr::Ge(Box::new(e.clone()), Box::new(Expr::Int(0)))),
                    Box::new(e),
                    Box::new(Expr::Neg(Box::new(
                        self.lower_expr(get(v, "expression", "exp")?, locals)?,
                    ))),
                ))
            }
            _ => Err(bad(format!("expression operator `{op}` is not supported"))),
        }
    }

    /// System synchronisation vectors to a global restriction set. JANI
    /// actions without a vector interleave, so the restriction is always
    /// explicit (unlike PRISM's default full synchronisation).
    fn lower_system(&self) -> Result<super::prism::SyncConfig, PrismError> {
        use super::prism::SyncConfig;

        let interleave_all = SyncConfig {
            restricted: Some(BTreeSet::new()),
        };
        let Some(sys) = self.root.get("system") else {
            return Ok(interleave_all);
        };
        if let Some(elements) = sys.get("elements") {
            let elements = elements
                .as_array()
                .ok_or_else(|| PrismError::Type("system `elements` must be an array".into()))?;
            for e in elements {
                let aname = match e {
                    Value::String(s) => s.clone(),
                    _ => as_str(get(e, "element", "automaton")?, "element automaton")?,
                };
                if !self.known_automaton(&aname) {
                    return Err(PrismError::Unknown(format!(
                        "automaton `{aname}` in system elements"
                    )));
                }
            }
        }
        let syncs = match sys.get("syncs") {
            None => return Ok(interleave_all),
            Some(s) => s
                .as_array()
                .ok_or_else(|| PrismError::Type("system `syncs` must be an array".into()))?,
        };
        let mut restricted = BTreeSet::new();
        for s in syncs {
            for key in ["inputs", "outputs"] {
                if s.get(key).is_some() {
                    return Err(bad(format!(
                        "synchronisation vectors with `{key}` are not supported"
                    )));
                }
            }
            let entries = get(s, "sync", "synchronise")?
                .as_array()
                .ok_or_else(|| PrismError::Type("sync `synchronise` must be an array".into()))?;
            if entries.is_empty() {
                return Err(PrismError::Type("sync `synchronise` is empty".into()));
            }
            // Every entry must name the same action (no renaming syncs);
            // object entries additionally pin the participating automata.
            let mut action: Option<String> = None;
            let mut participants: Option<BTreeSet<String>> = None;
            for e in entries {
                let (aname, act) = match e {
                    Value::String(a) => (None, a.clone()),
                    _ => (
                        Some(as_str(
                            get(e, "sync entry", "automaton")?,
                            "sync automaton",
                        )?),
                        as_str(get(e, "sync entry", "action")?, "sync action")?,
                    ),
                };
                if let Some(prev) = &action {
                    if prev != &act {
                        return Err(bad(format!(
                            "renaming synchronisation (`{prev}` with `{act}`) is not supported"
                        )));
                    }
                }
                action = Some(act);
                if let Some(aname) = aname {
                    if !self.known_automaton(&aname) {
                        return Err(PrismError::Unknown(format!(
                            "automaton `{aname}` in sync vector"
                        )));
                    }
                    participants.get_or_insert_with(BTreeSet::new).insert(aname);
                }
            }
            let action = action.expect("non-empty vector");
            if let Some(result) = s.get("result").and_then(|r| r.get("action")) {
                let result = as_str(result, "sync result action")?;
                if result != action {
                    return Err(bad(
                        "synchronisation vectors renaming the result action are not supported",
                    ));
                }
            }
            if let Some(result) = s.get("action") {
                let result = as_str(result, "sync action")?;
                if result != action {
                    return Err(bad(
                        "synchronisation vectors renaming the result action are not supported",
                    ));
                }
            }
            if let Some(listed) = participants {
                // Every automaton offering the action must take part;
                // otherwise the global label synchronisation would
                // over-synchronise the others.
                let offerers = self.action_offerers(&action);
                let missing: Vec<&str> = offerers
                    .iter()
                    .filter(|a| !listed.contains(*a))
                    .map(|a| a.as_str())
                    .collect();
                if !missing.is_empty() {
                    return Err(bad(format!(
                        "partial synchronisation vector for `{action}` (missing: {}); list every automaton offering the action",
                        missing.join(", ")
                    )));
                }
            }
            restricted.insert(action);
        }
        Ok(SyncConfig {
            restricted: Some(restricted),
        })
    }

    fn known_automaton(&self, name: &str) -> bool {
        self.root
            .get("automata")
            .and_then(Value::as_array)
            .is_some_and(|list| {
                list.iter()
                    .any(|a| a.get("name").and_then(Value::as_str) == Some(name))
            })
    }

    /// Automata with at least one edge labelled `action`. Only usable
    /// after edges exist — here approximated from the raw JSON (an edge
    /// lists `action` explicitly), which is exact for the subset.
    fn action_offerers(&self, action: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        if let Some(list) = self.root.get("automata").and_then(Value::as_array) {
            for a in list {
                let aname = a.get("name").and_then(Value::as_str).unwrap_or_default();
                let edges = a.get("edges").and_then(Value::as_array);
                if edges.is_some_and(|es| {
                    es.iter()
                        .any(|e| e.get("action").and_then(Value::as_str) == Some(action))
                }) {
                    out.insert(aname.to_owned());
                }
            }
        }
        out
    }
}

struct AutoInfo {
    name: String,
    loc_idx: HashMap<String, usize>,
    loc_var: usize,
    locals: HashMap<String, (String, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn coin() -> Value {
        json!({
            "jani-version": 1, "name": "coin", "type": "dtmc",
            "variables": {"s": {"type": {"kind": "bounded", "lower-bound": 0, "upper-bound": 2}, "initial-value": 0}},
            "automata": [{
                "name": "m", "locations": [{"name": "idle"}],
                "initial-location": "idle",
                "edges": [
                    {"location": "idle", "action": "toss", "guard": {"op": "=", "left": "s", "right": 0},
                     "destinations": [
                        {"location": "idle", "probability": 0.3, "assignments": [{"ref": "s", "value": 1}]},
                        {"location": "idle", "probability": 0.7, "assignments": [{"ref": "s", "value": 2}]}
                     ]},
                    {"location": "idle", "guard": {"op": "or", "left": {"op": "=", "left": "s", "right": 1}, "right": {"op": "=", "left": "s", "right": 2}},
                     "destinations": [{"location": "idle", "probability": 1.0, "assignments": [{"ref": "s", "value": 0}]}]}
                ]
            }],
            "system": {"elements": [{"automaton": "m"}], "syncs": []}
        })
    }

    #[test]
    fn dtmc_coin_with_locations() {
        let m = parse_jani(&serde_json::to_string(&coin()).unwrap()).unwrap();
        assert_eq!(m.kind, ModelKind::Dtmc);
        let d = m.derive().unwrap();
        assert_eq!(d.num_states(), 3);
        // Locations add a hidden variable; the coin still tosses 0.3/0.7.
        let init = d.lts.initial().unwrap();
        let mut probs: Vec<f64> = d
            .lts
            .successors(init)
            .iter()
            .map(|e| e.weight.value())
            .collect();
        probs.sort_by(f64::total_cmp);
        assert_eq!(probs, vec![0.3, 0.7]);
    }

    #[test]
    fn ctmc_two_state_and_sync_vectors() {
        let src = serde_json::to_string(&json!({
            "jani-version": 1, "name": "ab", "type": "ctmc",
            "constants": [{"name": "r", "type": "double", "value": 2.0}],
            "variables": {},
            "automata": [
                {"name": "a", "locations": [{"name": "x"}, {"name": "y"}],
                 "initial-location": 0,
                 "edges": [{"location": 0, "action": "tick", "rate": "r",
                            "destinations": [{"location": 1}]}]},
                {"name": "b", "locations": [{"name": "p"}],
                 "initial-location": "p",
                 "edges": [{"location": "p", "action": "tick", "rate": 3.0,
                            "destinations": [{"location": "p"}]}]}
            ],
            "system": {"elements": [{"automaton": "a"}, {"automaton": "b"}],
                       "syncs": [{"synchronise": ["tick", "tick"]}]}
        }))
        .unwrap();
        let m = parse_jani(&src).unwrap();
        let d = m.derive().unwrap();
        // Synchronised rate 2*3 from the single joint state pair.
        assert_eq!(d.num_states(), 2);
        let init = d.lts.initial().unwrap();
        let r: Vec<f64> = d
            .lts
            .successors(init)
            .iter()
            .map(|e| e.weight.value())
            .collect();
        assert_eq!(r, vec![6.0]);
    }

    #[test]
    fn unsupported_corners_error_clearly() {
        let mdp = serde_json::to_string(&json!({
            "jani-version": 1, "name": "x", "type": "mdp",
            "variables": {}, "automata": [],
            "system": {"elements": []}
        }))
        .unwrap();
        assert!(matches!(
            parse_jani(&mdp).unwrap_err(),
            PrismError::Unsupported(_)
        ));
        let unbounded = serde_json::to_string(&json!({
            "jani-version": 1, "name": "x", "type": "dtmc",
            "variables": {"n": {"type": "int", "initial-value": 0}},
            "automata": [{"name": "m", "locations": [{"name": "l"}],
                          "initial-location": "l", "edges": []}],
            "system": {"elements": [{"automaton": "m"}]}
        }))
        .unwrap();
        assert!(
            parse_jani(&unbounded)
                .unwrap_err()
                .to_string()
                .contains("bounded")
        );
        // No automata at all.
        let empty = serde_json::to_string(&json!({
            "jani-version": 1, "name": "x", "type": "dtmc",
            "variables": {}, "automata": [],
            "system": {"elements": []}
        }))
        .unwrap();
        assert!(parse_jani(&empty).is_err());
    }
}
