//! Lexer, parser and name resolution for the PRISM subset.
//!
//! Resolution folds constants, expands formulas (with cycle detection)
//! and replaces variable names by global indices, yielding
//! [`PrismModel`](super::PrismModel).

use std::collections::HashMap;

use crate::pepa::ParseError;

use super::ast::{
    Assign, Cmp, Command, Expr, ModelKind, Module, PrismError, PrismModel, ProbQuery, Query,
    RewardBlock, RewardItem, SteadyQuery, SyncConfig, Until, Update, Var, VarType,
};

/// Parses and resolves a PRISM model (`dtmc` or `ctmc`).
pub fn parse_prism(src: &str) -> Result<PrismModel, PrismError> {
    let tokens = lex(src)?;
    let raw = Parser { tokens, pos: 0 }.file()?;
    resolve(raw)
}

/// Parses a query against a model's variables: `P=? [F pred]`,
/// `P>=0.9 [F<=10 pred]`, `S=? [pred]`, `S<0.01 [pred]`.
///
/// Predicates use the same expression language as guards, over the
/// model's variables (constants and formulas must be inlined by hand).
pub fn parse_query(model: &PrismModel, src: &str) -> Result<Query, PrismError> {
    let tokens = lex(src)?;
    let raw = Parser { tokens, pos: 0 }.query()?;
    let vars: HashMap<String, usize> = model
        .vars
        .iter()
        .enumerate()
        .map(|(i, v)| (v.name.clone(), i))
        .collect();
    let pred = resolve_pred(&raw.pred, &vars)?;
    Ok(match raw.kind {
        RawQueryKind::Steady => Query::Steady(SteadyQuery {
            threshold: raw.threshold,
            pred,
        }),
        RawQueryKind::Prob { until } => Query::Prob(ProbQuery {
            threshold: raw.threshold,
            until: match until {
                Some(t) => Until::Bounded(t),
                None => Until::Eventually,
            },
            pred,
        }),
    })
}

/// Resolves a predicate: variables by index; anything else is an error.
fn resolve_pred(expr: &RawExpr, vars: &HashMap<String, usize>) -> Result<Expr, PrismError> {
    match expr {
        RawExpr::Int(i) => Ok(Expr::Int(*i)),
        RawExpr::Float(x) => Ok(Expr::Float(*x)),
        RawExpr::Bool(b) => Ok(Expr::Bool(*b)),
        RawExpr::Name(n) => vars.get(n).map(|i| Expr::Var(*i)).ok_or_else(|| {
            PrismError::Unknown(format!("{n} (not a variable; inline constants by hand)"))
        }),
        RawExpr::Neg(e) => Ok(Expr::Neg(Box::new(resolve_pred(e, vars)?))),
        RawExpr::Not(e) => Ok(Expr::Not(Box::new(resolve_pred(e, vars)?))),
        RawExpr::Bin(op, a, b) => Ok(bin(*op, resolve_pred(a, vars)?, resolve_pred(b, vars)?)),
        RawExpr::Call(f, args) => {
            let ea: Vec<Expr> = args
                .iter()
                .map(|a| resolve_pred(a, vars))
                .collect::<Result<_, _>>()?;
            call(f, ea)
        }
        RawExpr::Cond(c, a, b) => Ok(Expr::Cond(
            Box::new(resolve_pred(c, vars)?),
            Box::new(resolve_pred(a, vars)?),
            Box::new(resolve_pred(b, vars)?),
        )),
    }
}

// ---------- Lexer ----------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    Sym(&'static str),
    Eof,
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    line: usize,
    column: usize,
}

/// Two-character symbols first (longest match wins).
const SYMBOLS2: [&str; 7] = ["..", "->", "<=", ">=", "!=", "&&", "||"];
const SYMBOLS1: &str = "[](),;:!<>=+-*/%&|?'";

fn lex(src: &str) -> Result<Vec<Token>, ParseError> {
    let chars: Vec<char> = src.chars().collect();
    let (mut i, mut line, mut col) = (0, 1, 1);
    let mut out = Vec::new();
    let advance = |i: &mut usize, line: &mut usize, col: &mut usize, n: usize| {
        for _ in 0..n {
            if chars[*i] == '\n' {
                *line += 1;
                *col = 1;
            } else {
                *col += 1;
            }
            *i += 1;
        }
    };
    while i < chars.len() {
        let c = chars[i];
        let (start_line, start_col) = (line, col);
        let here = |tok| Token {
            tok,
            line: start_line,
            column: start_col,
        };
        if c.is_whitespace() {
            advance(&mut i, &mut line, &mut col, 1);
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                advance(&mut i, &mut line, &mut col, 1);
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'*') {
            advance(&mut i, &mut line, &mut col, 2);
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                advance(&mut i, &mut line, &mut col, 1);
            }
            if i >= chars.len() {
                return Err(err_at(start_line, start_col, "unterminated comment"));
            }
            advance(&mut i, &mut line, &mut col, 2);
        } else if c == '"' {
            // Quoted names (`rewards "name"`); no escapes.
            advance(&mut i, &mut line, &mut col, 1);
            let start = i;
            while i < chars.len() && chars[i] != '"' && chars[i] != '\n' {
                advance(&mut i, &mut line, &mut col, 1);
            }
            if chars.get(i) != Some(&'"') {
                return Err(err_at(start_line, start_col, "unterminated string"));
            }
            out.push(here(Tok::Str(chars[start..i].iter().collect())));
            advance(&mut i, &mut line, &mut col, 1);
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                advance(&mut i, &mut line, &mut col, 1);
            }
            out.push(here(Tok::Ident(chars[start..i].iter().collect())));
        } else if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                advance(&mut i, &mut line, &mut col, 1);
            }
            let mut is_float = false;
            if chars.get(i) == Some(&'.') && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()) {
                is_float = true;
                advance(&mut i, &mut line, &mut col, 1);
                while i < chars.len() && chars[i].is_ascii_digit() {
                    advance(&mut i, &mut line, &mut col, 1);
                }
            }
            if matches!(chars.get(i), Some('e' | 'E')) {
                let sign = usize::from(matches!(chars.get(i + 1), Some('+' | '-')));
                if chars.get(i + 1 + sign).is_some_and(|d| d.is_ascii_digit()) {
                    is_float = true;
                    advance(&mut i, &mut line, &mut col, 1 + sign);
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        advance(&mut i, &mut line, &mut col, 1);
                    }
                }
            }
            let text: String = chars[start..i].iter().collect();
            if is_float {
                let value = text
                    .parse()
                    .map_err(|_| err_at(start_line, start_col, "invalid number"))?;
                out.push(here(Tok::Float(value)));
            } else {
                let value = text
                    .parse()
                    .map_err(|_| err_at(start_line, start_col, "invalid number"))?;
                out.push(here(Tok::Int(value)));
            }
        } else {
            let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
            if let Some(sym) = SYMBOLS2.iter().find(|s| rest.starts_with(**s)) {
                out.push(here(Tok::Sym(sym)));
                advance(&mut i, &mut line, &mut col, sym.len());
            } else if SYMBOLS1.contains(c) {
                let sym = SYMBOLS1
                    .find(c)
                    .map(|k| match k {
                        0 => "[",
                        1 => "]",
                        2 => "(",
                        3 => ")",
                        4 => ",",
                        5 => ";",
                        6 => ":",
                        7 => "!",
                        8 => "<",
                        9 => ">",
                        10 => "=",
                        11 => "+",
                        12 => "-",
                        13 => "*",
                        14 => "/",
                        15 => "%",
                        16 => "&",
                        17 => "|",
                        18 => "?",
                        _ => "'",
                    })
                    .expect("indexed symbol");
                out.push(here(Tok::Sym(sym)));
                advance(&mut i, &mut line, &mut col, 1);
            } else {
                return Err(err_at(
                    start_line,
                    start_col,
                    &format!("unexpected character `{c}`"),
                ));
            }
        }
    }
    out.push(Token {
        tok: Tok::Eof,
        line,
        column: col,
    });
    Ok(out)
}

fn err_at(line: usize, column: usize, message: &str) -> ParseError {
    ParseError {
        line,
        column,
        message: message.to_owned(),
    }
}

// ---------- Raw AST (names unresolved) ----------

type Pos = (usize, usize);

#[derive(Debug, Clone)]
enum RawExpr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Name(String),
    Neg(Box<RawExpr>),
    Not(Box<RawExpr>),
    Bin(BinOp, Box<RawExpr>, Box<RawExpr>),
    Call(String, Vec<RawExpr>),
    Cond(Box<RawExpr>, Box<RawExpr>, Box<RawExpr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawType {
    Int,
    Double,
    Bool,
}

#[derive(Debug, Clone)]
struct RawConst {
    name: String,
    typ: RawType,
    expr: RawExpr,
}

#[derive(Debug, Clone)]
struct RawFormula {
    name: String,
    expr: RawExpr,
}

#[derive(Debug, Clone)]
enum RawVarType {
    Range(RawExpr, RawExpr),
    Bool,
}

#[derive(Debug, Clone)]
struct RawVar {
    name: String,
    typ: RawVarType,
    init: RawExpr,
}

#[derive(Debug, Clone)]
struct RawAssign {
    var: String,
    expr: RawExpr,
}

#[derive(Debug, Clone)]
struct RawUpdate {
    weight: RawExpr,
    assigns: Vec<RawAssign>,
}

#[derive(Debug, Clone)]
struct RawCommand {
    label: Option<String>,
    guard: RawExpr,
    updates: Vec<RawUpdate>,
}

#[derive(Debug, Clone)]
struct RawModule {
    name: String,
    vars: Vec<RawVar>,
    commands: Vec<RawCommand>,
}

#[derive(Debug, Clone)]
enum RawRewardItem {
    State {
        guard: RawExpr,
        value: RawExpr,
    },
    Trans {
        label: String,
        guard: RawExpr,
        value: RawExpr,
    },
}

#[derive(Debug, Clone)]
struct RawRewardBlock {
    name: String,
    items: Vec<RawRewardItem>,
}

#[derive(Debug, Clone)]
enum RawSysExpr {
    Module(String),
    Par(Box<RawSysExpr>, Box<RawSysExpr>),
    Sync(Box<RawSysExpr>, Box<RawSysExpr>, Vec<String>),
}

#[derive(Debug, Clone)]
struct RawFile {
    kind: ModelKind,
    consts: Vec<RawConst>,
    formulas: Vec<RawFormula>,
    modules: Vec<RawModule>,
    rewards: Vec<RawRewardBlock>,
    system: Option<RawSysExpr>,
}

// ---------- Parser ----------

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn peek_at(&self, k: usize) -> &Tok {
        &self.tokens[(self.pos + k).min(self.tokens.len() - 1)].tok
    }

    fn here(&self) -> Pos {
        let t = &self.tokens[self.pos];
        (t.line, t.column)
    }

    fn error<T>(&self, msg: &str) -> Result<T, ParseError> {
        let (line, column) = self.here();
        Err(err_at(line, column, msg))
    }

    fn bump(&mut self) -> Tok {
        let t = self.tokens[self.pos].tok.clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, sym: &str) -> bool {
        if matches!(self.peek(), Tok::Sym(s) if *s == sym) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, sym: &str) -> Result<(), ParseError> {
        if self.eat(sym) {
            Ok(())
        } else {
            self.error(&format!("expected `{sym}`"))
        }
    }

    fn ident(&mut self) -> Result<String, ParseError> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.bump();
                Ok(s)
            }
            _ => self.error("expected a name"),
        }
    }

    fn keyword(&mut self, word: &str) -> bool {
        if matches!(self.peek(), Tok::Ident(s) if s == word) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn file(mut self) -> Result<RawFile, ParseError> {
        let kind = match self.ident()?.as_str() {
            "dtmc" => ModelKind::Dtmc,
            "ctmc" => ModelKind::Ctmc,
            "mdp" => {
                return self.error("mdp models are not supported; use dtmc or ctmc");
            }
            other => {
                let (line, column) = self.here();
                return Err(err_at(
                    line,
                    column,
                    &format!("expected `dtmc` or `ctmc` first, found `{other}`"),
                ));
            }
        };
        let mut consts = Vec::new();
        let mut formulas = Vec::new();
        let mut modules = Vec::new();
        let mut rewards = Vec::new();
        let mut system = None;
        while *self.peek() != Tok::Eof {
            match self.peek().clone() {
                Tok::Ident(s) if s == "const" => consts.push(self.const_decl()?),
                Tok::Ident(s) if s == "formula" => formulas.push(self.formula_decl()?),
                Tok::Ident(s) if s == "module" => modules.push(self.module_decl()?),
                Tok::Ident(s) if s == "rewards" => rewards.push(self.rewards_decl()?),
                Tok::Ident(s) if s == "system" => {
                    if system.is_some() {
                        return self.error("only one system block is allowed");
                    }
                    system = Some(self.system_decl()?);
                }
                Tok::Ident(s) if s == "init" => {
                    return self.error(
                        "init constraints are not supported; give every variable an init value",
                    );
                }
                _ => {
                    return self
                        .error("expected `const`, `formula`, `module`, `rewards` or `system`");
                }
            }
        }
        if modules.is_empty() {
            return self.error("the model has no modules");
        }
        Ok(RawFile {
            kind,
            consts,
            formulas,
            modules,
            rewards,
            system,
        })
    }

    fn const_decl(&mut self) -> Result<RawConst, ParseError> {
        assert!(self.keyword("const"));
        let typ = match self.ident()?.as_str() {
            "int" => RawType::Int,
            "double" => RawType::Double,
            "bool" => RawType::Bool,
            other => {
                return self.error(&format!(
                    "expected `int`, `double` or `bool` after `const`, found `{other}`"
                ));
            }
        };
        let name = self.ident()?;
        self.expect("=")?;
        let expr = self.expr()?;
        self.expect(";")?;
        Ok(RawConst { name, typ, expr })
    }

    fn formula_decl(&mut self) -> Result<RawFormula, ParseError> {
        assert!(self.keyword("formula"));
        let name = self.ident()?;
        self.expect("=")?;
        let expr = self.expr()?;
        self.expect(";")?;
        Ok(RawFormula { name, expr })
    }

    fn module_decl(&mut self) -> Result<RawModule, ParseError> {
        assert!(self.keyword("module"));
        let name = self.ident()?;
        let mut vars = Vec::new();
        let mut commands = Vec::new();
        loop {
            if self.keyword("endmodule") {
                return Ok(RawModule {
                    name,
                    vars,
                    commands,
                });
            }
            // A variable starts `name :`; anything else is a command.
            let is_var =
                matches!(self.peek(), Tok::Ident(_)) && matches!(self.peek_at(1), Tok::Sym(":"));
            if is_var {
                vars.push(self.var_decl()?);
            } else if *self.peek() == Tok::Eof {
                return self.error("missing `endmodule`");
            } else {
                commands.push(self.command()?);
            }
        }
    }

    fn rewards_decl(&mut self) -> Result<RawRewardBlock, ParseError> {
        assert!(self.keyword("rewards"));
        let name = match self.bump() {
            Tok::Str(s) | Tok::Ident(s) => s,
            _ => return self.error("expected a reward name after `rewards`"),
        };
        let mut items = Vec::new();
        loop {
            if self.keyword("endrewards") {
                return Ok(RawRewardBlock { name, items });
            }
            if *self.peek() == Tok::Eof {
                return self.error("missing `endrewards`");
            }
            items.push(self.reward_item()?);
        }
    }

    fn reward_item(&mut self) -> Result<RawRewardItem, ParseError> {
        // `[label] guard : value` is a transition reward; otherwise the
        // item is a state reward `guard : value`.
        if self.eat("[") {
            let label = match self.peek().clone() {
                Tok::Ident(s) => {
                    self.bump();
                    s
                }
                _ => return self.error("transition rewards need a label, e.g. `[tick]`"),
            };
            self.expect("]")?;
            let guard = self.expr()?;
            self.expect(":")?;
            let value = self.expr()?;
            self.expect(";")?;
            return Ok(RawRewardItem::Trans {
                label,
                guard,
                value,
            });
        }
        let guard = self.expr()?;
        self.expect(":")?;
        let value = self.expr()?;
        self.expect(";")?;
        Ok(RawRewardItem::State { guard, value })
    }

    fn system_decl(&mut self) -> Result<RawSysExpr, ParseError> {
        assert!(self.keyword("system"));
        let expr = self.sys_expr()?;
        if !self.keyword("endsystem") {
            return self.error("expected `endsystem`");
        }
        Ok(expr)
    }

    // sys := factor (("||" | "|[" labels "]|") factor)*
    fn sys_expr(&mut self) -> Result<RawSysExpr, ParseError> {
        let mut left = self.sys_factor()?;
        loop {
            if self.eat("||") {
                left = RawSysExpr::Par(Box::new(left), Box::new(self.sys_factor()?));
            } else if self.eat("|") {
                // Restricted synchronisation `|[a, b]|`.
                if !self.eat("[") {
                    return self.error("expected `||` or `|[...]|` between modules");
                }
                let mut labels = Vec::new();
                if !matches!(self.peek(), Tok::Sym("]")) {
                    labels.push(self.ident()?);
                    while self.eat(",") {
                        labels.push(self.ident()?);
                    }
                }
                self.expect("]")?;
                self.expect("|")?;
                left = RawSysExpr::Sync(Box::new(left), Box::new(self.sys_factor()?), labels);
            } else {
                return Ok(left);
            }
        }
    }

    fn sys_factor(&mut self) -> Result<RawSysExpr, ParseError> {
        if self.eat("(") {
            let e = self.sys_expr()?;
            self.expect(")")?;
            return Ok(e);
        }
        let name = self.ident()?;
        // Renaming (`M[a=b]`) is not supported; catch it explicitly.
        if *self.peek() == Tok::Sym("[") {
            return self.error("renaming (M[...]) is not supported in system blocks");
        }
        Ok(RawSysExpr::Module(name))
    }

    fn var_decl(&mut self) -> Result<RawVar, ParseError> {
        let name = self.ident()?;
        self.expect(":")?;
        let typ = if self.eat("[") {
            let lo = self.expr()?;
            self.expect("..")?;
            let hi = self.expr()?;
            self.expect("]")?;
            RawVarType::Range(lo, hi)
        } else if matches!(self.peek(), Tok::Ident(s) if s == "bool") {
            self.bump();
            RawVarType::Bool
        } else {
            return self.error("expected `[lo..hi]` or `bool` after `:`");
        };
        if !self.keyword("init") {
            return self.error("expected `init <value>` after the variable type");
        }
        let init = self.expr()?;
        self.expect(";")?;
        Ok(RawVar { name, typ, init })
    }

    fn command(&mut self) -> Result<RawCommand, ParseError> {
        let label = if self.eat("[") {
            let label = match self.peek().clone() {
                Tok::Ident(s) => {
                    self.bump();
                    Some(s)
                }
                _ => None,
            };
            self.expect("]")?;
            label
        } else {
            None
        };
        let guard = self.expr()?;
        self.expect("->")?;
        let mut updates = vec![self.update()?];
        while self.eat("+") {
            updates.push(self.update()?);
        }
        self.expect(";")?;
        Ok(RawCommand {
            label,
            guard,
            updates,
        })
    }

    fn update(&mut self) -> Result<RawUpdate, ParseError> {
        // Bare `true` is the empty update (no assignments): either without
        // a weight (`-> true`, weight 1) or with one (`-> w : true`).
        if matches!(self.peek(), Tok::Ident(s) if s == "true")
            && matches!(self.peek_at(1), Tok::Sym(";") | Tok::Sym("+"))
        {
            self.bump();
            return Ok(RawUpdate {
                weight: RawExpr::Int(1),
                assigns: Vec::new(),
            });
        }
        let weight = self.expr()?;
        self.expect(":")?;
        if matches!(self.peek(), Tok::Ident(s) if s == "true")
            && matches!(self.peek_at(1), Tok::Sym(";") | Tok::Sym("+"))
        {
            self.bump();
            return Ok(RawUpdate {
                weight,
                assigns: Vec::new(),
            });
        }
        let mut assigns = vec![self.assign()?];
        while self.eat("&") || self.eat("&&") {
            assigns.push(self.assign()?);
        }
        Ok(RawUpdate { weight, assigns })
    }

    fn assign(&mut self) -> Result<RawAssign, ParseError> {
        self.expect("(")?;
        let var = self.ident()?;
        self.expect("'")?;
        self.expect("=")?;
        let expr = self.expr()?;
        self.expect(")")?;
        Ok(RawAssign { var, expr })
    }

    // expr := cond
    fn expr(&mut self) -> Result<RawExpr, ParseError> {
        self.cond()
    }

    // cond := or ('?' expr ':' expr)?
    fn cond(&mut self) -> Result<RawExpr, ParseError> {
        let c = self.or()?;
        if self.eat("?") {
            let a = self.expr()?;
            self.expect(":")?;
            let b = self.cond()?;
            return Ok(RawExpr::Cond(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    // or := and (('| finally '||') and)*
    fn or(&mut self) -> Result<RawExpr, ParseError> {
        let mut left = self.and()?;
        while self.eat("|") || self.eat("||") {
            left = RawExpr::Bin(BinOp::Or, Box::new(left), Box::new(self.and()?));
        }
        Ok(left)
    }

    // and := eq (('&'|'&&') eq)*
    fn and(&mut self) -> Result<RawExpr, ParseError> {
        let mut left = self.eq()?;
        while self.eat("&") || self.eat("&&") {
            left = RawExpr::Bin(BinOp::And, Box::new(left), Box::new(self.eq()?));
        }
        Ok(left)
    }

    // eq := cmp (('='|'!=') cmp)*
    fn eq(&mut self) -> Result<RawExpr, ParseError> {
        let mut left = self.cmp()?;
        loop {
            let op = if self.eat("=") {
                if self.eat("=") {
                    return self.error("use `=` for equality, not `==`");
                }
                BinOp::Eq
            } else if self.eat("!=") {
                BinOp::Ne
            } else {
                return Ok(left);
            };
            left = RawExpr::Bin(op, Box::new(left), Box::new(self.cmp()?));
        }
    }

    // cmp := add (('<'|'<='|'>'|'>=') add)*
    fn cmp(&mut self) -> Result<RawExpr, ParseError> {
        let mut left = self.add()?;
        loop {
            let op = if self.eat("<=") {
                BinOp::Le
            } else if self.eat(">=") {
                BinOp::Ge
            } else if self.eat("<") {
                BinOp::Lt
            } else if self.eat(">") {
                BinOp::Gt
            } else {
                return Ok(left);
            };
            left = RawExpr::Bin(op, Box::new(left), Box::new(self.add()?));
        }
    }

    // add := mul (('+'|'-') mul)*
    fn add(&mut self) -> Result<RawExpr, ParseError> {
        let mut left = self.mul()?;
        loop {
            let op = if self.eat("+") {
                BinOp::Add
            } else if self.eat("-") {
                BinOp::Sub
            } else {
                return Ok(left);
            };
            left = RawExpr::Bin(op, Box::new(left), Box::new(self.mul()?));
        }
    }

    // mul := unary (('*'|'/'|'mod'|'%') unary)*
    fn mul(&mut self) -> Result<RawExpr, ParseError> {
        let mut left = self.unary()?;
        loop {
            let op = if self.eat("*") {
                BinOp::Mul
            } else if self.eat("/") {
                BinOp::Div
            } else if self.eat("%") || matches!(self.peek(), Tok::Ident(s) if s == "mod") {
                if matches!(self.peek(), Tok::Ident(_)) {
                    self.bump();
                }
                BinOp::Mod
            } else {
                return Ok(left);
            };
            left = RawExpr::Bin(op, Box::new(left), Box::new(self.unary()?));
        }
    }

    // unary := '-' unary | '!' unary | primary
    fn unary(&mut self) -> Result<RawExpr, ParseError> {
        if self.eat("-") {
            return Ok(RawExpr::Neg(Box::new(self.unary()?)));
        }
        if self.eat("!") {
            return Ok(RawExpr::Not(Box::new(self.unary()?)));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<RawExpr, ParseError> {
        match self.bump() {
            Tok::Int(i) => Ok(RawExpr::Int(i)),
            Tok::Float(x) => Ok(RawExpr::Float(x)),
            Tok::Ident(s) if s == "true" => Ok(RawExpr::Bool(true)),
            Tok::Ident(s) if s == "false" => Ok(RawExpr::Bool(false)),
            Tok::Ident(s)
                if matches!(s.as_str(), "min" | "max" | "floor" | "ceil" | "pow")
                    && *self.peek() == Tok::Sym("(") =>
            {
                self.bump();
                let mut args = vec![self.expr()?];
                while self.eat(",") {
                    args.push(self.expr()?);
                }
                self.expect(")")?;
                Ok(RawExpr::Call(s, args))
            }
            Tok::Ident(s) => Ok(RawExpr::Name(s)),
            Tok::Sym("(") => {
                let e = self.expr()?;
                self.expect(")")?;
                Ok(e)
            }
            _ => {
                let (line, column) = self.here();
                Err(err_at(line, column, "expected an expression"))
            }
        }
    }

    fn number(&mut self) -> Result<f64, ParseError> {
        match self.bump() {
            Tok::Int(i) => Ok(i as f64),
            Tok::Float(x) => Ok(x),
            _ => self.error("expected a number"),
        }
    }

    // query := ('P' | 'S') ('=?' | cmp number) '[' ('F' ('<=' number)? expr | expr) ']'
    fn query(&mut self) -> Result<RawQuery, ParseError> {
        let prob = match self.bump() {
            Tok::Ident(s) if s == "P" => true,
            Tok::Ident(s) if s == "S" => false,
            _ => return self.error("expected `P` or `S`"),
        };
        let threshold = if self.eat("=") {
            if !self.eat("?") {
                return self.error("did you mean `=?`?");
            }
            None
        } else if self.eat(">=") {
            Some((Cmp::Ge, self.number()?))
        } else if self.eat("<=") {
            Some((Cmp::Le, self.number()?))
        } else if self.eat(">") {
            Some((Cmp::Gt, self.number()?))
        } else if self.eat("<") {
            Some((Cmp::Lt, self.number()?))
        } else {
            return self.error("expected `=?` or a threshold like `>= 0.9`");
        };
        self.expect("[")?;
        let kind = if prob {
            if !self.keyword("F") {
                return self.error("only `F` path formulas are supported");
            }
            let until = if self.eat("<=") {
                Some(self.number()?)
            } else {
                None
            };
            RawQueryKind::Prob { until }
        } else {
            RawQueryKind::Steady
        };
        let pred = self.expr()?;
        self.expect("]")?;
        if *self.peek() != Tok::Eof {
            return self.error("unexpected input after `]`");
        }
        Ok(RawQuery {
            kind,
            threshold,
            pred,
        })
    }
}

#[derive(Debug, Clone)]
enum RawQueryKind {
    Prob { until: Option<f64> },
    Steady,
}

#[derive(Debug, Clone)]
struct RawQuery {
    kind: RawQueryKind,
    threshold: Option<(Cmp, f64)>,
    pred: RawExpr,
}

// ---------- Resolution ----------

use super::ast::{Value, VarKind};

/// Closed-world evaluation for constants, bounds and init values:
/// only earlier constants are visible.
fn eval_closed(expr: &RawExpr, consts: &HashMap<String, Value>) -> Result<Value, PrismError> {
    match expr {
        RawExpr::Int(i) => Ok(Value::Int(*i)),
        RawExpr::Float(x) => Ok(Value::Float(*x)),
        RawExpr::Bool(b) => Ok(Value::Bool(*b)),
        RawExpr::Name(n) => consts
            .get(n)
            .copied()
            .ok_or_else(|| PrismError::Unknown(n.clone())),
        RawExpr::Neg(e) => match eval_closed(e, consts)? {
            Value::Int(i) => i
                .checked_neg()
                .map(Value::Int)
                .ok_or_else(|| PrismError::Arithmetic("negation overflows".to_owned())),
            Value::Float(x) => x
                .is_finite()
                .then_some(Value::Float(-x))
                .ok_or_else(|| PrismError::Arithmetic("negation is not finite".to_owned())),
            Value::Bool(_) => Err(PrismError::Type("cannot negate a boolean".to_owned())),
        },
        RawExpr::Not(e) => match eval_closed(e, consts)? {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            _ => Err(PrismError::Type("`!` needs a boolean".to_owned())),
        },
        RawExpr::Bin(op, a, b) => {
            let ea = rewrite_closed(a, consts)?;
            let eb = rewrite_closed(b, consts)?;
            super::ast::eval(&bin(*op, ea, eb), &[], &[])
                .ok_or_else(|| PrismError::Arithmetic("constant is not finite".to_owned()))
        }
        RawExpr::Call(f, args) => {
            let ea: Vec<Expr> = args
                .iter()
                .map(|a| rewrite_closed(a, consts))
                .collect::<Result<_, _>>()?;
            super::ast::eval(&call(f, ea)?, &[], &[])
                .ok_or_else(|| PrismError::Arithmetic(format!("constant `{f}(…)` is not finite")))
        }
        RawExpr::Cond(c, a, b) => {
            if eval_closed(c, consts)?
                .as_bool()
                .ok_or_else(|| PrismError::Type("`?:` condition must be boolean".to_owned()))?
            {
                eval_closed(a, consts)
            } else {
                eval_closed(b, consts)
            }
        }
    }
}

/// Resolves names shared by closed and open expression rewriting.
fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    match op {
        BinOp::Add => Expr::Add(Box::new(a), Box::new(b)),
        BinOp::Sub => Expr::Sub(Box::new(a), Box::new(b)),
        BinOp::Mul => Expr::Mul(Box::new(a), Box::new(b)),
        BinOp::Div => Expr::Div(Box::new(a), Box::new(b)),
        BinOp::Mod => Expr::Mod(Box::new(a), Box::new(b)),
        BinOp::Eq => Expr::Eq(Box::new(a), Box::new(b)),
        BinOp::Ne => Expr::Ne(Box::new(a), Box::new(b)),
        BinOp::Lt => Expr::Lt(Box::new(a), Box::new(b)),
        BinOp::Le => Expr::Le(Box::new(a), Box::new(b)),
        BinOp::Gt => Expr::Gt(Box::new(a), Box::new(b)),
        BinOp::Ge => Expr::Ge(Box::new(a), Box::new(b)),
        BinOp::And => Expr::And(Box::new(a), Box::new(b)),
        BinOp::Or => Expr::Or(Box::new(a), Box::new(b)),
    }
}

fn call(f: &str, args: Vec<Expr>) -> Result<Expr, PrismError> {
    match f {
        "min" if !args.is_empty() => Ok(Expr::Min(args)),
        "max" if !args.is_empty() => Ok(Expr::Max(args)),
        "floor" if args.len() == 1 => {
            Ok(Expr::Floor(Box::new(args.into_iter().next().expect("one"))))
        }
        "ceil" if args.len() == 1 => {
            Ok(Expr::Ceil(Box::new(args.into_iter().next().expect("one"))))
        }
        "pow" if args.len() == 2 => {
            let mut it = args.into_iter();
            let (a, b) = (it.next().expect("two"), it.next().expect("two"));
            Ok(Expr::Pow(Box::new(a), Box::new(b)))
        }
        _ => Err(PrismError::Type(format!(
            "unknown function or wrong arity: `{f}` (min/max take 1+, floor/ceil 1, pow 2)"
        ))),
    }
}

/// Rewrites a closed expression, folding constants to literals.
fn rewrite_closed(expr: &RawExpr, consts: &HashMap<String, Value>) -> Result<Expr, PrismError> {
    Ok(match eval_closed(expr, consts)? {
        Value::Int(i) => Expr::Int(i),
        Value::Float(x) => Expr::Float(x),
        Value::Bool(b) => Expr::Bool(b),
    })
}

struct Ctx<'a> {
    vars: &'a HashMap<String, usize>,
    consts: &'a HashMap<String, Value>,
    formulas: &'a HashMap<String, RawExpr>,
}

/// Resolves an open expression: folds constants, expands formulas,
/// replaces variables by indices.
fn resolve_expr(expr: &RawExpr, ctx: &Ctx, stack: &mut Vec<String>) -> Result<Expr, PrismError> {
    match expr {
        RawExpr::Int(i) => Ok(Expr::Int(*i)),
        RawExpr::Float(x) => Ok(Expr::Float(*x)),
        RawExpr::Bool(b) => Ok(Expr::Bool(*b)),
        RawExpr::Name(n) => {
            if let Some(v) = ctx.consts.get(n) {
                return Ok(match *v {
                    Value::Int(i) => Expr::Int(i),
                    Value::Float(x) => Expr::Float(x),
                    Value::Bool(b) => Expr::Bool(b),
                });
            }
            if let Some(i) = ctx.vars.get(n) {
                return Ok(Expr::Var(*i));
            }
            if let Some(body) = ctx.formulas.get(n) {
                if stack.contains(n) {
                    return Err(PrismError::CyclicFormula(n.clone()));
                }
                stack.push(n.clone());
                let e = resolve_expr(body, ctx, stack)?;
                stack.pop();
                return Ok(e);
            }
            Err(PrismError::Unknown(n.clone()))
        }
        RawExpr::Neg(e) => Ok(Expr::Neg(Box::new(resolve_expr(e, ctx, stack)?))),
        RawExpr::Not(e) => Ok(Expr::Not(Box::new(resolve_expr(e, ctx, stack)?))),
        RawExpr::Bin(op, a, b) => Ok(bin(
            *op,
            resolve_expr(a, ctx, stack)?,
            resolve_expr(b, ctx, stack)?,
        )),
        RawExpr::Call(f, args) => {
            let ea: Vec<Expr> = args
                .iter()
                .map(|a| resolve_expr(a, ctx, stack))
                .collect::<Result<_, _>>()?;
            call(f, ea)
        }
        RawExpr::Cond(c, a, b) => Ok(Expr::Cond(
            Box::new(resolve_expr(c, ctx, stack)?),
            Box::new(resolve_expr(a, ctx, stack)?),
            Box::new(resolve_expr(b, ctx, stack)?),
        )),
    }
}

fn resolve(raw: RawFile) -> Result<PrismModel, PrismError> {
    // Constants first: each sees only earlier ones.
    let mut consts: HashMap<String, Value> = HashMap::new();
    for c in &raw.consts {
        if consts.contains_key(&c.name) {
            return Err(PrismError::Duplicate(c.name.clone()));
        }
        let v = eval_closed(&c.expr, &consts)?;
        match (c.typ, v) {
            (RawType::Int, Value::Int(_))
            | (RawType::Double, Value::Int(_) | Value::Float(_))
            | (RawType::Bool, Value::Bool(_)) => {}
            _ => {
                return Err(PrismError::Type(format!(
                    "constant `{}` is not a {}",
                    c.name,
                    match c.typ {
                        RawType::Int => "integer",
                        RawType::Double => "number",
                        RawType::Bool => "boolean",
                    }
                )));
            }
        }
        // Normalise `const double x = 2` to a float.
        let v = match (c.typ, v) {
            (RawType::Double, Value::Int(i)) => Value::Float(i as f64),
            _ => v,
        };
        consts.insert(c.name.clone(), v);
    }

    let mut formulas: HashMap<String, RawExpr> = HashMap::new();
    for f in &raw.formulas {
        if formulas.contains_key(&f.name) || consts.contains_key(&f.name) {
            return Err(PrismError::Duplicate(f.name.clone()));
        }
        formulas.insert(f.name.clone(), f.expr.clone());
    }

    // Variables get global indices; names must be unique across modules.
    let mut var_ids: HashMap<String, usize> = HashMap::new();
    let mut vars: Vec<Var> = Vec::new();
    let mut kinds: Vec<VarKind> = Vec::new();
    for m in &raw.modules {
        for v in &m.vars {
            if var_ids.contains_key(&v.name)
                || consts.contains_key(&v.name)
                || formulas.contains_key(&v.name)
            {
                return Err(PrismError::Duplicate(v.name.clone()));
            }
            let (typ, kind, init) = match &v.typ {
                RawVarType::Bool => {
                    let init = match eval_closed(&v.init, &consts)? {
                        Value::Bool(b) => i64::from(b),
                        Value::Int(0) => 0,
                        Value::Int(1) => 1,
                        _ => {
                            return Err(PrismError::Type(format!(
                                "initial value of `{}` is not boolean",
                                v.name
                            )));
                        }
                    };
                    (VarType::Bool, VarKind::Bool, init)
                }
                RawVarType::Range(lo, hi) => {
                    let (lo, hi) = match (eval_closed(lo, &consts)?, eval_closed(hi, &consts)?) {
                        (Value::Int(lo), Value::Int(hi)) => (lo, hi),
                        _ => {
                            return Err(PrismError::Type(format!(
                                "bounds of `{}` are not integers",
                                v.name
                            )));
                        }
                    };
                    if lo > hi {
                        return Err(PrismError::Type(format!(
                            "empty range [{lo}..{hi}] of `{}`",
                            v.name
                        )));
                    }
                    let init = match eval_closed(&v.init, &consts)? {
                        Value::Int(i) => i,
                        _ => {
                            return Err(PrismError::Type(format!(
                                "initial value of `{}` is not an integer",
                                v.name
                            )));
                        }
                    };
                    if init < lo || init > hi {
                        return Err(PrismError::OutOfBounds {
                            var: v.name.clone(),
                            value: init,
                        });
                    }
                    (VarType::Int { lo, hi }, VarKind::Int, init)
                }
            };
            var_ids.insert(v.name.clone(), vars.len());
            vars.push(Var {
                name: v.name.clone(),
                typ,
                init,
            });
            kinds.push(kind);
        }
    }

    let ctx = Ctx {
        vars: &var_ids,
        consts: &consts,
        formulas: &formulas,
    };
    let mut modules = Vec::new();
    for m in &raw.modules {
        let mut commands = Vec::new();
        for c in &m.commands {
            let mut stack = Vec::new();
            let guard = resolve_expr(&c.guard, &ctx, &mut stack)?;
            let mut updates = Vec::new();
            for u in &c.updates {
                let weight = resolve_expr(&u.weight, &ctx, &mut stack)?;
                let mut seen = std::collections::BTreeSet::new();
                let mut assigns = Vec::new();
                for a in &u.assigns {
                    let idx = *var_ids
                        .get(&a.var)
                        .ok_or_else(|| PrismError::Unknown(format!("{} (assigned here)", a.var)))?;
                    if !seen.insert(idx) {
                        return Err(PrismError::Type(format!(
                            "`{}` is assigned twice in one update",
                            a.var
                        )));
                    }
                    assigns.push(Assign {
                        var: idx,
                        expr: resolve_expr(&a.expr, &ctx, &mut stack)?,
                    });
                }
                updates.push(Update { weight, assigns });
            }
            if updates.is_empty() {
                return Err(PrismError::Type(
                    "a command needs at least one update".to_owned(),
                ));
            }
            commands.push(Command {
                label: c.label.clone(),
                guard,
                updates,
            });
        }
        // Module-local variable indices for convenience.
        let module_vars: Vec<usize> = m.vars.iter().map(|v| var_ids[&v.name]).collect();
        modules.push(Module {
            name: m.name.clone(),
            vars: module_vars,
            commands,
        });
        if m.name.is_empty() {
            return Err(PrismError::Type("module names cannot be empty".to_owned()));
        }
    }
    // Module names must be unique (checked late so variable errors win).
    {
        let mut seen = std::collections::BTreeSet::new();
        for m in &modules {
            if !seen.insert(m.name.clone()) {
                return Err(PrismError::Duplicate(m.name.clone()));
            }
        }
    }
    let rewards = resolve_rewards(&raw.rewards, &ctx)?;
    let sync = resolve_system(raw.system.as_ref(), &modules)?;
    Ok(PrismModel {
        kind: raw.kind,
        modules,
        vars,
        rewards,
        sync,
    })
}

/// Resolves reward guards and values (constants, formulas and variables
/// are all visible here).
fn resolve_rewards(raw: &[RawRewardBlock], ctx: &Ctx) -> Result<Vec<RewardBlock>, PrismError> {
    let mut out = Vec::new();
    for b in raw {
        let mut items = Vec::new();
        for item in &b.items {
            let mut stack = Vec::new();
            items.push(match item {
                RawRewardItem::State { guard, value } => RewardItem::State {
                    guard: resolve_expr(guard, ctx, &mut stack)?,
                    value: resolve_expr(value, ctx, &mut stack)?,
                },
                RawRewardItem::Trans {
                    label,
                    guard,
                    value,
                } => RewardItem::Trans {
                    label: label.clone(),
                    guard: resolve_expr(guard, ctx, &mut stack)?,
                    value: resolve_expr(value, ctx, &mut stack)?,
                },
            });
        }
        out.push(RewardBlock {
            name: b.name.clone(),
            items,
        });
    }
    Ok(out)
}

/// Resolves a `system` block to a global sync configuration: one
/// top-level parallel composition, either unrestricted (`||`) or with a
/// single shared restriction set (`|[...]|` everywhere). Anything nested
/// or mixed is rejected explicitly.
fn resolve_system(raw: Option<&RawSysExpr>, modules: &[Module]) -> Result<SyncConfig, PrismError> {
    use std::collections::BTreeSet;

    let Some(expr) = raw else {
        return Ok(SyncConfig::default());
    };
    fn walk(
        e: &RawSysExpr,
        modules: &[Module],
        seen: &mut BTreeSet<String>,
        sets: &mut Vec<BTreeSet<String>>,
    ) -> Result<(), PrismError> {
        match e {
            RawSysExpr::Module(name) => {
                if !modules.iter().any(|m| &m.name == name) {
                    return Err(PrismError::Unknown(format!(
                        "module `{name}` in system block"
                    )));
                }
                if !seen.insert(name.clone()) {
                    return Err(PrismError::Duplicate(format!(
                        "module `{name}` appears twice in system block"
                    )));
                }
                Ok(())
            }
            RawSysExpr::Par(l, r) => {
                walk(l, modules, seen, sets)?;
                walk(r, modules, seen, sets)
            }
            RawSysExpr::Sync(l, r, labels) => {
                sets.push(labels.iter().cloned().collect());
                walk(l, modules, seen, sets)?;
                walk(r, modules, seen, sets)
            }
        }
    }
    let mut seen = BTreeSet::new();
    let mut sets = Vec::new();
    walk(expr, modules, &mut seen, &mut sets)?;
    if seen.len() != modules.len() {
        let missing: Vec<&str> = modules
            .iter()
            .filter(|m| !seen.contains(&m.name))
            .map(|m| m.name.as_str())
            .collect();
        return Err(PrismError::Type(format!(
            "system block omits modules: {}",
            missing.join(", ")
        )));
    }
    let mut sets = sets.into_iter();
    let Some(first) = sets.next() else {
        return Ok(SyncConfig::default());
    };
    if sets.any(|s| s != first) {
        return Err(PrismError::Unsupported(
            "nested system composition with different restrictions is not supported; use one top-level `||` or `|[...]|`".to_owned(),
        ));
    }
    Ok(SyncConfig {
        restricted: Some(first),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_commands_formulas_and_functions() {
        let m = parse_prism(
            "ctmc
             const double beta = 0.3;
             const int N = 10;
             formula spreading = s > 0 & i > 0;
             module sir
               s : [0..N] init 9;
               i : [0..N] init 1;
               flag : bool init false;
               [infect] spreading -> beta*s : (s'=s-1) & (i'=i+1) & (flag'=true);
               [] flag -> 1 : true;
             endmodule",
        )
        .unwrap();
        assert_eq!(m.kind, ModelKind::Ctmc);
        assert_eq!(m.vars.len(), 3);
        assert_eq!(m.modules[0].commands.len(), 2);
        // `N` folded into the bound, `beta` into the weight.
        assert!(matches!(m.vars[0].typ, VarType::Int { lo: 0, hi: 10 }));
        assert_eq!(m.initial(), vec![9, 1, 0]);
    }

    #[test]
    fn reports_positions_cycles_and_keywords() {
        let e = parse_prism("module m x : [0..1] init 0; endmodule").unwrap_err();
        assert!(
            e.to_string().contains("expected `dtmc` or `ctmc` first"),
            "got {e}"
        );
        let e = parse_prism("mdp module m x : [0..1] init 0; endmodule").unwrap_err();
        assert!(e.to_string().contains("mdp"));
        let e = parse_prism(
            "dtmc formula a = b; formula b = a; module m x : [0..1] init 0; [t] a -> 1 : (x'=1); endmodule",
        )
        .unwrap_err();
        assert!(matches!(e, PrismError::CyclicFormula(_)));
        let e = parse_prism("dtmc module m x : [0..1] init 0; [t] x -> 1 : (y'=1); endmodule")
            .unwrap_err();
        assert!(matches!(e, PrismError::Unknown(_)));
        let e = parse_prism("ctmc module m x : [0..1] init 5; endmodule").unwrap_err();
        assert!(matches!(e, PrismError::OutOfBounds { .. }));
    }

    fn query_model() -> PrismModel {
        parse_prism(
            "ctmc module m x : [0..2] init 0; y : [0..1] init 1;
             [] x=0 -> 1 : (x'=1); endmodule",
        )
        .unwrap()
    }

    #[test]
    fn parses_probability_and_steady_state_queries() {
        let m = query_model();
        let q = parse_query(&m, "P=? [F x=2]").unwrap();
        assert!(matches!(
            q,
            Query::Prob(ProbQuery {
                threshold: None,
                until: Until::Eventually,
                ..
            })
        ));
        let q = parse_query(&m, "P>=0.9 [F<=10 x=2 & y=1]").unwrap();
        match q {
            Query::Prob(ProbQuery {
                threshold: Some((Cmp::Ge, b)),
                until: Until::Bounded(t),
                ..
            }) => {
                assert!((b - 0.9).abs() < 1e-12);
                assert!((t - 10.0).abs() < 1e-12);
            }
            other => panic!("unexpected query {other:?}"),
        }
        let q = parse_query(&m, "S<0.01 [x=2]").unwrap();
        assert!(matches!(
            q,
            Query::Steady(SteadyQuery {
                threshold: Some((Cmp::Lt, _)),
                ..
            })
        ));
        // Unknown names and bad shapes error with guidance.
        let e = parse_query(&m, "P=? [F z=0]").unwrap_err();
        assert!(matches!(e, PrismError::Unknown(_)));
        assert!(parse_query(&m, "P=0.5 [F x=1]").is_err());
        assert!(parse_query(&m, "P=? [G x=1]").is_err());
        assert!(parse_query(&m, "P=? [F x=1]").is_ok());
    }
}
