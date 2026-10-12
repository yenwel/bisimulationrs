//! Parser for the textual PEPA syntax of the PEPA Eclipse plug-in.
//!
//! ```text
//! r = 0.02;                         // rate parameter (lower-case name)
//! P = (a, r).Q + (b, 2 * r).P;      // sequential component (upper-case name)
//! Q = (a, infty).P;                 // passive rate; also `w * infty`
//! Dead = 0;                         // deadlocked / absorbing
//! Sys = P[3] <a> Q;                 // named model component (inlined)
//! Sys / {b}                         // system equation (last, no `=`)
//! ```
//!
//! Cooperation: `P <a, b> Q`, interleaving: `P || Q` or `P <> Q`, hiding
//! `P / {a}` (binds tighter than cooperation), arrays `P[n]`. Comments use
//! `//` and `/* … */`.

use std::collections::HashMap;
use std::fmt;

use super::ast::{ActionSet, ModelTerm, ParamId, PepaModel, RateExpr, Seq, SeqId, Sym};
use super::validate::PepaError;

/// A syntax error with a 1-based position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            // Errors found after parsing have no single source position.
            write!(f, "{}", self.message)
        } else {
            write!(f, "{}:{}: {}", self.line, self.column, self.message)
        }
    }
}

/// Parses and validates a PEPA model.
pub fn parse_pepa(src: &str) -> Result<PepaModel<Sym>, PepaError> {
    let tokens = lex(src)?;
    let raw = Parser { tokens, pos: 0 }.file()?;
    let model = resolve(raw)?;
    model.validate()?;
    Ok(model)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Sym(&'static str),
    Eof,
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    line: usize,
    column: usize,
}

const SYMBOLS: [&str; 15] = [
    "||", "=", ";", "(", ")", ",", ".", "+", "-", "*", "/", "<", ">", "[", "]",
];

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
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '\'')
            {
                advance(&mut i, &mut line, &mut col, 1);
            }
            out.push(Token {
                tok: Tok::Ident(chars[start..i].iter().collect()),
                line: start_line,
                column: start_col,
            });
        } else if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                advance(&mut i, &mut line, &mut col, 1);
            }
            // A '.' is a decimal point only if a digit follows (else: prefix dot).
            if chars.get(i) == Some(&'.') && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
                advance(&mut i, &mut line, &mut col, 1);
                while i < chars.len() && chars[i].is_ascii_digit() {
                    advance(&mut i, &mut line, &mut col, 1);
                }
            }
            if matches!(chars.get(i), Some('e' | 'E')) {
                let sign = usize::from(matches!(chars.get(i + 1), Some('+' | '-')));
                if chars.get(i + 1 + sign).is_some_and(char::is_ascii_digit) {
                    advance(&mut i, &mut line, &mut col, 1 + sign);
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        advance(&mut i, &mut line, &mut col, 1);
                    }
                }
            }
            let text: String = chars[start..i].iter().collect();
            let value = text
                .parse()
                .map_err(|_| err_at(start_line, start_col, "invalid number"))?;
            out.push(Token {
                tok: Tok::Num(value),
                line: start_line,
                column: start_col,
            });
        } else if c == '{' || c == '}' {
            out.push(Token {
                tok: Tok::Sym(if c == '{' { "{" } else { "}" }),
                line: start_line,
                column: start_col,
            });
            advance(&mut i, &mut line, &mut col, 1);
        } else {
            let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
            let Some(sym) = SYMBOLS.iter().find(|s| rest.starts_with(**s)) else {
                return Err(err_at(
                    start_line,
                    start_col,
                    &format!("unexpected character `{c}`"),
                ));
            };
            out.push(Token {
                tok: Tok::Sym(sym),
                line: start_line,
                column: start_col,
            });
            advance(&mut i, &mut line, &mut col, sym.len());
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

/// Rate expression before parameter names are resolved.
#[derive(Debug, Clone)]
enum RawRate {
    Num(f64),
    Name(String, (usize, usize)),
    Infty,
    Bin(char, Box<RawRate>, Box<RawRate>),
}

/// Process expression before names are classified and resolved.
#[derive(Debug, Clone)]
enum Expr {
    Prefix(String, RawRate, Box<Expr>),
    Choice(Vec<Expr>),
    Name(String, (usize, usize)),
    Stop,
    Coop(Box<Expr>, Vec<String>, Box<Expr>),
    Hide(Box<Expr>, Vec<String>),
    Array(String, u32, (usize, usize)),
}

#[derive(Debug)]
struct RawFile {
    params: Vec<(String, RawRate)>,
    defs: Vec<(String, Expr, (usize, usize))>,
    system: Expr,
}

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

    fn here(&self) -> (usize, usize) {
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

    fn file(mut self) -> Result<RawFile, ParseError> {
        let mut params = Vec::new();
        let mut defs = Vec::new();
        let mut system = None;
        while *self.peek() != Tok::Eof {
            let at = self.here();
            if matches!(self.peek(), Tok::Ident(_)) && *self.peek_at(1) == Tok::Sym("=") {
                let name = self.ident()?;
                self.bump();
                if name.starts_with(|c: char| c.is_ascii_lowercase()) {
                    params.push((name, self.rate()?));
                } else {
                    defs.push((name, self.expr()?, at));
                }
                self.expect(";")?;
            } else {
                if system.is_some() {
                    return self.error("only one system equation is allowed");
                }
                system = Some(self.expr()?);
                if !self.eat(";") && *self.peek() != Tok::Eof {
                    return self.error("expected `;` or end of input after the system equation");
                }
            }
        }
        let system = system.ok_or_else(|| {
            let (line, column) = self.here();
            err_at(line, column, "missing system equation")
        })?;
        Ok(RawFile {
            params,
            defs,
            system,
        })
    }

    // expr := hide (('<' actions? '>' | '||') hide)*
    fn expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.hide()?;
        loop {
            let set = if self.eat("||") {
                Vec::new()
            } else if self.eat("<") {
                let set = if *self.peek() == Tok::Sym(">") {
                    Vec::new()
                } else {
                    self.actions()?
                };
                self.expect(">")?;
                set
            } else {
                return Ok(left);
            };
            let right = self.hide()?;
            left = Expr::Coop(Box::new(left), set, Box::new(right));
        }
    }

    // hide := choice ('/' '{' actions '}')*
    fn hide(&mut self) -> Result<Expr, ParseError> {
        let mut e = self.choice()?;
        while self.eat("/") {
            self.expect("{")?;
            let set = self.actions()?;
            self.expect("}")?;
            e = Expr::Hide(Box::new(e), set);
        }
        Ok(e)
    }

    fn actions(&mut self) -> Result<Vec<String>, ParseError> {
        let mut out = vec![self.ident()?];
        while self.eat(",") {
            out.push(self.ident()?);
        }
        Ok(out)
    }

    // choice := prefix ('+' prefix)*
    fn choice(&mut self) -> Result<Expr, ParseError> {
        let first = self.prefix()?;
        if *self.peek() != Tok::Sym("+") {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.eat("+") {
            items.push(self.prefix()?);
        }
        Ok(Expr::Choice(items))
    }

    // prefix := '(' action ',' rate ')' '.' prefix | atom
    fn prefix(&mut self) -> Result<Expr, ParseError> {
        let is_prefix = *self.peek() == Tok::Sym("(")
            && matches!(self.peek_at(1), Tok::Ident(_))
            && *self.peek_at(2) == Tok::Sym(",");
        if !is_prefix {
            return self.atom();
        }
        self.bump();
        let action = self.ident()?;
        self.expect(",")?;
        let rate = self.rate()?;
        self.expect(")")?;
        self.expect(".")?;
        let next = self.prefix()?;
        Ok(Expr::Prefix(action, rate, Box::new(next)))
    }

    // atom := Name ('[' int ']')? | '0' | '(' expr ')'
    fn atom(&mut self) -> Result<Expr, ParseError> {
        let at = self.here();
        match self.peek().clone() {
            Tok::Ident(name) => {
                self.bump();
                if self.eat("[") {
                    let n = match self.bump() {
                        Tok::Num(v) if v >= 0.0 && v.fract() == 0.0 && v <= u32::MAX as f64 => {
                            v as u32
                        }
                        _ => return self.error("expected a non-negative integer array size"),
                    };
                    self.expect("]")?;
                    Ok(Expr::Array(name, n, at))
                } else {
                    Ok(Expr::Name(name, at))
                }
            }
            Tok::Num(0.0) => {
                self.bump();
                Ok(Expr::Stop)
            }
            Tok::Sym("(") => {
                self.bump();
                let e = self.expr()?;
                self.expect(")")?;
                Ok(e)
            }
            _ => self.error("expected a prefix `(action, rate).P`, a component name, or `0`"),
        }
    }

    // rate := term (('+' | '-') term)*
    fn rate(&mut self) -> Result<RawRate, ParseError> {
        let mut left = self.rate_term()?;
        loop {
            let op = if self.eat("+") {
                '+'
            } else if self.eat("-") {
                '-'
            } else {
                return Ok(left);
            };
            left = RawRate::Bin(op, Box::new(left), Box::new(self.rate_term()?));
        }
    }

    fn rate_term(&mut self) -> Result<RawRate, ParseError> {
        let mut left = self.rate_factor()?;
        loop {
            let op = if self.eat("*") {
                '*'
            } else if self.eat("/") {
                '/'
            } else {
                return Ok(left);
            };
            left = RawRate::Bin(op, Box::new(left), Box::new(self.rate_factor()?));
        }
    }

    fn rate_factor(&mut self) -> Result<RawRate, ParseError> {
        let at = self.here();
        match self.bump() {
            Tok::Num(v) => Ok(RawRate::Num(v)),
            Tok::Ident(s) if s == "infty" || s == "T" => Ok(RawRate::Infty),
            Tok::Ident(s) => Ok(RawRate::Name(s, at)),
            Tok::Sym("(") => {
                let r = self.rate()?;
                self.expect(")")?;
                Ok(r)
            }
            _ => Err(err_at(at.0, at.1, "expected a rate")),
        }
    }
}

struct Resolver {
    params: HashMap<String, ParamId>,
    seq_ids: HashMap<String, SeqId>,
    seqs: Vec<(String, Option<Seq<Sym>>)>,
    models: HashMap<String, Expr>,
}

fn resolve(raw: RawFile) -> Result<PepaModel<Sym>, PepaError> {
    let mut r = Resolver {
        params: HashMap::new(),
        seq_ids: HashMap::new(),
        seqs: Vec::new(),
        models: HashMap::new(),
    };
    let mut params = Vec::new();
    for (name, _) in &raw.params {
        r.params.insert(name.clone(), ParamId(params.len() as u32));
        params.push((name.clone(), 0.0));
    }
    for (i, (_, rate)) in raw.params.iter().enumerate() {
        let expr = r.rate(rate)?;
        if matches!(expr, RateExpr::Passive(_)) {
            return Err(PepaError::InvalidRate {
                component: params[i].0.clone(),
                action: "parameter".into(),
            });
        }
        // Parameters may refer to earlier parameters.
        let value =
            expr.eval(&params)
                .map(|r| r.value())
                .ok_or_else(|| PepaError::InvalidRate {
                    component: params[i].0.clone(),
                    action: "parameter".into(),
                })?;
        params[i].1 = value;
    }

    // Classify definitions: model components (cooperation, hiding, arrays)
    // are inlined; everything else is a sequential component.
    let mut is_model: HashMap<String, bool> = raw
        .defs
        .iter()
        .map(|(n, e, _)| (n.clone(), syntactically_model(e)))
        .collect();
    loop {
        let mut changed = false;
        for (name, e, _) in &raw.defs {
            if !is_model[name] && mentions_model(e, &is_model) {
                is_model.insert(name.clone(), true);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (name, e, _) in &raw.defs {
        if is_model[name] {
            r.models.insert(name.clone(), e.clone());
        } else {
            r.seq_id(name);
        }
    }
    for (name, e, _) in &raw.defs {
        if !is_model[name] {
            let body = r.seq(e)?;
            let id = r.seq_ids[name];
            r.seqs[id.0 as usize].1 = Some(body);
        }
    }
    let system = r.term(&raw.system, &mut Vec::new())?;
    Ok(PepaModel {
        params,
        seqs: r.seqs,
        system,
    })
}

fn syntactically_model(e: &Expr) -> bool {
    match e {
        Expr::Coop(..) | Expr::Hide(..) | Expr::Array(..) => true,
        Expr::Prefix(..) | Expr::Name(..) | Expr::Stop => false,
        Expr::Choice(xs) => xs.iter().any(syntactically_model),
    }
}

fn mentions_model(e: &Expr, is_model: &HashMap<String, bool>) -> bool {
    match e {
        Expr::Name(n, _) => is_model.get(n).copied().unwrap_or(false),
        // Behind a prefix only sequential components may follow.
        Expr::Prefix(..) | Expr::Stop => false,
        Expr::Choice(xs) => xs.iter().any(|x| mentions_model(x, is_model)),
        Expr::Coop(..) | Expr::Hide(..) | Expr::Array(..) => true,
    }
}

impl Resolver {
    fn seq_id(&mut self, name: &str) -> SeqId {
        if let Some(&id) = self.seq_ids.get(name) {
            return id;
        }
        let id = SeqId(self.seqs.len() as u32);
        self.seq_ids.insert(name.to_owned(), id);
        self.seqs.push((name.to_owned(), None));
        id
    }

    fn rate(&self, r: &RawRate) -> Result<RateExpr, PepaError> {
        // `infty` may only appear as `infty`, `w * infty` or `infty * w`.
        match r {
            RawRate::Infty => return Ok(RateExpr::Passive(Box::new(RateExpr::Lit(1.0)))),
            RawRate::Bin('*', a, b) if matches!(**b, RawRate::Infty) => {
                return Ok(RateExpr::Passive(Box::new(self.active_rate(a)?)));
            }
            RawRate::Bin('*', a, b) if matches!(**a, RawRate::Infty) => {
                return Ok(RateExpr::Passive(Box::new(self.active_rate(b)?)));
            }
            _ => {}
        }
        self.active_rate(r)
    }

    fn active_rate(&self, r: &RawRate) -> Result<RateExpr, PepaError> {
        Ok(match r {
            RawRate::Num(v) => RateExpr::Lit(*v),
            RawRate::Name(n, (line, column)) => {
                RateExpr::Param(*self.params.get(n).ok_or_else(|| {
                    PepaError::Parse(err_at(
                        *line,
                        *column,
                        &format!("unknown rate parameter `{n}`"),
                    ))
                })?)
            }
            RawRate::Infty => {
                return Err(PepaError::InvalidRate {
                    component: "expression".into(),
                    action: "infty".into(),
                });
            }
            RawRate::Bin(op, a, b) => {
                let (a, b) = (
                    Box::new(self.active_rate(a)?),
                    Box::new(self.active_rate(b)?),
                );
                match op {
                    '+' => RateExpr::Add(a, b),
                    '-' => RateExpr::Sub(a, b),
                    '*' => RateExpr::Mul(a, b),
                    _ => RateExpr::Div(a, b),
                }
            }
        })
    }

    fn seq(&mut self, e: &Expr) -> Result<Seq<Sym>, PepaError> {
        Ok(match e {
            Expr::Prefix(a, rate, next) => {
                Seq::Prefix(a.clone(), self.rate(rate)?, Box::new(self.seq(next)?))
            }
            Expr::Choice(xs) => {
                Seq::Choice(xs.iter().map(|x| self.seq(x)).collect::<Result<_, _>>()?)
            }
            Expr::Name(n, (line, column)) => {
                if self.models.contains_key(n) {
                    return Err(PepaError::Parse(err_at(
                        *line,
                        *column,
                        &format!(
                            "model component `{n}` cannot be used inside a sequential component"
                        ),
                    )));
                }
                Seq::Const(self.seq_id(n))
            }
            Expr::Stop => Seq::Choice(Vec::new()),
            Expr::Coop(..) | Expr::Hide(..) | Expr::Array(..) => {
                return Err(PepaError::Parse(err_at(
                    0,
                    0,
                    "cooperation, hiding and arrays cannot appear inside a sequential component",
                )));
            }
        })
    }

    fn term(&mut self, e: &Expr, stack: &mut Vec<String>) -> Result<ModelTerm<Sym>, PepaError> {
        let set = |xs: &[String]| xs.iter().cloned().collect::<ActionSet<Sym>>();
        Ok(match e {
            Expr::Coop(l, xs, r) => ModelTerm::Coop(
                Box::new(self.term(l, stack)?),
                set(xs),
                Box::new(self.term(r, stack)?),
            ),
            Expr::Hide(x, xs) => ModelTerm::Hide(Box::new(self.term(x, stack)?), set(xs)),
            Expr::Array(n, k, (line, column)) => {
                if self.models.contains_key(n) {
                    return Err(PepaError::Parse(err_at(
                        *line,
                        *column,
                        &format!("arrays of model components (`{n}[{k}]`) are not supported"),
                    )));
                }
                ModelTerm::Array(self.seq_id(n), *k)
            }
            Expr::Name(n, (line, column)) => match self.models.get(n).cloned() {
                Some(body) => {
                    if stack.contains(n) {
                        return Err(PepaError::Parse(err_at(
                            *line,
                            *column,
                            &format!("model component `{n}` is defined in terms of itself"),
                        )));
                    }
                    stack.push(n.clone());
                    let t = self.term(&body, stack)?;
                    stack.pop();
                    t
                }
                None => ModelTerm::Component(self.seq_id(n)),
            },
            Expr::Prefix(..) | Expr::Choice(..) | Expr::Stop => {
                return Err(PepaError::Parse(err_at(
                    0,
                    0,
                    "the system equation must be built from named components; \
                     define the sequential component first",
                )));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cooperation_hiding_and_arrays() {
        let m = parse_pepa(
            "/* toy */ r = 0.5; s = 2 * r;
             P = (a, r).Q + (b, s).P;   // comment
             Q = (a, infty).P;
             R = (a, 1e-1).R;
             Sys = P[2] <a> R;
             Sys / {b}",
        )
        .unwrap();
        assert_eq!(
            m.params().map(|(_, _, v)| v).collect::<Vec<_>>(),
            vec![0.5, 1.0]
        );
        let ModelTerm::Hide(inner, hidden) = m.system() else {
            panic!("expected hiding at the top");
        };
        assert!(hidden.contains("b"));
        assert!(matches!(**inner, ModelTerm::Coop(..)));
    }

    #[test]
    fn reports_positions_and_semantic_errors() {
        let e = parse_pepa("P = (a, r).P;\nP").unwrap_err();
        assert_eq!(e.to_string(), "1:9: unknown rate parameter `r`");

        let e = parse_pepa("P = (a, 1).Q;\nP").unwrap_err();
        assert_eq!(e, PepaError::UndefinedComponent("Q".into()));

        let e = parse_pepa("P = Q;\nQ = P + (a, 1).P;\nP").unwrap_err();
        assert!(matches!(e, PepaError::UnguardedRecursion(_)));

        let e = parse_pepa("P = (a, infty).P;\nP").unwrap_err();
        assert_eq!(e, PepaError::UnsynchronisedPassive("a".into()));

        let e = parse_pepa("P = (a, 1).P;\nP;\nP").unwrap_err();
        assert!(e.to_string().contains("only one system equation"));
    }
}
