//! Mutation fuzzing on stable: mangle known-good models and check that
//! parsing and deriving never panic (errors are fine).

use std::panic::{AssertUnwindSafe, catch_unwind};

use bisim_algebra::jani::parse_jani;
use bisim_algebra::pepa::{DeriveOptions, parse_pepa};
use bisim_algebra::prism::{parse_prism, parse_query};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const PIECES: &[&str] = &[
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "|",
    "||",
    "|[",
    "]|",
    "+",
    "-",
    "*",
    "/",
    "%",
    ",",
    ";",
    ":",
    ".",
    "=",
    "'",
    "#",
    "?",
    "!",
    "&",
    "0",
    "1",
    "2.5",
    "1e400",
    "-1",
    "4294967295",
    "infty",
    "T",
    "P",
    "Q",
    "s",
    "x",
    "true",
    "false",
    "min",
    "max",
    "pow",
    "floor",
    "ceil",
    "mod",
    "ctmc",
    "dtmc",
    "module",
    "endmodule",
    "rewards",
    "endrewards",
    "system",
    "endsystem",
    "formula",
    "const",
    "int",
    "double",
    "bool",
    "init",
    "label",
    "\"",
    "é",
    "∞",
    "\u{0}",
    "\n",
];

fn mutate(rng: &mut Rng, seed: &str) -> String {
    let mut chars: Vec<char> = seed.chars().collect();
    for _ in 0..1 + rng.below(4) {
        if chars.is_empty() {
            chars.push('x');
        }
        let i = rng.below(chars.len());
        match rng.below(6) {
            0 => {
                chars.remove(i);
            }
            1 => {
                let c = chars[i];
                chars.insert(i, c);
            }
            2 => {
                let j = rng.below(chars.len());
                chars.swap(i, j);
            }
            3 => {
                let piece = PIECES[rng.below(PIECES.len())];
                for (k, c) in piece.chars().enumerate() {
                    chars.insert(i + k, c);
                }
            }
            4 => {
                // Duplicate a slice (nesting, repetition).
                let j = (i + 1 + rng.below(20)).min(chars.len());
                let slice: Vec<char> = chars[i..j].to_vec();
                for (k, c) in slice.into_iter().enumerate() {
                    chars.insert(j + k, c);
                }
            }
            _ => chars.truncate(i),
        }
    }
    chars.into_iter().collect()
}

fn fuzz(name: &str, seeds: &[&str], iterations: usize, run: impl Fn(&str)) {
    std::panic::set_hook(Box::new(|_| {}));
    let mut rng = Rng(0xF00D_CAFE ^ name.len() as u64);
    let mut crashes: Vec<String> = Vec::new();
    for i in 0..iterations {
        let seed = seeds[i % seeds.len()];
        let input = mutate(&mut rng, seed);
        if catch_unwind(AssertUnwindSafe(|| run(&input))).is_err() {
            crashes.push(input);
            if crashes.len() >= 3 {
                break;
            }
        }
    }
    let _ = std::panic::take_hook();
    assert!(crashes.is_empty(), "{name} panicked on:\n{crashes:#?}");
}

#[test]
fn pepa_never_panics() {
    let seeds = [
        include_str!("../../../examples/sir/sir.pepa"),
        include_str!("../../../examples/sir/sir_fun.pepa"),
        include_str!("../../../examples/garden/tomatoes.pepa"),
        "r = 1; P1 = (start, 2 * r).P2; P2 = (run, r).P3 + (x, infty).P1; P3 = (stop, r).P1;\n\
         Q = (run, infty).Q; (P1[2] <run> Q) / {stop}",
    ];
    fuzz("pepa", &seeds, 30_000, |src| {
        if let Ok(m) = parse_pepa(src) {
            let _ = m.to_string();
            let _ = m.derive_with(DeriveOptions { max_states: 300 });
            if let Ok(f) = m.fluid_model() {
                let _ = f.fluid_drift(&f.initial_counts());
            }
        }
    });
}

const QUERY_MODEL: &str = "ctmc module m s : [0..2] init 0; [] s<2 -> 1 : (s'=s+1); endmodule";

#[test]
fn prism_never_panics() {
    let seeds = [
        include_str!("../../../examples/sir/sir.prism"),
        "ctmc const double r = 2; formula g = x < 3;\n\
         module a x : [0..3] init 0; [go] g -> r : (x'=x+1); [] x=3 -> 1 : (x'=0); endmodule\n\
         module b y : [0..1] init 0; [go] y=0 -> 1.5 : (y'=1); endmodule\n\
         rewards \"n\" x=1 : 2; [go] true : 1; endrewards",
        "P>=0.9 [F<=10 s=2]",
        "S=? [s=1]",
        "dtmc module m s : [0..2] init 0; [] s=0 -> 1/2 : (s'=1) + 1/2 : (s'=2); [] s>0 -> 1 : (s'=s); endmodule",
    ];
    fuzz("prism", &seeds, 30_000, |src| {
        if let Ok(m) = parse_prism(src) {
            let _ = m.derive_with(DeriveOptions { max_states: 300 });
        }
        if let Ok(m) = parse_prism(QUERY_MODEL) {
            let _ = parse_query(&m, src);
        }
    });
}

#[test]
fn jani_never_panics() {
    let seeds = [
        r#"{"jani-version":1,"name":"c","type":"ctmc","constants":[{"name":"N","type":"int","value":2}],
           "variables":[{"name":"x","type":{"kind":"bounded","lower-bound":0,"upper-bound":"N"},"initial-value":0}],
           "automata":[{"name":"m","locations":[{"name":"a"},{"name":"b"}],"initial-locations":["a"],
             "edges":[{"location":"a","action":"go","rate":{"exp":1.5},"guard":{"op":"<","left":"x","right":"N"},
               "destinations":[{"location":"b","assignments":[{"ref":"x","value":{"op":"+","left":"x","right":1}}]}]},
              {"location":"b","rate":2.0,"destinations":[{"location":"a"}]}]}],
           "system":{"elements":[{"automaton":"m"}],"syncs":[{"synchronise":["go"],"result":"go"}]}}"#,
    ];
    fuzz("jani", &seeds, 30_000, |src| {
        if let Ok(m) = parse_jani(src) {
            let _ = m.derive_with(DeriveOptions { max_states: 300 });
        }
    });
}
