//! Emit Lean certificates for "same behaviour" answers.
//!
//! For a few bundled models, compute the coarsest Markovian bisimulation with
//! `refine_fast` and write it, with the transition system it is about, to
//! `lean/Bisim/Certificates.lean`. The Lean kernel checks every certificate
//! (`decide +kernel`), and `Bisim.Cert.sound` turns an accepted one into a
//! proof that the lumping is a bisimulation (`Bisim.IsBisim`).
//!
//! Rates are scaled to integers by one common power of two, which is exact
//! for floating-point numbers. Before writing anything, the same check runs
//! here in integer arithmetic, so a lumping that only holds up to rounding is
//! reported instead of emitted.
//!
//! Run: `cargo run -p bisimulationrs --example emit_certificates [-- OUT]`
//! (CI regenerates the file and fails if it differs from the committed one).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::core::{Real, Wlts};
use bisimulationrs::refine::refine_fast;

type Label = bisimulationrs::algebra::pepa::Act<String>;

const CYCLE: &str = "r = 1;
P1 = (start, r).P2;
P2 = (run, r).P3;
P3 = (stop, r).P1;

P1 || P1";

const HERD: &str = "r_disease = 0.02;
Healthy = (disease, r_disease).Sick;
Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
Dead = 0;
Vet = (healing, infty).Vet;
Healthy || Healthy || Healthy <healing> Vet";

const CLOCK_A: &str = "A0 = (tick, 1).A1;
A1 = (tock, 1).A0;
A0";

const CLOCK_B: &str = "B0 = (tick, 1).B1;
B1 = (tock, 1).B2;
B2 = (tick, 1).B3;
B3 = (tock, 1).B0;
B0";

/// A transition system with its state names, ready to certify.
struct System {
    lts: Wlts<Label, Real>,
    names: Vec<String>,
}

fn derive(src: &str) -> Result<System, Box<dyn std::error::Error>> {
    let model = parse_pepa(src)?.derive()?;
    let names = model.lts.states().map(|s| model.state_name(s)).collect();
    Ok(System {
        lts: model.lts,
        names,
    })
}

/// Both systems side by side; states of `right` come after those of `left`.
/// Returns the union and the two initial states in it.
fn union(left: &System, right: &System) -> (System, usize, usize) {
    let offset = left.lts.num_states();
    let s = left
        .lts
        .initial()
        .expect("derived models have an initial state");
    let t = right
        .lts
        .initial()
        .expect("derived models have an initial state");
    let names = left
        .names
        .iter()
        .map(|n| format!("left {n}"))
        .chain(right.names.iter().map(|n| format!("right {n}")))
        .collect();
    let lts = left.lts.disjoint_union(&right.lts);
    (System { lts, names }, s.index(), t.index() + offset)
}

/// `x = m · 2^e` with `m` odd; exact for every positive finite `f64`.
fn split(x: f64) -> (u64, i32) {
    assert!(x.is_finite() && x > 0.0, "rates are positive and finite");
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, e) = if exp == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp - 1075)
    };
    let tz = m.trailing_zeros();
    (m >> tz, e + tz as i32)
}

/// A certificate: integer edges, the representative of every state, the
/// labels in use and the representatives of the blocks.
struct Cert {
    n: usize,
    /// `(src, lbl, tgt, w)`.
    edges: Vec<(usize, usize, usize, u128)>,
    labels: Vec<String>,
    used: Vec<usize>,
    rep: Vec<usize>,
    blocks: Vec<usize>,
    /// Every weight is its rate times `2^scale`.
    scale: i32,
}

impl Cert {
    fn new(sys: &Wlts<Label, Real>) -> Result<Self, String> {
        let raw: Vec<_> = sys
            .transitions()
            .map(|(s, l, t, w)| {
                let lbl = sys.label_id(l).expect("label of an edge").index();
                (s.index(), lbl, t.index(), split(w.0))
            })
            .collect();
        let low = raw.iter().map(|e| e.3.1).min().unwrap_or(0);
        let mut edges = Vec::with_capacity(raw.len());
        for (s, a, t, (m, e)) in raw {
            let shift = (e - low) as u32;
            if shift + (64 - m.leading_zeros()) > 127 {
                return Err("rates span too many binary orders for 128-bit weights".into());
            }
            edges.push((s, a, t, u128::from(m) << shift));
        }
        let mut used: Vec<usize> = edges.iter().map(|e| e.1).collect();
        used.sort_unstable();
        used.dedup();

        let block_of = refine_fast(sys, None);
        let mut first = vec![usize::MAX; block_of.num_blocks()];
        for (s, &b) in block_of.as_slice().iter().enumerate() {
            first[b as usize] = first[b as usize].min(s);
        }
        let rep: Vec<usize> = block_of
            .as_slice()
            .iter()
            .map(|&b| first[b as usize])
            .collect();
        let mut blocks = first;
        blocks.sort_unstable();
        Ok(Cert {
            n: sys.num_states(),
            edges,
            labels: sys.labels().iter().map(ToString::to_string).collect(),
            used,
            rep,
            blocks,
            scale: -low,
        })
    }

    /// The check `Bisim.Cert.Valid` makes, in exact integer arithmetic.
    /// `None` if it passes, else the first state that differs from its
    /// representative.
    fn first_failure(&self, rep: &[usize]) -> Result<Option<usize>, String> {
        let mut sums = vec![BTreeMap::<(usize, usize), u128>::new(); self.n];
        for &(s, a, t, w) in &self.edges {
            let slot = sums[s].entry((a, rep[t])).or_insert(0);
            *slot = slot.checked_add(w).ok_or("weight sum overflows 128 bits")?;
        }
        Ok((0..self.n).find(|&p| sums[p] != sums[rep[p]]))
    }

    /// The partition with the two blocks of smallest representative merged:
    /// coarser than the coarsest bisimulation, so it must be rejected.
    fn tampered(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        let (&keep, &drop) = (self.blocks.first()?, self.blocks.get(1)?);
        let rep = self
            .rep
            .iter()
            .map(|&r| if r == drop { keep } else { r })
            .collect();
        let blocks = self.blocks.iter().copied().filter(|&k| k != drop).collect();
        Some((rep, blocks))
    }
}

fn list(xs: impl IntoIterator<Item = String>) -> String {
    let mut out = String::from("[");
    let mut width = 0;
    for (i, x) in xs.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
            if width + x.len() > 64 {
                out.push_str("\n   ");
                width = 0;
            }
            out.push(' ');
        }
        width += x.len() + 2;
        out.push_str(&x);
    }
    out.push(']');
    out
}

fn nums(xs: &[usize]) -> String {
    list(xs.iter().map(ToString::to_string))
}

/// What to prove about one system.
enum Claim {
    /// The coarsest bisimulation is a lumping of the model.
    Lumping,
    /// The two initial states of a union are bisimilar.
    Same(usize, usize),
}

fn emit(
    out: &mut String,
    id: &str,
    title: &str,
    sources: &[&str],
    sys: &System,
    claim: Claim,
) -> Result<(), String> {
    let cert = Cert::new(&sys.lts)?;
    if let Some(p) = cert.first_failure(&cert.rep)? {
        return Err(format!(
            "{id}: the lumping holds only up to rounding (state {p} differs from state {})",
            cert.rep[p]
        ));
    }
    if let Claim::Same(s, t) = claim {
        if cert.rep[s] != cert.rep[t] {
            return Err(format!("{id}: states {s} and {t} are not bisimilar"));
        }
    }
    let n = cert.n;

    writeln!(out, "/-! ## {title}\n").unwrap();
    for src in sources {
        writeln!(out, "```\n{src}\n```\n").unwrap();
    }
    writeln!(
        out,
        "{n} states, {} blocks. Weights are rates × 2^{}.\n",
        cert.blocks.len(),
        cert.scale
    )
    .unwrap();
    let labels: Vec<String> = cert
        .used
        .iter()
        .map(|&a| format!("{a} = {}", cert.labels[a]))
        .collect();
    writeln!(out, "Labels: {}.\n", labels.join(", ")).unwrap();
    writeln!(out, "States:").unwrap();
    for (s, name) in sys.names.iter().enumerate() {
        writeln!(out, "* {s}: `{name}` (block of {})", cert.rep[s]).unwrap();
    }
    writeln!(out, "-/\n").unwrap();

    let edges = cert
        .edges
        .iter()
        .map(|(s, a, t, w)| format!("⟨{s}, {a}, {t}, {w}⟩"))
        .collect::<Vec<_>>();
    writeln!(out, "def {id}Edges : List Edge := [").unwrap();
    for (i, e) in edges.iter().enumerate() {
        let sep = if i + 1 < edges.len() { "," } else { "" };
        writeln!(out, "  {e}{sep}").unwrap();
    }
    writeln!(out, "]\n").unwrap();
    writeln!(out, "def {id}Rep : List ℕ := {}\n", nums(&cert.rep)).unwrap();
    writeln!(out, "def {id}Labels : List ℕ := {}\n", nums(&cert.used)).unwrap();
    writeln!(out, "def {id}Blocks : List ℕ := {}\n", nums(&cert.blocks)).unwrap();
    writeln!(
        out,
        "theorem {id}_valid :\n    Valid {n} {id}Edges (lookup {id}Rep) {id}Labels {id}Blocks := by\n  decide +kernel\n"
    )
    .unwrap();
    match claim {
        Claim::Lumping => writeln!(
            out,
            "/-- The library's lumping of this model is a bisimulation. -/\n\
             theorem {id}_lumping : IsBisim (Rel {n} (lookup {id}Rep)) (toWLTS {n} {id}Edges) :=\n  \
             sound _ _ _ _ _ {id}_valid\n"
        )
        .unwrap(),
        Claim::Same(s, t) => writeln!(
            out,
            "/-- The two initial states, {s} and {t}, are bisimilar. -/\n\
             theorem {id}_bisimilar : ∃ (E : Fin {n} → Fin {n} → Prop) (_ : DecidableRel E),\n    \
             IsBisim E (toWLTS {n} {id}Edges) ∧ E {s} {t} :=\n  \
             bisimilar_of_valid _ _ _ _ _ {id}_valid {s} {t} (by decide +kernel)\n"
        )
        .unwrap(),
    }

    if let Some((rep, blocks)) = cert.tampered() {
        if cert.first_failure(&rep)?.is_none() {
            return Err(format!("{id}: merging two blocks still passed the check"));
        }
        writeln!(
            out,
            "/-- Merging blocks {} and {} is rejected: the checker is not a rubber stamp. -/\n\
             example : ¬ Valid {n} {id}Edges (lookup {})\n    {id}Labels {} := by\n  decide +kernel\n",
            cert.blocks[0],
            cert.blocks[1],
            nums(&rep),
            nums(&blocks)
        )
        .unwrap();
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../lean/Bisim/Certificates.lean"
        )
        .into()
    });

    let mut out = String::from(
        "import Bisim.Cert\n\n\
         /-!\n\
         # Checked certificates for bundled models\n\n\
         Generated by `cargo run -p bisimulationrs --example emit_certificates`; do not edit.\n\n\
         Each section is one answer of the library (\"these states behave the same\") with its\n\
         evidence: the derived transition system, with rates scaled exactly to natural numbers,\n\
         and the partition `refine_fast` computed, as a representative for every state.\n\
         `decide +kernel` checks the certificate and `Bisim.Cert.sound` makes it a theorem.\n\
         -/\n\n\
         namespace Bisim.Certificates\n\n\
         open Bisim.Cert\n\n",
    );

    let cycle = derive(CYCLE)?;
    emit(
        &mut out,
        "cycle",
        "Two machines in a start-run-stop cycle: lumping",
        &[CYCLE],
        &cycle,
        Claim::Lumping,
    )?;

    let herd = derive(HERD)?;
    emit(
        &mut out,
        "herd",
        "Three animals and one vet: lumping",
        &[HERD],
        &herd,
        Claim::Lumping,
    )?;

    let array_src = CYCLE.replace("P1 || P1", "P1[2]");
    let (both, s, t) = union(&cycle, &derive(&array_src)?);
    emit(
        &mut out,
        "cycleArray",
        "`P1 || P1` and the counted `P1[2]` behave the same",
        &[CYCLE, &array_src],
        &both,
        Claim::Same(s, t),
    )?;

    let (clocks, s, t) = union(&derive(CLOCK_A)?, &derive(CLOCK_B)?);
    emit(
        &mut out,
        "clocks",
        "A two-state and a four-state clock behave the same",
        &[CLOCK_A, CLOCK_B],
        &clocks,
        Claim::Same(s, t),
    )?;

    out.push_str("end Bisim.Certificates\n");
    std::fs::write(&path, out)?;
    println!("wrote {path}");
    Ok(())
}
