//! Compare two systems that are observed piece by piece.
//!
//! A violation is only reported when it can no longer be repaired: the
//! observed move has no answer from a state declared *complete*. Until then
//! the verdict is "holds so far".
//!
//! Run: `cargo run -p bisimulationrs --example prefix_stream`

use bisimulationrs::core::Verdict;
use bisimulationrs::refine::PrefixChecker;

fn show(step: &str, c: &PrefixChecker<&str, char>) {
    let v = c.verdict();
    let word = match &v.bisimilar {
        Verdict::Holds => "holds".to_owned(),
        Verdict::HoldsSoFar => "holds so far".to_owned(),
        Verdict::Violated { witness } => format!("VIOLATED: {witness}"),
        other => format!("{other:?}"),
    };
    println!("{step:<44} bisimilar: {word}");
}

fn main() {
    let mut c = PrefixChecker::new("expected", "actual");
    // The plan: do `a`, then `b`.
    c.observe_left([("expected", 'a', "e1"), ("e1", 'b', "e2")]);
    for s in ["expected", "e1", "e2"] {
        c.complete_left(s);
    }
    show("plan known, nothing observed yet", &c);

    c.observe_right([("actual", 'a', "a1")]);
    c.complete_right("actual");
    show("observed `a`", &c);

    // `a1` is finished without ever doing `b`: the difference is final.
    c.complete_right("a1");
    show("observed that nothing follows `a`", &c);
}
