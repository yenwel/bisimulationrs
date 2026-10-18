//! Compare two labelled transition systems and explain the difference.
//!
//! `a.(b + c) + a.b` and `a.(b + c)` simulate each other but are not
//! bisimilar: the first can commit to the `a` that leads to a state with no
//! `c`. The distinguishing formula says exactly that.
//!
//! Run: `cargo run -p bisimulationrs --example compare_systems`

use bisimulationrs::core::NamedBuilder;
use bisimulationrs::refine::compare;

fn main() {
    let (p, _) = NamedBuilder::from_triples([
        ("p", 'a', "p1"),
        ("p", 'a', "p2"),
        ("p1", 'b', "end"),
        ("p1", 'c', "end"),
        ("p2", 'b', "end"),
    ]);
    let (q, _) =
        NamedBuilder::from_triples([("q", 'a', "q1"), ("q1", 'b', "end"), ("q1", 'c', "end")]);

    let v = compare(&p, &q);
    println!("q simulates p:       {:?}", v.right_simulates_left);
    println!("mutually similar:    {:?}", v.mutually_similar);
    println!("bisimilar:           {:?}", v.bisimilar);
    if let bisimulationrs::core::Verdict::Violated { witness } = &v.bisimilar {
        // ⟨a⟩¬⟨c⟩tt: "p can do a and then be unable to do c".
        println!("witness (true in p, false in q): {witness}");
    }
}
