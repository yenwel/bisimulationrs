//! Comparison up to causality through the API and the JSON protocol.

use bisim_api::{Options, causal_compare, compare, handle_json};

/// Two workers, one per task, working independently.
const TWO_WORKERS: &str =
    "Waterer = (water, 1).Waterer; Weeder = (weed, 1).Weeder; Waterer || Weeder";
/// One farmer doing both tasks, one after the other.
const ONE_FARMER: &str = "Farmer = (water, 1).Farmer + (weed, 1).Farmer; Farmer";

#[test]
fn rates_and_sequences_agree_but_causality_does_not() {
    let o = Options::default();
    // Even the rates agree: the Markovian comparison calls them equal.
    let ordinary = compare(TWO_WORKERS, ONE_FARMER, o).unwrap();
    assert!(ordinary.markovian_bisimilar.holds, "{ordinary:?}");
    let c = causal_compare(TWO_WORKERS, ONE_FARMER, o).unwrap();
    assert!(c.interleaving_bisimilar);
    assert_eq!(c.causal, "violated", "{c:?}");
    let w = c.causal_witness.clone().unwrap();
    // Some step of one kind that does not depend on an earlier one.
    assert!(w.contains('∅'), "{w}");
    assert!(c.explanation.contains("same sequences"));
    assert_eq!(c.nets[0].components, 2);
    assert_eq!(c.nets[1].components, 1);
    // A model is causally equal to itself, and to a reordering of its components.
    let same = causal_compare(
        TWO_WORKERS,
        "Waterer = (water, 1).Waterer; Weeder = (weed, 1).Weeder; Weeder || Waterer",
        o,
    )
    .unwrap();
    assert_eq!(same.causal, "holds", "{same:?}");
}

#[test]
fn one_vet_chains_the_healings_two_vets_do_not() {
    let herd = |vets: &str| {
        format!(
            "Healthy = (disease, 0.02).Sick; Sick = (healing, 0.1).Healthy;
             Vet = (healing, infty).Vet; Healthy[2] <healing> ({vets})"
        )
    };
    let c = causal_compare(&herd("Vet"), &herd("Vet || Vet"), Options::default()).unwrap();
    // As sequences of actions a second vet changes nothing ...
    assert!(c.interleaving_bisimilar, "{c:?}");
    // ... but with one vet every healing depends on the previous one.
    assert_eq!(c.causal, "violated", "{c:?}");
}

#[test]
fn json_and_refusals() {
    let req = serde_json::json!({ "op": "causal", "a": TWO_WORKERS, "b": ONE_FARMER });
    let r: serde_json::Value = serde_json::from_str(&handle_json(&req.to_string())).unwrap();
    assert_eq!(r["ok"]["causal"], "violated", "{r}");
    let guild = include_str!("../../../examples/permaculture/guild.pepa");
    let req = serde_json::json!({ "op": "causal", "a": guild, "b": guild });
    let r: serde_json::Value = serde_json::from_str(&handle_json(&req.to_string())).unwrap();
    let msg = r["error"]["a"][0]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("functional rate"), "{r}");
}
