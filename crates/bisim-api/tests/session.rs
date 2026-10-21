//! Sessions through the JSON protocol: open, push, close, caps and errors.

use bisim_api::{MAX_SESSIONS, handle_json};
use serde_json::{Value, json};

/// The session registry is shared by the process, and one test fills it up:
/// the tests in this file take turns.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn call(request: Value) -> Value {
    serde_json::from_str(&handle_json(&request.to_string())).unwrap()
}

fn open(spec: Value) -> u64 {
    let r = call(json!({ "op": "session_open", "session": spec }));
    r["ok"]["id"].as_u64().unwrap_or_else(|| panic!("{r}"))
}

#[test]
fn prefix_session_reports_changes_and_rejects_late_edges() {
    let _turn = serial();
    let id = open(json!({ "kind": "prefix", "left": "p0", "right": "q0" }));
    let push = |events: Value| call(json!({ "op": "session_push", "id": id, "events": events }));
    let r = push(
        json!([{ "left": ["p0", "a", "p1"] }, { "complete_left": "p0" }, { "complete_left": "p1" }]),
    );
    assert_eq!(
        r["ok"]["verdict"]["bisimilar"]["verdict"], "holds_so_far",
        "{r}"
    );
    let r = push(json!([{ "complete_right": "q0" }]));
    assert_eq!(r["ok"]["changed"], true);
    assert_eq!(
        r["ok"]["verdict"]["bisimilar"]["verdict"], "violated",
        "{r}"
    );
    assert_eq!(r["ok"]["verdict"]["bisimilar"]["witness"], "⟨a⟩tt");
    let r = push(json!([{ "right": ["q0", "a", "q1"] }]));
    assert!(
        r["error"][0]["message"]
            .as_str()
            .unwrap()
            .contains("declared complete"),
        "{r}"
    );
    assert_eq!(
        call(json!({ "op": "session_close", "id": id }))["ok"]["closed"],
        id
    );
    let r = push(json!([]));
    assert!(
        r["error"][0]["message"]
            .as_str()
            .unwrap()
            .contains("no open session"),
        "{r}"
    );
}

#[test]
fn observe_session_matches_the_one_shot_check() {
    let _turn = serial();
    let source = "Tomato = (grow, 0.1).Big + (lose, 0.01).Lost; Big = (lose, 0.01).Lost; Lost = 0; Tomato[4]";
    let timeline = json!({ "start": "15/05/2026", "unit": "day" });
    let obs = json!([
        { "date": "25/05/2026", "state": "Big", "count": 2 },
        { "date": "14/06/2026", "state": "Lost", "count": 1 }
    ]);
    let once = call(
        json!({ "op": "observe", "source": source, "timeline": timeline, "observations": obs }),
    );
    let id = open(json!({ "kind": "observe", "source": source, "timeline": timeline }));
    call(json!({ "op": "session_push", "id": id, "events": [obs[0]] }));
    let last = call(json!({ "op": "session_push", "id": id, "events": [obs[1]] }));
    assert_eq!(
        last["ok"]["explanation"], once["ok"]["explanation"],
        "{last}"
    );
    assert_eq!(last["ok"]["rows"].as_array().unwrap().len(), 2);
    call(json!({ "op": "session_close", "id": id }));
}

#[test]
fn stream_session_accumulates_steps() {
    let _turn = serial();
    let id = open(json!({
        "kind": "stream_distance", "states_a": ["ok", "sick"], "states_b": ["ok", "sick"],
        "values": { "ok": 0.0, "sick": 1.0 }, "gamma": 0.8
    }));
    let step = json!({ "a": ["ok", "sick"], "b": ["ok", "ok"] });
    let r = call(json!({ "op": "session_push", "id": id, "events": [step, step] }));
    assert_eq!(r["ok"]["steps"], 2, "{r}");
    let bad = call(
        json!({ "op": "session_push", "id": id, "events": [{ "a": ["ok", "nope"], "b": ["ok", "ok"] }] }),
    );
    assert!(
        bad["error"][0]["message"]
            .as_str()
            .unwrap()
            .contains("nope")
    );
    let r = call(json!({ "op": "session_push", "id": id, "events": [step] }));
    assert_eq!(r["ok"]["steps"], 3, "a rejected push adds nothing: {r}");
    call(json!({ "op": "session_close", "id": id }));
}

#[test]
fn the_number_of_sessions_is_capped() {
    let _turn = serial();
    // Other tests may hold a few sessions; fill up to the cap and one more.
    let mut ids = Vec::new();
    let mut refused = false;
    for _ in 0..=MAX_SESSIONS {
        let r = call(
            json!({ "op": "session_open", "session": { "kind": "prefix", "left": "a", "right": "b" } }),
        );
        match r["ok"]["id"].as_u64() {
            Some(id) => ids.push(id),
            None => {
                assert!(
                    r["error"][0]["message"]
                        .as_str()
                        .unwrap()
                        .contains("close one")
                );
                refused = true;
                break;
            }
        }
    }
    assert!(refused);
    for id in ids {
        call(json!({ "op": "session_close", "id": id }));
    }
}
