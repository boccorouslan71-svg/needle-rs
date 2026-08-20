//! Grammar-constrained decoding on the v2 path, against real weights.
//!
//! Requires `weights/needle2.cact`; skips with a notice otherwise.
//!
//! The interesting cases are tools whose names the model would not guess: an
//! unconstrained run is free to invent a plausible name, a constrained run is
//! not. Checking only well-known tools would pass whether the constraint worked
//! or not.
//!
//! Run: cargo test -p needle-infer --release --test v2_constrained -- --nocapture

use needle_infer::v2_engine::{GenerateOptions, V2Engine};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");

fn engine() -> Option<V2Engine> {
    if !std::path::Path::new(CACT).exists() {
        eprintln!("skipping v2 constrained tests: missing {CACT}");
        return None;
    }
    Some(V2Engine::load(CACT).expect("load"))
}

fn opts(constrain: bool) -> GenerateOptions {
    GenerateOptions { max_new_tokens: 80, constrain, ..Default::default() }
}

/// Pull every `"name":"..."` out of a tool-call payload.
fn names(payload: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = payload;
    while let Some(i) = rest.find("\"name\":\"") {
        rest = &rest[i + 8..];
        match rest.find('"') {
            Some(j) => {
                out.push(rest[..j].to_string());
                rest = &rest[j..];
            }
            None => break,
        }
    }
    out
}

/// Pull the keys of the `arguments` object.
fn arg_keys(payload: &str) -> Vec<String> {
    let Some(i) = payload.find("\"arguments\":{") else { return Vec::new() };
    let mut rest = &payload[i + 13..];
    let mut out = Vec::new();
    // Keys are the quoted strings immediately after `{` or `,` at this depth.
    let mut depth = 0i32;
    let bytes = rest.as_bytes();
    let mut k = 0usize;
    let mut expect_key = true;
    while k < bytes.len() {
        match bytes[k] {
            b'{' | b'[' => {
                depth += 1;
                expect_key = false;
            }
            b'}' | b']' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            b',' if depth == 0 => expect_key = true,
            b'"' if expect_key && depth == 0 => {
                let start = k + 1;
                let end = rest[start..].find('"').map(|e| start + e);
                if let Some(end) = end {
                    out.push(rest[start..end].to_string());
                    k = end;
                }
                expect_key = false;
            }
            b':' if depth == 0 => expect_key = false,
            _ => {}
        }
        k += 1;
    }
    let _ = &mut rest;
    out
}

/// Tools with names and argument keys the model cannot have memorised, but whose
/// descriptions are clear enough that it still makes the call. This is the case
/// that discriminates: a constrained run cannot invent a name, an unconstrained
/// one can.
const ODD_TOOLS: &str = r#"[{"name":"qq_weather_lookup","description":"Get current weather for a city","parameters":{"type":"object","properties":{"wibble_place":{"type":"string"},"city":{"type":"string"}},"required":["wibble_place"]}}]"#;

/// A tool whose *name* gives no hint of its purpose. The model reads the name as
/// well as the description and, unaided, decides no suitable tool exists —
/// emitting an empty call. Kept as a named constant because that behaviour is
/// asserted below rather than worked around.
const OPAQUE_TOOLS: &str = r#"[{"name":"zx_quibble_frobnicate","description":"Look up the weather for a city","parameters":{"type":"object","properties":{"wibble_place":{"type":"string"}},"required":["wibble_place"]}}]"#;

#[test]
fn constrained_names_come_from_the_declared_tools() {
    let Some(e) = engine() else { return };
    let r = e.generate("What's the weather in Paris?", ODD_TOOLS, &opts(true), |_, _| {});
    let payload = r.tool_call.clone().unwrap_or_default();
    eprintln!("constrained payload: {payload}");
    let got = names(&payload);
    assert!(!got.is_empty(), "no tool name emitted; text was {:?}", r.text);
    for n in &got {
        assert_eq!(
            n, "qq_weather_lookup",
            "constrained decoding emitted an undeclared name {n:?}"
        );
    }
}

/// Declining to call anything is a valid output, and the constraint must not
/// turn it into a forced call: the grammar restricts *which* name may be written,
/// not whether one is written at all.
#[test]
fn an_unrecognised_tool_may_still_yield_an_empty_call() {
    let Some(e) = engine() else { return };
    let free = e.generate("What's the weather in Paris?", OPAQUE_TOOLS, &opts(false), |_, _| {});
    let bound = e.generate("What's the weather in Paris?", OPAQUE_TOOLS, &opts(true), |_, _| {});
    eprintln!("unconstrained: {:?}", free.text);
    eprintln!("constrained:   {:?}", bound.text);
    // Whatever it decides, the constraint must not change it, and any name it
    // does emit must be the declared one.
    for n in names(&bound.tool_call.clone().unwrap_or_default()) {
        assert_eq!(n, "zx_quibble_frobnicate");
    }
    assert_eq!(bound.text, free.text, "the constraint altered a no-call decision");
}

/// The point of the constraint: the unconstrained run is free to invent a name,
/// so if both runs are already correct the test above proves nothing. This
/// records which happened rather than asserting the model misbehaves.
#[test]
fn constraint_is_what_forces_the_odd_name() {
    let Some(e) = engine() else { return };
    let free = e.generate("What's the weather in Paris?", ODD_TOOLS, &opts(false), |_, _| {});
    let bound = e.generate("What's the weather in Paris?", ODD_TOOLS, &opts(true), |_, _| {});
    let free_names = names(&free.tool_call.clone().unwrap_or_default());
    let bound_names = names(&bound.tool_call.clone().unwrap_or_default());
    eprintln!("unconstrained names: {free_names:?}");
    eprintln!("constrained   names: {bound_names:?}");

    // Whatever the unconstrained run did, the constrained one must be valid.
    for n in &bound_names {
        assert_eq!(n, "qq_weather_lookup");
    }
    if free_names.iter().any(|n| n != "qq_weather_lookup") {
        eprintln!("  -> the constraint corrected an invalid name");
    } else {
        eprintln!("  -> the model already produced a valid name unaided");
    }
}

#[test]
fn constrained_argument_keys_come_from_the_schema() {
    let Some(e) = engine() else { return };
    let r = e.generate("What's the weather in Berlin?", ODD_TOOLS, &opts(true), |_, _| {});
    let payload = r.tool_call.clone().unwrap_or_default();
    eprintln!("payload: {payload}");
    let keys = arg_keys(&payload);
    eprintln!("argument keys: {keys:?}");
    assert!(!keys.is_empty(), "no argument keys emitted from {payload}");
    for k in keys {
        assert!(
            k == "wibble_place" || k == "city",
            "undeclared argument key {k:?} in {payload}"
        );
    }
}

/// No key may repeat while another declared key is still unused.
///
/// This is the guarantee the grammar can give. Once *every* declared key has been
/// written, the model has already emitted the `,` that commits it to another key,
/// and there is no legal continuation left to steer it toward — closing the
/// object would have to be forced one token earlier, at the value boundary, which
/// the state machine does not model. So a payload can still end with a repeat
/// after exhausting the schema; it cannot repeat before that.
/// `unique_arg_keys_excludes_a_written_key` pins the mechanism directly.
#[test]
fn no_key_repeats_while_another_remains_unused() {
    let Some(e) = engine() else { return };
    let r = e.generate("What's the weather in Berlin?", ODD_TOOLS, &opts(true), |_, _| {});
    let payload = r.tool_call.clone().unwrap_or_default();
    let keys = arg_keys(&payload);
    eprintln!("payload: {payload}");
    eprintln!("keys:    {keys:?}");

    // Walk the keys; a repeat is only tolerated once the schema is exhausted.
    let declared = ["wibble_place", "city"];
    let mut used: Vec<&str> = Vec::new();
    for k in &keys {
        let all_used = declared.iter().all(|d| used.contains(d));
        if used.iter().any(|u| u == k) {
            assert!(
                all_used,
                "key {k:?} repeated while {:?} was still unused, in {payload}",
                declared.iter().find(|d| !used.contains(*d))
            );
        }
        if let Some(d) = declared.iter().find(|d| **d == k.as_str()) {
            if !used.contains(d) {
                used.push(d);
            }
        }
    }
}

/// Constraining must not change the answer when the model is already right —
/// the mask only ever removes tokens the grammar forbids.
#[test]
fn constraint_is_inert_on_conventional_tools() {
    let Some(e) = engine() else { return };
    const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
    for q in ["What's the weather in Paris?", "Is it raining in Tokyo?"] {
        let free = e.generate(q, TOOLS, &opts(false), |_, _| {});
        let bound = e.generate(q, TOOLS, &opts(true), |_, _| {});
        assert_eq!(bound.text, free.text, "constraint changed the output for {q:?}");
    }
}

/// Multiple tools: the name must still be one of them, and matched to the query.
#[test]
fn constrained_selection_across_several_tools() {
    let Some(e) = engine() else { return };
    const TOOLS: &str = r#"[{"name":"qq_weather_lookup","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"qq_mail_dispatch","description":"Send an email to a recipient","parameters":{"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}]"#;
    let valid = ["qq_weather_lookup", "qq_mail_dispatch"];
    for q in [
        "What's the weather in Paris?",
        "Email bob@example.com that the build passed",
    ] {
        let r = e.generate(q, TOOLS, &opts(true), |_, _| {});
        let payload = r.tool_call.clone().unwrap_or_default();
        let got = names(&payload);
        eprintln!("{q:?} -> {got:?}");
        assert!(!got.is_empty(), "no name for {q:?}; text {:?}", r.text);
        for n in &got {
            assert!(valid.contains(&n.as_str()), "undeclared name {n:?} for {q:?}");
        }
    }
}

/// Constrained output must still be parseable JSON-ish and terminate.
#[test]
fn constrained_output_terminates_and_is_well_formed() {
    let Some(e) = engine() else { return };
    let r = e.generate("Weather in Cairo?", ODD_TOOLS, &opts(true), |_, _| {});
    assert!(r.stopped_naturally(), "did not terminate: {:?}", r.stop_reason);
    let payload = r.tool_call.expect("tool call present");
    let opens = payload.bytes().filter(|&b| b == b'{').count();
    let closes = payload.bytes().filter(|&b| b == b'}').count();
    assert_eq!(opens, closes, "unbalanced braces in {payload}");
    assert!(payload.starts_with('['), "payload should be an array: {payload}");
}

/// An empty tool list must disable the constraint rather than forbid everything.
#[test]
fn empty_tool_list_leaves_decoding_free() {
    let Some(e) = engine() else { return };
    let bound = e.generate("What's the weather in Paris?", "[]", &opts(true), |_, _| {});
    let free = e.generate("What's the weather in Paris?", "[]", &opts(false), |_, _| {});
    assert_eq!(bound.text, free.text);
}
