//! Exercises the `needle_v2_*` C ABI through the same pointers a C caller uses.
//!
//! Requires `weights/needle2.cact`; the null-argument and lifecycle checks run
//! regardless, since they must not need a model.
//!
//! Run: cargo test -p needle-c --release --test v2_ffi_smoke -- --nocapture

use needle_c::v2::*;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");
const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;

fn have_weights() -> bool {
    if std::path::Path::new(CACT).exists() {
        return true;
    }
    eprintln!("skipping: missing {CACT}");
    false
}

unsafe fn load() -> *mut NeedleV2Handle {
    let p = CString::new(CACT).unwrap();
    let h = needle_v2_load(p.as_ptr());
    assert!(!h.is_null(), "load failed: {}", last_error());
    h
}

fn last_error() -> String {
    unsafe {
        let p = needle_c::needle_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

unsafe fn take(s: *mut c_char) -> String {
    assert!(!s.is_null(), "null string: {}", last_error());
    let out = CStr::from_ptr(s).to_string_lossy().into_owned();
    needle_c::needle_free_str(s);
    out
}

#[test]
fn load_run_and_free() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let q = CString::new("What's the weather in Paris?").unwrap();
        let t = CString::new(TOOLS).unwrap();

        let text = take(needle_v2_run(h, q.as_ptr(), t.as_ptr()));
        assert!(text.contains("get_weather"), "unexpected text: {text}");

        let payload = take(needle_v2_run_json(h, q.as_ptr(), t.as_ptr()));
        assert!(payload.starts_with('['), "payload: {payload}");
        assert!(payload.contains("\"city\""), "payload: {payload}");

        needle_v2_free(h);
    }
}

#[test]
fn load_from_bytes_matches_load_from_path() {
    if !have_weights() {
        return;
    }
    unsafe {
        let bytes = std::fs::read(CACT).unwrap();
        let hb = needle_v2_load_bytes(bytes.as_ptr(), bytes.len());
        assert!(!hb.is_null(), "{}", last_error());
        let hp = load();

        let q = CString::new("Weather in Berlin?").unwrap();
        let t = CString::new(TOOLS).unwrap();
        let a = take(needle_v2_run(hb, q.as_ptr(), t.as_ptr()));
        let b = take(needle_v2_run(hp, q.as_ptr(), t.as_ptr()));
        assert_eq!(a, b);

        needle_v2_free(hb);
        needle_v2_free(hp);
    }
}

#[test]
fn generate_honours_settings() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let q = CString::new("What's the weather in Paris?").unwrap();
        let t = CString::new(TOOLS).unwrap();

        // Greedy is deterministic.
        let a = take(needle_v2_generate(h, q.as_ptr(), t.as_ptr(), 48, 0.0, 0, 0));
        let b = take(needle_v2_generate(h, q.as_ptr(), t.as_ptr(), 48, 0.0, 0, 0));
        assert_eq!(a, b);

        // Constrained must still be valid.
        let c = take(needle_v2_generate(h, q.as_ptr(), t.as_ptr(), 48, 0.0, 0, 1));
        assert!(c.contains("get_weather"), "{c}");

        // Same seed, same sample.
        let s1 = take(needle_v2_generate(h, q.as_ptr(), t.as_ptr(), 32, 0.8, 7, 0));
        let s2 = take(needle_v2_generate(h, q.as_ptr(), t.as_ptr(), 32, 0.8, 7, 0));
        assert_eq!(s1, s2, "sampling should be seed-deterministic");

        needle_v2_free(h);
    }
}

/// Counts callback invocations and reassembles the stream.
struct Sink {
    pieces: String,
    calls: usize,
}

unsafe extern "C" fn collect(_id: u32, piece: *const c_char, ud: *mut c_void) {
    let sink = &mut *(ud as *mut Sink);
    sink.calls += 1;
    if !piece.is_null() {
        sink.pieces.push_str(&CStr::from_ptr(piece).to_string_lossy());
    }
}

#[test]
fn streaming_callback_reassembles_the_output() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let q = CString::new("What's the weather in Rome?").unwrap();
        let t = CString::new(TOOLS).unwrap();
        let mut sink = Sink { pieces: String::new(), calls: 0 };
        let text = take(needle_v2_run_stream(
            h,
            q.as_ptr(),
            t.as_ptr(),
            Some(collect),
            &mut sink as *mut Sink as *mut c_void,
        ));
        assert!(sink.calls > 0, "callback never fired");
        assert_eq!(sink.pieces, text, "streamed pieces should rebuild the text");
        needle_v2_free(h);
    }
}

/// A null callback must be tolerated, not dereferenced.
#[test]
fn streaming_without_a_callback_still_returns() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let q = CString::new("Weather in Oslo?").unwrap();
        let t = CString::new(TOOLS).unwrap();
        let text = take(needle_v2_run_stream(h, q.as_ptr(), t.as_ptr(), None, ptr::null_mut()));
        assert!(!text.is_empty());
        needle_v2_free(h);
    }
}

#[test]
fn heads_are_reachable_through_the_abi() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let dim = needle_v2_contrastive_dim(h);
        assert!(dim > 0, "expected a contrastive head");

        let text = CString::new("get_weather: Get current weather for a city").unwrap();
        let mut emb = vec![0.0f32; dim];
        assert!(needle_v2_encode_contrastive(h, text.as_ptr(), emb.as_mut_ptr(), dim));
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "not unit norm: {norm}");

        // Too small a buffer must fail rather than overflow.
        let mut tiny = vec![0.0f32; dim - 1];
        assert!(!needle_v2_encode_contrastive(h, text.as_ptr(), tiny.as_mut_ptr(), dim - 1));
        assert!(last_error().contains("too small"), "{}", last_error());

        let mut conf = 0.0f32;
        assert!(needle_v2_confidence(h, text.as_ptr(), &mut conf));
        assert!(conf.is_finite());

        needle_v2_free(h);
    }
}

#[test]
fn retrieve_tools_writes_ranked_results() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let q = CString::new("what is the weather in Paris").unwrap();
        let descs: Vec<CString> = [
            "get_weather: Get current weather for a city",
            "send_email: Send an email to a recipient",
            "play_music: Play a song",
        ]
        .iter()
        .map(|s| CString::new(*s).unwrap())
        .collect();
        let ptrs: Vec<*const c_char> = descs.iter().map(|c| c.as_ptr()).collect();

        let mut idx = vec![usize::MAX; 3];
        let mut scores = vec![0.0f32; 3];
        let n = needle_v2_retrieve_tools(
            h,
            q.as_ptr(),
            ptrs.as_ptr(),
            ptrs.len(),
            3,
            idx.as_mut_ptr(),
            scores.as_mut_ptr(),
        );
        assert_eq!(n, 3);
        assert_eq!(idx[0], 0, "weather query should rank get_weather first: {idx:?}");
        assert!(scores[0] >= scores[1] && scores[1] >= scores[2], "{scores:?}");
        needle_v2_free(h);
    }
}

/// Every entry point must reject null inputs without dereferencing them. Runs
/// even without weights, because that is exactly when a caller passes null.
#[test]
fn null_arguments_are_rejected() {
    unsafe {
        assert!(needle_v2_load(ptr::null()).is_null());
        assert!(needle_v2_load_bytes(ptr::null(), 0).is_null());

        let q = CString::new("q").unwrap();
        let t = CString::new("[]").unwrap();
        assert!(needle_v2_run(ptr::null_mut(), q.as_ptr(), t.as_ptr()).is_null());
        assert!(needle_v2_run_json(ptr::null_mut(), q.as_ptr(), t.as_ptr()).is_null());
        assert!(needle_v2_generate(ptr::null_mut(), q.as_ptr(), t.as_ptr(), 8, 0.0, 0, 0).is_null());
        assert!(needle_v2_run_stream(ptr::null_mut(), q.as_ptr(), t.as_ptr(), None, ptr::null_mut())
            .is_null());
        assert_eq!(needle_v2_contrastive_dim(ptr::null_mut()), 0);

        let mut f = 0.0f32;
        assert!(!needle_v2_confidence(ptr::null_mut(), q.as_ptr(), &mut f));
        assert!(!needle_v2_encode_contrastive(ptr::null_mut(), q.as_ptr(), &mut f, 1));
        assert_eq!(
            needle_v2_retrieve_tools(
                ptr::null_mut(),
                q.as_ptr(),
                ptr::null(),
                0,
                1,
                ptr::null_mut(),
                ptr::null_mut()
            ),
            0
        );

        // Freeing null is a no-op, as C callers expect.
        needle_v2_free(ptr::null_mut());
    }
}
