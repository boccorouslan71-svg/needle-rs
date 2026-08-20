//! C ABI for the Needle v2 path.
//!
//! A `.cact` container carries its weights, geometry and tokenizer, so loading
//! takes one path and there is no vocabulary argument — the v1 entry points are
//! kept unchanged alongside these.
//!
//! ```text
//! needle_v2_load(cact_path)                                  -> *NeedleV2Handle
//! needle_v2_load_bytes(data, len)                            -> *NeedleV2Handle
//! needle_v2_run(h, query, tools_json)                        -> *char
//! needle_v2_run_json(h, query, tools_json)                   -> *char  (payload only)
//! needle_v2_generate(h, query, tools, max, temp, seed, constrain) -> *char
//! needle_v2_run_stream(h, query, tools, cb, userdata)        -> *char
//! needle_v2_encode_contrastive(h, text, out, dim)            -> bool
//! needle_v2_contrastive_dim(h)                               -> usize
//! needle_v2_confidence(h, text, out)                         -> bool
//! needle_v2_confidence_for(h, query, tools, completion, out) -> bool
//! needle_v2_retrieve_tools(h, query, descs, n, k, idx, scores) -> usize
//! needle_v2_free(h)
//! ```
//!
//! Strings are freed with `needle_free_str` and errors read with
//! `needle_last_error`, both shared with the v1 surface.

use crate::{clear_last_error, set_last_error};
use needle_infer::v2_engine::{GenerateOptions, V2Engine};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

/// Opaque handle to a loaded v2 engine.
pub struct NeedleV2Handle {
    engine: V2Engine,
}

unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

fn out_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => {
            set_last_error("result contained a null byte");
            ptr::null_mut()
        }
    }
}

/// Load a v2 model from a `.cact` file. Null on failure; free with
/// `needle_v2_free`.
///
/// # Safety
/// `cact_path` must be a valid, null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_load(cact_path: *const c_char) -> *mut NeedleV2Handle {
    clear_last_error();
    let Some(path) = cstr(cact_path) else {
        set_last_error("cact_path is null or not UTF-8");
        return ptr::null_mut();
    };
    match V2Engine::load(path) {
        Ok(engine) => Box::into_raw(Box::new(NeedleV2Handle { engine })),
        Err(e) => {
            set_last_error(format!("load error: {e}"));
            ptr::null_mut()
        }
    }
}

/// Load a v2 model from an in-memory `.cact` image.
///
/// # Safety
/// `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_load_bytes(
    data: *const u8,
    len: usize,
) -> *mut NeedleV2Handle {
    clear_last_error();
    if data.is_null() {
        set_last_error("data is null");
        return ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data, len).to_vec();
    match V2Engine::from_bytes(bytes) {
        Ok(engine) => Box::into_raw(Box::new(NeedleV2Handle { engine })),
        Err(e) => {
            set_last_error(format!("load error: {e}"));
            ptr::null_mut()
        }
    }
}

unsafe fn handle_ref<'a>(h: *mut NeedleV2Handle) -> Option<&'a V2Engine> {
    if h.is_null() {
        set_last_error("handle is null");
        return None;
    }
    Some(&(*h).engine)
}

/// Greedy tool call. Returns the full decoded text; free with `needle_free_str`.
///
/// # Safety
/// `handle` must come from `needle_v2_load`/`needle_v2_load_bytes`; `query` and
/// `tools_json` must be valid null-terminated UTF-8 C strings.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_run(
    handle: *mut NeedleV2Handle,
    query: *const c_char,
    tools_json: *const c_char,
) -> *mut c_char {
    clear_last_error();
    let (Some(engine), Some(q), Some(t)) =
        (handle_ref(handle), cstr(query), cstr(tools_json))
    else {
        set_last_error("invalid arguments");
        return ptr::null_mut();
    };
    out_string(engine.run(q, t).text)
}

/// As `needle_v2_run`, but returns only the `<tool_call>` payload — `"[]"` when
/// no declared tool fits, which is a deliberate abstention rather than an error.
/// Null (with no error set) only when the output carried no markers at all.
///
/// # Safety
/// As `needle_v2_run`.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_run_json(
    handle: *mut NeedleV2Handle,
    query: *const c_char,
    tools_json: *const c_char,
) -> *mut c_char {
    clear_last_error();
    let (Some(engine), Some(q), Some(t)) =
        (handle_ref(handle), cstr(query), cstr(tools_json))
    else {
        set_last_error("invalid arguments");
        return ptr::null_mut();
    };
    match engine.run(q, t).tool_call {
        Some(tc) => out_string(tc),
        None => ptr::null_mut(),
    }
}

/// Generation with explicit settings. `temperature <= 0` is greedy;
/// `constrain != 0` restricts the tool-call payload to the declared schema.
///
/// # Safety
/// As `needle_v2_run`.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_generate(
    handle: *mut NeedleV2Handle,
    query: *const c_char,
    tools_json: *const c_char,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
    constrain: i32,
) -> *mut c_char {
    clear_last_error();
    let (Some(engine), Some(q), Some(t)) =
        (handle_ref(handle), cstr(query), cstr(tools_json))
    else {
        set_last_error("invalid arguments");
        return ptr::null_mut();
    };
    let opts = GenerateOptions {
        max_new_tokens: if max_new_tokens == 0 { 128 } else { max_new_tokens },
        temperature: temperature.max(0.0),
        seed,
        constrain: constrain != 0,
        ..Default::default()
    };
    out_string(engine.generate(q, t, &opts, |_, _| {}).text)
}

/// Streaming generation. `callback(token_id, piece_utf8, userdata)` fires per
/// token; `piece_utf8` is only valid for the duration of the call.
///
/// # Safety
/// As `needle_v2_run`. `callback` must be a valid function pointer or null.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_run_stream(
    handle: *mut NeedleV2Handle,
    query: *const c_char,
    tools_json: *const c_char,
    callback: Option<unsafe extern "C" fn(u32, *const c_char, *mut c_void)>,
    userdata: *mut c_void,
) -> *mut c_char {
    clear_last_error();
    let (Some(engine), Some(q), Some(t)) =
        (handle_ref(handle), cstr(query), cstr(tools_json))
    else {
        set_last_error("invalid arguments");
        return ptr::null_mut();
    };
    let result = engine.run_stream(q, t, |id, piece| {
        if let Some(cb) = callback {
            // A piece containing a null byte cannot be handed to C; skip it
            // rather than truncating the stream.
            if let Ok(c) = CString::new(piece) {
                cb(id, c.as_ptr(), userdata);
            }
        }
    });
    out_string(result.text)
}

/// Write the L2-normalised contrastive embedding into `out` (`dim` floats).
/// Returns false if there is no contrastive head or `dim` is too small.
///
/// # Safety
/// `out` must point to `dim` writable floats.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_encode_contrastive(
    handle: *mut NeedleV2Handle,
    text: *const c_char,
    out: *mut f32,
    dim: usize,
) -> bool {
    clear_last_error();
    let (Some(engine), Some(text)) = (handle_ref(handle), cstr(text)) else {
        set_last_error("invalid arguments");
        return false;
    };
    if out.is_null() {
        set_last_error("out is null");
        return false;
    }
    let Some(e) = engine.encode_contrastive(text) else {
        set_last_error("this model has no contrastive head");
        return false;
    };
    if dim < e.len() {
        set_last_error(format!("out too small: need {}, got {dim}", e.len()));
        return false;
    }
    ptr::copy_nonoverlapping(e.as_ptr(), out, e.len());
    true
}

/// Width of the contrastive embedding, or 0 if the model has no such head.
///
/// # Safety
/// `handle` must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_contrastive_dim(handle: *mut NeedleV2Handle) -> usize {
    match handle_ref(handle) {
        Some(e) => e.contrastive_dim(),
        None => 0,
    }
}

/// Write the raw confidence logit for `text` to `out`. Higher is more
/// confident; apply a sigmoid for a probability. False if the model has no
/// confidence head.
///
/// This is the primitive. The head scores a judgement already made, so a bare
/// query reads near zero however good it is — use
/// `needle_v2_confidence_for` unless you are assembling the prompt yourself.
///
/// # Safety
/// `out` must be a writable float.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_confidence(
    handle: *mut NeedleV2Handle,
    text: *const c_char,
    out: *mut f32,
) -> bool {
    clear_last_error();
    let (Some(engine), Some(text)) = (handle_ref(handle), cstr(text)) else {
        set_last_error("invalid arguments");
        return false;
    };
    if out.is_null() {
        set_last_error("out is null");
        return false;
    }
    match engine.confidence(text) {
        Some(v) => {
            *out = v;
            true
        }
        None => {
            set_last_error("this model has no confidence head");
            false
        }
    }
}

/// Write the probability that `completion` is the right answer for
/// `(query, tools_json)` to `out`, in `(0, 1)`.
///
/// Assembles what the head was trained on — the formatted prompt followed by
/// the completion — so pass the output of `needle_v2_run` for the run being
/// judged. False if the model has no confidence head.
///
/// # Safety
/// `query`, `tools_json` and `completion` must be valid null-terminated UTF-8
/// C strings; `out` must be a writable float.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_confidence_for(
    handle: *mut NeedleV2Handle,
    query: *const c_char,
    tools_json: *const c_char,
    completion: *const c_char,
    out: *mut f32,
) -> bool {
    clear_last_error();
    let (Some(engine), Some(q), Some(t), Some(c)) =
        (handle_ref(handle), cstr(query), cstr(tools_json), cstr(completion))
    else {
        set_last_error("invalid arguments");
        return false;
    };
    if out.is_null() {
        set_last_error("out is null");
        return false;
    }
    match engine.confidence_for(q, t, c) {
        Some(v) => {
            *out = v;
            true
        }
        None => {
            set_last_error("this model has no confidence head");
            false
        }
    }
}

/// Rank `n` tool descriptions against `query` by cosine similarity.
///
/// Writes up to `top_k` results into `out_indices` and `out_scores`, descending,
/// and returns how many were written. Zero if the model has no contrastive head.
///
/// # Safety
/// `descriptions` must point to `n` valid C strings; the output arrays must each
/// hold `top_k` elements.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_retrieve_tools(
    handle: *mut NeedleV2Handle,
    query: *const c_char,
    descriptions: *const *const c_char,
    n: usize,
    top_k: usize,
    out_indices: *mut usize,
    out_scores: *mut f32,
) -> usize {
    clear_last_error();
    let (Some(engine), Some(q)) = (handle_ref(handle), cstr(query)) else {
        set_last_error("invalid arguments");
        return 0;
    };
    if descriptions.is_null() || out_indices.is_null() || out_scores.is_null() {
        set_last_error("null output or description array");
        return 0;
    }
    let mut descs: Vec<&str> = Vec::with_capacity(n);
    for i in 0..n {
        match cstr(*descriptions.add(i)) {
            Some(s) => descs.push(s),
            None => {
                set_last_error(format!("description {i} is null or not UTF-8"));
                return 0;
            }
        }
    }
    let ranked = engine.retrieve_tools(q, &descs, top_k);
    for (i, (idx, score)) in ranked.iter().enumerate() {
        *out_indices.add(i) = *idx;
        *out_scores.add(i) = *score;
    }
    ranked.len()
}

/// Free a handle from `needle_v2_load`/`needle_v2_load_bytes`.
///
/// # Safety
/// `handle` must be null or a handle not already freed.
#[no_mangle]
pub unsafe extern "C" fn needle_v2_free(handle: *mut NeedleV2Handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}
