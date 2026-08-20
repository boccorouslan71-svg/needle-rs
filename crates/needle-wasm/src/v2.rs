//! WASM bindings for the Needle v2 path.
//!
//! A `.cact` container carries its weights, geometry and tokenizer, so loading
//! takes one byte array and no vocabulary string:
//!
//! ```js
//! import init, { NeedleV2Wasm } from './needle_wasm.js';
//! await init();
//! const engine = NeedleV2Wasm.load(new Uint8Array(cactBuffer));
//!
//! engine.run(query, toolsJson);              // full decoded text
//! engine.run_json(query, toolsJson);         // just the <tool_call> payload
//! engine.generate(query, toolsJson, {max_new_tokens: 64, constrain: true});
//! engine.run_stream(query, toolsJson, (id, piece) => out(piece));
//! engine.encode_contrastive(text);           // Float32Array | null
//! engine.confidence(text);                   // number | null (logit)
//! engine.retrieve_tools(query, descsJson, k) // JSON [[index, score], ...]
//! ```

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use needle_infer::v2_engine::{GenerateOptions, V2Engine};

/// WASM handle for a loaded v2 model.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct NeedleV2Wasm {
    engine: V2Engine,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl NeedleV2Wasm {
    /// Load from a `.cact` image. Returns `null` on failure.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = load))]
    pub fn load(cact_bytes: Vec<u8>) -> Option<NeedleV2Wasm> {
        match V2Engine::from_bytes(cact_bytes) {
            Ok(engine) => Some(NeedleV2Wasm { engine }),
            Err(e) => {
                log(&format!("needle v2 load failed: {e}"));
                None
            }
        }
    }

    /// Greedy tool call; returns the full decoded text.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = run))]
    pub fn run(&self, query: &str, tools_json: &str) -> String {
        self.engine.run(query, tools_json).text
    }

    /// Just the `<tool_call>` payload, or an empty string if none was emitted.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = run_json))]
    pub fn run_json(&self, query: &str, tools_json: &str) -> String {
        self.engine
            .run(query, tools_json)
            .tool_call
            .unwrap_or_default()
    }

    /// Generation with explicit settings. `temperature <= 0` is greedy.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = generate))]
    pub fn generate(
        &self,
        query: &str,
        tools_json: &str,
        max_new_tokens: usize,
        temperature: f32,
        seed: f64,
        constrain: bool,
    ) -> String {
        let opts = GenerateOptions {
            max_new_tokens: if max_new_tokens == 0 {
                128
            } else {
                max_new_tokens
            },
            temperature: temperature.max(0.0),
            // JS numbers are f64; clamp rather than wrap on a negative or huge seed.
            seed: seed.max(0.0).min(u64::MAX as f64) as u64,
            constrain,
            ..Default::default()
        };
        self.engine
            .generate(query, tools_json, &opts, |_, _| {})
            .text
    }

    /// L2-normalised contrastive embedding, or `null` without such a head.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = encode_contrastive))]
    pub fn encode_contrastive(&self, text: &str) -> Option<Vec<f32>> {
        self.engine.encode_contrastive(text)
    }

    /// Width of the contrastive embedding, 0 without such a head.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = contrastive_dim))]
    pub fn contrastive_dim(&self) -> usize {
        self.engine.contrastive_dim()
    }

    /// Probability in `(0, 1)` that `completion` is the right answer for this
    /// query, or `null` without a confidence head.
    ///
    /// The head scores a judgement already made, so it needs the completion:
    /// pass what [`run`] returned. Gate execution on it — act above a threshold
    /// you pick, escalate below it.
    ///
    /// [`run`]: NeedleV2Wasm::run
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = confidence_for))]
    pub fn confidence_for(&self, query: &str, tools_json: &str, completion: &str) -> Option<f32> {
        self.engine.confidence_for(query, tools_json, completion)
    }

    /// Raw confidence logit for an arbitrary string, or `null` without a
    /// confidence head. Higher is more confident; apply a sigmoid.
    ///
    /// This is the primitive. A bare query reads near zero however answerable
    /// it is, because the completion is what gets scored — prefer
    /// [`confidence_for`] unless you are assembling the prompt yourself.
    ///
    /// [`confidence_for`]: NeedleV2Wasm::confidence_for
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = confidence))]
    pub fn confidence(&self, text: &str) -> Option<f32> {
        self.engine.confidence(text)
    }

    /// Rank tool descriptions against a query.
    ///
    /// `tool_descs_json` is a JSON array of strings; the result is a JSON array
    /// of `[index, score]` pairs, descending.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = retrieve_tools))]
    pub fn retrieve_tools(&self, query: &str, tool_descs_json: &str, top_k: usize) -> String {
        let descs: Vec<String> = match serde_json::from_str(tool_descs_json) {
            Ok(v) => v,
            Err(e) => {
                log(&format!("retrieve_tools: bad JSON: {e}"));
                return "[]".to_string();
            }
        };
        let refs: Vec<&str> = descs.iter().map(String::as_str).collect();
        let ranked = self.engine.retrieve_tools(query, &refs, top_k);
        let parts: Vec<String> = ranked.iter().map(|(i, s)| format!("[{i},{s}]")).collect();
        format!("[{}]", parts.join(","))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl NeedleV2Wasm {
    /// Streaming generation. `on_token(tokenId, piece)` fires per token.
    #[wasm_bindgen(js_name = run_stream)]
    pub fn run_stream(&self, query: &str, tools_json: &str, on_token: &js_sys::Function) -> String {
        let this = JsValue::NULL;
        self.engine
            .run_stream(query, tools_json, |id, piece| {
                let _ = on_token.call2(
                    &this,
                    &JsValue::from_f64(id as f64),
                    &JsValue::from_str(piece),
                );
            })
            .text
    }
}

#[cfg(target_arch = "wasm32")]
fn log(msg: &str) {
    web_sys::console::error_1(&JsValue::from_str(msg));
}

#[cfg(not(target_arch = "wasm32"))]
fn log(msg: &str) {
    eprintln!("{msg}");
}
