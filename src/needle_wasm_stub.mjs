/**
 * Runtime no-op fallback for the needle-rs WebAssembly bindings.
 *
 * `vite.config.ts` resolves the "needle-rs" import against this stub when the
 * gitignored `pkg/` artifact has not been generated with:
 *   wasm-pack build crates/needle-wasm --target web
 * When the real bindings exist, the genuine module is used instead and the
 * on-device Needle v2 extractor is enabled.
 *
 * The no-op keeps the module graph resolvable (dev server + production build)
 * in environments without a Rust toolchain. The application is fully
 * functional without it: `initWasmEngine` returns false and the resilient
 * French regex extractors in `src/ai_engine.ts` take over.
 *
 * MUST stay in sync with the ambient declarations in `src/needle_wasm.d.ts`.
 */
export default async function init() {}

export class NeedleWasm {
  static load(_weights, _vocab) {
    return null;
  }

  run(_query, _toolsJson) {
    return null;
  }

  retrieve_tools(_query, _descriptions, _topK) {
    return '';
  }

  contrastive_dim() {
    return 0;
  }
}

export class NeedleV2Wasm {
  static load(_bytes) {
    return null;
  }

  run_json(_voiceTranscript, _schemaJson) {
    return null;
  }
}