/**
 * Type declarations for the needle-rs WebAssembly bindings.
 *
 * These mirror the wasm-bindgen generated declarations emitted in
 * `pkg/needle_wasm.d.ts` when the Rust crate is built with:
 *   wasm-pack build crates/needle-wasm --target web
 *
 * This committed declaration keeps `tsc` green even when the gitignored
 * `pkg/` artifact has not been generated (e.g. environments without a Rust
 * toolchain, such as Google AI Studio). At runtime, `vite.config.ts` selects
 * the real module when `pkg/needle_wasm.js` exists and a tiny no-op stub
 * (`src/needle_wasm_stub.mjs`) otherwise, so the app compiles and runs
 * everywhere. `initWasmEngine` returns false without the WASM engine and the
 * resilient French regex extractors in `src/ai_engine.ts` take over — the
 * app is fully functional in that mode.
 */
declare module 'needle-rs' {
  export default function init(
    module_or_path?:
      | RequestInfo
      | URL
      | Response
      | WebAssembly.Module
      | BufferSource,
  ): Promise<unknown>;

  export class NeedleWasm {
    private constructor();
    static load(weights: Uint8Array, vocab: string): NeedleWasm | null;
    run(query: string, toolsJson: string): string | null;
    retrieve_tools(query: string, descriptions: string[], topK: number): string | null;
    contrastive_dim(): number;
  }

  export class NeedleV2Wasm {
    private constructor();
    static load(bytes: Uint8Array): NeedleV2Wasm | null;
    run_json(voiceTranscript: string, schemaJson: string): string | null;
  }
}