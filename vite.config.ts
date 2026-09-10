import { defineConfig } from "vite";
import { resolve } from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const rootDir = fileURLToPath(new URL(".", import.meta.url));
const pkg = resolve(rootDir, "pkg/needle_wasm.js");
const stub = resolve(rootDir, "src/needle_wasm_stub.mjs");

// Prefer the real wasm-pack bindings when the gitignored `pkg/` artifact is
// present; otherwise fall back to a no-op stub so dev/build never fail to
// resolve the import (the resilient regex extractors run without the WASM).
const needleRs = existsSync(pkg) ? pkg : stub;

export default defineConfig({
  resolve: {
    alias: {
      "needle-rs": needleRs,
      "./pkg/needle_wasm.js": needleRs,
      "/pkg/needle_wasm.js": needleRs,
    },
  },
  server: {
    host: "0.0.0.0",
    port: 3000,
    fs: {
      allow: [rootDir],
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 3000,
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(rootDir, "index.html"),
        playground: resolve(rootDir, "playground.html"),
      },
    },
  },
  assetsInclude: ["**/*.wasm"],
});
