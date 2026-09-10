import { defineConfig } from "vite";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = fileURLToPath(new URL(".", import.meta.url));
const pkg = resolve(rootDir, "pkg/needle_wasm.js");

export default defineConfig({
  resolve: {
    alias: {
      "needle-rs": pkg,
      "./pkg/needle_wasm.js": pkg,
      "/pkg/needle_wasm.js": pkg,
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
