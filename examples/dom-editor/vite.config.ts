import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

// The WASM package is built from this repo, not installed from npm: the
// published `needle-rs` package and the local `pkg/` directory expose the same
// module, so the source imports "needle-rs" either way and this alias points
// that name at the local build. Run wasm-pack first — see the README.
const pkg = fileURLToPath(new URL("../../pkg/needle_wasm.js", import.meta.url));
const repoRoot = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  resolve: { alias: { "needle-rs": pkg } },
  // The alias resolves outside the project root, so Vite's dev server has to
  // be allowed to serve it.
  server: { fs: { allow: [repoRoot] } },
  // wasm-bindgen's glue fetches the .wasm at runtime; keep it a real asset
  // rather than inlining it into the bundle.
  assetsInclude: ["**/*.wasm"],
});
