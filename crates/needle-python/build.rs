fn main() {
    // On macOS an `extension-module` cdylib deliberately leaves the CPython
    // symbols undefined — the interpreter supplies them at import time. Without
    // `-undefined dynamic_lookup` the standalone link step fails, which made
    // `cargo build -p needle-python` error out even though the crate compiles.
    // This emits those flags for this crate only, so no other target in the
    // workspace loses undefined-symbol checking.
    pyo3_build_config::add_extension_module_link_args();
}
