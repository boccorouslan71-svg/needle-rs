# Releasing

One tag drives everything. Pushing `v*` runs
[`.github/workflows/release.yml`](../.github/workflows/release.yml), which builds
the artifacts and publishes to four registries. Hugging Face is the one manual
step.

## Channels

| Channel | Package | Built from | Job |
|---|---|---|---|
| crates.io | `needle-core`, `needle-infer`, `needle-c`, `needle-rs-cli` | the workspace | `crates-publish` |
| npm | `needle-rs` | `crates/needle-wasm` via wasm-pack | `npm-publish` |
| PyPI | `needle-rs` | `crates/needle-python` via maturin + cibuildwheel | `python-publish` |
| GitHub Releases | CLI + C libs + `.wasm`, per target | `build-native`, `build-wasm` | `github-release` |
| Cloudflare Pages | [needle-rs.pages.dev](https://needle-rs.pages.dev) | `examples/browser-demo` + `pkg/` | `deploy-demo` |
| Hugging Face | `Abdalrahman/needle-rs-safetensors` | **manual** | — |

Names that are not what you would guess, and why:

- **`needle-rs-cli`**, not `needle-cli`. Both `needle-cli` and `needle-rs` on
  crates.io belong to unrelated projects. The installed binary is still
  `needle-rs`. Never "fix" the crate name back — `cargo install needle-cli`
  fetches a stranger's tool.
- **`needle-wasm` and `needle-python` are `publish = false`.** They ship as the
  npm and PyPI packages both called `needle-rs`; neither is a crates.io artifact.
- The npm and PyPI packages **share the name `needle-rs`** and each carries its
  own README (`crates/needle-wasm/README.md`, `crates/needle-python/README.md`).
  Those files exist because the repo README's image links are repo-relative and
  render broken on a registry page. Keep every link in them absolute — CI fails
  the npm build if a relative one appears.

## Credentials

All five live as **GitHub Actions repository secrets** (Settings → Secrets and
variables → Actions). Nothing is stored in the repo, and no workflow echoes a
secret value.

| Secret | Used by | Scope needed |
|---|---|---|
| `CARGO_REGISTRY_TOKEN` | `crates-publish` | publish-update, **plus publish-new** |
| `NPM_TOKEN` | `npm-publish` | automation token, publish on `needle-rs` |
| `PYPI_TOKEN` | `python-publish-upload` | project-scoped API token for `needle-rs` |
| `CF_API_TOKEN` | `deploy-demo`, `wasm-demo` | Cloudflare Pages: Edit |
| `CF_ACCOUNT_ID` | `deploy-demo`, `wasm-demo` | account identifier, not a secret as such |

`needle-rs-cli` has never been published, so the crates.io token needs the
**publish-new** scope for the 0.2.0 release, not just publish-update. A token
scoped to updates only will publish the other three crates and then fail on the
last one, leaving the release half-done — and the earlier three cannot be
unpublished.

Rotate the registry tokens on any maintainer change; all three registries treat a
published version as permanent, so a leaked token cannot be undone by yanking.

## Cutting a release

1. **Bump the version in both manifests.** `Cargo.toml`
   (`workspace.package.version`) and `pyproject.toml` (`project.version`). They
   must match the tag: the `version-guard` job fails the whole run before
   anything uploads if they don't. This guard exists because npm takes its
   version from the tag while crates.io and PyPI take theirs from the manifests —
   a tag that outran a bump would publish three different versions, none
   retractable.
2. **Update `CHANGELOG.md`** — move `Unreleased` to the new version, and refresh
   the link definitions at the bottom.
3. **Re-verify locally**, in both feature configurations:
   ```bash
   cargo test --workspace --release
   cargo test --workspace --release --features needle-core/parallel
   cargo clippy --workspace --all-targets --release -- -D warnings
   cargo clippy -p needle-core --no-default-features --release -- -D warnings
   cargo +1.87 check --workspace          # MSRV
   wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg-nodejs/
   node crates/needle-wasm/tests/node_e2e_v2.js
   ```
   The parity suites need `weights/needle2.cact`, `weights/needle.safetensors`
   and `weights/vocab.txt`; each skips with a printed notice if absent, so a run
   that is silently green may have tested nothing. Check the counts.
4. **Tag and push.**
   ```bash
   git tag -a v0.2.0 -m "needle-rs 0.2.0" && git push origin v0.2.0
   ```
5. **Watch the run.** `version-guard` gates everything; if a publish job fails
   partway, re-running is safe — every publish step is idempotent (npm checks
   `npm view` first, `cargo publish` treats "already exists" as success).
6. **Upload the Hugging Face weights** if the v1 conversion changed. Follow the
   checklist in [hf-model-card.md](hf-model-card.md) — banner before README, or
   the CDN caches a broken image. v2 needs nothing here: needle-rs reads
   upstream's own `.cact` container, so there is no conversion to host.

## Verifying a release landed

```bash
cargo search needle-infer                     # crates.io
npm view needle-rs version                    # npm
pip index versions needle-rs                  # PyPI
curl -sI https://needle-rs.pages.dev | head -1  # demo
```

crates.io's API rejects requests without a `User-Agent`; if you query it
directly, send one, or you will read a "not published" answer that is wrong.

## Known rough edges

- `wasm-opt` is not run by wasm-pack (the crate sets `wasm-opt = false`, since
  the binary wasm-pack downloads fails in some environments). Both release
  workflows run it as an explicit step. A local `wasm-pack build` therefore
  produces a 462 KB module where CI produces 414 KB.
- Python wheels are built for CPython 3.8–3.12 by cibuildwheel 2.19.2, and Linux
  aarch64 is skipped because QEMU emulation is too slow. Both are worth
  revisiting: 3.8 is long EOL and 3.13+ gets no wheel.
- `needle-cli`'s directory name still differs from its published name. Renaming
  the directory would be churn for no gain, but it does surprise people.
