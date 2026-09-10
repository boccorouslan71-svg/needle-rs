#!/usr/bin/env bash
# Run the needle-rs CLI against a single tool definition, on either model version.
#
# Usage:
#   ./run.sh                      # Needle v2, default query
#   ./run.sh "your query"         # Needle v2
#   ./run.sh --v1 "your query"    # Needle v1
#   ./run.sh --both "your query"  # run both and compare
#
# v2 needs one file (weights/needle2.cact); v1 needs two
# (weights/needle.safetensors + weights/vocab.txt). Missing files are fetched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$REPO_ROOT/target/release/needle-rs"
WEIGHTS_DIR="$REPO_ROOT/weights"

V2_CACT="$WEIGHTS_DIR/needle2.cact"
V1_WEIGHTS="$WEIGHTS_DIR/needle.safetensors"
V1_VOCAB="$WEIGHTS_DIR/vocab.txt"

HF_V2="https://huggingface.co/Cactus-Compute/needle2/resolve/main"
HF_V1="https://huggingface.co/Abdalrahman/needle-rs-safetensors/resolve/main"

MODE="v2"
case "${1:-}" in
  --v1)   MODE="v1";   shift ;;
  --v2)   MODE="v2";   shift ;;
  --both) MODE="both"; shift ;;
esac

QUERY="${1:-What is the weather in Paris?}"

TOOLS='[{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "description": "celsius or fahrenheit"}
    },
    "required": ["city"]
  }
}]'

if [ ! -f "$CLI" ]; then
  echo "Building the CLI…"
  cargo build --release -p needle-rs-cli --manifest-path "$REPO_ROOT/Cargo.toml"
fi

mkdir -p "$WEIGHTS_DIR"

fetch() { # fetch <url> <dest> <label>
  [ -f "$2" ] && return 0
  echo "Fetching $3 → $2"
  curl -fSL --progress-bar -o "$2" "$1"
}

run_v2() {
  fetch "$HF_V2/needle2.cact" "$V2_CACT" "Needle v2 (13.7 MB)"
  echo
  echo "── Needle v2  ·  one .cact file, tokenizer included"
  # A .cact carries its own geometry and tokenizer, so no vocabulary argument.
  # --constrain restricts the payload to the declared schema.
  "$CLI" --constrain "$V2_CACT" "$QUERY" "$TOOLS"
}

run_v1() {
  fetch "$HF_V1/needle.safetensors" "$V1_WEIGHTS" "Needle v1 weights (22 MB)"
  fetch "$HF_V1/vocab.txt" "$V1_VOCAB" "Needle v1 vocabulary"
  echo
  echo "── Needle v1  ·  weights + separate vocabulary"
  "$CLI" "$V1_WEIGHTS" "$V1_VOCAB" "$QUERY" "$TOOLS"
}

echo "Query: $QUERY"
case "$MODE" in
  v2)   run_v2 ;;
  v1)   run_v1 ;;
  both) run_v2; run_v1 ;;
esac
