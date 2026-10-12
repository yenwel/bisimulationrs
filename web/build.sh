#!/usr/bin/env sh
# Builds PEPA Studio for local use: compiles the Rust engine to WebAssembly
# and writes web/dist/ (index.html + bisim_wasm.wasm).
#
#   sh web/build.sh && python3 -m http.server -d web/dist 8000
#   open http://localhost:8000
set -eu
cd "$(dirname "$0")/.."
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
cargo build -p bisim-wasm --release --target wasm32-unknown-unknown
mkdir -p web/dist
cp target/wasm32-unknown-unknown/release/bisim_wasm.wasm web/bisim_wasm.wasm
cp web/bisim_wasm.wasm web/dist/bisim_wasm.wasm
# pepa-studio.html is a page body (it is also published as a claude.ai
# artifact, which adds the document skeleton); add the skeleton here.
{
  printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
  printf '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
  printf '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
  cat web/pepa-studio.html
  printf '\n</body>\n</html>\n'
} > web/dist/index.html
echo "Built web/dist. Serve it with: python3 -m http.server -d web/dist 8000"
