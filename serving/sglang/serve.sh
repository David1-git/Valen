#!/usr/bin/env bash
# Run inside the validated official SGLang environment after install.sh.
set -euo pipefail
MODEL="${MODEL:-Valen-Team/Valen-2B}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-valen}"
revision=()
if [[ -n "${REVISION:-}" ]]; then
  revision=(--revision "$REVISION")
fi
export FLASHINFER_VERSION=0.6.18
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
exec python3 -m valen_sglang.launch_server \
  --model-path "$MODEL" "${revision[@]}" \
  --served-model-name "$MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --attention-backend fa3 \
  "$@"
