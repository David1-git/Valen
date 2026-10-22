#!/usr/bin/env bash
set -euo pipefail
KIT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
shopt -s nullglob
wheels=("$KIT"/valen_vllm-*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Expected exactly one valen_vllm wheel in the built kit" >&2
  exit 1
fi
if [[ -f "$KIT/SHA256SUMS" ]]; then
  (cd "$KIT" && sha256sum -c SHA256SUMS)
fi
python3 -m pip install --no-deps --force-reinstall "${wheels[0]}"
python3 -m valen_vllm.compatibility
echo "Valen plugin installed. Run bash serve.sh."
