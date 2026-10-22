#!/usr/bin/env bash
# Install into the pinned official SGLang environment, never a Valen source overlay.
set -euo pipefail
KIT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_NO_CACHE_DIR=1
shopt -s nullglob
wheels=("$KIT"/valen_sglang-*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Expected one valen_sglang wheel in $KIT; use the built serving-kit directory." >&2
  exit 1
fi
if [[ -f "$KIT/SHA256SUMS" ]]; then
  (cd "$KIT" && sha256sum -c SHA256SUMS)
fi
python3 -m pip install --no-deps --force-reinstall "${wheels[0]}"
# Stop before changing runtime libraries if the engine does not match.
python3 -m valen_sglang.compatibility --engine-only
python3 -m pip uninstall -y flashinfer-jit-cache
python3 - <<'PY'
import shutil
import sysconfig
shutil.rmtree(sysconfig.get_path('purelib') + '/flashinfer_jit_cache', ignore_errors=True)
PY
python3 -m pip install \
  torch==2.14.1 torchvision==0.29.1 sglang-kernel==0.4.10 \
  transformers==5.19.0 tokenizers==0.23.3
# The pinned SGLang base already supplies these libraries' dependencies.
# Keep this tested kernel stack together instead of resolving a new one.
python3 -m pip install --no-deps \
  flashinfer-python==0.6.18 \
  nvidia-cutlass-dsl==4.6.2 \
  nvidia-cutlass-dsl-libs-base==4.6.2 \
  nvidia-cutlass-dsl-libs-core==4.6.2 \
  nvidia-cutlass-dsl-libs-cu12==4.6.2 \
  nvidia-cutlass-dsl-libs-cu13==4.6.2 \
  quack-kernels==0.6.4 apache-tvm-ffi==0.1.11
python3 -m pip install --no-deps \
  --index-url https://flashinfer.ai/whl flashinfer-cubin==0.6.18
python3 -m valen_sglang.compatibility
python3 - <<'PY'
from importlib.metadata import entry_points
assert any(e.name == 'valen' for e in entry_points(group='sglang.srt.plugins'))
PY
printf '%s\n' 'Valen plugin installed. Run bash serve.sh from the kit directory.'
