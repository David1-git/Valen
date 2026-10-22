# Valen SGLang serving kit

Install native Valen decision support as a small `valen-sglang` wheel in an existing, compatible SGLang environment. This is a deployment alternative to the prebuilt Valen-Team image and direct Transformers inference; it uses the same SGLang engine as the image.

See `VALIDATION.md` for the exact environment and completed checks before using it in a deployment.

## What gets installed

The wheel contains the Valen config, shared-state prompt compiler, FP32 Mixer head, and model/request adapters. Its `sglang.srt.plugins` entry point registers these through SGLang's general-plugin hooks. It does not include a SGLang source tree, replace the engine package, modify checkpoint files, or patch installed engine files.

The adapters reuse stock Qwen3.5 execution and the stock embedding scheduler. They extend the existing `/v1/systemone` route. Other models keep the stock implementation; generative decision requests retain their ten-level score limit.

## Supported environment

- Linux x86-64, an NVIDIA H200, and a compatible CUDA 13 runtime. Other GPUs have not been validated with this kit.
- **Unmodified official SGLang commit `c1c287a42928495166e14dbcef31b9e1f1dac199`.** The matching official base is `docker.io/lmsysorg/sglang@sha256:a8eedda74b54f7dfecc445ee66d298f8d8f6f50192201a53b0a6dd7571ff8d26`.
- The pinned runtime in `valen_sglang/runtime-lock.json`, including PyTorch 2.14.1, Transformers 5.19.0, sglang-kernel 0.4.10, FlashInfer 0.6.18, and CUTLASS DSL 4.6.2.

This first version deliberately checks ten critical engine source files by SHA256 and refuses incompatible releases, including the Valen fork/image that already embeds Valen support. Do not bypass that check or assume a fresh `pip install sglang` gives this exact engine.

`install.sh` is intended for that prepared SGLang environment, not an empty Python venv. It replaces selected runtime package versions. Keep Transformers training/inference dependencies in a separate environment.

## Install and run

A built checkpoint-side kit contains:

```text
sglang/
  valen_sglang-<version>-py3-none-any.whl
  install.sh
  serve.sh
  example_client.py
  validate.py
  README.md
  README_zh.md
  VALIDATION.md
  LICENSE
  NOTICE
  SHA256SUMS
```

Inside the supported environment, use the checkpoint's kit directory:

```bash
bash /path/to/checkpoint/sglang/install.sh
MODEL=/path/to/checkpoint \
  CUDA_VISIBLE_DEVICES=0 \
  bash /path/to/checkpoint/sglang/serve.sh
```

No Valen repository installation, SGLang fork checkout, custom Valen image build, or `trust_remote_code` is needed by the server. An existing compatible environment can be a Python installation or an official SGLang container. If using a container, mount the complete checkpoint and kit into it first; set `HOST=0.0.0.0` inside the container and bind the published host port to localhost.

The installer verifies bundle checksums, installs one plugin wheel, checks engine compatibility, pins the tested runtime, removes an incompatible FlashInfer JIT cache, and checks runtime versions. FlashInfer cubin uses `https://flashinfer.ai/whl`; the other packages use the environment's pip index. For offline installs, predownload all required wheels and set `PIP_NO_INDEX=1 PIP_FIND_LINKS=/path/to/wheels`. A complete SGLang environment is still required.

Kernel packages are installed without re-resolving their transitive dependencies: the pinned official environment supplies them. This is a curated runtime, not a guarantee that arbitrary pip dependency combinations or a global `pip check` will pass.

`serve.sh` exposes these settings:

| Variable | Default |
| --- | --- |
| `MODEL` | `Valen-Team/Valen-2B` |
| `REVISION` | unset; use a fixed Hub revision for reproducibility |
| `MODEL_NAME` | `valen` |
| `HOST` | `127.0.0.1` |
| `PORT` | `30000` |
| `CUDA_VISIBLE_DEVICES` | select one GPU in the caller |

Extra arguments are passed to the engine. The script uses FA3. The launcher checks the engine and runtime before startup and requires every Valen hook to load successfully. If `SGLANG_PLUGINS` is set, it must include `valen`.

## Send requests

Wait for the startup warmup to finish, then use a second terminal in the same client environment:

```bash
python /path/to/checkpoint/sglang/example_client.py
python /path/to/checkpoint/sglang/example_client.py --image /path/to/cat.jpg
```

The endpoint and JSON contract match the Valen-Team image:

```bash
curl http://127.0.0.1:30000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","state":"A cat is on the sofa.","questions":{"on_sofa":{"type":"noul","instructions":"The cat is on the sofa."}}}'
```

The response's `answers` object is keyed by question ID. `choice` returns named candidate probabilities; `noul` returns the probability of the statement; `score` returns an expected zero-based level and its distribution. Use `criteria` mappings for choice, ordered criteria lists for score, and no criteria override for noul.

This is the local API, distinct from the hosted `/v1/decisions` payload.

## Scope

- Published full Valen Qwen3.5 Mixer checkpoints, `shared_state` execution, one GPU, TP=PP=1.
- Text and images; choice with 1–255 candidates, noul, score with 2–255 levels.
- BF16 backbone for the published checkpoint's auto dtype; FP32 decision-head parameters and pooling.
- No direct `video_url`, speculative decoding, distributed decision heads, quantization, or adapter-only checkpoints in this version.
- One complete prefill per request; prefix reuse, chunked prefill and CUDA graphs are disabled for the decision path.

## Validate a new checkpoint

```bash
python /path/to/checkpoint/sglang/validate.py \
  --model-path /path/to/checkpoint --output-dir /path/to/results
```

The harness runs the checkpoint's Transformers reference and plugin server in separate processes. It compares token counts, choices and probabilities for text/image requests, then runs concurrent requests. Its default absolute-probability tolerance is 0.02. Keep the full result files, runtime freeze, engine fingerprint report, checkpoint revision and hardware information. A new checkpoint or engine change needs new validation.

## Build the kit from source

The source lives in `Valen/serving/sglang`; it is a separate Python distribution from `valen`.

```bash
python -m pip install build
python -m build --wheel /path/to/Valen/serving/sglang
python /path/to/Valen/serving/sglang/build_kit.py \
  /path/to/Valen/serving/sglang/dist/valen_sglang-<version>-py3-none-any.whl \
  /path/to/exported-checkpoint/sglang
```

The destination must not exist. The builder copies the wheel, bilingual instructions, validation notes, scripts and license notices, and writes `SHA256SUMS`. It does not copy or upload model weights. The resulting `sglang/` directory can be published with the checkpoint after its tests pass.

The engine source lock is intentional. Updating it means reviewing hook APIs and behavior, rebuilding the wheel, and repeating CPU/GPU checks; editing hashes alone does not establish compatibility.
