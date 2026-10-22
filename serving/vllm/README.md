# Valen vLLM plugin

Serve Valen text/image decisions through `/v1/systemone` with the `valen-vllm` plugin.

## Requirements

Linux x86-64, Python 3.12, glibc 2.34+, a CUDA 13-compatible NVIDIA GPU environment, and official vLLM `0.32.1rc1.dev253+g30aded62d`. Use the pinned [runtime dependencies](runtime-requirements.txt) in a separate environment from Transformers training or SGLang.

The plugin supports single-GPU `shared_state` execution with Choice, Noul and Score. It has been tested with Valen-2B on H200. For direct video inference, use [Transformers](../../cookbook/local_inference.ipynb).

## Build the kit

Run from the Valen repository root:

```bash
python -m pip install build
python -m build --wheel serving/vllm
python serving/vllm/build_kit.py \
  serving/vllm/dist/valen_vllm-0.1.0.dev1-py3-none-any.whl ./vllm-kit
```

Copy `vllm-kit/` and a complete Valen checkpoint to the GPU host.

## Install and start

<details>
<summary>Prepare a fresh vLLM environment</summary>

```bash
python3.12 -m venv .venv-valen-vllm
source .venv-valen-vllm/bin/activate
python -m pip install --no-deps -r /path/to/vllm-kit/runtime-requirements.txt
python -m pip install --no-deps \
  'https://wheels.vllm.ai/30aded62d46dfcf52c815925a8acec5a8bebddc6/vllm-0.32.1rc1.dev253%2Bg30aded62d-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=8c424b1a95e4f893f1c31f2073001e07c2e645d17e7f0745e1abdddaa864fb94'
```

The requirements file contains the complete pinned dependency set. For offline installation, prepare these wheels and set `PIP_NO_INDEX=1 PIP_FIND_LINKS=/path/to/wheels`.

</details>

Inside the required vLLM environment:

```bash
bash /path/to/vllm-kit/install.sh
MODEL=/path/to/Valen-2B CUDA_VISIBLE_DEVICES=0 bash /path/to/vllm-kit/serve.sh
```

The installer checks engine and runtime compatibility. It installs the Valen plugin only; prepare vLLM first.

Defaults: `127.0.0.1:8000`, model name `valen`, maximum input length 8192. Set `HOST`, `PORT` or `MODEL_NAME` to override. `MODEL` also accepts a Hub ID; use `REVISION` to pin the download. Additional launch options include `--max-model-len` and `--gpu-memory-utilization`.

## Send a request

Once the model is ready, run in another terminal:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","state":"A cat is on the sofa.","questions":{"on_sofa":{"type":"noul","instructions":"The cat is on the sofa."}}}'
```

`answers.on_sofa.noul` is the probability that the statement is true. The request uses the same local format as the SGLang plugin: `state`, a question mapping, `criteria` and `noul`. The hosted `/v1/decisions` format is different.

The client includes Choice, Noul, Score and image examples:

```bash
python /path/to/vllm-kit/example_client.py
python /path/to/vllm-kit/example_client.py --image /path/to/cat.jpg
```

Images must be PNG/JPEG/WebP data URLs; the client encodes local files automatically. Each request allows up to 16 images, each up to 24 MiB encoded and 32 megapixels.

For compatibility details and checkpoint comparison commands, see [VALIDATION.md](VALIDATION.md).
