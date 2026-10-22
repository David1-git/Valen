# Valen vLLM candidate image

This image combines the official CUDA 13.0.1 runtime, the pinned official vLLM
wheel and the Valen plugin. No vLLM fork, source overlay or model weights are
included. It targets Linux AMD64 with Python 3.12 and a CUDA 13-compatible driver.

The candidate has not yet completed container GPU validation. Do not treat the
plugin's existing validation receipt as an image validation receipt.

## Build

From the Valen repository root:

```bash
docker build --platform linux/amd64 \
  -f serving/vllm/docker/Dockerfile \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t valen-vllm:candidate .
```

The build checks every runtime wheel's SHA256, verifies the pinned vLLM source
and dependency compatibility, and records the plugin wheel SHA256, engine source
fingerprint, package versions and source revision under `/opt/valen/`.
The plugin wheel is built from the image's source revision; its SHA256 is
recorded separately from the earlier standalone plugin validation receipt.

## Run

```bash
docker run --rm --gpus device=0 --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/Valen-2B:/models/Valen-2B:ro \
  valen-vllm:candidate \
  --model-path /models/Valen-2B --served-model-name valen
```

The server listens on `0.0.0.0:8000` inside the container. Bind the host port to
localhost as above. Launch arguments match `valen-vllm --help`.

For Hub downloads, mount a writable cache instead of a local checkpoint:

```bash
docker run --rm --gpus device=0 --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  valen-vllm:candidate \
  --model-path Valen-Team/Valen-2B \
  --revision fb9205c2541abc97ce276670e66f79b78d9068fc
```

Send the requests described in the [plugin guide](../README.md).

## Validate the actual image

Run on an otherwise idle supported GPU, without installing or replacing packages
inside the container:

```bash
mkdir -p image-validation
docker run --rm --gpus device=0 --ipc=host \
  -v /path/to/Valen-2B:/models/Valen-2B:ro \
  -v "$PWD/image-validation:/results" \
  --entrypoint python3 valen-vllm:candidate \
  /opt/valen/plugin/docker/validate_image.py \
  --model-path /models/Valen-2B --output-dir /results
```

For published candidates, replace the local tag with the exact GHCR digest.
The validation captures environment evidence, runs the 21 source tests, compares
five text/image cases with Transformers, checks 20 concurrent requests and
confirms engine sources are unchanged. The candidate must pass before a release
tag or public image deployment guide is added.
