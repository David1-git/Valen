# Valen vLLM image

## Requirements

Linux x86-64, a CUDA 13-compatible NVIDIA driver, and Docker with NVIDIA Container Toolkit.

## Download and start

```bash
docker pull ghcr.io/valen-team/vllm:valen-v0.1.0
docker run --rm --gpus device=0 --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/valen-team/vllm:valen-v0.1.0 \
  --model-path Valen-Team/Valen-2B --served-model-name valen
```

Weights download on first launch and remain in the mounted cache. For local weights, mount the complete checkpoint directory and set `--model-path` to its path inside the container.

## Send a request

Wait for model loading, then run in another terminal:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","state":"A cat is on the sofa.","questions":{"on_sofa":{"type":"noul","instructions":"The cat is on the sofa."}}}'
```

`answers.on_sofa.noul` is the probability that the statement is true. See the [image request example](https://valen-team.github.io/docs/valen/vllm-image/#send-an-image-request).
