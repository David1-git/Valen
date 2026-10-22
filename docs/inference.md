# Local HTTP deployment

Choose a deployment route:

| Route | Environment | Instructions |
| --- | --- | --- |
| Valen-Team image | Docker with NVIDIA Container Toolkit | [Image deployment](https://valen-team.github.io/docs/valen/sglang-image/) |
| Stock SGLang plugin | A compatible official SGLang environment | [SGLang plugin kit](../serving/sglang/README.md) |
| vLLM plugin | The pinned official vLLM nightly environment | [vLLM plugin kit](../serving/vllm/README.md) |

All routes expose `/v1/systemone` for single-GPU text/image choice, noul, and score decisions. For direct video inference, use [Transformers](../cookbook/local_inference.ipynb).

SGLang defaults to port `30000`; vLLM defaults to port `8000`. Each plugin requires its documented engine version.
