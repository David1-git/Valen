# SGLang deployment

Choose a deployment route:

| Route | Environment | Instructions |
| --- | --- | --- |
| Valen-Team image | Docker with NVIDIA Container Toolkit | [Image deployment](https://valen-team.github.io/docs/valen/sglang-image/) |
| Stock SGLang plugin | A compatible official SGLang environment | [Plugin serving kit](../serving/sglang/README.md) |

Both routes expose `/v1/systemone` for single-GPU text/image choice, noul, and score decisions. For direct video inference, use [Transformers](../cookbook/local_inference.ipynb).
