# 本地 HTTP 部署

选择一种部署方式：

| 方式 | 环境 | 操作说明 |
| --- | --- | --- |
| Valen-Team SGLang 镜像 | 已配置 NVIDIA Container Toolkit 的 Docker | [镜像部署](https://valen-team.github.io/zh-Hans/docs/valen/sglang-image/) |
| 原版 SGLang 插件 | 兼容的官方 SGLang 环境 | [SGLang 插件部署包](../serving/sglang/README_zh.md) |
| Valen-Team vLLM 镜像 | 已配置 NVIDIA Container Toolkit 的 Docker | [vLLM 镜像](../serving/vllm/docker/README_zh.md) |
| vLLM 插件 | 固定版本的官方 vLLM nightly 环境 | [vLLM 插件部署包](../serving/vllm/README_zh.md) |

以上方式均提供 `/v1/systemone`，支持单 GPU 文本/图片 choice、noul、score 决策。直接视频推理请使用 [Transformers](../cookbook/local_inference_zh.ipynb)。

SGLang 默认使用 `30000` 端口，vLLM 默认使用 `8000`。插件需配合各自文档指定的引擎版本。
