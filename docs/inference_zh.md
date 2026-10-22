# SGLang 部署

选择一种部署方式：

| 方式 | 环境 | 操作说明 |
| --- | --- | --- |
| Valen-Team 镜像 | 已配置 NVIDIA Container Toolkit 的 Docker | [镜像部署](https://valen-team.github.io/zh-Hans/docs/valen/sglang-image/) |
| 原版 SGLang 插件 | 兼容的官方 SGLang 环境 | [插件部署包](../serving/sglang/README_zh.md) |

两种方式均提供 `/v1/systemone`，支持单 GPU 文本/图片 choice、noul、score 决策。直接视频推理请使用 [Transformers](../cookbook/local_inference_zh.ipynb)。
