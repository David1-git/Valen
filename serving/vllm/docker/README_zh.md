# Valen vLLM 镜像

## 环境要求

Linux x86-64、兼容 CUDA 13 的 NVIDIA 驱动，以及已配置 NVIDIA Container Toolkit 的 Docker。

## 下载并启动

```bash
docker pull ghcr.io/valen-team/vllm:valen-v0.1.0
docker run --rm --gpus device=0 --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/valen-team/vllm:valen-v0.1.0 \
  --model-path Valen-Team/Valen-2B --served-model-name valen
```

首次启动会下载模型，挂载的缓存目录用于复用权重。使用本地权重时，挂载完整模型目录，并将 `--model-path` 改为容器内的路径。

## 发送请求

等待模型加载完成，在另一个终端运行：

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","state":"A cat is on the sofa.","questions":{"on_sofa":{"type":"noul","instructions":"The cat is on the sofa."}}}'
```

`answers.on_sofa.noul` 表示该陈述为真的概率。图片输入见[图片请求示例](https://valen-team.github.io/zh-Hans/docs/valen/vllm-image/#发送图片请求)。
