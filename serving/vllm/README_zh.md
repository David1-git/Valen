# Valen vLLM 插件

使用 `valen-vllm` 插件，通过 `/v1/systemone` 提供 Valen 文本/图片决策服务。

## 环境要求

Linux x86-64、Python 3.12、glibc 2.34 及以上、兼容 CUDA 13 的 NVIDIA GPU 环境，以及官方 vLLM `0.32.1rc1.dev253+g30aded62d`。使用[固定依赖列表](runtime-requirements.txt)，与 Transformers 训练、SGLang 分别建立环境。

支持单 GPU、`shared_state` 和 Choice、Noul、Score，已在 H200 上测试 Valen-2B。直接视频推理请使用 [Transformers](../../cookbook/local_inference_zh.ipynb)。

## 构建部署包

在 Valen 仓库根目录运行：

```bash
python -m pip install build
python -m build --wheel serving/vllm
python serving/vllm/build_kit.py \
  serving/vllm/dist/valen_vllm-0.1.0.dev1-py3-none-any.whl ./vllm-kit
```

将生成的 `vllm-kit/` 和完整模型目录复制到 GPU 服务器。

## 安装并启动

<details>
<summary>从空环境安装指定版本的 vLLM</summary>

```bash
python3.12 -m venv .venv-valen-vllm
source .venv-valen-vllm/bin/activate
python -m pip install --no-deps -r /path/to/vllm-kit/runtime-requirements.txt
python -m pip install --no-deps \
  'https://wheels.vllm.ai/30aded62d46dfcf52c815925a8acec5a8bebddc6/vllm-0.32.1rc1.dev253%2Bg30aded62d-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=8c424b1a95e4f893f1c31f2073001e07c2e645d17e7f0745e1abdddaa864fb94'
```

依赖列表包含完整的固定版本组合。离线安装时，提前准备对应 wheel，并设置 `PIP_NO_INDEX=1 PIP_FIND_LINKS=/path/to/wheels`。

</details>

在上述 vLLM 环境中执行：

```bash
bash /path/to/vllm-kit/install.sh
MODEL=/path/to/Valen-2B CUDA_VISIBLE_DEVICES=0 bash /path/to/vllm-kit/serve.sh
```

安装器会检查引擎与运行时版本，只安装 Valen 插件；vLLM 环境需提前准备。

默认地址 `127.0.0.1:8000`，模型名 `valen`，最大输入长度 8192。可通过 `HOST`、`PORT`、`MODEL_NAME` 修改。`MODEL` 也可填写 Hub 模型 ID，并用 `REVISION` 固定下载版本。额外启动参数包括 `--max-model-len` 和 `--gpu-memory-utilization`。

## 发送请求

等待模型加载完成，在另一个终端运行：

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","state":"A cat is on the sofa.","questions":{"on_sofa":{"type":"noul","instructions":"The cat is on the sofa."}}}'
```

`answers.on_sofa.noul` 表示该陈述为真的概率。请求格式与 SGLang 插件相同，使用 `state`、问题映射、`criteria` 和 `noul`；托管 `/v1/decisions` 的格式不同。

客户端包含 Choice、Noul、Score 和图片示例：

```bash
python /path/to/vllm-kit/example_client.py
python /path/to/vllm-kit/example_client.py --image /path/to/cat.jpg
```

图片使用 PNG/JPEG/WebP data URL，客户端会自动编码本地文件。每请求最多 16 张，每张编码后不超过 24 MiB、3200 万像素。

兼容性详情与检查点对照命令见 [VALIDATION.md](VALIDATION.md)。
