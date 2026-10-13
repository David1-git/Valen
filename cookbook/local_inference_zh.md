# 下载 Valen 并在本地运行

[English](local_inference.md) · **简体中文** · [Cookbook 首页](README_zh.md)

这条路线适合在自己的机器上处理文本、图片和视频。建议从 **Valen-0.8B** 开始。准备 Python 3.10+、足够存放完整模型的磁盘空间，推荐使用支持 BF16 的 NVIDIA GPU。显存需求与模型大小、图片分辨率、视频长度和问题数量有关，权重文件大小不能直接当作最低显存要求。

[先获取仓库](README_zh.md#获取示例)，以下命令都从仓库根目录执行。安装步骤面向 Linux NVIDIA GPU 环境；CPU 模式可以用于较慢的流程检查。

## 1. 安装本地环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -e .
```

检查 GPU 是否可见：

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

本教程建议使用 Python **3.10 或 3.11**。项目在 [pyproject.toml](../pyproject.toml) 中固定了 Transformers **5.4.0** 等依赖版本。先使用 SDPA 注意力即可；Flash Attention 是可选项，运行本教程不需要安装。

## 2. 下载完整模型

如果只想下载文件，安装 `python -m pip install huggingface-hub==1.7.1` 后执行本节即可；下载本身不需要 GPU 或 PyTorch。

```bash
# 可选：先查看文件列表统计和下载大小。
python cookbook/examples/download_model.py --dry-run

# 下载到 models/Valen-0.8B。
python cookbook/examples/download_model.py
```

下载包含权重、tokenizer、processor 和自定义模型代码。完整的 Valen 模型**不需要**再单独下载 Qwen 基座。脚本会跳过可选的历史 `unmerged/` 权重包。

其他尺寸使用同样的命令：

```bash
python cookbook/examples/download_model.py --model Valen-Team/Valen-2B
python cookbook/examples/download_model.py --model Valen-Team/Valen-4B
```

重复执行可以复用已完成的文件。若 Hugging Face 要求鉴权，先运行 `hf auth login`。需要固定模型版本时，使用 `--revision <commit-hash>`。下载和缓存机制可参考 [Hub 官方下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)。

## 3. 获得第一个本地结果

```bash
python cookbook/examples/local_inference.py \
  --model models/Valen-0.8B --local-files-only
```

这使用与 API 教程相同的“猫在沙发上”文本，一次提出 Choice、Noul 和 Score 三道题。本地响应的 `answers` 是**以问题 ID 为键的对象**：

```json
{
  "answers": {
    "animal": {"type": "choice", "choice": "cat"},
    "on_sofa": {"type": "noul", "noul": 0.9},
    "cat_count": {"type": "score", "score": 1.0}
  }
}
```

这是说明字段的简化响应，实际数值会变化。`noul` 表示陈述为真的概率；提供三个评分等级时，`score` 是范围在 0–2 内的概率加权索引。

也可以让加载器自动下载并缓存 Hub 模型：

```bash
python cookbook/examples/local_inference.py --model Valen-Team/Valen-0.8B
```

首次运行包含模型加载和下载。下载其他尺寸后，用 `--model models/Valen-2B` 或 `--model models/Valen-4B` 切换。

## 4. 运行图片和视频

下面的小型素材已经随仓库提供：

```bash
python cookbook/examples/local_inference.py --model models/Valen-0.8B \
  --image data/smoke/green.png --local-files-only

python cookbook/examples/local_inference.py --model models/Valen-0.8B \
  --video data/smoke/red_to_green.mp4 --local-files-only
```

图片示例判断绿灯；视频由红灯变为绿灯，问题要求判断**最后一帧**。这些合成输入用于检查推理流程，不用于评估模型性能。换成自己的素材时，也要修改问题，使它们与内容相符。

本地媒体路径相对于 `media_root` 解析；使用 `--request` 时，则相对于请求 JSON 所在目录。这些脚本读取本地文件，网络图片或视频需要先下载。已发布 Qwen 模型默认采样 16 帧视频，推理沿用模型保存的媒体设置。

## 5. 在 Python 中使用

将下面的代码保存为仓库根目录下的 `my_local.py`，执行 `python my_local.py`：

```python
import torch
from transformers import AutoModel

torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False
model = AutoModel.from_pretrained(
    "models/Valen-0.8B", trust_remote_code=True,
    dtype="auto", attn_implementation="sdpa",
).to("cuda").eval()

request = {
    "state": {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Observe the traffic light."},
        {"type": "image_url", "image_url": {"url": "data/smoke/green.png"}},
    ]}]},
    "questions": {
        "colour": {
            "type": "choice", "instructions": "Which colour is the traffic light?",
            "criteria": {"red": "Red", "green": "Green", "unknown": "No traffic light is visible"},
        },
        "green": {"type": "noul", "instructions": "The traffic light is green."},
    },
}
result = model.predict(request, media_root=".", execution="shared_state")
print(result["answers"])
```

`dtype="auto"` 沿用 checkpoint 记录的精度；`trust_remote_code=True` 加载模型附带的自定义决策实现。结构化决策通过 `model.predict()` 调用。

文本输入只需把 `state` 换成字符串。视频输入将图片条目替换为 `{"type": "video_url", "video_url": {"url": "data/smoke/red_to_green.mp4"}}`，并调整问题。

### 自定义问题与共享 state

`questions` 中每道题使用唯一的键。Choice 的 `criteria` 是候选名 → 描述的对象；Noul 只需提供陈述；Score 的 `criteria` 是按顺序排列的描述列表。更多字段见[完整数据格式](../docs/data-format.md)。

将本地格式的请求保存为 `custom_local.json`，再执行：

```bash
python cookbook/examples/local_inference.py --model models/Valen-0.8B \
  --request custom_local.json --local-files-only > result.json
```

JSON 包含上述 Python 示例中的 `state` 和 `questions`，也支持训练数据使用的外层 `request` 包装。媒体路径相对于这个 JSON 所在目录。

默认 `--execution shared_state` 对共同输入编码一次，再读取所有问题。`--execution question` 则逐题独立处理。安装了兼容的 CUDA/Flash Attention 后，可设置 `--attn-implementation flash_attention_2`。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| CUDA 不可用 | 检查 `nvidia-smi`、GPU 可见性和 PyTorch 安装版本。没有 GPU 可用 [API](api_zh.md)，或加 `--device cpu` 做较慢的流程检查。 |
| CUDA 显存不足 | 从 0.8B 开始，使用更小图片、更短视频和更少问题；除了权重，还要为中间计算预留显存。 |
| `qwen3_5` 或自定义代码导入错误 | 激活本地环境，安装 `pyproject.toml` 中指定的版本，并下载完整模型目录。 |
| 没有 Flash Attention | 使用默认的 `sdpa`；安装兼容版本后再切换。 |
| 找不到媒体文件 | 检查路径和 `media_root`；使用 `--request` 时，路径相对于 JSON 文件。 |
| 离线加载失败 | 下载完整快照，再传入本地目录及 `--local-files-only`。 |
| Hugging Face 缓存目录没有写入权限 | 将 `HF_HOME` 设置为可写目录，再重新运行。 |
| 本地推理拒绝 API 字段 | 本地使用 `state`、问题 ID、`criteria` 和 `noul`；[查看两种格式的对照](api_zh.md#api-与本地请求的区别)。 |

[下一步：评估 VisualDecisionBench](../README_zh.md#快速开始) · [训练指南](../valen/training/README.md)
