# 调用 Valen API

[English](api.md) · **简体中文** · [Cookbook 首页](README_zh.md)

这条路线不需要 GPU，也不用下载模型权重。准备 Python 3.10+ 和网络即可。[先获取仓库](README_zh.md#获取示例)，再从仓库根目录执行以下命令。终端命令使用 Linux/macOS 语法。

## 1. 只安装一个依赖

```bash
python3 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install -r cookbook/requirements-api.txt
```

Windows 使用 `python -m venv .venv-api` 创建环境，在 PowerShell 中用 `.venv-api\Scripts\Activate.ps1` 激活。

## 2. 获得第一个结果

```bash
python cookbook/examples/api.py
```

默认请求使用 `valen-0-8b`，输入文本为 **“A cat is resting on a sofa. There is no dog.”**，即“一只猫在沙发上休息，没有狗”。一次请求提出三个问题：出现了什么动物、猫是否在沙发上、提到了几只猫。

脚本会显示 HTTP 状态、请求耗时和响应 JSON。重点查看 `answers` 中的字段：

```json
{
  "answers": [
    {"type": "choice", "name": "animal", "choice": "cat"},
    {"type": "predicate", "name": "on_sofa", "probability": 0.9},
    {"type": "score", "name": "cat_count", "score": 1.0}
  ]
}
```

这是用于说明结构的简化响应，实际数值会变化。响应还包含 `model_id` 和 `revision`，可以记录它们来识别服务部署的模型版本。API 的 `answers` 是**数组**，通过每项的 `name` 对应原问题。

### 切换模型或替换文本

```bash
python cookbook/examples/api.py --model valen-2b
python cookbook/examples/api.py --model valen-4b
python cookbook/examples/api.py --model valen-preview-0923
python cookbook/examples/api.py --text "Two cats are resting on a sofa. There is no dog."
```

API 支持的名称为 `valen-preview-0923`、`valen-0-8b`、`valen-2b` 和 `valen-4b`。接口地址为 `https://llm-proxy.openxxlab.com/v1/decisions`，当前这些示例无需密钥即可调用。若你使用的部署需要鉴权，脚本会读取环境变量 `VALEN_API_KEY`，作为 Bearer token 发送。

## 3. 试一张图片

```bash
python cookbook/examples/api.py --image data/smoke/green.png
```

这会针对同一图片询问交通灯颜色，以及是否为绿灯。可以替换成自己的 PNG、JPEG 或 WebP；换成其他主题的图片时，也要相应修改问题，因为示例问题专门针对交通灯。

下面是展示图片请求格式的完整 Python 示例。保存为仓库根目录下的 `my_api.py`，执行 `python my_api.py`：

```python
import base64
import os
from pathlib import Path
import requests

image_b64 = base64.b64encode(Path("data/smoke/green.png").read_bytes()).decode("ascii")
payload = {
    "model": "valen-0-8b",
    "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "Answer using only the visible image."},
        {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
    ]}],
    "questions": [{
        "name": "colour", "type": "choice",
        "instructions": "Which colour is the traffic light?",
        "choices": [
            {"value": "red", "description": "Red"},
            {"value": "green", "description": "Green"},
            {"value": "unknown", "description": "No traffic light is visible"},
        ],
    }],
    "temperature": 1.0,
}
headers = {}
if os.environ.get("VALEN_API_KEY"):
    headers["Authorization"] = "Bearer " + os.environ["VALEN_API_KEY"]
response = requests.post(
    "https://llm-proxy.openxxlab.com/v1/decisions",
    headers=headers, json=payload, timeout=120,
)
print("HTTP:", response.status_code)
print(response.text)
response.raise_for_status()
```

JPEG 图片应使用 `data:image/jpeg;base64,...`；可运行脚本会自动根据扩展名选择类型。视频请使用[本地推理教程](local_inference_zh.md#4-运行图片和视频)，本节 API 示例覆盖文本和图片。

## 4. 提出自己的问题

在仓库根目录将以下内容保存为 `custom_api.json`：

```json
{
  "input": "There are two cats in the room and no dogs.",
  "questions": [
    {
      "name": "animal", "type": "choice",
      "instructions": "Which animal is present?",
      "choices": [
        {"value": "cat", "description": "A cat"},
        {"value": "dog", "description": "A dog"}
      ]
    },
    {"name": "has_dog", "type": "predicate", "instructions": "There is a dog in the room."},
    {
      "name": "cats", "type": "score",
      "instructions": "How many cats are in the room?",
      "levels": [
        {"label": "none", "description": "No cats"},
        {"label": "one", "description": "Exactly one cat"},
        {"label": "multiple", "description": "Two or more cats"}
      ]
    }
  ]
}
```

```bash
python cookbook/examples/api.py --model valen-2b --request custom_api.json
python cookbook/examples/api.py --request custom_api.json > result.json
```

每道题使用唯一的 `name`。评分等级按列表顺序对应数值：上面三个标签的索引为 **0、1、2**。`score` 是按预测概率加权的平均索引，因此小数结果是正常的。`--model` 会覆盖 JSON 文件中的模型名称。

## API 与本地请求的区别

| 含义 | API | 本地 `model.predict()` |
| --- | --- | --- |
| 输入 state | `input`：字符串或消息数组 | `state`：字符串或 `{"messages": [...]}` |
| 图片内容 | `input_image`，`image_url` 为 data URI 字符串 | `image_url`，内容为 `{"url": "local/path.png"}` |
| 多个问题 | 数组，每项包含 `name` | 以问题 ID 为键的对象 |
| 选择题候选 | `choices`：`{value, description}` 数组 | `criteria`：候选名 → 描述的对象 |
| 判断题 | `predicate` | `noul` |
| 评分等级 | `levels`：`{label, description}` 数组 | `criteria`：描述列表 |

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| HTTP 400 提到 `image`、`context`、`criteria` 或 `noul` | 使用上面的 API 格式：`input`、问题数组、`choices`/`levels` 和 `predicate`。 |
| 提示模型不存在 | 使用 `valen-0-8b` 等 API 名称，不能直接填写 Hugging Face 仓库 ID。 |
| `model_id` 或 `revision` 与预期不符 | 请服务维护者确认该名称配置的模型权重和模型元信息。 |
| HTTP 401/403 | 检查部署的访问要求及 `VALEN_API_KEY`。 |
| 超时或连接失败 | 检查网络和 `--url`；服务正在加载模型或排队时，可增大 `--timeout`。 |
| 找不到图片 | 在仓库根目录执行，或传入图片的完整本地路径。 |
| 答案不符合预期 | 确认问题和候选描述与输入相关，温度先使用 `1.0`。 |

其他部署可通过 `VALEN_API_URL` 或 `--url` 设置地址。通过 IP 代理访问且需要 Host 请求头时，将 `VALEN_API_HOST` 设置为代理域名。默认 HTTPS 地址会自动处理 Host。

参考：[Requests 官方快速入门](https://requests.readthedocs.io/en/latest/user/quickstart/)。[继续了解本地模型 →](local_inference_zh.md)
