# Valen Cookbook

[English](README.md) · **简体中文** · [项目首页](../README_zh.md)

从这里获得你的第一个 Valen 结果。Valen 接收一段输入（state）和你的问题，返回结构化的决策与概率。选择题需要你提供候选答案，判断题提供待判断的陈述，评分题提供从低到高排列的等级。

## 选择上手路线

| 你的需求 | 从这里开始 | 需要准备什么 |
| --- | --- | --- |
| 体验 Valen，或接入自己的应用 | [调用 API](api_zh.md) | Python 3.10+、网络；不需要 GPU，也不用下载模型 |
| 在自己的机器上运行模型，包括视频推理 | [下载模型并本地推理](local_inference_zh.md) | Python 3.10+、模型权重；推荐支持 BF16 的 NVIDIA GPU |

第一次接触模型，建议先用 API。本地推理建议从 **Valen-0.8B** 开始，跑通后用相同的示例切换到 2B 或 4B。

### 获取示例

在终端执行以下命令。如果已经下载了 Valen 仓库，直接进入仓库目录即可。

```bash
git clone https://github.com/Liuziyu77/Valen.git
cd Valen
```

Cookbook 的命令都在这个仓库根目录执行。图片和视频示例使用 `data/smoke/` 中随仓库提供的小型合成素材，可以先跑通，再换成自己的文件。

## 上手教程

1. **[第一次调用 API](api_zh.md)：** 最少依赖、文本和图片、四个模型名称、多问题请求、HTTP 报错处理。
2. **[下载模型并运行](local_inference_zh.md)：** 下载完整模型、本地文本/图片/视频推理、共享 state 多题处理、GPU 常见问题。

可以直接运行的脚本：

- [api.py](examples/api.py)：文本和图片 API 调用，也支持自己的请求 JSON。
- [download_model.py](examples/download_model.py)：下载完整模型，或先查看下载大小。
- [local_inference.py](examples/local_inference.py)：本地文本、图片和视频决策。

运行 `python <脚本路径> --help` 查看参数。脚本将响应 JSON 输出到标准输出，推理耗时输出到标准错误，因此可以用 `> result.json` 保存有效的 JSON 响应。

## API 名称与模型下载地址

| API 的 `model` 值 | Hugging Face 模型 |
| --- | --- |
| `valen-0-8b` | [Valen-Team/Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) |
| `valen-2b` | [Valen-Team/Valen-2B](https://huggingface.co/Valen-Team/Valen-2B) |
| `valen-4b` | [Valen-Team/Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) |
| `valen-preview-0923` | [Valen-Team/Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923)，早期版本 |

调用 API 用左列，下载模型用右列。本文的 Qwen 本地推理教程面向当前的 0.8B、2B 和 4B 模型。

## 理解三种决策

| 任务 | 你需要提供什么 | API 返回 | 本地返回 |
| --- | --- | --- | --- |
| Choice 选择题 | 候选值及其描述 | `choice` 和各候选的概率 | `choice` 和各候选的概率 |
| 判断题 | 待判断的陈述 | `predicate` → `probability`，表示陈述为真的概率 | `noul` → `noul`，表示陈述为真的概率 |
| Score 评分题 | 从低到高排列的等级 | `score`，等级索引的期望值 | `score`，等级索引的期望值 |

提供三个评分等级时，数值范围是 **0–2**，结果可以是小数。概率是模型的估计，实际使用前可以用自己的任务样例检查效果。温度建议从 `1.0` 开始；很小的温度会让概率更接近 0 或 1。

**API 和本地推理的字段名称不同。** 请使用对应教程中的请求格式。两者都可以针对同一输入提出多个问题；本地 Qwen 推理通过 `execution="shared_state"` 对它们共有的输入编码一次。

## 后续使用

- [评估 VisualDecisionBench](../README_zh.md#快速开始)。
- [准备自己的数据](../docs/data-format.md)。
- [训练或微调 Valen](../valen/training/README.md)。
- [反馈问题](https://github.com/Liuziyu77/Valen/issues)：附上模型名称、Python/依赖版本、最小请求和报错，分享前移除凭证。
