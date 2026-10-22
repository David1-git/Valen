# Valen SGLang 部署套件

在兼容的现有 SGLang 环境中安装一个小型 `valen-sglang` wheel，即可获得 Valen 原生决策支持。这是预构建 Valen-Team 镜像和 Transformers 直接推理之外的一种部署方式，底层仍使用 SGLang 引擎。

部署前请阅读 `VALIDATION.md`，确认已验证的环境及检查范围。

## 安装了什么

wheel 包含 Valen 配置、共享状态 prompt 编译器、FP32 Mixer 决策头和模型/请求适配。通过 `sglang.srt.plugins` entry point 和 SGLang 的通用插件 hooks 注册。包内没有整棵 SGLang 源码，不替换引擎包，也不修改检查点或已安装的引擎文件。

适配复用官方 Qwen3.5 执行路径及 embedding 调度，在现有 `/v1/systemone` 路由上处理 Valen 请求。其他模型沿用原有实现；生成式决策模型仍限制为十个 Score 等级。

## 支持的环境

- Linux x86-64、NVIDIA H200 和兼容的 CUDA 13 运行时。该套件尚未验证其他 GPU。
- **未经修改的官方 SGLang 提交 `c1c287a42928495166e14dbcef31b9e1f1dac199`。** 对应官方基础镜像为 `docker.io/lmsysorg/sglang@sha256:a8eedda74b54f7dfecc445ee66d298f8d8f6f50192201a53b0a6dd7571ff8d26`。
- `valen_sglang/runtime-lock.json` 固定的依赖，包括 PyTorch 2.14.1、Transformers 5.19.0、sglang-kernel 0.4.10、FlashInfer 0.6.18、CUTLASS DSL 4.6.2。

第一版会检查十个关键引擎文件的 SHA256，拒绝不兼容的版本，也会拒绝已内置适配的 Valen fork/镜像。不要跳过检查，也不能假设新运行一次 `pip install sglang` 就会得到正确版本。

`install.sh` 面向已经准备好的上述 SGLang 环境，不负责从空 venv 安装整套依赖，并会更换其中部分运行时版本。请与 Transformers 训练/推理环境隔离。

## 安装和启动

构建后的检查点侧套件如下：

```text
sglang/
  valen_sglang-<version>-py3-none-any.whl
  install.sh
  serve.sh
  example_client.py
  validate.py
  README.md
  README_zh.md
  VALIDATION.md
  LICENSE
  NOTICE
  SHA256SUMS
```

在支持的环境中运行：

```bash
bash /path/to/checkpoint/sglang/install.sh
MODEL=/path/to/checkpoint \
  CUDA_VISIBLE_DEVICES=0 \
  bash /path/to/checkpoint/sglang/serve.sh
```

服务端不需要安装 Valen 仓库、克隆 SGLang fork、构建 Valen 专用镜像，也不需要 `trust_remote_code`。现有环境可以是 Python 安装，也可以是官方 SGLang 容器。使用容器时，先挂载完整检查点和套件，将容器内的 `HOST` 设为 `0.0.0.0`，宿主机发布端口则绑定 localhost。

安装器先验证套件校验和、安装一个插件 wheel 并检查引擎，再固定运行时、清理不兼容的 FlashInfer JIT cache、检查依赖版本。FlashInfer cubin 使用 `https://flashinfer.ai/whl`，其余包使用环境的 pip 源。离线安装需要预先准备全部 wheel，并设置 `PIP_NO_INDEX=1 PIP_FIND_LINKS=/path/to/wheels`；完整的基础 SGLang 环境仍然必需。

内核包按固定版本安装，不重新求解其传递依赖，依赖基础环境已提供的组件。这是指定环境中的固定组合，不代表任意 pip 组合可用，也没有承诺整个环境的 `pip check` 通过。

`serve.sh` 支持：

| 环境变量 | 默认值 |
| --- | --- |
| `MODEL` | `Valen-Team/Valen-2B` |
| `REVISION` | 不设置；使用 Hub 时建议填写固定 revision |
| `MODEL_NAME` | `valen` |
| `HOST` | `127.0.0.1` |
| `PORT` | `30000` |
| `CUDA_VISIBLE_DEVICES` | 由调用者指定一张 GPU |

额外参数会传给引擎。脚本使用 FA3。启动器会在启动前检查引擎和运行时，并要求所有 Valen hooks 完整加载。若设置了 `SGLANG_PLUGINS`，其中必须包含 `valen`。

## 发送请求

等待服务启动预热结束，在另一个客户端终端运行：

```bash
python /path/to/checkpoint/sglang/example_client.py
python /path/to/checkpoint/sglang/example_client.py --image /path/to/cat.jpg
```

接口和 JSON 格式与 Valen-Team 镜像相同：

```bash
curl http://127.0.0.1:30000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","state":"A cat is on the sofa.","questions":{"on_sofa":{"type":"noul","instructions":"The cat is on the sofa."}}}'
```

响应 `answers` 按问题 ID 返回：`choice` 给出候选概率，`noul` 给出命题为真的概率，`score` 给出从零开始的等级期望及概率分布。Choice 使用 `criteria` 映射，Score 使用有序列表，Noul 不覆盖 criteria。

这是本地接口，与托管 `/v1/decisions` 的请求体不同。

## 支持范围

- 已发布的完整 Valen Qwen3.5 Mixer 检查点，`shared_state` 执行，单 GPU，TP=PP=1。
- 文本和图片；Choice 支持 1–255 个候选，Noul 是非判断，Score 支持 2–255 个等级。
- 已发布检查点 auto dtype 下的 BF16 backbone；决策头参数与池化为 FP32。
- 本版不支持直接 `video_url`、推测解码、分布式决策头、量化或仅含 adapter 的检查点。
- 每个请求执行一次完整 prefill；决策路径关闭 prefix 复用、分块 prefill 和 CUDA graph。

## 验证新检查点

```bash
python /path/to/checkpoint/sglang/validate.py \
  --model-path /path/to/checkpoint --output-dir /path/to/results
```

脚本分进程运行检查点的 Transformers 参考实现和插件服务，对照文本/图片请求的 token 数、选择与概率，再检查并发请求。默认概率绝对误差容差为 0.02。保留完整结果、runtime freeze、引擎指纹、检查点 revision 和硬件信息。新检查点或新引擎都需要重新验证。

## 从源码构建并附加到检查点

源码位于 `Valen/serving/sglang`，是独立于 `valen` 的 Python distribution。

```bash
python -m pip install build
python -m build --wheel /path/to/Valen/serving/sglang
python /path/to/Valen/serving/sglang/build_kit.py \
  /path/to/Valen/serving/sglang/dist/valen_sglang-<version>-py3-none-any.whl \
  /path/to/exported-checkpoint/sglang
```

目标目录必须不存在。构建器复制 wheel、中英文说明、验证记录、脚本和许可文件，并生成 `SHA256SUMS`，不复制或上传模型权重。通过该检查点的测试后，可将生成的 `sglang/` 目录随检查点发布。

引擎源码锁是有意保留的约束。更新时需要审查 hook API 与行为，重新构建并完成 CPU/GPU 测试；只更新哈希不能证明兼容。
