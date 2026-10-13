# Valen technical notes

**English** · [简体中文](technical_zh.md) · [README](../README.md)

This page covers the model architecture, output protocol, training workflow and full experiment results. All commands run from the repository root.

[Output types](#output-types) · [Model architecture](#model-architecture) · [Requirements](#requirements) · [Training](#training) · [Full experiments](#full-experiments)

## Output types

| Type | Use | Returns |
| --- | --- | --- |
| **Choice** | Pick one of 1–255 candidates | Chosen candidate, full probability distribution, confidence |
| **Noul** | Decide whether a condition holds | Probability of `true` |
| **Score** | Rate against 2–10 ordered descriptions | Expected grade index, grade probabilities, legend, confidence |

Candidates are passed in with each question, and all tasks and candidate counts share the same decision head. `confidence` describes how concentrated the distribution is.

## Model architecture

<p align="center">
  <img src="../assets/figures/readme-architecture.png" alt="Valen architecture: multimodal inputs and candidates pass through the Qwen3.5 backbone, an additional shared decision head and softmax to produce Choice, Noul or Score outputs." width="1000">
</p>

Choice and Noul compute all candidates of a question in one branch; Score runs a separate backbone forward pass for each grade, then normalizes the logits together. The shared state is encoded only once but is recomputed in every backbone branch. See the [architecture notes](architecture.md) and the [response fields and formulas](evaluation.md).

<a id="requirements"></a>

## Requirements

- Linux, Python 3.10+, NVIDIA GPU.
- PyTorch 2.6.0, torchvision 0.21.0, Transformers 5.4.0, PEFT 0.18.1.
- See [pyproject.toml](../pyproject.toml) for the full dependency list; the setup script installs everything automatically.

```bash
git clone https://github.com/Liuziyu77/Valen.git Valen
cd Valen
bash scripts/setup/bootstrap.sh
source .venv/bin/activate
```

If you already have a Conda or virtualenv environment, see the [manual installation guide](../scripts/README.md#installation). The setup script creates `.venv`; model weights must be downloaded separately.

## Local training and smoke checks

Install the dependencies as described in [Requirements](#requirements) first, then download the model weights.

```bash
# Download the Qwen3.5-2B base model.
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# Train the decision head on the synthetic samples shipped with the repository.
python -m valen.train \
  --config configs/train/qwen/sft_warmup.json

# Run inference with the resulting checkpoint.
python -m valen.inference \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/predictions.jsonl

# Check the evaluation pipeline on the same synthetic samples.
python -m valen.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval
```

`data/smoke` contains a handful of simple questions and is only meant to check that everything runs.

<details>
<summary>A labeled record with an image input</summary>

Each line of a JSONL file is one record. The example below uses the repository's [evaluation overview figure](../assets/figures/evaluation-results.png) and assumes it is saved as `example.jsonl` at the repository root.

```json
{
  "group_id": "evaluation-general-2b",
  "request": {
    "state": {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "Compare the accuracy and average per-question latency of the 2B models on the General task in the figure."},
          {"type": "image_url", "image_url": {"url": "assets/figures/evaluation-results.png"}}
        ]
      }]
    },
    "questions": {
      "best_2b": {
        "type": "choice",
        "instructions": "On the General task, which 2B model with an average per-question latency below 200 ms has the highest accuracy?",
        "criteria": {
          "qwen": "Qwen3.5-2B",
          "valen": "Valen-Preview-0923"
        }
      }
    }
  },
  "targets": {
    "best_2b": {"probabilities": {"qwen": 0.0, "valen": 1.0}}
  }
}
```
</details>

<a id="training"></a>

## Training

Valen currently supports two training `method`s, `SFT` and `RLCD`. `stage` decides which parameters are updated, and four stage configurations are currently supported.

<p align="center">
  <img src="../assets/figures/readme-training-workflow.png" alt="Train with SFT first, optionally initialize RLCD from its checkpoint, then evaluate on a held-out test set." width="1000">
</p>

Both SFT and RLCD use labeled training data. Checkpoints from either method can be evaluated on a held-out test set, and loading them also requires the matching base model (the current implementation uses LoRA; full fine-tuning may be considered later).

| Stage | Updated parameters | SFT | RLCD |
| --- | --- | --- | --- |
| `warmup` | Decision head | [config](../configs/train/qwen/sft_warmup.json) | [config](../configs/train/qwen/rlcd_warmup.json) |
| `text` | LLM LoRA, decision head; uses text-only data | [config](../configs/train/qwen/sft_text.json) | [config](../configs/train/qwen/rlcd_text.json) |
| `joint` | LLM LoRA, ViT merger, decision head | [config](../configs/train/qwen/sft_joint.json) | [config](../configs/train/qwen/rlcd_joint.json) |
| `vision_top` | `joint` plus the last 4 layers of the vision encoder | [config](../configs/train/qwen/sft_vision_top.json) | [config](../configs/train/qwen/rlcd_vision_top.json) |

Before real training, copy a config and set `model_path`, `data`, `output`, `epochs` and `max_steps`. `tokens_per_step` is a **per-GPU** budget, counted as the total tokens of all question branches in a state. Every GPU holds the full model, using **data parallelism**. See the [configuration reference](configuration.md) for the defaults and examples of changing them.

```bash
# Single node, eight GPUs; use a new output directory for each independent experiment.
VALEN_GPUS=8 bash scripts/train/launch.sh configs/train/qwen/sft_warmup.json

# Initialize RLCD from a trained SFT decision-head checkpoint.
VALEN_GPUS=8 bash scripts/train/launch.sh configs/train/qwen/rlcd_warmup.json \
  --initialize output/sft_warmup/latest
```

SFT currently uses label supervision. RLCD takes a GRPO-style form whose reward accounts for both correctness and confidence error, with a KL constraint toward a fixed reference policy and an optional Brier loss. See the [training guide](../valen/training/README.md) for the formulas, initialization and resume commands.

A checkpoint saves the values of the trainable parameters and the training state under `<output>/latest/`; the frozen base model is loaded separately. Each save overwrites `latest/` and keeps no earlier versions. `--initialize` starts a new experiment from existing weights; `--resume` restores the optimizer, data cursor and per-rank state, and requires the same number of processes. `launch.sh` serves both SFT and RLCD, and `VALEN_PYTHON` selects the interpreter.

<a id="full-experiments"></a>

## Full experiments

In the tables, `Valen-Base-*` are general-task models and `Valen-Sokoban-*` are Sokoban models; RLCD is this repository's reinforcement-learning training method. Each group of experiments uses checkpoints trained on its own task.

### Model Card

We ran ablations over the training method and the training data:

| Model | Training method | Data | Dataset |
| --- | --- | --- | --- |
| Valen-Base-SFT-0.8B | SFT | General SFT 100k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-SFT-2B | SFT | General SFT 100k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-RLCD-0.8B | SFT + RLCD | General SFT 70k + General RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-RLCD-2B | SFT + RLCD | General SFT 70k + General RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Sokoban-SFT-2B | SFT + SFT | General SFT 100k + Sokoban SFT 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k)&nbsp;[![Sokoban](https://img.shields.io/badge/Sokoban-8A63B8?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) |
| Valen-Sokoban-RLCD-2B | SFT + RLCD | General SFT 100k + Sokoban RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k)&nbsp;[![Sokoban](https://img.shields.io/badge/Sokoban-8A63B8?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) |

The Sokoban training and evaluation sets are both included in [Valen-Eval-Game](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game); training and evaluation each use their own split.

#### Training Loss

<p align="center">
  <a href="../assets/figures/training/sft100k_loss.png"><img src="../assets/figures/training/sft100k_loss.png" alt="Training loss curves of Valen-Base-SFT-0.8B and 2B on the General 100k data." width="1000"></a><br>
  <sub>General SFT: training loss of 0.8B and 2B.</sub>
</p>

<p align="center">
  <a href="../assets/figures/training/sokoban_rlcd_loss_reward.png"><img src="../assets/figures/training/sokoban_rlcd_loss_reward.png" alt="Loss and mean reward of the Valen-Sokoban-RLCD 0.8B and 2B models during training." width="1000"></a><br>
  <sub>Sokoban RLCD: training loss and reward of 0.8B and 2B.</sub>
</p>
