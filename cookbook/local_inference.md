# Download and run Valen locally

**English** · [简体中文](local_inference_zh.md) · [Cookbook home](README.md)

Use this route to keep inference on your machine and process text, images or video. Start with **Valen-0.8B**. You need Python 3.10+, enough disk space for the complete model, and preferably an NVIDIA GPU with BF16 support. GPU memory use depends on model size, image resolution, video length and question count; the weight-file size alone is not a minimum VRAM requirement.

[Clone the repository](README.md#get-the-examples) first. Run all commands from its root. The installation below targets a Linux NVIDIA GPU environment; CPU mode is available for slow smoke checks.

## 1. Install the local environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -e .
```

Check that your GPU is visible:

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Use Python **3.10 or 3.11** for this recipe. The project pins Transformers **5.4.0** and other dependencies in [pyproject.toml](../pyproject.toml). Start with SDPA attention; Flash Attention is optional and is not needed for these examples.

## 2. Download the complete model

For downloading alone, install `python -m pip install huggingface-hub==1.7.1` and run this step. Downloading does not require a GPU or PyTorch.

```bash
# Optional: inspect the files and download size first.
python cookbook/examples/download_model.py --dry-run

# Downloads to models/Valen-0.8B.
python cookbook/examples/download_model.py
```

The download includes weights, tokenizer, processor and custom model code. A complete published Valen model does **not** require a separate Qwen base-model download. The script skips optional historical `unmerged/` bundles.

Other sizes use the same command:

```bash
python cookbook/examples/download_model.py --model Valen-Team/Valen-2B
python cookbook/examples/download_model.py --model Valen-Team/Valen-4B
```

Downloads can be rerun to reuse completed files. If Hugging Face requires authentication, run `hf auth login`. Use `--revision <commit-hash>` when you need a fixed model version. See the [Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download) for download and cache details.

## 3. Run your first local prediction

```bash
python cookbook/examples/local_inference.py \
  --model models/Valen-0.8B --local-files-only
```

This uses the same cat-on-a-sofa text example as the API recipe and asks Choice, Noul and Score questions together. The response's `answers` is an **object keyed by question ID**:

```json
{
  "answers": {
    "animal": {"type": "choice", "choice": "cat"},
    "on_sofa": {"type": "noul", "noul": 0.9},
    "cat_count": {"type": "score", "score": 1.0}
  }
}
```

This shortened response illustrates the fields; actual values vary. `noul` is the probability that the statement is true. With three Score levels, the score is a probability-weighted index in the range 0–2.

You can also let the loader download/cache a Hub model automatically:

```bash
python cookbook/examples/local_inference.py --model Valen-Team/Valen-0.8B
```

The first run includes model loading and downloads. To switch sizes after downloading, pass `--model models/Valen-2B` or `--model models/Valen-4B`.

## 4. Run images and videos

These small files are already in the repository:

```bash
python cookbook/examples/local_inference.py --model models/Valen-0.8B \
  --image data/smoke/green.png --local-files-only

python cookbook/examples/local_inference.py --model models/Valen-0.8B \
  --video data/smoke/red_to_green.mp4 --local-files-only
```

The image example asks about a green traffic light. The video changes from red to green, and the example questions ask about its **final frame**. These synthetic inputs check the inference flow; they are not a benchmark. Replace the media paths and change the questions to suit your own task.

Local media paths are resolved relative to `media_root`, or the request JSON's directory when using `--request`. These scripts accept local files. Download web images/videos first. Published Qwen model configurations use 16 sampled video frames by default; inference retains the model's saved media settings.

## 5. Use the model in Python

Save the following as `my_local.py` in the repository root, then run `python my_local.py`:

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

`dtype="auto"` keeps the dtype recorded in the checkpoint. `trust_remote_code=True` loads the model's custom decision implementation. Use `model.predict()` for these structured decisions.

For text, replace `state` with a string. For video, replace the image part with `{"type": "video_url", "video_url": {"url": "data/smoke/red_to_green.mp4"}}` and adjust the questions.

### Customize questions and shared-state inference

Use a unique key for each question in `questions`. Choice `criteria` is a candidate-name → description object; Noul needs only the statement; Score `criteria` is an ordered list of descriptions. See the [complete data format](../docs/data-format.md).

Save a local-format request as `custom_local.json`, then run:

```bash
python cookbook/examples/local_inference.py --model models/Valen-0.8B \
  --request custom_local.json --local-files-only > result.json
```

The JSON contains `state` and `questions`, as in the Python example. It can also contain the outer `request` wrapper used in training data. For media, paths are relative to this JSON file's directory.

The default `--execution shared_state` encodes the shared input once and reads all its questions. Use `--execution question` to run questions independently. For a compatible CUDA/Flash Attention installation, select `--attn-implementation flash_attention_2`.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| CUDA unavailable | Check `nvidia-smi`, GPU visibility and the installed PyTorch build. Use the [API](api.md) without a GPU, or add `--device cpu` for a slow smoke check. |
| CUDA out of memory | Start with 0.8B, smaller images, a shorter video and fewer questions. Keep free memory for activations as well as weights. |
| `qwen3_5` or custom-code import error | Activate the local environment and install the versions in `pyproject.toml`; download the complete model directory. |
| Missing Flash Attention | Use the default `sdpa`. Install a compatible Flash Attention build before selecting it. |
| Media file not found | Check the path and `media_root`; with `--request`, paths are relative to the JSON file. |
| Offline loading fails | Download the full snapshot, then pass its local path and `--local-files-only`. |
| Hugging Face cache permission error | Set `HF_HOME` to a directory you can write to, then rerun the command. |
| API fields rejected by local inference | Use local `state`, question IDs, `criteria` and `noul`; [compare the formats](api.md#request-formats-api-versus-local). |

[Next: evaluate VisualDecisionBench](../README.md#quick-start) · [Training guide](../valen/training/README.md)
