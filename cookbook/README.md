# Valen Cookbook

**English** · [简体中文](README_zh.md) · [Project home](../README.md)

Start here to get your first Valen result. Valen takes a state and your questions, then returns structured decisions and probabilities. You provide the possible answers for a Choice question, a statement for a yes/no question, or ordered levels for a Score question.

## Choose your route

| What you want | Start here | What you need |
| --- | --- | --- |
| Try Valen or connect it to your app | [Call the API](api.md) | Python 3.10+, internet access; no GPU or model download |
| Run models on your own machine, including video | [Download and run locally](local_inference.md) | Python 3.10+, model weights; a BF16-capable NVIDIA GPU is recommended |

If you are new to models, start with the API. For local inference, start with **Valen-0.8B**, then switch to 2B or 4B using the same examples.

### Get the examples

Run these commands in your terminal. If you already cloned Valen, open its directory instead.

```bash
git clone https://github.com/Liuziyu77/Valen.git
cd Valen
```

All cookbook commands run from this repository root. The image and video examples use small synthetic files already included in `data/smoke/`, so you can run them before preparing your own media.

## Recipes

1. **[Your first API call](api.md):** minimal installation, text and images, all four model aliases, multiple questions, and HTTP errors.
2. **[Download and run a model](local_inference.md):** complete model downloads, local text/image/video inference, shared-state questions, and GPU troubleshooting.

Runnable scripts:

- [api.py](examples/api.py) — text/image API calls, or your own request JSON.
- [download_model.py](examples/download_model.py) — download a complete model or preview its download size.
- [local_inference.py](examples/local_inference.py) — local text, image and video decisions.

Use `python <script> --help` to see its options. Scripts print response JSON to standard output; inference timing goes to standard error, so `> result.json` saves a valid JSON response.

## API names and downloadable models

| API `model` value | Hugging Face model |
| --- | --- |
| `valen-0-8b` | [Valen-Team/Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) |
| `valen-2b` | [Valen-Team/Valen-2B](https://huggingface.co/Valen-Team/Valen-2B) |
| `valen-4b` | [Valen-Team/Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) |
| `valen-preview-0923` | [Valen-Team/Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923), earlier release |

Use the left column for API requests and the right column for downloads. The local Qwen recipes below target the current 0.8B, 2B and 4B models.

## Understand the three decisions

| Task | What you supply | API answer | Local answer |
| --- | --- | --- | --- |
| Choice | Candidate values and descriptions | `choice` and candidate probabilities | `choice` and candidate probabilities |
| Yes/no | A statement to judge | `predicate` → `probability` that the statement is true | `noul` → `noul`, the probability that the statement is true |
| Score | Levels ordered from low to high | `score`, the expected level index | `score`, the expected level index |

For three Score levels, the numerical range is **0–2** and the result can be fractional. Probability is a model estimate; check performance on examples from your own task. A temperature of `1.0` is a useful starting point; very small values make probabilities more extreme.

**API and local requests use different field names.** Follow the request format in the recipe you choose. Both can ask multiple questions about the same input; local Qwen inference uses `execution="shared_state"` to encode their shared state once.

## Next steps

- [Evaluate VisualDecisionBench](../README.md#quick-start).
- [Prepare your own data](../docs/data-format.md).
- [Train or fine-tune Valen](../valen/training/README.md).
- [Report a problem](https://github.com/Liuziyu77/Valen/issues): include your model name, Python/dependency versions, a small request and the error message. Remove credentials before sharing.
