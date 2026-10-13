# Call the Valen API

**English** · [简体中文](api_zh.md) · [Cookbook home](README.md)

This route needs no GPU or model weights. You need Python 3.10+ and internet access. [Clone the repository](README.md#get-the-examples) first, then run the commands below from its root directory. Shell commands below use Linux/macOS syntax.

## 1. Install one dependency

```bash
python3 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install -r cookbook/requirements-api.txt
```

On Windows, use `python -m venv .venv-api` and activate with `.venv-api\Scripts\Activate.ps1` in PowerShell.

## 2. Get your first result

```bash
python cookbook/examples/api.py
```

The default request uses `valen-0-8b` and the text **“A cat is resting on a sofa. There is no dog.”** It asks three questions in one request: which animal is present, whether the cat is on the sofa, and how many cats are mentioned.

The script prints the HTTP status and request time, followed by JSON. Look in `answers` for these fields:

```json
{
  "answers": [
    {"type": "choice", "name": "animal", "choice": "cat"},
    {"type": "predicate", "name": "on_sofa", "probability": 0.9},
    {"type": "score", "name": "cat_count", "score": 1.0}
  ]
}
```

This shortened response shows the structure; actual values vary. The response also includes `model_id` and `revision`, which you can record to identify the deployed model version. API `answers` is an **array**; match each answer using its `name`.

### Choose a model or supply your own text

```bash
python cookbook/examples/api.py --model valen-2b
python cookbook/examples/api.py --model valen-4b
python cookbook/examples/api.py --model valen-preview-0923
python cookbook/examples/api.py --text "Two cats are resting on a sofa. There is no dog."
```

The API names are `valen-preview-0923`, `valen-0-8b`, `valen-2b` and `valen-4b`. The endpoint is `https://llm-proxy.openxxlab.com/v1/decisions`. It currently accepts these examples without a key. If your deployment requires authentication, the script reads `VALEN_API_KEY` from the environment and sends a Bearer token.

## 3. Try an image

```bash
python cookbook/examples/api.py --image data/smoke/green.png
```

This asks two questions about the same image: the traffic-light colour and whether it is green. Replace the file with your own PNG, JPEG or WebP. For different subjects, change the questions too; the example questions are specifically about traffic lights.

The following complete Python example shows the image request format. Save it as `my_api.py` in the repository root and run `python my_api.py`:

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

For JPEG files, use `data:image/jpeg;base64,...`; the runnable script selects the MIME type for you. For video, use the [local inference recipe](local_inference.md#4-run-images-and-videos); the API examples here cover text and images.

## 4. Ask your own questions

Save the following as `custom_api.json` in the repository root:

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

Give each question a unique `name`. Score levels follow list order: the three labels above correspond to numerical indices **0, 1, 2**. `score` is their probability-weighted average, so a fractional result is valid. `--model` overrides a model name stored in the JSON file.

## Request formats: API versus local

| Meaning | API | Local `model.predict()` |
| --- | --- | --- |
| State | `input`: string or message array | `state`: string or `{"messages": [...]}` |
| Image part | `input_image`, string `image_url` containing a data URI | `image_url`, `{"url": "local/path.png"}` |
| Questions | Array, each with a `name` | Object keyed by question ID |
| Choice candidates | `choices`: array of `{value, description}` | `criteria`: object of name → description |
| Yes/no task | `predicate` | `noul` |
| Score levels | `levels`: array of `{label, description}` | `criteria`: list of descriptions |

## Troubleshooting

| Problem | What to do |
| --- | --- |
| HTTP 400 mentioning `image`, `context`, `criteria` or `noul` | Use the API format above: `input`, a question array, `choices`/`levels`, and `predicate`. |
| Model not found | Use an API alias such as `valen-0-8b`, rather than a Hugging Face repository ID. |
| Unexpected `model_id` or `revision` | Ask the service operator to confirm the weights and model metadata configured for that alias. |
| HTTP 401/403 | Check the deployment's access requirements and `VALEN_API_KEY`. |
| Timeout or connection error | Check network access and `--url`; increase `--timeout` if the service is loading a model or queued. |
| Image file not found | Run from the repository root or supply the image's full local path. |
| Unexpected answers | Confirm that the questions and candidate descriptions match your input. Start with temperature `1.0`. |

For another deployment, set `VALEN_API_URL` or pass `--url`. When connecting through an IP-based proxy that requires a Host header, set `VALEN_API_HOST` to its hostname. The default HTTPS URL handles this automatically.

Reference: [Requests quickstart](https://requests.readthedocs.io/en/latest/user/quickstart/). [Continue with local models →](local_inference.md)
