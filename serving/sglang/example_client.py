"""Text and optional image example for the local System One endpoint."""

import argparse
import base64
import json
import mimetypes
from pathlib import Path

import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:30000")
parser.add_argument("--model", default="valen")
parser.add_argument("--image", type=Path)
args = parser.parse_args()
state = "A cat is sitting on the sofa."
if args.image:
    mime = mimetypes.guess_type(args.image.name)[0]
    if not mime or not mime.startswith("image/"):
        parser.error("--image must be a recognized image file")
    data = base64.b64encode(args.image.read_bytes()).decode("ascii")
    state = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"},
                    },
                    {"type": "text", "text": "Inspect this image."},
                ],
            }
        ]
    }
response = requests.post(
    args.base_url.rstrip("/") + "/v1/systemone",
    json={
        "model": args.model,
        "state": state,
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal is present?",
                "criteria": {"cat": "A cat", "dog": "A dog"},
            },
            "on_sofa": {"type": "noul", "instructions": "The cat is on the sofa."},
            "clarity": {
                "type": "score",
                "instructions": "How clearly is the animal identified?",
                "criteria": ["Unclear", "Partially clear", "Clear"],
            },
        },
    },
    timeout=120,
)
response.raise_for_status()
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
