"""Call Valen with text or an image. / 使用文本或图片调用 Valen API。"""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import sys
import time


def example_payload(image=None, text=None):
    if image is None:
        return {
            "input": text or "A cat is resting on a sofa. There is no dog.",
            "questions": [
                {"name": "animal", "type": "choice", "instructions": "Which animal is present?",
                 "choices": [{"value": "cat", "description": "A cat"},
                             {"value": "dog", "description": "A dog"},
                             {"value": "unknown", "description": "Cannot be determined"}]},
                {"name": "on_sofa", "type": "predicate", "instructions": "The cat is on the sofa."},
                {"name": "cat_count", "type": "score", "instructions": "How many cats are mentioned?",
                 "levels": [{"label": "none", "description": "No cats"},
                            {"label": "one", "description": "Exactly one cat"},
                            {"label": "multiple", "description": "Two or more cats"}]},
            ],
        }
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(image.suffix.lower())
    if mime is None:
        raise ValueError("Use a PNG, JPEG or WebP image")
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return {
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Answer using only the visible image."},
            {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"},
        ]}],
        "questions": [
            {"name": "colour", "type": "choice", "instructions": "Which colour is the traffic light?",
             "choices": [{"value": "red", "description": "Red"},
                         {"value": "green", "description": "Green"},
                         {"value": "unknown", "description": "No traffic light is visible"}]},
            {"name": "green", "type": "predicate", "instructions": "The traffic light is green."},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="API model name, e.g. valen-0-8b (default when omitted)")
    parser.add_argument("--url", default=os.environ.get("VALEN_API_URL", "https://llm-proxy.openxxlab.com/v1/decisions"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Local PNG, JPEG or WebP image")
    source.add_argument("--text", help="Text state; defaults to a short cat example")
    source.add_argument("--request", type=Path, help="JSON file containing an API payload")
    parser.add_argument("--temperature", type=float, help="Positive calibration temperature (default: 1.0)")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        import requests
    except ImportError:
        parser.exit(1, "Install the API dependency: python -m pip install -r cookbook/requirements-api.txt\n")
    try:
        payload = json.loads(args.request.read_text(encoding="utf-8")) if args.request else example_payload(args.image, args.text)
        if not isinstance(payload, dict):
            raise ValueError("The request JSON must be an object")
        payload["model"] = args.model or payload.get("model", "valen-0-8b")
        temperature = args.temperature if args.temperature is not None else payload.get("temperature", 1.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        payload["temperature"] = temperature
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Invalid input: {exc}\n")
    headers = {}
    # 凭证从环境变量读取。 / Read credentials from the environment.
    if os.environ.get("VALEN_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VALEN_API_KEY"]
    if os.environ.get("VALEN_API_HOST"):
        headers["Host"] = os.environ["VALEN_API_HOST"]
    started = time.perf_counter()
    try:
        response = requests.post(args.url, headers=headers, json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        parser.exit(1, f"Request failed: {exc}\nCheck --url and your network connection.\n")
    print(f"HTTP {response.status_code}; {time.perf_counter() - started:.2f}s including network time", file=sys.stderr)
    try:
        body = response.json()
    except ValueError:
        parser.exit(1, "The server returned a non-JSON response. Check the endpoint URL.\n")
    if not response.ok:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        parser.exit(1, f"API error: {detail}\n")
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
