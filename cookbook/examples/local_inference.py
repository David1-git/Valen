"""Run local text, image or video decisions. / 在本地执行文本、图片或视频决策。"""
import argparse
import json
from pathlib import Path
import sys
import time


def example_request(text=None, image=None, video=None):
    if image is None and video is None:
        return {"state": text or "A cat is resting on a sofa. There is no dog.", "questions": {
            "animal": {"type": "choice", "instructions": "Which animal is present?",
                       "criteria": {"cat": "A cat", "dog": "A dog", "unknown": "Cannot be determined"}},
            "on_sofa": {"type": "noul", "instructions": "The cat is on the sofa."},
            "cat_count": {"type": "score", "instructions": "How many cats are mentioned?",
                          "criteria": ["No cats", "Exactly one cat", "Two or more cats"]},
        }}
    path = (image or video).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Media path is not a file: {path}")
    kind = "image_url" if image else "video_url"
    when = "" if image else " in the final frame"
    return {"state": {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Observe the traffic light" + when + "."},
        {"type": kind, kind: {"url": str(path)}},
    ]}]}, "questions": {
        "colour": {"type": "choice", "instructions": "Which colour is the traffic light" + when + "?",
                   "criteria": {"red": "Red", "green": "Green", "unknown": "No traffic light is visible"}},
        "green": {"type": "noul", "instructions": "The traffic light is green" + when + "."},
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Valen-Team/Valen-0.8B", help="Hub repository ID or downloaded model directory")
    parser.add_argument("--revision", help="Optional Hub branch, tag or commit hash")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text state; defaults to a short cat example")
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    source.add_argument("--request", type=Path, help="Local-format request JSON; relative media paths use this file's directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--execution", choices=("shared_state", "question"), default="shared_state")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8")) if args.request else example_request(args.text, args.image, args.video)
        if not isinstance(request, dict):
            raise ValueError("The request JSON must be an object")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Invalid input: {exc}\n")
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        parser.exit(1, "Install the local inference dependencies; see cookbook/local_inference.md.\n")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.exit(1, "CUDA is unavailable. Check nvidia-smi and your PyTorch installation, use --device cpu for a slow smoke test, or use the API recipe.\n")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        parser.exit(1, "This recipe requires a GPU with BF16 support. Use the API recipe or --device cpu.\n")
    if args.device == "cpu" and args.attn_implementation == "flash_attention_2":
        parser.error("Flash Attention requires CUDA; use --attn-implementation sdpa on CPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    # 按 checkpoint 保存的精度加载。 / Preserve the dtype recorded in the checkpoint.
    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, dtype="auto",
        attn_implementation=args.attn_implementation, local_files_only=args.local_files_only,
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    media_root = args.request.resolve().parent if args.request else Path.cwd()
    result = model.predict(request, media_root=media_root, execution=args.execution, processor=processor)
    print(f"Loaded model and completed inference in {time.perf_counter() - started:.2f}s", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
