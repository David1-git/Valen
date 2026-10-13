"""Download a complete Valen model. / 下载完整的 Valen 模型。"""
import argparse
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Valen-Team/Valen-0.8B", help="Hugging Face repository ID")
    parser.add_argument("--output", type=Path, help="Default: models/<model name>")
    parser.add_argument("--revision", default="main", help="Branch, tag or commit hash")
    parser.add_argument("--dry-run", action="store_true", help="Show file count and download size without downloading weights")
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.exit(1, "Install the downloader: python -m pip install huggingface-hub==1.7.1\n")
    output = args.output or Path("models") / args.model.rsplit("/", 1)[-1]
    # 下载权重、处理器和自定义代码。 / Include weights, processor and custom code.
    result = snapshot_download(args.model, revision=args.revision, local_dir=output,
                               ignore_patterns=["unmerged/**"], dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({"model": args.model, "revision": result[0].commit_hash if result else None,
                          "files": len(result), "total_gib": round(sum(f.file_size for f in result) / 2**30, 2),
                          "download_gib": round(sum(f.file_size for f in result if f.will_download) / 2**30, 2),
                          "output": str(output)}, indent=2))
    else:
        print(f"Model downloaded to: {result}")
        print(f"Run: python cookbook/examples/local_inference.py --model {shlex.quote(str(output))}")


if __name__ == "__main__":
    main()
