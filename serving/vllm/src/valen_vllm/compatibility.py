"""Check the pinned engine API before plugin registration; never patch files."""

import argparse
import hashlib
import importlib.util
import json
from importlib import metadata
from pathlib import Path


def verify_engine(root=None):
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError(
                "Install the pinned vLLM environment described in README.md"
            )
        root = Path(spec.origin).resolve().parent
    root = Path(root)
    lock = json.loads(Path(__file__).with_name("engine-lock.json").read_text())
    mismatch = []
    for name, expected in lock["files"].items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            mismatch.append(name)
    if mismatch:
        raise RuntimeError(
            f"Unsupported vLLM source; requires {lock['upstream_commit']}. Mismatched files: {', '.join(mismatch)}"
        )
    return {
        "upstream_commit": lock["upstream_commit"],
        "verified_files": len(lock["files"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", action="store_true")
    parser.add_argument("--engine-only", action="store_true")
    args = parser.parse_args()
    result = verify_engine()
    if not args.engine_only:
        result["runtime"] = verify_runtime()
    if args.source_tree:
        root = Path(importlib.util.find_spec("vllm").origin).resolve().parent
        digest = hashlib.sha256()
        count = 0
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
                count += 1
        result["source_tree"] = {"python_files": count, "sha256": digest.hexdigest()}
    print(json.dumps(result, indent=2))


def verify_runtime():
    lock = json.loads(Path(__file__).with_name("runtime-lock.json").read_text())
    actual, mismatch = {}, {}
    for name, expected in lock.items():
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        actual[name] = version
        matches = (
            version == expected if name == "vllm" else version.split("+")[0] == expected
        )
        if not matches:
            mismatch[name] = {"expected": expected, "actual": version}
    if mismatch:
        raise RuntimeError("Unsupported vLLM runtime: " + json.dumps(mismatch))
    return actual


if __name__ == "__main__":
    main()
