"""Fail before registration when the engine does not match the tested API surface."""

import argparse
import hashlib
import importlib.util
import json
from importlib import metadata
from pathlib import Path


def runtime_lock():
    return json.loads(Path(__file__).with_name("runtime-lock.json").read_text())


def engine_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("SGLang is not installed; use the serving kit's install.sh")
    return Path(spec.origin).resolve().parent


def verify_engine(root=None):
    """Read engine files, never alter them. Reject forks and drifting releases."""
    root = Path(root) if root is not None else engine_root()
    lock = runtime_lock()
    mismatch = []
    for name, expected in lock["files"].items():
        path = root / name
        actual = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if actual != expected:
            mismatch.append(name)
    if mismatch:
        raise RuntimeError(
            "Unsupported SGLang source. This kit requires unmodified upstream "
            f"{lock['upstream_commit']}; mismatched files: {', '.join(mismatch)}. "
            "Use the pinned official environment described in README.md."
        )
    return {
        "engine_root": str(root),
        "upstream_commit": lock["upstream_commit"],
        "verified_files": len(lock["files"]),
    }


def source_tree_fingerprint(root=None):
    """Fingerprint all installed engine Python files, excluding bytecode caches."""
    root = Path(root) if root is not None else engine_root()
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    return {"python_files": count, "sha256": digest.hexdigest()}


def verify_runtime():
    mismatch = {}
    versions = {}
    for name, expected in runtime_lock()["runtime"].items():
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        versions[name] = version
        if version.split("+")[0] != expected:
            mismatch[name] = {"expected": expected, "actual": version}
    if mismatch:
        raise RuntimeError(
            "Runtime version mismatch; run install.sh: " + json.dumps(mismatch)
        )
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-only",
        action="store_true",
        help="Check source compatibility without GPU runtime packages",
    )
    parser.add_argument(
        "--source-tree",
        action="store_true",
        help="Include a fingerprint of all engine Python source files",
    )
    args = parser.parse_args()
    report = verify_engine()
    if not args.engine_only:
        report["runtime"] = verify_runtime()
    if args.source_tree:
        report["source_tree"] = source_tree_fingerprint()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
