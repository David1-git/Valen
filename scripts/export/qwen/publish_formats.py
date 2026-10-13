"""Publish only manifested, validated Valen exports and verify every remote file."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
from huggingface_hub import HfApi, CommitOperationAdd, get_token, hf_hub_download
from huggingface_hub.utils import disable_progress_bars
if __package__:
    from .export_formats import DEFAULT_MODELS, DEFAULT_MODELS_DIR, DEFAULT_WORK_DIR, file_hash
else:
    from export_formats import DEFAULT_MODELS, DEFAULT_MODELS_DIR, DEFAULT_WORK_DIR, file_hash

disable_progress_bars()


def publish(name, fmt, api, token, *, exports_dir=DEFAULT_MODELS_DIR, work_dir=DEFAULT_WORK_DIR, namespace=None):
    directory = Path(exports_dir) / (name + "-" + fmt)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != fmt:
        raise ValueError("Conversion format differs from requested format: " + str(directory))
    if namespace:
        manifest["target_repo"] = namespace + "/" + name + "-" + fmt
    manifest["files"] = [item for item in manifest["files"] if item["path"] != ".gitattributes"]
    manifest["hub_managed_files"] = [".gitattributes"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validation = json.loads((directory / "validation.json").read_text())
    if not manifest.get("validated") or not validation.get("passed"):
        raise ValueError("Refusing an unvalidated export: " + str(directory))
    expected = {item["path"]: item for item in manifest["files"]}
    expected["conversion_manifest.json"] = {"path": "conversion_manifest.json", "bytes": manifest_path.stat().st_size,
                                             "sha256": file_hash(manifest_path)}
    for item in expected.values():
        path = directory / item["path"]
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("Prepared file changed after validation: " + str(path))
    # Check the source card before any upload. / 上传前核对原始模型说明。
    source_card_path = hf_hub_download(manifest["source_repo"], "README.md", revision=manifest["source_revision"],
                                     token=token, cache_dir=str(work_dir / "hf-cache"))
    source_card = Path(source_card_path).read_text()
    declaration = f"\n\nThis is the **{fmt}** version of [{manifest['source_repo']}](https://huggingface.co/{manifest['source_repo']})."
    if (directory / "README.md").read_text().replace(declaration, "", 1) != source_card:
        raise ValueError("Converted model card differs from the source beyond its format declaration")
    repo = manifest["target_repo"]
    receipt_path = work_dir / (name + "-" + fmt + "-publication.json")
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    api.create_repo(repo, repo_type="model", exist_ok=True)
    before = api.model_info(repo, revision="main")
    unchanged = previous and previous.get("repo_id") == repo and previous.get("manifest_sha256") == file_hash(manifest_path) and before.sha == previous["commit_sha"]
    if unchanged and previous.get("status") == "verified":
        print("ALREADY_VERIFIED", repo, previous["commit_url"], flush=True)
        return previous
    if unchanged and previous.get("status") == "uploaded":
        operations = [CommitOperationAdd(path_in_repo="conversion_manifest.json", path_or_fileobj=manifest_path)]
    else:
        operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=directory / path) for path in expected]
        operations.append(CommitOperationAdd(path_in_repo=".gitattributes", path_or_fileobj=directory / ".gitattributes"))
    print("UPLOADING", repo, "files", len(operations), "bytes", sum(f["bytes"] for f in expected.values()), flush=True)
    commit = api.create_commit(repo_id=repo, repo_type="model", revision="main", operations=operations,
        commit_message=f"Release complete {name} {fmt} model with validated inference runtime",
        commit_description=(f"Convert {manifest['source_repo']} at {manifest['source_revision']} to {fmt}.\n\n"
                            "Include the vision encoder, language backbone, trained Mixer decision head, tokenizer, "
                            "processor, dedicated inference runtime, and conversion validation. Preserve the source "
                            "model card with only a format declaration."),
        parent_commit=before.sha, create_pr=False, num_threads=4)
    receipt = {"repo_id": repo, "commit_sha": commit.oid, "commit_url": commit.commit_url,
               "bytes": sum(f["bytes"] for f in expected.values()), "files": len(expected), "status": "uploaded",
               "manifest_sha256": file_hash(manifest_path)}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("COMMITTED", repo, commit.oid, flush=True)
    remote = api.model_info(repo, revision=commit.oid, files_metadata=True)
    siblings = {f.rfilename: f for f in remote.siblings}
    for path, item in expected.items():
        info = siblings.get(path)
        if info is None or info.size != item["bytes"]:
            raise ValueError("Remote file is missing or has a different size: " + path)
        if info.lfs:
            if info.lfs.sha256 != item["sha256"]:
                raise ValueError("Remote LFS hash differs: " + path)
        else:
            content = (directory / path).read_bytes()
            blob_sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if info.blob_id != blob_sha:
                raise ValueError("Remote Git blob differs: " + path)
    attributes = hf_hub_download(repo, ".gitattributes", revision=commit.oid, token=token,
                                 cache_dir=str(work_dir / "hf-cache"), force_download=True)
    (directory / ".gitattributes").write_bytes(Path(attributes).read_bytes())
    receipt["status"] = "verified"
    receipt["manifest_sha256"] = file_hash(manifest_path)
    receipt["verified_at"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("VERIFIED", repo, commit.commit_url, flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--formats", nargs="+", choices=["ONNX", "GGUF"], default=["ONNX", "GGUF"])
    parser.add_argument("--exports-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory containing converted model folders")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Directory for publication receipts and caches")
    parser.add_argument("--namespace", help="Publish to this Hugging Face namespace; otherwise use each manifest's target")
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise ValueError("Hugging Face authentication is unavailable")
    api = HfApi(token=token, headers={"Cache-Control": "no-cache"})
    receipts = []
    for name in args.models:
        for fmt in args.formats:
            receipts.append(publish(name, fmt, api, token, exports_dir=args.exports_dir.expanduser().resolve(),
                                    work_dir=args.work_dir.expanduser().resolve(), namespace=args.namespace))
    print(json.dumps(receipts, indent=2), flush=True)


if __name__ == "__main__":
    main()
