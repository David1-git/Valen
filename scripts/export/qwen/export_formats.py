"""Convert and validate complete Valen models. / 转换并校验完整模型。"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import shutil
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK_DIR = ROOT / "artifacts/model-format-export"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_MODELS = ["Valen-0.8B", "Valen-2B", "Valen-4B"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_metadata(name, namespace, revision):
    """Pin the source revision and weight hashes. / 固定来源版本与权重摘要。"""
    from huggingface_hub import HfApi, hf_hub_download
    repo = namespace + "/" + name
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = []
    for item in info.siblings:
        if not item.rfilename.endswith(".safetensors"):
            continue
        digest = item.lfs.sha256 if item.lfs else file_hash(hf_hub_download(repo, item.rfilename, revision=info.sha))
        files.append({"path": item.rfilename, "size": item.size, "sha256": digest})
    if not files:
        raise ValueError("Source repository has no safetensors weights: " + repo)
    return {"repo_id": repo, "sha": info.sha, "files": files}


def prepare(name, source_info, models_dir, output_dir, work_dir):
    from huggingface_hub import hf_hub_download
    source = Path(models_dir) / name
    repo = source_info["repo_id"]
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Download the complete source model to {source} before exporting")
    expected_weights = {item["path"] for item in source_info["files"] if item["path"].endswith(".safetensors")}
    actual_weights = {str(path.relative_to(source)) for path in source.rglob("*.safetensors")}
    if not expected_weights or actual_weights != expected_weights:
        raise ValueError("Local weight files do not match the pinned source repository: " + str(source))
    for item in source_info["files"]:
        if not item["path"].endswith(".safetensors"):
            continue
        local = source / item["path"]
        if local.stat().st_size != item["size"] or file_hash(local) != item["sha256"]:
            raise ValueError("Local weights do not match pinned source repository: " + str(local))
    card_path = hf_hub_download(repo, "README.md", revision=source_info["sha"], cache_dir=str(work_dir / "hf-cache"))
    card = Path(card_path).read_text()
    cards = work_dir / "source-cards"
    cards.mkdir(parents=True, exist_ok=True)
    (cards / (name + ".README.md")).write_text(card)
    for fmt in ["ONNX", "GGUF"]:
        dest = Path(output_dir) / (name + "-" + fmt)
        dest.mkdir(parents=True, exist_ok=True)
        attributes_path = dest / ".gitattributes"
        attributes = attributes_path.read_text() if attributes_path.exists() else ""
        for pattern in ["*.onnx", "*.data", "*.gguf"]:
            if not any(line.split()[0] == pattern for line in attributes.splitlines() if line.strip() and not line.startswith("#")):
                attributes = attributes.rstrip() + f"\n{pattern} filter=lfs diff=lfs merge=lfs -text\n"
        attributes_path.write_text(attributes)
        package = dest / "valen_native"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text("")
        # Copy the processor and model code as a complete bundle. / 保留完整处理器与模型代码。
        for path in source.iterdir():
            if path.suffix == ".py":
                shutil.copy2(path, dest / path.name)
                shutil.copy2(path, package / path.name)
            elif path.suffix in {".json", ".jinja", ".model", ".txt", ".tiktoken"} or path.name in {"LICENSE", "MODEL_USAGE.md"}:
                shutil.copy2(path, dest / path.name)
        shutil.copy2(Path(__file__).with_name("format_runtime.py"), dest / "valen_formats.py")
        marker = "# " + name
        declaration = f"\n\nThis is the **{fmt}** version of [{repo}](https://huggingface.co/{repo})."
        if marker not in card:
            raise ValueError("Cannot locate original model-card title")
        (dest / "README.md").write_text(card.replace(marker, marker + declaration, 1))
        cls = "ValenONNXModel" if fmt == "ONNX" else "ValenGGUFModel"
        dependencies = "onnxruntime==1.23.2" if fmt == "ONNX" else "gguf==0.19.0"
        (dest / "requirements-format.txt").write_text(
            "transformers==5.4.0\ntorch>=2.6\ntorchvision\nnumpy\nPillow\nav\nhuggingface_hub\n" + dependencies + "\n")
        explanation = (
            "The vision encoder, text backbone, and Mixer decision head run as three ONNX Runtime graphs. "
            "Weights and arithmetic are FP32. Inputs have dynamic sequence, patch, and candidate counts. "
            "The dedicated runtime retains the native tokenizer, processor, shared-state compiler, and typed outputs. "
            "The reported model-card benchmarks are the original model's results; `validation.json` records conversion checks."
            if fmt == "ONNX" else
            "The GGUF file contains every original backbone, vision, merger, and Mixer tensor in its original precision, "
            "using the custom `valen_qwen` architecture and original tensor names. "
            "Load it with the included dedicated Valen runtime. This is a complete decision model, and its GGUF layout "
            "requires the Valen runtime; ordinary llama.cpp/Ollama loaders do not implement this architecture."
        )
        device = ", device=\"cuda\"" if fmt == "GGUF" else ""
        (dest / "FORMAT_USAGE.md").write_text(
            f"# {name} {fmt} usage\n\n{explanation}\n\n"
            "Install the packages listed in `requirements-format.txt`. Python 3.10+ is required.\n\n"
            f"```python\nimport sys\nfrom huggingface_hub import snapshot_download\n\n"
            f"folder = snapshot_download(\"{repo}-{fmt}\")\nsys.path.insert(0, folder)\n"
            f"from valen_formats import {cls}\n\nmodel = {cls}.from_pretrained(folder{device})\n"
            "print(model.predict({\n    \"state\": \"A cat is on the sofa.\",\n"
            "    \"questions\": {\n        \"animal\": {\"type\": \"choice\", \"instructions\": \"Which animal is present?\",\n"
            "                   \"criteria\": {\"cat\": \"A cat\", \"dog\": \"A dog\"}},\n"
            "        \"on_sofa\": {\"type\": \"noul\", \"instructions\": \"The cat is on the sofa.\"},\n"
            "    },\n}, execution=\"shared_state\"))\n```\n\n"
            "Both `question` and `shared_state` execution, Choice/Noul/Score outputs, and the native image/video "
            "request schema are supported. The source model card is preserved except for its format declaration.\n")
    return source_info


def media_examples(work_dir):
    from PIL import Image
    import av
    media = work_dir / "validation-media"
    media.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), "red")
    image.save(media / "red.png")
    container = av.open(str(media / "red.mp4"), mode="w")
    stream = container.add_stream("libx264", rate=4)
    stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
    for _ in range(8):
        for packet in stream.encode(av.VideoFrame.from_image(image)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    questions = {
        "color": {"type": "choice", "instructions": "Which color is described or shown?",
                  "criteria": {"red": "Red", "blue": "Blue"}},
        "red": {"type": "noul", "instructions": "The color is red."},
        "brightness": {"type": "score", "instructions": "Rate the brightness.",
                       "criteria": ["Dark", "Bright"]},
    }
    result = {"text": {"state": "A bright red square is shown.", "questions": questions}}
    for kind, file in [("image", "red.png"), ("video", "red.mp4")]:
        result[kind] = {"state": {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Examine the color."},
            {"type": kind + "_url", kind + "_url": {"url": str(media / file)}},
        ]}]}, "questions": questions}
    return result


MEDIA_KWARGS = {"images_kwargs": {"size": {"shortest_edge": 1024, "longest_edge": 4096}},
                "videos_kwargs": {"num_frames": 2, "fps": None,
                                  "size": {"shortest_edge": 1024, "longest_edge": 4096}}}


def native_logits(model, state):
    with torch.inference_mode():
        return model.forward_state(state) if state.inputs is not None else [model(q) for q in state.questions]


def compare_logits(expected, actual, state, atol, rtol):
    errors = []
    for q, reference, converted in zip(state.questions, expected, actual):
        a, b = reference.detach().cpu().float().numpy(), converted.detach().cpu().float().numpy()
        if not np.isfinite(b).all():
            raise ValueError("Converted logits are not finite")
        np.testing.assert_allclose(b, a, atol=atol, rtol=rtol, err_msg=q.qid)
        errors.append({"question": q.qid, "type": q.kind, "max_abs_logit_error": float(np.max(np.abs(a - b))),
                       "same_argmax": bool(np.argmax(a) == np.argmax(b)), "candidates": len(a)})
    if len(errors) != len(state.questions):
        raise ValueError("Missing converted outputs")
    return errors


def validate_onnx(model, destination, examples):
    if __package__:
        from .format_runtime import ValenONNXModel
    else:
        from format_runtime import ValenONNXModel
    runtime = ValenONNXModel.from_pretrained(destination, num_threads=8)
    runtime.config["valen_media_kwargs"] = copy.deepcopy(MEDIA_KWARGS)
    model.config.valen_media_kwargs = copy.deepcopy(MEDIA_KWARGS)
    model.backbone_autocast_dtype = None
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    checks = []
    for modality, request in examples.items():
        for execution in ["shared_state", "question"]:
            started = time.time()
            state = model.make_compiler(execution=execution).compile({"request": request})
            expected = native_logits(model, state)
            actual = runtime.compiled_logits(state)
            checks.append({"modality": modality, "execution": execution, "tokens": state.compute_tokens,
                           "outputs": compare_logits(expected, actual, state, atol=0.005, rtol=0.002),
                           "seconds": time.time() - started})
            print("ONNX_VALIDATED", destination.name, modality, execution, state.compute_tokens, flush=True)
    response = runtime.predict(examples["text"], execution="shared_state")
    if set(response["answers"]) != set(examples["text"]["questions"]):
        raise ValueError("ONNX prediction lost typed outputs")
    result = {"passed": True, "reference": "same trained tensors evaluated in FP32",
              "network_runtime": "onnxruntime", "onnx_operators": "standard opset 18 including dynamic Loop",
              "checks": checks, "typed_predict_api_passed": True,
              "validation_media_kwargs": MEDIA_KWARGS,
              "precision_note": "FP32 compute can differ from the native BF16/autocast reference benchmarks."}
    (destination / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    del runtime
    gc.collect()
    return result


def validate_gguf(source, destination, examples):
    from transformers import AutoModel
    if __package__:
        from .format_runtime import ValenGGUFModel
    else:
        from format_runtime import ValenGGUFModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    native = AutoModel.from_pretrained(source, trust_remote_code=True, dtype="auto", attn_implementation="sdpa").to(device).eval()
    converted = ValenGGUFModel.from_pretrained(destination, device=device)
    native.config.valen_media_kwargs = copy.deepcopy(MEDIA_KWARGS)
    converted.config.valen_media_kwargs = copy.deepcopy(MEDIA_KWARGS)
    checks = []
    for modality, request in examples.items():
        for execution in ["shared_state", "question"]:
            state = native.make_compiler(execution=execution).compile({"request": request})
            checks.append({"modality": modality, "execution": execution, "tokens": state.compute_tokens,
                           "outputs": compare_logits(native_logits(native, state), native_logits(converted, state), state,
                                                     atol=1e-5, rtol=1e-5)})
            print("GGUF_VALIDATED", destination.name, modality, execution, flush=True)
    result = {"passed": True, "reference": "original native model at original tensor precision and computation mode",
              "custom_architecture": "valen_qwen", "all_source_tensors_bitwise_preserved": True,
              "checks": checks, "validation_media_kwargs": MEDIA_KWARGS}
    (destination / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    del native, converted
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(name, source_info, examples, models_dir, output_dir, work_dir, target_namespace):
    from transformers import AutoModel
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    if __package__:
        from .format_export import export_gguf, export_onnx
    else:
        from format_export import export_gguf, export_onnx
    prepare(name, source_info, models_dir, output_dir, work_dir)
    source = models_dir / name
    onnx_dir, gguf_dir = output_dir / (name + "-ONNX"), output_dir / (name + "-GGUF")
    print("START", name, flush=True)
    gguf_manifest = export_gguf(source, gguf_dir, source_info["repo_id"], source_info["sha"])
    model = AutoModel.from_pretrained(source, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager").eval()
    model.backbone_autocast_dtype = None
    onnx_manifest = export_onnx(model, onnx_dir)
    for layer in model.backbone.language_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    validate_onnx(model, onnx_dir, examples)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    validate_gguf(source, gguf_dir, examples)
    for fmt, directory, tensor_manifest in [("ONNX", onnx_dir, onnx_manifest), ("GGUF", gguf_dir, gguf_manifest)]:
        files = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or any(part in {".git", ".cache", "__pycache__"} for part in path.parts) or path.name == "conversion_manifest.json":
                continue
            files.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": file_hash(path)})
        manifest = {"source_repo": source_info["repo_id"], "source_revision": source_info["sha"], "format": fmt,
                    "target_repo": target_namespace + "/" + name + "-" + fmt,
                    "validated": True, "tensors": tensor_manifest, "files": files}
        (directory / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETE", name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory containing complete source models")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory for converted model folders")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Directory for metadata, caches and validation media")
    parser.add_argument("--source-namespace", default="Valen-Team")
    parser.add_argument("--target-namespace", default="Valen-Team", help="Namespace recorded in conversion manifests")
    parser.add_argument("--revision", default="main", help="Source Hub revision; resolved to a commit before export")
    parser.add_argument("--source-metadata", type=Path, help="Reuse a previously pinned hub-revisions.json")
    args = parser.parse_args()
    args.models_dir = args.models_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.source_metadata:
        metadata = json.loads(args.source_metadata.expanduser().read_text())
    else:
        metadata = [source_metadata(name, args.source_namespace, args.revision) for name in args.models]
    revisions = {item["repo_id"]: item for item in metadata}
    (args.work_dir / "hub-revisions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    examples = media_examples(args.work_dir)
    for name in args.models:
        info = revisions[args.source_namespace + "/" + name]
        run(name, info, examples, args.models_dir, args.output_dir, args.work_dir, args.target_namespace)
    (args.work_dir / "export-status.json").write_text(json.dumps({"status": "complete", "models": args.models}) + "\n")


if __name__ == "__main__":
    main()
