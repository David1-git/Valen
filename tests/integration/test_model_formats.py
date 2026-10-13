"""Portable export preparation, publication checks and small-format round trips."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.export.qwen import export_formats, format_export, publish_formats


def source_bundle(tmp_path, monkeypatch):
    source = tmp_path / "source" / "Example"
    source.mkdir(parents=True)
    (source / "config.json").write_text('{}')
    (source / "model.safetensors").write_bytes(b"source-weight-fixture")
    (source / "modeling_valen.py").write_text('# model code\n')
    (source / "preprocessor_config.json").write_text('{"image":true}')
    (source / "video_preprocessor_config.json").write_text('{"video":true}')
    (source / "tokenizer.model").write_bytes(b"tokenizer-fixture")
    card = tmp_path / "source-card.md"
    card.write_text('# Example\n\nOriginal model card.\n')
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(card))
    weight = source / "model.safetensors"
    info = {"repo_id": "test/Example", "sha": "pinned-commit", "files": [
        {"path": weight.name, "size": weight.stat().st_size, "sha256": export_formats.file_hash(weight)},
    ]}
    return source, info


def test_prepare_creates_fresh_export_folders_and_keeps_processor_files(tmp_path, monkeypatch):
    source, info = source_bundle(tmp_path, monkeypatch)
    output, work = tmp_path / "new" / "exports", tmp_path / "new" / "work"
    export_formats.prepare(source.name, info, source.parent, output, work)
    for fmt in ["ONNX", "GGUF"]:
        folder = output / (source.name + "-" + fmt)
        assert (folder / "valen_native" / "modeling_valen.py").read_bytes() == (source / "modeling_valen.py").read_bytes()
        assert (folder / "valen_formats.py").is_file()
        for name in ["preprocessor_config.json", "video_preprocessor_config.json", "tokenizer.model"]:
            assert (folder / name).read_bytes() == (source / name).read_bytes()
        assert "*." + fmt.lower() in (folder / ".gitattributes").read_text()
        assert "Original model card." in (folder / "README.md").read_text()


@pytest.mark.parametrize("change", ["changed", "extra"])
def test_prepare_rejects_weights_that_differ_from_the_pinned_source(tmp_path, monkeypatch, change):
    source, info = source_bundle(tmp_path, monkeypatch)
    if change == "changed":
        (source / "model.safetensors").write_bytes(b"different-weights")
    else:
        (source / "stale.safetensors").write_bytes(b"stale-weights")
    output = tmp_path / "exports"
    with pytest.raises(ValueError, match="[Ll]ocal weight"):
        export_formats.prepare(source.name, info, source.parent, output, tmp_path / "work")
    assert not output.exists()


def test_source_metadata_uses_the_resolved_commit_and_hub_weight_hash(monkeypatch):
    weight = SimpleNamespace(rfilename="model.safetensors", size=42, lfs=SimpleNamespace(sha256="weight-sha"))
    readme = SimpleNamespace(rfilename="README.md", size=12, lfs=None)
    class Hub:
        def model_info(self, repo, **kwargs):
            assert repo == "test/Example" and kwargs == {"revision": "release", "files_metadata": True}
            return SimpleNamespace(sha="pinned-commit", siblings=[weight, readme])
    monkeypatch.setattr("huggingface_hub.HfApi", Hub)
    assert export_formats.source_metadata("Example", "test", "release") == {
        "repo_id": "test/Example", "sha": "pinned-commit",
        "files": [{"path": "model.safetensors", "size": 42, "sha256": "weight-sha"}],
    }


def publication_bundle(tmp_path, monkeypatch):
    folder = tmp_path / "exports" / "Example-GGUF"
    folder.mkdir(parents=True)
    source_card = tmp_path / "source.md"
    source_card.write_text('# Example\n\nOriginal description.\n')
    declaration = '\n\nThis is the **GGUF** version of [test/Example](https://huggingface.co/test/Example).'
    (folder / "README.md").write_text(source_card.read_text().replace('# Example', '# Example' + declaration, 1))
    (folder / ".gitattributes").write_text('*.gguf filter=lfs diff=lfs merge=lfs -text\n')
    (folder / "validation.json").write_text('{"passed":true}')
    (folder / "model.gguf").write_bytes(b"verified-weight-fixture")
    manifest = {"source_repo": "test/Example", "source_revision": "source-commit", "target_repo": "test/Example-GGUF",
                "format": "GGUF", "validated": True, "files": [
                    {"path": p.name, "bytes": p.stat().st_size, "sha256": export_formats.file_hash(p)}
                    for p in folder.iterdir()
                ]}
    (folder / "conversion_manifest.json").write_text(json.dumps(manifest))

    class Hub:
        def __init__(self):
            self.created = []
            self.commits = []
            self.files = {}
        def create_repo(self, repo, **kwargs):
            self.created.append(repo)
        def model_info(self, repo, revision, files_metadata=False):
            if not files_metadata:
                return SimpleNamespace(sha="published" if self.commits else "initial")
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename=name, size=len(data), lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
                for name, data in self.files.items()
            ])
        def create_commit(self, **kwargs):
            self.commits.append(kwargs)
            self.files.update({op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in kwargs["operations"]})
            return SimpleNamespace(oid="published", commit_url="https://huggingface.co/test/Example-GGUF/commit/published")

    hub = Hub()
    def download(repo, name, **kwargs):
        if name == "README.md":
            assert repo == "test/Example" and kwargs["revision"] == "source-commit"
            return str(source_card)
        assert name == ".gitattributes"
        path = tmp_path / "remote-attributes"
        path.write_bytes(hub.files[name])
        return str(path)
    monkeypatch.setattr(publish_formats, "hf_hub_download", download)
    return folder, hub


def test_publisher_supports_custom_paths_and_namespace_and_skips_verified_upload(tmp_path, monkeypatch):
    folder, hub = publication_bundle(tmp_path, monkeypatch)
    kwargs = {"exports_dir": folder.parent, "work_dir": tmp_path / "custom" / "receipts", "namespace": "new-owner"}
    first = publish_formats.publish("Example", "GGUF", hub, "test-token", **kwargs)
    assert first["status"] == "verified" and first["repo_id"] == "new-owner/Example-GGUF"
    assert hub.commits[0]["repo_id"] == first["repo_id"]
    second = publish_formats.publish("Example", "GGUF", hub, "test-token", **kwargs)
    assert second == first and len(hub.commits) == 1


@pytest.mark.parametrize("change", ["weight", "card", "validation"])
def test_publisher_rejects_changed_or_unvalidated_files_before_remote_writes(tmp_path, monkeypatch, change):
    folder, hub = publication_bundle(tmp_path, monkeypatch)
    if change == "weight":
        (folder / "model.gguf").write_bytes(b"changed")
    elif change == "validation":
        (folder / "validation.json").write_text('{"passed":false}')
    else:
        (folder / "README.md").write_text('Changed model card.\n')
        manifest_path = folder / "conversion_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        item = next(item for item in manifest["files"] if item["path"] == "README.md")
        item.update(bytes=(folder / "README.md").stat().st_size, sha256=export_formats.file_hash(folder / "README.md"))
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        publish_formats.publish("Example", "GGUF", hub, "test-token", exports_dir=folder.parent, work_dir=tmp_path / "work")
    assert not hub.created and not hub.commits


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gguf_preserves_all_tensor_bits_and_shapes(tmp_path, dtype):
    gguf = pytest.importorskip("gguf")
    from safetensors.torch import save_file
    source, output = tmp_path / "Example", tmp_path / "new" / "gguf"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"dtype": "bfloat16" if dtype == torch.bfloat16 else "float32"}))
    tensors = {"backbone.embedding": torch.arange(24).reshape(4, 6).to(dtype),
               "backbone.visual.kernel": torch.arange(32).reshape(2, 2, 2, 2, 2).to(dtype),
               "head.roles": torch.tensor([[1., -2., 0., 0.125]], dtype=dtype)}
    save_file(tensors, source / "model.safetensors")
    manifest = format_export.export_gguf(source, output, "test/Example", "pinned-commit")
    reader = gguf.GGUFReader(str(output / manifest["file"]), mode="c")
    shapes = json.loads(reader.get_field("valen.tensor_shapes_json").contents())
    stored = {tensor.name: tensor for tensor in reader.tensors}
    assert set(stored) == set(tensors)
    for name, tensor in tensors.items():
        expected = tensor.view(torch.uint16).numpy() if dtype == torch.bfloat16 else tensor.numpy()
        restored = stored[name].data.view(np.uint16) if dtype == torch.bfloat16 else stored[name].data
        np.testing.assert_array_equal(restored.reshape(shapes[name]), expected)


def test_onnx_mixer_matches_native_with_dynamic_candidate_counts(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from valen.modeling.qwen.heads import build_head
    torch.manual_seed(17)
    head = build_head(8, {"head_type": "mixer", "head_width": 4, "head_layers": 1,
                          "head_token_hidden_dim": 3, "head_channel_hidden_dim": 8}).eval()
    path = tmp_path / "new" / "head.onnx"
    format_export.save_external_graph(format_export.HeadGraph(head), (torch.randn(3, 4, 8),),
        ["features"], ["logits"], {"features": {0: "candidates"}, "logits": {0: "candidates"}}, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for candidates in [1, 2, 7]:
        features = torch.randn(candidates, 4, 8)
        with torch.inference_mode():
            expected = head.score_features(features).numpy()
        actual = runtime.run(None, {"features": features.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
