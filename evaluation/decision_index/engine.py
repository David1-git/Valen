"""Native Valen engine for apolinario/decision-index. / 复用已发布的决策头。"""
import gc
import hashlib
import importlib.metadata
from pathlib import Path

import torch
from decision_index.engines.base import Engine, Unsupported, text
from huggingface_hub import snapshot_download

from valen.evaluation.inference import predict
from valen.modeling.factory import build_compiler, build_model, normalize_model_config
from valen.modeling.qwen.attention import ATTENTION_IMPLEMENTATIONS
from valen.modeling.qwen.pretrained import published_config


def render_questions(questions):
    """Preserve IDs/order and use the upstream renderer for structured values."""
    output = {}
    for key, question in questions.items():
        q = dict(question)
        q["instructions"] = text(q["instructions"])
        if q["type"] == "choice":
            # Null descriptions use their option key, as in the reference engine.
            # null 描述使用原选项名；字典和列表沿用上游紧凑 JSON 格式。
            q["criteria"] = {name: name if value is None else text(value)
                             for name, value in q["criteria"].items()}
        output[key] = q
    return output


class ValenEngine(Engine):
    name = "valen"

    def __init__(self, model, device="cuda", attention="sdpa", revision="main"):
        super().__init__(model=str(model), device=device, attention=attention, revision=revision)
        if attention not in ATTENTION_IMPLEMENTATIONS:
            raise ValueError(f"attention must be one of {ATTENTION_IMPLEMENTATIONS}")
        self.device = torch.device(device)
        folder = Path(model).expanduser()
        local = folder.is_dir()
        if not local:
            folder = Path(snapshot_download(repo_id=str(model), revision=revision))
        folder = folder.resolve()
        if published_config(folder) is None:
            raise ValueError("Expected a complete published Valen Qwen model with model_type='valen_qwen'")
        self.config = normalize_model_config(dict(
            architecture="qwen", model_path=str(folder), stage="warmup",
            dtype="bf16" if self.device.type == "cuda" else "fp32",
            device=str(self.device), attn_implementation=attention,
            qwen_execution="shared_state", gradient_checkpointing=False))
        # warmup selects a frozen loader; the saved Mixer and backbone are retained.
        # warmup 仅选择冻结加载路径，不训练、不重建已保存的 Mixer。
        self.model = build_model(self.config).eval().requires_grad_(False)
        self.compiler = build_compiler(self.config)
        self.provenance = {
            "kind": "native_decision_head", "model": str(model), "model_path": str(folder),
            "revision": None if local else revision,
            "resolved_revision": folder.name if not local and folder.parent.name == "snapshots" else None,
            "config_sha256": hashlib.sha256((folder / "config.json").read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "qwen_execution": self.config["qwen_execution"],
            "max_length": self.config["max_length"], "temperature": 1.0,
            "attention_backends": self.model.attention_backends,
            "parameter_dtype": str(next(self.model.backbone.parameters()).dtype),
            "backbone_autocast_dtype": str(self.model.backbone_autocast_dtype),
            "state_serialization": "unchanged string; compact UTF-8 JSON for other values",
            "option_policy": "preserve every supplied option and its order; no truncation",
            "question_serialization": "upstream text(): preserve strings, compact UTF-8 JSON for structured values; null option descriptions use their key",
        }

    @torch.inference_mode()
    def __call__(self, state, questions):
        # Only inputs reach the compiler; labels remain with the upstream scorer.
        # 通用 state 字典转为文本；标准答案始终留在计分侧。
        try:
            compiled = self.compiler.compile({"request": {"state": text(state), "questions": render_questions(questions)}})
        except ValueError as exc:
            if "exceed max_length=" in str(exc) or "Reserved tokenizer control token" in str(exc):
                raise Unsupported(str(exc)) from exc
            raise
        return predict(self.model, compiled, temperature=1.0), None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def runtime(self):
        return {
            "hardware": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "cuda": torch.version.cuda,
            "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "decision-index")},
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
        }

    def close(self):
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
