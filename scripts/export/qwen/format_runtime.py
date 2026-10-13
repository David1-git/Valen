"""Complete Valen inference from ONNX graphs or a lossless custom GGUF bundle."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch


def resolve_directory(path):
    directory = Path(path)
    if directory.is_dir():
        return directory.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(path), ignore_patterns=["*.safetensors", ".git/*"]))


def vision_inputs(pixel_values, grid_thw, config):
    """Compute untrained grid geometry; learned interpolation runs inside ONNX."""
    merge = config["spatial_merge_size"]
    side = int(config["num_position_embeddings"] ** 0.5)
    indices, weights, coordinates, frame_ids = [], [], [], []
    frame_number = 0
    for frames, height, width in grid_thw.tolist():
        h = torch.linspace(0, side - 1, height)
        w = torch.linspace(0, side - 1, width)
        hf, wf = h.int(), w.int()
        hc, wc = (hf + 1).clamp(max=side - 1), (wf + 1).clamp(max=side - 1)
        dh, dw = h - hf, w - wf
        corner_indices = [
            hf[:, None] * side + wf[None], hf[:, None] * side + wc[None],
            hc[:, None] * side + wf[None], hc[:, None] * side + wc[None],
        ]
        corner_weights = [
            (1 - dh[:, None]) * (1 - dw[None]), (1 - dh[:, None]) * dw[None],
            dh[:, None] * (1 - dw[None]), dh[:, None] * dw[None],
        ]
        def reorder(value):
            return value.repeat(frames, 1, 1).reshape(
                frames, height // merge, merge, width // merge, merge
            ).permute(0, 1, 3, 2, 4).reshape(-1)
        indices.append(torch.stack([reorder(v) for v in corner_indices]).long())
        weights.append(torch.stack([reorder(v) for v in corner_weights]).float())
        rows = torch.arange(height)[:, None].expand(height, width)
        cols = torch.arange(width)[None].expand(height, width)
        coordinates.append(torch.stack([reorder(rows), reorder(cols)], dim=-1))
        frame_ids.extend(np.repeat(np.arange(frame_number, frame_number + frames), height * width))
        frame_number += frames
    coordinates = torch.cat(coordinates)
    head_dim = config["hidden_size"] // config["num_heads"]
    inv_freq = 1 / (10000.0 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))
    table = torch.outer(torch.arange(int(coordinates.max()) + 1).float(), inv_freq)
    rotary = table[coordinates].flatten(1)
    rotary = torch.cat([rotary, rotary], dim=-1)
    frame_ids = np.asarray(frame_ids)
    mask = np.where(frame_ids[:, None] == frame_ids[None], 0, np.finfo(np.float32).min).astype(np.float32)
    return {
        "pixel_values": pixel_values.detach().cpu().float().numpy(),
        "position_indices": torch.cat(indices, dim=1).numpy(),
        "position_weights": torch.cat(weights, dim=1).numpy(),
        "rotary_cos": rotary.cos().numpy(), "rotary_sin": rotary.sin().numpy(),
        "attention_mask": mask,
    }


class ValenONNXModel:
    """The vision backbone, text backbone, and decision head all run in ONNX Runtime."""
    def __init__(self, directory, providers=None, num_threads=4):
        import onnxruntime as ort
        from transformers import AutoProcessor
        from valen_native.compiler import Compiler
        from valen_native.heads import role_features
        from valen_native.responses import answer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        from valen_native.configuration_valen import ValenQwenConfig
        self.directory = resolve_directory(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        self.processor = AutoProcessor.from_pretrained(self.directory, trust_remote_code=True)
        self.Compiler, self.role_features, self.answer = Compiler, role_features, answer
        self.geometry = SimpleNamespace(config=ValenQwenConfig(**self.config))
        self.geometry.get_vision_position_ids = Qwen3_5Model.get_vision_position_ids.__get__(self.geometry)
        self.geometry.get_rope_index = Qwen3_5Model.get_rope_index.__get__(self.geometry)
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_cpu_mem_arena = False
        self.sessions = {}
        for name in ["text", "vision", "head"]:
            self.sessions[name] = ort.InferenceSession(
                str(self.directory / "onnx" / (name + ".onnx")), sess_options=options,
                providers=providers or ["CPUExecutionProvider"],
            )

    @classmethod
    def from_pretrained(cls, directory, **kwargs):
        directory = resolve_directory(directory)
        import sys
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        return cls(directory, **kwargs)

    def make_compiler(self, media_root=".", execution=None, processor=None):
        return self.Compiler(
            processor or self.processor, media_root, self.config.get("valen_max_length", 32768),
            self.config.get("valen_media_kwargs", {}),
            execution=execution or self.config.get("qwen_execution", "shared_state"),
        )

    def hidden(self, inputs):
        input_ids = inputs["input_ids"].cpu().long()
        if input_ids.shape[0] != 1 or not torch.all(inputs["attention_mask"] == 1):
            raise ValueError("Use one unpadded compiled state at a time")
        visual_positions, visual_features = [], []
        for pixels, grid, token in [
            ("pixel_values", "image_grid_thw", "image_token_id"),
            ("pixel_values_videos", "video_grid_thw", "video_token_id"),
        ]:
            if pixels not in inputs:
                continue
            features = self.sessions["vision"].run(None, vision_inputs(
                inputs[pixels], inputs[grid].cpu(), self.config["vision_config"]
            ))[0]
            positions = np.flatnonzero(input_ids.numpy()[0] == self.config[token]).astype(np.int64)
            if len(positions) != len(features):
                raise ValueError("Visual placeholder count does not match encoded patches")
            visual_positions.append(positions)
            visual_features.append(features)
        if visual_positions:
            positions = np.concatenate(visual_positions)
            features = np.concatenate(visual_features)
        else:
            positions = np.empty((0,), dtype=np.int64)
            features = np.empty((0, self.config["text_config"]["hidden_size"]), dtype=np.float32)
        if "image_grid_thw" in inputs or "video_grid_thw" in inputs:
            position_ids, _ = self.geometry.get_rope_index(
                input_ids, inputs["mm_token_type_ids"].cpu(),
                image_grid_thw=inputs.get("image_grid_thw"), video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=inputs["attention_mask"].cpu(),
            )
        else:
            position_ids = torch.arange(input_ids.shape[1]).reshape(1, 1, -1).expand(3, 1, -1)
        return self.sessions["text"].run(None, {
            "input_ids": input_ids.numpy(), "position_ids": position_ids.numpy(),
            "visual_positions": positions, "visual_features": features.astype(np.float32),
        })[0]

    def branch_logits(self, hidden, branch):
        features = self.role_features(torch.from_numpy(hidden), branch).numpy()
        return torch.from_numpy(self.sessions["head"].run(None, {"features": features})[0])

    def compiled_logits(self, state):
        if state.inputs is not None:
            hidden = self.hidden(state.inputs)
            return [torch.cat([self.branch_logits(hidden, b) for b in q.branches]) for q in state.questions]
        return [torch.cat([self.branch_logits(self.hidden(b.inputs), b) for b in q.branches]) for q in state.questions]

    def predict(self, request, *, media_root=".", execution=None, temperature=1.0, processor=None):
        request = request.get("request", request)
        state = self.make_compiler(media_root, execution, processor).compile({"request": request})
        logits = self.compiled_logits(state) if state.questions else []
        answers = {
            q.qid: self.answer(q, z, temperature[q.kind] if isinstance(temperature, dict) else temperature)
            for q, z in zip(state.questions, logits)
        }
        return {"model": "Valen", "answers": answers,
                "usage": {"input_tokens": state.logical_tokens, "output_tokens": 0},
                "internal_usage": {"compute_tokens": state.compute_tokens}}


class ValenGGUFModel:
    """Load all native Valen tensors from its custom, lossless GGUF architecture."""
    @classmethod
    def from_pretrained(cls, directory, *, device="cpu", attn_implementation="sdpa"):
        import sys
        import gguf
        directory = resolve_directory(directory)
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        from valen_native.configuration_valen import ValenQwenConfig
        from valen_native.modeling_valen import ValenQwenForDecisionMaking
        from transformers import AutoProcessor
        files = sorted(directory.glob("*.gguf"))
        if len(files) != 1:
            raise ValueError("Expected exactly one complete Valen GGUF file")
        reader = gguf.GGUFReader(str(files[0]), mode="c")
        if reader.get_field("general.architecture").contents() != "valen_qwen":
            raise ValueError("Expected a complete custom valen_qwen GGUF bundle")
        shape_field = reader.get_field("valen.tensor_shapes_json")
        shapes = json.loads(shape_field.contents()) if shape_field else {}
        state = {}
        for tensor in reader.tensors:
            shape = shapes.get(tensor.name, tuple(reversed(tensor.shape.tolist())))
            if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                array = tensor.data.view(np.uint16).reshape(shape)
                state[tensor.name] = torch.from_numpy(array).view(torch.bfloat16)
            elif tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                state[tensor.name] = torch.from_numpy(tensor.data.reshape(shape))
            else:
                raise ValueError(f"Unsupported tensor precision: {tensor.tensor_type}")
        config = ValenQwenConfig(**json.loads((directory / "config.json").read_text()))
        model, loading = ValenQwenForDecisionMaking.from_pretrained(
            None, config=config, state_dict=state, dtype="auto", attn_implementation=attn_implementation,
            output_loading_info=True,
        )
        if any(loading.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
            raise ValueError(f"GGUF tensors do not match the complete Valen configuration: {loading}")
        model = model.to(device).eval()
        model._gguf_reader = reader
        model._processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True)
        return model
