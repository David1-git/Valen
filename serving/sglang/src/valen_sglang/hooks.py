"""Small, Valen-gated hooks; no engine source replacement or on-disk patching."""

import logging
import os
import time

import torch

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.layers.joint_schema_head import max_joint_prompt_tokens
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_schedule
from valen_sglang.head import ValenMixerHead, ValenPooler, parse_valen_layout

logger = logging.getLogger(__name__)


def normalize_config(config, *args, **kwargs):
    if getattr(config, "architectures", None) != ["ValenQwenForDecisionMaking"]:
        return config
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.model_type = "qwen3_5"
    dtype = config.valen_backbone_autocast_dtype or "float32"
    config.text_config.dtype = getattr(torch, dtype)
    config.text_config._valen_backbone_dtype = dtype
    return config


def resolve_dtype(original, config, dtype):
    # Stock auto dtype may downcast FP32 checkpoints. Valen's stored autocast
    # setting, including no autocast, is part of its trained execution contract.
    if dtype == "auto" and hasattr(config, "_valen_backbone_dtype"):
        dtype = config._valen_backbone_dtype
    return original(config, dtype)


def model_config_ready(result, self, *args, **kwargs):
    self.valen_head_config = getattr(self.hf_config, "valen_head", None)
    if self.valen_head_config is not None and self.joint_head_config is not None:
        raise ValueError(
            "A Valen checkpoint cannot also contain a Clef joint_head_config.json"
        )


def configure_capabilities(original, server_args):
    result = original(server_args)
    model_config = model_config_of(server_args)
    if getattr(model_config, "valen_head_config", None) is None:
        return result
    cfg = resolving_view(server_args)
    if cfg.tp_size != 1 or cfg.pp_size != 1:
        raise ValueError("Valen supports --tp-size 1 and --pp-size 1 only")
    if cfg.speculative_algorithm is not None:
        raise ValueError(
            "Valen decisions have no autoregressive decode; speculative decoding is unsupported"
        )
    if cfg.quantization is not None:
        raise ValueError(
            "This Valen serving kit validates unquantized checkpoints only"
        )
    for key, value in (
        ("is_embedding", True),
        ("disable_radix_cache", True),
        ("chunked_prefill_size", -1),
    ):
        declare_resolution(
            server_args, "_handle_model_capability_adjustments", **{key: value}
        )
    for phase in (Phase.DECODE, Phase.PREFILL):
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, phase, backend=Backend.DISABLED
            ),
        )
    logger.info(
        "Valen plugin: one complete prefill, FP32 head, no decode/cache/chunking/CUDA graphs"
    )
    return result


def initialize_head(result, self, config, *args, **kwargs):
    if getattr(config, "valen_head", None) is not None:
        self.pooler = ValenPooler(
            ValenMixerHead(config.text_config.hidden_size, config.valen_head)
        )


def load_weights(original, self, weights, *args, **kwargs):
    if not isinstance(self.pooler, ValenPooler):
        return original(self, weights, *args, **kwargs)
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    params = dict(self.pooler.head.named_parameters())
    loaded = set()

    def backbone_weights():
        for name, weight in weights:
            if name.startswith("head."):
                key = name.removeprefix("head.")
                if key not in params:
                    raise ValueError(f"Unknown Valen head parameter: {name}")
                if key in loaded:
                    raise ValueError(f"Duplicate Valen head parameter: {name}")
                default_weight_loader(params[key], weight)
                loaded.add(key)
            elif name.startswith("backbone."):
                yield "model." + name.removeprefix("backbone."), weight
            else:
                raise ValueError(f"Unexpected Valen checkpoint parameter: {name}")

    result = original(self, backbone_weights(), *args, **kwargs)
    if missing := params.keys() - loaded:
        raise ValueError(f"Missing Valen head weights: {sorted(missing)}")
    if any(parameter.dtype != torch.float32 for parameter in params.values()):
        raise RuntimeError("Valen decision-head parameters must remain FP32")
    if isinstance(result, set):
        result.update(f"pooler.head.{name}" for name in loaded)
    return result


def prepare_image_processor(self, hf_config, server_args, _processor, *args, **kwargs):
    if getattr(hf_config, "valen_head", None) is not None:
        image_kwargs = hf_config.valen_media_kwargs.get("images_kwargs", {})
        if "size" in image_kwargs:
            _processor.image_processor.size = dict(image_kwargs["size"])


def validate_request(original, self, obj, input_ids):
    if getattr(self.model_config, "valen_head_config", None) is None:
        return original(self, obj, input_ids)
    if isinstance(obj, GenerateReqInput) or obj.decision_layout is None:
        if (
            isinstance(obj, EmbeddingReqInput)
            and isinstance(obj.rid, str)
            and obj.rid.startswith(HEALTH_CHECK_RID_PREFIX)
        ):
            return
        raise ValueError("This checkpoint answers Valen decisions; use /v1/systemone")
    limit = min(
        self.model_config.hf_config.valen_max_length,
        max_joint_prompt_tokens(
            context_len=self.context_len,
            num_reserved_tokens=self.num_reserved_tokens,
            max_req_input_len=self.max_req_input_len,
            max_prefill_tokens=get_schedule().max_prefill_tokens,
        ),
    )
    if len(input_ids) > limit:
        raise ValueError(
            f"The prompt has {len(input_ids)} tokens; Valen permits at most {limit}"
        )
    parse_valen_layout(obj.decision_layout, len(input_ids))


def warmup(original, server_args):
    # Import only after protocol and serving class hooks have been applied.
    from sglang.srt.entrypoints import http_server as server
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_serving

    manager = server._global_state.tokenizer_manager
    if getattr(manager.model_config, "valen_head_config", None) is None:
        return original(server_args)
    headers = {}
    if get_serving().api_key:
        headers["Authorization"] = f"Bearer {get_serving().api_key}"
    verify = server.ssl_verify_of(server_args)
    url = server_args.url()
    try:
        for attempt in range(120):
            try:
                response = server.requests.get(
                    url + "/model_info", headers=headers, timeout=5, verify=verify
                )
                response.raise_for_status()
                break
            except server.requests.exceptions.RequestException:
                if attempt == 119:
                    raise
                time.sleep(1)
        timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
        response = server.requests.post(
            url + "/v1/systemone",
            json={
                "model": manager.served_model_name,
                "state": "The capital city of France is Paris.",
                "questions": {
                    "warmup": {"type": "noul", "instructions": "Is the state true?"}
                },
            },
            headers=headers,
            timeout=timeout if timeout > 0 else 600,
            verify=verify,
        )
        response.raise_for_status()
        manager.server_status = server.ServerStatus.Up
        return True
    except Exception:
        logger.exception("Valen warmup failed")
        server.kill_process_tree(os.getpid())
        return False


def resolve_warmup_callback(original, *args, **kwargs):
    """Resolve a callback captured in upstream function defaults before hooks ran.

    Replacing a module attribute cannot change a function's pre-bound defaults.
    Preserve callers' explicit custom warmups; refresh only the stock callback.
    """
    import inspect

    from sglang.srt.entrypoints import http_server

    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    callback = bound.arguments["execute_warmup_func"]
    if (
        getattr(callback, "__module__", None) == http_server.__name__
        and getattr(callback, "__name__", None) == "_execute_server_warmup"
    ):
        bound.arguments["execute_warmup_func"] = http_server._execute_server_warmup
    return original(*bound.args, **bound.kwargs)
