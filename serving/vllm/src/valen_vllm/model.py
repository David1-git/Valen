"""Native vLLM Qwen3.5 execution with Valen's FP32 Mixer readout."""

import torch
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from valen_vllm.head import ValenMixerHead
from valen_vllm.pooler import ValenPooler


class ValenProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs):
        processor = super().get_hf_processor(**kwargs)
        image_kwargs = self.get_hf_config().valen_media_kwargs.get("images_kwargs", {})
        if "size" in image_kwargs:
            processor.image_processor.size = dict(image_kwargs["size"])
        return processor


_Base = as_embedding_model(Qwen3_5ForConditionalGeneration)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=ValenProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ValenQwenForDecisionMaking(_Base):
    supports_multimodal_pruning = False

    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        config = vllm_config.model_config.hf_config
        if (
            config.valen_backbone_autocast_dtype != "bfloat16"
            or vllm_config.model_config.dtype != torch.bfloat16
        ):
            raise ValueError(
                "This build requires a published BF16-compute Valen checkpoint"
            )
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Valen currently requires TP=PP=1")
        if (
            vllm_config.quant_config is not None
            or vllm_config.speculative_config is not None
        ):
            raise ValueError("Quantized and speculative Valen execution is unsupported")
        if vllm_config.lora_config is not None:
            raise ValueError(
                "Valen requires complete merged checkpoints, not runtime LoRA adapters"
            )
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("Disable chunked prefill for Valen")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _init_pooler(self, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        return ValenPooler(
            ValenMixerHead(config.text_config.hidden_size, config.valen_head)
        )

    def load_weights(self, weights):
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

        params = dict(self.pooler.head.named_parameters())
        seen = set()

        def backbone():
            for name, value in weights:
                if name.startswith("head."):
                    key = name.removeprefix("head.")
                    if key not in params or key in seen:
                        raise ValueError(f"Unexpected or duplicate head weight: {name}")
                    default_weight_loader(params[key], value)
                    seen.add(key)
                elif name.startswith("backbone."):
                    yield "model." + name.removeprefix("backbone."), value
                else:
                    raise ValueError(f"Unexpected checkpoint weight: {name}")

        loaded = Qwen3_5ForConditionalGeneration.load_weights(self, backbone())
        if missing := params.keys() - seen:
            raise ValueError(f"Missing Valen head weights: {sorted(missing)}")
        if any(p.dtype != torch.float32 for p in params.values()):
            raise RuntimeError("Valen head must remain FP32")
        return loaded | {"pooler.head." + key for key in seen}
