"""Compute per-candidate logits on the GPU; only logits leave the worker."""

import torch
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate

from valen_vllm.head import pool_features


class ValenPooler(Pooler):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def get_supported_tasks(self):
        return {"plugin"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    @torch.no_grad()
    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        lengths = cursor.num_scheduled_tokens_cpu.tolist()
        outputs, offset = [], 0
        for index, (params, length, full_length, finished) in enumerate(
            zip(
                pooling_metadata.pooling_params,
                lengths,
                pooling_metadata.prompt_lens.tolist(),
                cursor.get_finished_mask(),
            )
        ):
            hidden = hidden_states[offset : offset + length]
            offset += length
            layout = (params.extra_kwargs or {}).get("valen_layout")
            if layout is None:
                # vLLM's memory profiler calls the pooler without request metadata.
                token_ids = pooling_metadata.get_prompt_token_ids_cpu()[index]
                if token_ids.numel() == 0 or torch.any(token_ids != 0):
                    raise ValueError(
                        "Valen decisions require /v1/systemone and decision layout metadata"
                    )
                outputs.append(hidden_states.new_zeros(1, dtype=torch.float32))
                continue
            if length != full_length or not finished:
                raise ValueError(
                    "Valen needs a complete prefill, without prefix reuse or chunking"
                )
            outputs.append(self.head(pool_features(hidden, layout)))
        return outputs
