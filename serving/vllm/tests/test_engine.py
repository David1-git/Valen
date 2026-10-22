"""Integration boundaries; run with the installed wheel and real pinned vLLM."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


@unittest.skipUnless(
    os.environ.get("VALEN_TEST_VLLM") == "1", "requires real pinned vLLM runtime"
)
class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from valen_vllm import register

        register()
        from valen_vllm.model import ValenQwenForDecisionMaking
        from valen_vllm.pooler import ValenPooler

        cls.model_cls = ValenQwenForDecisionMaking
        cls.pooler_cls = ValenPooler

    def test_registration_and_native_pooling_model(self):
        from valen_vllm import register
        from valen_vllm.compatibility import verify_engine
        from vllm import ModelRegistry
        from vllm.model_executor.models.interfaces import supports_multimodal
        from vllm.model_executor.models.interfaces_base import is_pooling_model

        register()
        register()
        verify_engine()
        self.assertIn("ValenQwenForDecisionMaking", ModelRegistry.get_supported_archs())
        self.assertTrue(is_pooling_model(self.model_cls))
        self.assertTrue(supports_multimodal(self.model_cls))

    def test_config_loads_without_remote_code(self):
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(
            model=os.environ["VALEN_TEST_MODEL"],
            runner="pooling",
            dtype="bfloat16",
            max_model_len=8192,
            max_num_batched_tokens=8192,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            limit_mm_per_prompt={"image": 16, "video": 0},
        )
        config = args.create_engine_config()
        self.assertEqual(
            config.model_config.hf_config.architectures, ["ValenQwenForDecisionMaking"]
        )
        self.assertEqual(config.model_config.dtype, torch.bfloat16)
        self.assertFalse(config.scheduler_config.enable_chunked_prefill)

    def test_real_metadata_batching_and_prefix_rejection(self):
        import numpy as np
        from valen_vllm.head import (
            ValenMixerHead,
            ValenQuestion,
            pack_valen_layout,
            pool_features,
        )
        from vllm.pooling_params import PoolingParams
        from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

        head = ValenMixerHead(
            6,
            {
                "head_width": 4,
                "head_layers": 1,
                "head_token_hidden_dim": 3,
                "head_channel_hidden_dim": 8,
            },
        )
        pooler = self.pooler_cls(head)
        layouts = [
            pack_valen_layout(n, 3, [ValenQuestion((3, 5), ((5, 8),), n - 1)])
            for n in (12, 15)
        ]
        params = [
            PoolingParams(task="plugin", extra_kwargs={"valen_layout": l})
            for l in layouts
        ]
        metadata = PoolingMetadata(
            prompt_lens=torch.tensor([12, 15]),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=params,
            pooling_states=[PoolingStates(), PoolingStates()],
        )
        metadata.build_pooling_cursor(
            np.array([12, 15]), torch.tensor([12, 15]), torch.device("cpu")
        )
        hidden = torch.randn(27, 6, dtype=torch.bfloat16)
        got = pooler(hidden, metadata)
        for result, h, l in zip(got, [hidden[:12], hidden[12:]], layouts):
            torch.testing.assert_close(result, head(pool_features(h, l)))
        metadata.build_pooling_cursor(
            np.array([11, 15]), torch.tensor([12, 15]), torch.device("cpu")
        )
        with self.assertRaisesRegex(ValueError, "complete prefill"):
            pooler(hidden, metadata)

    def test_weight_mapping_and_head_completeness(self):
        from valen_vllm.head import ValenMixerHead

        head = ValenMixerHead(
            4,
            {
                "head_width": 4,
                "head_layers": 1,
                "head_token_hidden_dim": 3,
                "head_channel_hidden_dim": 8,
            },
        )
        owner = SimpleNamespace(pooler=self.pooler_cls(head))
        head_weights = [
            ("head." + n, torch.randn_like(p)) for n, p in head.named_parameters()
        ]
        received = []

        def load_backbone(self, weights):
            received.extend(list(weights))
            return {"visual.fake"}

        with patch(
            "vllm.model_executor.models.qwen3_5.Qwen3_5ForConditionalGeneration.load_weights",
            load_backbone,
        ):
            loaded = self.model_cls.load_weights(
                owner, iter(head_weights + [("backbone.visual.fake", torch.ones(1))])
            )
            self.assertEqual(received[0][0], "model.visual.fake")
            self.assertIn("pooler.head.roles", loaded)
            for n, value in head_weights:
                torch.testing.assert_close(dict(head.named_parameters())[n[5:]], value)
            with self.assertRaisesRegex(ValueError, "Missing Valen head"):
                self.model_cls.load_weights(owner, iter(head_weights[:-1]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.model_cls.load_weights(
                    owner, iter(head_weights + [head_weights[0]])
                )
            with self.assertRaisesRegex(ValueError, "Unexpected checkpoint"):
                self.model_cls.load_weights(owner, iter([("unknown", torch.ones(1))]))
