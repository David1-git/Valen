"""Numerical and contract tests independent of CUDA/vLLM native extensions."""

import base64
import importlib.util
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from valen_vllm import compatibility
from valen_vllm.answers import answers
from valen_vllm.compatibility import verify_engine, verify_runtime
from valen_vllm.configuration import ValenQwenConfig
from valen_vllm.endpoint import ValenEndpointPlugin, read_image
from valen_vllm.head import (
    ValenMixerHead,
    ValenQuestion,
    pack_valen_layout,
    parse_valen_layout,
    pool_features,
)
from valen_vllm.protocol import SystemOneRequest


def request():
    return SystemOneRequest(
        model="valen",
        state="An invoice was charged twice.",
        questions={
            "route": {
                "type": "choice",
                "instructions": "Who?",
                "criteria": {"billing": "Billing", "support": "Support"},
            },
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "severity": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["Low", "Medium", "High"],
            },
        },
    )


class TestNumerics(unittest.TestCase):
    def test_roles_match_trained_reference_with_image_shift(self):
        torch.manual_seed(27)
        h = torch.randn(36, 12, dtype=torch.bfloat16)
        questions = [
            ValenQuestion((5, 8), ((10, 13), (14, 16)), 22),
            ValenQuestion((8, 10), ((16, 18),), 24),
        ]
        layout = pack_valen_layout(30, 5, questions)
        actual = pool_features(h, layout)
        expected = []
        for q in questions:
            for a, b in q.candidates:
                expected.append(
                    torch.stack(
                        [
                            h[:11].float().mean(0),
                            h[q.instruction[0] + 6 : q.instruction[1] + 6]
                            .float()
                            .mean(0),
                            h[a + 6 : b + 6].float().mean(0),
                            h[q.decision + 6].float(),
                        ]
                    )
                )
        torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)

    def test_mixer_matches_training_head(self):
        root = Path(__file__).resolve().parents[3]
        ref = root / "valen/modeling/qwen/heads.py"
        if not ref.is_file():
            model = os.environ.get("VALEN_TEST_MODEL")
            ref = Path(model or ".") / "heads.py"
        if not ref.is_file():
            self.skipTest("training/reference head source not available")
        spec = importlib.util.spec_from_file_location("training_heads", ref)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(9)
        trained = module.MixerHead(
            12, width=8, layers=2, token_hidden=5, channel_hidden=16
        ).float()
        native = ValenMixerHead(
            12,
            {
                "head_width": 8,
                "head_layers": 2,
                "head_token_hidden_dim": 5,
                "head_channel_hidden_dim": 16,
            },
        )
        native.load_state_dict(trained.state_dict(), strict=True)
        features = torch.randn(11, 4, 12, dtype=torch.bfloat16)
        torch.testing.assert_close(
            native(features), trained.score_features(features), rtol=0, atol=0
        )
        self.assertTrue(all(p.dtype == torch.float32 for p in native.parameters()))

    def test_head_stays_fp32_under_backbone_default_dtype(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            head = ValenMixerHead(4, {})
        finally:
            torch.set_default_dtype(previous)
        self.assertTrue(all(p.dtype == torch.float32 for p in head.parameters()))

    def test_layout_rejects_bad_or_partial_spans(self):
        good = pack_valen_layout(12, 3, [ValenQuestion((3, 5), ((5, 8),), 10)])
        for bad, length in [
            (good, 11),
            (good + [0], 12),
            ([12, 3, 1], 12),
            ([12, 0, 1, 3, 5, 10, 1, 5, 8], 12),
            ([12, 3, 1, 3, 5, 10, 1, 8, 7], 12),
            ([12, 3, True], 12),
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_valen_layout(bad, length)

    def test_answer_semantics(self):
        got = answers(request(), [0, 0, 1000, -1000, 0, 0, 0])
        self.assertEqual(got["route"]["choice"], "billing")
        self.assertEqual(got["route"]["confidence"], 0)
        self.assertEqual(got["urgent"]["noul"], 1)
        self.assertAlmostEqual(got["severity"]["score"], 1)
        self.assertEqual(list(got["severity"]["probabilities"]), ["0", "1", "2"])
        for logits in ([0], [math.nan] * 7, [math.inf] * 7):
            with self.assertRaises(RuntimeError):
                answers(request(), logits)


class TestContract(unittest.TestCase):
    def test_invalid_questions_fail(self):
        for q in [
            {"type": "choice", "instructions": "x", "criteria": {}},
            {"type": "score", "instructions": "x", "criteria": ["Only"]},
            {"type": "noul", "instructions": "x", "criteria": ["wrong"]},
        ]:
            with self.subTest(q=q), self.assertRaises(ValidationError):
                SystemOneRequest(model="valen", state="x", questions={"q": q})

    def test_model_and_questions_required(self):
        for bad in [
            {"state": "x", "questions": {}},
            {"model": "valen", "state": "x", "questions": {}},
        ]:
            with self.assertRaises(ValidationError):
                SystemOneRequest(**bad)

    def test_reserved_hosted_fields_rejected(self):
        with self.assertRaises(ValidationError):
            SystemOneRequest(**request().model_dump(), input="x")

    def test_native_config_preserves_architecture_and_precision(self):
        c = ValenQwenConfig(
            architectures=["ValenQwenForDecisionMaking"],
            valen_backbone_autocast_dtype="bfloat16",
        )
        self.assertEqual(c.architectures, ["ValenQwenForDecisionMaking"])
        self.assertEqual(c.dtype, torch.bfloat16)
        self.assertEqual(c.text_config.dtype, torch.bfloat16)
        for kwargs in [
            {"qwen_execution": "question"},
            {"valen_head": {"head_type": "bilinear"}},
            {"valen_backbone_autocast_dtype": "float16"},
        ]:
            with self.assertRaises(ValueError):
                ValenQwenConfig(**kwargs)

    def test_endpoint_schema_and_not_ready(self):
        app = FastAPI()
        ValenEndpointPlugin().attach_router(app)
        client = TestClient(app)
        response = client.post("/v1/systemone", json=request().model_dump())
        self.assertEqual(response.status_code, 503)
        response = client.post("/v1/systemone", json={"model": "valen"})
        self.assertEqual(response.status_code, 422)

    def test_image_decoding_and_no_server_path_fetching(self):
        buf = io.BytesIO()
        Image.new("RGB", (12, 8), "red").save(buf, format="PNG")
        source = SimpleNamespace(
            url="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        )
        self.assertEqual(read_image(source).size, (12, 8))
        for url in [
            "/etc/passwd",
            "file:///etc/passwd",
            "https://example.org/image.png",
            "data:image/png;base64,%%%",
        ]:
            with self.assertRaises(ValueError):
                read_image(SimpleNamespace(url=url))

    def test_lock_rejects_changed_source(self):
        source = os.environ.get("VALEN_TEST_VLLM_SOURCE")
        if not source:
            spec = importlib.util.find_spec("vllm")
            source = Path(spec.origin).parent if spec and spec.origin else None
        if not source:
            self.skipTest("source snapshot not provided")
        lock = json.loads(
            Path(compatibility.__file__).with_name("engine-lock.json").read_text()
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in lock["files"]:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((Path(source) / name).read_bytes())
            verify_engine(root)
            first = next(iter(lock["files"]))
            (root / first).write_text("bad source")
            with self.assertRaisesRegex(RuntimeError, "Unsupported vLLM"):
                verify_engine(root)

    def test_runtime_drift_is_rejected(self):
        lock = json.loads(
            Path(compatibility.__file__).with_name("runtime-lock.json").read_text()
        )
        with patch.object(
            compatibility.metadata, "version", side_effect=lambda name: lock[name]
        ):
            self.assertEqual(verify_runtime(), lock)
        with (
            patch.object(compatibility.metadata, "version", return_value="0.0.0"),
            self.assertRaisesRegex(RuntimeError, "Unsupported vLLM runtime"),
        ):
            verify_runtime()
