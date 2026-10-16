"""Check the public adapter contract with the actual upstream validator."""
from copy import deepcopy
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("decision_index")
from decision_index.engines import Unsupported, load_engine, validate

from evaluation.decision_index import engine as adapter
from evaluation.decision_index._compat import ensure_file_digest


@pytest.fixture
def engine(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "valen_qwen", "valen_max_length": 32768}))

    class Model(torch.nn.Module):
        architecture = "qwen"
        backbone_autocast_dtype = None
        attention_backends = {"text": "sdpa", "vision": "sdpa"}

        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1)

        def forward_state(self, compiled):
            return [torch.arange(len(q.keys), dtype=torch.float32) for q in compiled.questions]

    class Compiler:
        def compile(self, record):
            self.seen = record
            questions = [SimpleNamespace(qid=qid, kind=q["type"],
                keys=list(q["criteria"]) if q["type"] == "choice" else ["true", "false"])
                for qid, q in record["request"]["questions"].items()]
            return SimpleNamespace(questions=questions, inputs={}, logical_tokens=10, compute_tokens=10)

    monkeypatch.setattr(adapter, "build_model", lambda config: Model())
    monkeypatch.setattr(adapter, "build_compiler", lambda config: Compiler())
    return load_engine("evaluation.decision_index.engine:ValenEngine", model=str(tmp_path), device="cpu")


def test_structured_requests_preserve_options_and_use_native_probabilities(engine):
    state = {"scene": ["猫", "sofa"]}
    questions = {
        "animal": {"type": "choice", "instructions": {"task": "Which animal?"},
                   "criteria": {"cat": {"label": "猫"}, "dog": None, "bird": ["鸟"]}},
        "truth": {"type": "noul", "instructions": "A cat is present."},
    }
    before = deepcopy((state, questions))
    response, raw = engine(state, questions)
    validate(questions, response)
    assert (state, questions) == before
    request = engine.compiler.seen["request"]
    assert request["state"] == '{"scene":["猫","sofa"]}'
    choice = request["questions"]["animal"]
    assert choice["instructions"] == '{"task":"Which animal?"}'
    assert list(choice["criteria"].items()) == [("cat", '{"label":"猫"}'), ("dog", "dog"), ("bird", '["鸟"]')]
    assert request["questions"]["truth"] == questions["truth"]
    assert list(response["answers"]) == list(questions)
    assert response["answers"]["animal"]["choice"] == "bird"
    assert response["answers"]["truth"]["noul"] == pytest.approx(torch.tensor([0., 1.]).softmax(0)[0].item())
    assert raw is None
    assert engine.config["qwen_execution"] == "shared_state" and engine.config["stage"] == "warmup"
    assert not any(p.requires_grad for p in engine.model.parameters())


@pytest.mark.parametrize("message,expected", [
    ("Shared state: 33000 tokens exceed max_length=32768; no truncation", Unsupported),
    ("Reserved tokenizer control token in state", Unsupported),
    ("Invalid question schema", ValueError),
])
def test_capacity_limits_and_runtime_errors_remain_distinct(engine, monkeypatch, message, expected):
    def fail(record):
        raise ValueError(message)
    monkeypatch.setattr(engine.compiler, "compile", fail)
    with pytest.raises(expected) as exc:
        engine("State", {})
    assert type(exc.value) is expected


def test_hub_revision_and_local_snapshot_are_recorded(engine, monkeypatch):
    folder = engine.config["model_path"]
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return folder
    monkeypatch.setattr(adapter, "snapshot_download", download)
    hub = adapter.ValenEngine("Valen-Team/Valen-2B", revision="fixed-commit", device="cpu")
    assert calls == [{"repo_id": "Valen-Team/Valen-2B", "revision": "fixed-commit"}]
    assert hub.provenance["revision"] == "fixed-commit"
    assert hub.provenance["config_sha256"] == engine.provenance["config_sha256"]
    assert engine.provenance["revision"] is None
    assert engine.runtime()["hardware"] == "CPU"


def test_upstream_runner_records_failures_and_keeps_labels_out(engine, monkeypatch, tmp_path):
    from decision_index.runner import run

    seen = []
    compile_request = engine.compiler.compile
    def compile(record):
        seen.append(deepcopy(record))
        if record["request"]["state"] == "overlong":
            raise ValueError("Shared state: 33000 tokens exceed max_length=32768; no truncation")
        return compile_request(record)
    monkeypatch.setattr(engine.compiler, "compile", compile)
    monkeypatch.setattr(adapter, "build_compiler", lambda config: engine.compiler)
    questions = {"q": {"type": "choice", "instructions": "Pick", "criteria": {"a": {"label": "A"}, "b": None}}}
    rows = [{"state": state, "questions": questions, "expected": {"q": "secret-label"},
             "_evaluation": {"run_id": str(i), "catalog_id": 24, "dataset": "fixture"}}
            for i, state in enumerate([{"scene": "猫"}, "overlong"])]
    source = tmp_path / "rows.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "results"
    status = run("evaluation.decision_index.engine:ValenEngine",
                 {"model": engine.config["model_path"], "device": "cpu"},
                 source, out, warm=False, compact=True)
    assert status["completed"] == 2 and status["counts"] == {"ok": 1, "unsupported": 1}
    predictions = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [row["status"] for row in predictions] == ["ok", "unsupported"]
    assert all(set(record) == {"request"} and set(record["request"]) == {"state", "questions"} for record in seen)
    assert "secret-label" not in json.dumps(seen)
    environment = json.loads((out / "environment.json").read_text())
    assert environment["model_source"]["qwen_execution"] == "shared_state"
    assert environment["model_source"]["max_length"] == 32768


def test_base_qwen_cannot_silently_create_an_untrained_head(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    with pytest.raises(ValueError, match="complete published Valen"):
        adapter.ValenEngine(str(tmp_path), device="cpu")


def test_python310_digest_fallback_matches_standard_sha256(monkeypatch):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    ensure_file_digest()
    payload = b"frozen-suite" * 100001
    assert hashlib.file_digest(io.BytesIO(payload), "sha256").hexdigest() == hashlib.sha256(payload).hexdigest()
    assert hashlib.file_digest(io.BytesIO(payload), hashlib.sha256).hexdigest() == hashlib.sha256(payload).hexdigest()
