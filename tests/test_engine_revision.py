"""Pinning the model revision. A branch name is not a version."""

import sys
import types

import pytest

from inference.engine import VlmEngine


class FakeLoaded:
    """Stands in for a loaded model or processor, recording how it was asked for."""

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.config = types.SimpleNamespace(_commit_hash="a" * 40)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


@pytest.fixture
def fake_transformers(monkeypatch):
    calls = {}

    def recorder(name):
        class Auto:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls[name] = (model_id, kwargs)
                return FakeLoaded(model_id, **kwargs)
        return Auto

    mod = types.ModuleType("transformers")
    mod.AutoProcessor = recorder("processor")
    mod.AutoModelForImageTextToText = recorder("model")
    monkeypatch.setitem(sys.modules, "transformers", mod)

    torch = types.ModuleType("torch")
    torch.float16 = "float16-dtype"
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def test_the_revision_reaches_both_from_pretrained_calls(fake_transformers):
    VlmEngine("org/model", revision="deadbeef")._load()
    assert fake_transformers["processor"][1]["revision"] == "deadbeef"
    assert fake_transformers["model"][1]["revision"] == "deadbeef"


def test_no_revision_passes_none_rather_than_inventing_one(fake_transformers):
    VlmEngine("org/model")._load()
    assert fake_transformers["model"][1]["revision"] is None


def test_describe_reports_the_commit_that_loaded_not_the_one_requested(fake_transformers):
    engine = VlmEngine("org/model", revision="main")
    engine._load()
    described = engine.describe()
    assert described["revision_requested"] == "main"
    assert described["revision_loaded"] == "a" * 40


def test_describe_before_load_reports_no_commit_rather_than_guessing():
    described = VlmEngine("org/model", revision="main").describe()
    assert described["revision_requested"] == "main"
    assert described["revision_loaded"] is None
