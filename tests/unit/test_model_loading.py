import types
import sys

import pytest

import scripts.training.train_model as train_module


class DummyModel:
    def __init__(self):
        self.config = types.SimpleNamespace(model_type="qwen3")
        self.parameter_count = 123
    
    def num_parameters(self):
        return self.parameter_count


class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"


def make_base_config():
    return {
        "model": {
            "path": "dummy-model",
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        },
        "data": {"max_length": 128},
        "lora": {
            "rank": 16,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj"],
        },
        "training": {"gradient_checkpointing": True},
    }


def test_load_model_and_tokenizer_hf_backend(monkeypatch):
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()
    
    monkeypatch.setattr(
        train_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: dummy_model,
    )
    monkeypatch.setattr(
        train_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    
    model, tokenizer, backend = train_module.load_model_and_tokenizer(make_base_config())
    assert backend == "hf"
    assert model is dummy_model
    assert tokenizer.pad_token == tokenizer.eos_token == "<eos>"


class _FastLanguageModelStub:
    last_from_pretrained = {}
    last_get_peft_model = {}
    
    @staticmethod
    def from_pretrained(model_name, **kwargs):
        _FastLanguageModelStub.last_from_pretrained = {"model_name": model_name, **kwargs}
        return DummyModel(), DummyTokenizer()
    
    @staticmethod
    def get_peft_model(model, **kwargs):
        _FastLanguageModelStub.last_get_peft_model = kwargs
        return model


def test_load_model_and_tokenizer_unsloth_backend(monkeypatch):
    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = _FastLanguageModelStub
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    
    cfg = make_base_config()
    cfg["unsloth"] = {"enabled": True}
    
    model, tokenizer, backend = train_module.load_model_and_tokenizer(cfg)
    assert backend == "unsloth"
    assert isinstance(model, DummyModel)
    assert tokenizer.pad_token == "<eos>"
    assert _FastLanguageModelStub.last_from_pretrained["model_name"] == "dummy-model"
    assert _FastLanguageModelStub.last_from_pretrained["load_in_4bit"] is True
    assert _FastLanguageModelStub.last_from_pretrained["max_seq_length"] == 128


def test_apply_lora_adapters_unsloth(monkeypatch):
    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = _FastLanguageModelStub
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    
    cfg = make_base_config()
    cfg["unsloth"] = {"enabled": True}
    
    model = DummyModel()
    result = train_module.apply_lora_adapters(model, cfg, backend="unsloth")
    assert result is model
    assert _FastLanguageModelStub.last_get_peft_model["lora_dropout"] == 0.1


def test_apply_lora_adapters_hf(monkeypatch):
    applied_marker = object()
    
    def fake_setup_lora_model(model, lora_cfg):
        return applied_marker
    
    monkeypatch.setattr(train_module, "setup_lora_model", fake_setup_lora_model)
    
    cfg = make_base_config()
    result = train_module.apply_lora_adapters(DummyModel(), cfg, backend="hf")
    assert result is applied_marker
