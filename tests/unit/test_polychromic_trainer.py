import time
from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from transformers import TrainingArguments

from src.training.trainers.polychromic import PolychromicConfig, PolychromicTrainer


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])}


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        filtered = [t for t in tokens if t != self.pad_token_id]
        return " ".join(str(t) for t in filtered)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.generate_call_count = 0
        self.last_generate_kwargs = {}

    def forward(self, **kwargs):
        # Minimal forward pass returning a scalar loss
        return SimpleNamespace(loss=torch.tensor(0.0, requires_grad=True))

    def generate(self, input_ids, attention_mask=None, num_return_sequences=1, **kwargs):
        self.generate_call_count += 1
        self.last_generate_kwargs = {
            "num_return_sequences": num_return_sequences,
            "input_shape": tuple(input_ids.shape),
        }

        batch_size, prompt_len = input_ids.shape
        new_tokens = torch.full(
            (batch_size * num_return_sequences, 2),
            fill_value=99,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        repeated_input = input_ids.repeat_interleave(num_return_sequences, dim=0)
        return torch.cat([repeated_input, new_tokens], dim=1)


def make_trainer(tmp_path, config_overrides=None):
    config_kwargs = {
        "diversity_metric": "distinct",
        "cache_diversity_encoder": False,
    }
    if config_overrides:
        config_kwargs.update(config_overrides)
    config = PolychromicConfig(**config_kwargs)

    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        report_to=[],
    )

    trainer = PolychromicTrainer(
        polychromic_config=config,
        model=DummyModel(),
        args=args,
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
        tokenizer=DummyTokenizer(),
        data_collator=lambda x: x,
    )
    # Avoid nltk dependency within tests
    trainer.compute_diversity_score = lambda texts: 1.0
    return trainer


def test_generate_multiple_replies_vectorized(tmp_path):
    trainer = make_trainer(tmp_path)

    prompt = torch.tensor([[1, 2, 3]])
    replies = trainer.generate_multiple_replies(prompt, n=3)

    assert len(replies) == 3
    assert trainer.model.generate_call_count == 1
    assert trainer.model.last_generate_kwargs["num_return_sequences"] == 3
    # Trainer should restore original cache flag and training mode
    assert trainer.model.config.use_cache is False
    assert trainer.model.training is True


def test_compute_batch_diversity_records_example_time(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, {"n_generations": 2})

    # Provide deterministic timing
    timestamps = iter([100.0, 100.1, 100.2])

    monkeypatch.setattr(time, "time", lambda: next(timestamps))

    inputs = {
        "input_ids": torch.tensor([[7, 8, 9, 10]]),
        "labels": torch.tensor([[-100, -100, 11, 12]]),
    }

    score = trainer._compute_batch_diversity(inputs)

    assert score == pytest.approx(1.0)
    assert trainer.model.generate_call_count == 1
    # We should record per-example generation timing
    assert len(trainer.example_generation_times) == 1
    assert trainer.example_generation_times[-1] == pytest.approx(0.1)
