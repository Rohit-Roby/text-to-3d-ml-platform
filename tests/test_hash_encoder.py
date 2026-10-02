import torch

from src.features.text_encoder import HashTextEncoder


def test_hash_encoder_is_deterministic():
    encoder = HashTextEncoder(embedding_dim=32)
    first = encoder.encode(["a wooden chair"])
    second = encoder.encode(["a wooden chair"])
    assert first.shape == (1, 32)
    assert torch.equal(first, second)
