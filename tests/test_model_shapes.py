import torch

from src.models.cgan import ConditionalDiscriminator, ConditionalGenerator


def test_generator_output_shape():
    g = ConditionalGenerator(text_dim=768, noise_dim=128)
    out = g(torch.randn(2, 768), torch.randn(2, 128))
    assert out.shape == (2, 1, 32, 32, 32)


def test_discriminator_output_shape():
    d = ConditionalDiscriminator(text_dim=768)
    out = d(torch.randn(2, 1, 32, 32, 32), torch.randn(2, 768))
    assert out.shape == (2, 1)
