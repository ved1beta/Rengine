"""
Core inference engine tests
"""

import torch
import pytest
from RLLM.inference.engine import (
    ModelConfig,
    Transformer,
    InferenceEngine,
    KVCache,
    top_k_filter,
    top_p_filter,
    sample,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=1000,
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        intermediate_dim=256,
        max_seq_len=512,
    )


@pytest.fixture
def model(config, device):
    return Transformer(config).to(device)


@pytest.fixture
def engine(model, device):
    return InferenceEngine(model, device)


def test_model_config():
    """Test model config initialization."""
    cfg = ModelConfig(
        vocab_size=1000,
        num_layers=4,
        num_heads=8,
        hidden_dim=512,
        intermediate_dim=2048,
    )
    assert cfg.head_dim == 64
    assert cfg.max_seq_len == 2048


def test_kv_cache_creation(config, device):
    """Test KV cache initialization and reset."""
    cache = KVCache(
        num_layers=config.num_layers,
        batch=2,
        max_seq_len=128,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        device=device,
    )
    
    assert len(cache.K) == config.num_layers
    assert len(cache.V) == config.num_layers
    assert cache.seq_len == 0
    
    cache.seq_len = 10
    cache.reset()
    assert cache.seq_len == 0


def test_forward_step(engine, device):
    """Test single forward step."""
    batch_size = 2
    seq_len = 5
    tokens = torch.randint(0, engine.config.vocab_size, (batch_size, seq_len)).to(device)
    
    cache = engine.create_cache(batch_size, seq_len + 10)
    logits, hidden = engine.forward_step(tokens, cache)
    
    assert logits.shape == (batch_size, engine.config.vocab_size)
    assert hidden.shape == (batch_size, engine.config.hidden_dim)


def test_generate(engine, device):
    """Test autoregressive generation."""
    batch_size = 2
    prompt_len = 5
    max_new_tokens = 10
    
    prompt = torch.randint(0, engine.config.vocab_size, (batch_size, prompt_len)).to(device)
    
    tokens, logprobs = engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
    )
    
    assert tokens.shape[0] == batch_size
    assert tokens.shape[1] <= max_new_tokens
    assert logprobs.shape == tokens.shape


def test_greedy_sampling(device):
    """Test greedy decoding (temperature=0)."""
    logits = torch.randn(4, 100).to(device)
    tokens = sample(logits, temperature=0.0)
    
    assert tokens.shape == (4,)
    assert torch.all(tokens == logits.argmax(dim=-1))


def test_top_k_filter(device):
    """Test top-k filtering."""
    logits = torch.randn(2, 100).to(device)
    filtered = top_k_filter(logits, k=10)
    
    # Check that only k values per batch are not -inf
    for i in range(2):
        valid_count = (filtered[i] != float("-inf")).sum()
        assert valid_count == 10


def test_top_p_filter(device):
    """Test nucleus (top-p) sampling."""
    logits = torch.randn(2, 100).to(device)
    filtered = top_p_filter(logits, p=0.9)
    
    # Check that some values are filtered
    assert (filtered == float("-inf")).any()
    
    # p=1.0 should not filter anything
    unfiltered = top_p_filter(logits, p=1.0)
    assert torch.allclose(logits, unfiltered)


def test_generation_with_eos(engine, device):
    """Test early stopping with EOS token."""
    batch_size = 1
    prompt_len = 3
    eos_token = 0
    
    prompt = torch.randint(1, engine.config.vocab_size, (batch_size, prompt_len)).to(device)
    
    tokens, logprobs = engine.generate(
        prompt,
        max_new_tokens=100,
        temperature=0.0,
        eos_token_id=eos_token,
    )
    
    # Should stop when EOS is generated
    assert tokens.shape[1] <= 100


def test_deterministic_generation(engine, device):
    """Test that greedy generation is deterministic."""
    prompt = torch.tensor([[1, 2, 3]]).to(device)
    
    tokens1, _ = engine.generate(prompt, max_new_tokens=10, temperature=0.0)
    tokens2, _ = engine.generate(prompt, max_new_tokens=10, temperature=0.0)
    
    assert torch.equal(tokens1, tokens2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
