# Rengine

A lightweight inference engine for reinforcement-learning rollouts, built to outperform general-purpose generation APIs in the regime that matters for RL: small batches, short-to-medium sequences, and strict per-token overhead.

## The Problem

RL rollouts (PPO, DPO, RLHF) typically run at batch sizes 1-8 with tight throughput requirements. At this scale, frameworks like HuggingFace `model.generate()` spend more time in Python abstraction overhead — logits processors, dynamic cache management, attention mask construction — than in actual GPU compute. Rengine strips all of that out and provides a focused, inference-only decoding path.

## Benchmark

**Generation throughput (tok/s) — 35M param model, fp16, RTX 3050**

| Batch | Prompt | Gen | Rengine | HF `generate()` | Speedup |
|-------|--------|-----|---------|------------------|---------|
| 1     | 32     | 64  | 491     | 332              | 1.48x   |
| 4     | 32     | 64  | 461     | 318              | 1.45x   |
| 8     | 32     | 64  | 463     | 316              | 1.47x   |
| 1     | 128    | 128 | 501     | 340              | 1.47x   |
| 8     | 128    | 128 | 461     | 317              | 1.45x   |

Rengine holds a consistent ~1.45x advantage across batch sizes because decode steps remain Python-overhead bound at this model scale.

**Memory tradeoff:**

| Batch | Rengine | HF      |
|-------|---------|---------|
| 1     | 163 MB  | 155 MB  |
| 8     | 245 MB  | 181 MB  |

Rengine pre-allocates a static KV cache for the full sequence length. This trades higher memory for predictable latency — no dynamic allocation during decode.

## Current Optimizations

- Minimal Python logic per decode step (~15 lines in the hot loop)
- Static KV cache — slice writes, no object churn or reallocation
- No attention mask construction during single-token decode
- No logits processor pipeline overhead
- Direct `torch.argmax` / `torch.multinomial` — no abstraction layers

## Features

- Decoder-only Transformer with RoPE and SwiGLU
- Flash Attention via `F.scaled_dot_product_attention`
- Static pre-allocated KV cache
- Prefill + autoregressive decoding
- Per-token log-probabilities (RL-ready)
- Temperature, top-k, and nucleus (top-p) sampling
- Weight tying (embedding / lm_head)
- INT8 dynamic quantization


## Usage

```python
from Rengine.inference import InferenceEngine, ModelConfig, Transformer

config = ModelConfig(
    vocab_size=32000,
    num_layers=6,
    num_heads=8,
    hidden_dim=512,
    intermediate_dim=1376,
)

model = Transformer(config).cuda().half()
engine = InferenceEngine(model, device=torch.device("cuda"))

tokens, logprobs = engine.generate(
    prompt_tokens,       # [batch, prompt_len]
    max_new_tokens=128,
    temperature=0.0,     # greedy
)
# tokens:   [batch, num_generated]
# logprobs: [batch, num_generated]  <-- feed directly into PPO loss
```

## When to Use Rengine

**Good fit:** RLHF / PPO / DPO rollout collection, small-batch inference (1-8), fixed-length prompts, throughput-sensitive evaluation loops.

**Not a good fit:** production serving with continuous batching, variable-length padded batches, beam search, multi-tenant inference.

## Comparison

| | Rengine | HF `generate()` |
|---|---|---|
| RL-focused (logprobs) | Yes | Partial |
| Dynamic KV cache | No | Yes |
| Flash Attention / SDPA | Yes | Yes |
| RoPE | Yes | Yes |
| Weight tying | Yes | Yes |
| Continuous batching | No | Yes |
| Attention mask (padding) | No | Yes |
| Beam search | No | Yes |
| Speculative decoding | No | Yes |


## Requirements

- Python >= 3.10
- PyTorch >= 2.4
- `transformers` (for benchmarking comparison)

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest tests/ -v
```

## Roadmap

**Planned:** larger-model benchmarks (200M+), KV-cache memory optimization, reward-model inference path, PPO/TRL integration examples.

**Not planned:** distributed serving, beam search, speculative decoding.

## Status

Experimental, research-grade inference engine. APIs may change. Designed for learning, benchmarking, and RL experimentation.

## License

MIT
