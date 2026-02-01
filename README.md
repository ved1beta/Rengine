# RLLM

**Reinforcement Learning infrastructure with inference and quantization support.**

RLLM is a minimal, educational implementation of Proximal Policy Optimization (PPO) with a focus on clean code, inference optimization, and model quantization.

## Features

- **PPO Training**: Full implementation of PPO with GAE (Generalized Advantage Estimation)
- **Modular Architecture**: models, rollout buffers, and losses
- **Inference Pipeline**: Optimized inference with benchmarking
- **INT8 Quantization**: Dynamic quantization support for efficient deployment
- **Checkpoint Management**: Save/resume training with full RNG state preservation

## Installation

### From Source

```bash
git clone https://github.com/ved1beta/RLLM.git
cd RLLM
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With Quantization Support

```bash
pip install -e ".[quantization]"
```

## Quick Start

### Training

Train a PPO agent on CartPole:

```bash
# Using the CLI entry point
rllm-train

# Or run directly
python -m RLLM.main
```

Training will:
- Run for 500 updates by default
- Save checkpoints every 50 updates to `checkpoints/`
- Log metrics every 10 updates

### Inference

Run inference with a trained checkpoint:

```bash
# Using the CLI entry point
rllm-inference --checkpoint checkpoints/ckpt_000300.pt --episodes 10

# Or run directly
python -m RLLM.inference.run_inference --checkpoint checkpoints/ckpt_000300.pt
```

**Options:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `checkpoints/ckpt_000300.pt` | Path to checkpoint file |
| `--episodes` | `10` | Number of evaluation episodes |
| `--warmup` | `100` | Number of warmup inference steps |
| `--seed` | `123` | Random seed for reproducibility |

**Example Output:**
```
==================================================
RLLM Inference Results
==================================================
Device:              cuda
Checkpoint:          checkpoints/ckpt_000300.pt
Episodes:            10
==================================================
Latency (avg):       0.125 ms
Mean Reward:         487.50
Mean Episode Length: 487.5
==================================================
```

### Quantization

Quantize a trained model to INT8 for efficient inference:

```bash
# Using the CLI entry point
rllm-quantize --checkpoint checkpoints/ckpt_000300.pt --output artifacts/policy_int8.pt

# Or run directly
python -m RLLM.inference.quant --checkpoint checkpoints/ckpt_000300.pt
```

**Options:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `checkpoints/ckpt_000300.pt` | Path to trained checkpoint |
| `--output` | `artifacts/policy_int8.pt` | Output path for quantized model |
| `--obs-dim` | `4` | Observation dimension |
| `--action-dim` | `2` | Action dimension |

**Example Output:**
```
==================================================
RLLM Policy Quantization
==================================================
Loading checkpoint: checkpoints/ckpt_000300.pt
Original model size: 0.042 MB
Quantizing with dtype: torch.qint8
Quantized model size: 0.015 MB
Compression ratio: 2.80x
Saved to: artifacts/policy_int8.pt
==================================================
```

## Project Structure

```
RLLM/
├── src/RLLM/
│   ├── __init__.py           # Package exports, version
│   ├── main.py               # Training entry point
│   ├── rollout.py            # Rollout buffer for PPO
│   ├── losses.py             # PPO loss functions (policy, value, GAE)
│   ├── chkpt.py              # Checkpoint save/load utilities
│   ├── envs/
│   │   ├── base.py           # Abstract environment interface
│   │   └── cartpole.py       # CartPole environment wrapper
│   ├── models/
│   │   └── policy.py         # ActorCritic neural network
│   └── inference/
│       ├── __init__.py       # Inference module exports
│       ├── load_model.py     # Model loading utilities
│       ├── run_inference.py  # Inference benchmark script
│       └── quant.py          # INT8 quantization utilities
├── tests/
│   └── test_env.py           # Unit tests
├── checkpoints/              # Saved training checkpoints
├── artifacts/                # Quantized models and exports
├── requirements.txt          # Dependencies
└── setup.py                  # Package configuration
```

## API Reference

### Training

```python
from RLLM.main import train

# Start training
train(rank=0, world_size=1)
```

### Inference

```python
from RLLM.inference import load_policy, run_inference

# Load a trained policy
model = load_policy(
    "checkpoints/ckpt_000300.pt",
    obs_dim=4,
    action_dim=2,
    device="cuda",
)

# Run inference benchmark
results = run_inference(
    checkpoint_path="checkpoints/ckpt_000300.pt",
    num_episodes=10,
    warmup_steps=100,
)
# Returns: {"latency_ms": float, "mean_reward": float, "mean_length": float, ...}
```

### Quantization

```python
from RLLM.inference import quantize_policy, load_quantized_policy

# Quantize a trained model
results = quantize_policy(
    checkpoint_path="checkpoints/ckpt_000300.pt",
    output_path="artifacts/policy_int8.pt",
)
# Returns: {"original_size_mb": float, "quantized_size_mb": float, "compression_ratio": float}

# Load quantized model for inference
model = load_quantized_policy("artifacts/policy_int8.pt", obs_dim=4, action_dim=2)
```

### Custom Environments

Extend `BaseEnv` to add new environments:

```python
from RLLM.envs.base import BaseEnv

class MyEnv(BaseEnv):
    def reset(self):
        # Return initial observation as tensor
        ...
    
    def step(self, action):
        # Return (obs, reward, done, info)
        ...
    
    @property
    def obs_dim(self):
        return 4
    
    @property
    def action_dim(self):
        return 2
```

## Configuration

### Model Architecture

The `ActorCritic` model ([policy.py](src/RLLM/models/policy.py)):

```
MLPEncoder: obs_dim → 64 → Tanh → 128 → Tanh
PolicyHead: 128 → action_dim (logits)
ValueHead:  128 → 1 (value)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.4
- Gymnasium
- NumPy

See [requirements.txt](requirements.txt) for full dependencies.

## License

MIT License - see [LICENSE](LICENSE) for details.
