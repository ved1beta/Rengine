"""Quantization utilities for RLLM policies.

Supports INT8 dynamic quantization for reduced model size and faster inference.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch

from Rengine.models.policy import ActorCritic


def validate_numeric(value: float, name: str) -> float:
    """Validate that a numeric value is finite and non-negative."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value)}")
    if not (value >= 0 and value < float('inf')):
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def quantize_policy(
    checkpoint_path: str,
    output_path: str,
    obs_dim: int = 4,
    action_dim: int = 2,
    dtype: torch.dtype = torch.qint8,
    verbose: bool = True,
) -> dict:
    """Quantize a trained policy to INT8 for efficient inference.
    
    Args:
        checkpoint_path: Path to the trained checkpoint.
        output_path: Path to save the quantized model.
        obs_dim: Observation dimension (default: 4 for CartPole).
        action_dim: Action dimension (default: 2 for CartPole).
        dtype: Quantization dtype (default: torch.qint8).
        verbose: Whether to print progress.
        
    Returns:
        Dictionary with quantization metrics:
        - original_size_mb: Original model size in MB.
        - quantized_size_mb: Quantized model size in MB.
        - compression_ratio: Size reduction ratio.
    """
    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")
    
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    # Calculate original model size
    original_size_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    original_size_mb = original_size_bytes / (1024 * 1024)
    original_size_mb = validate_numeric(original_size_mb, "original_size_mb")
    
    if verbose:
        print(f"Original model size: {original_size_mb:.3f} MB")
        print(f"Quantizing with dtype: {dtype}")
    
    # Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=dtype,
    )
    
    # Save quantized model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(quantized_model.state_dict(), output_path)
    
    # Calculate quantized model size (from file)
    quantized_size_bytes = output_path.stat().st_size
    quantized_size_mb = quantized_size_bytes / (1024 * 1024)
    quantized_size_mb = validate_numeric(quantized_size_mb, "quantized_size_mb")
    
    compression_ratio = original_size_bytes / quantized_size_bytes if quantized_size_bytes > 0 else 0.0
    compression_ratio = validate_numeric(compression_ratio, "compression_ratio")
    
    results = {
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": compression_ratio,
        "output_path": str(output_path),
    }
    
    if verbose:
        print(f"Quantized model size: {quantized_size_mb:.3f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Saved to: {output_path}")
    
    return results


def load_quantized_policy(
    quantized_path: str,
    obs_dim: int = 4,
    action_dim: int = 2,
) -> torch.nn.Module:
    """Load a quantized policy for inference.
    
    Args:
        quantized_path: Path to the quantized model file.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        
    Returns:
        Quantized model ready for inference.
    """
    # Create and quantize a fresh model (structure needed for loading)
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    model.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    
    state_dict = torch.load(quantized_path, map_location="cpu", weights_only=False)
    quantized_model.load_state_dict(state_dict)
    
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description="Quantize RLLM policy to INT8")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ckpt_000300.pt",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/policy_int8.pt",
        help="Output path for quantized model",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=4,
        help="Observation dimension",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=2,
        help="Action dimension",
    )
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    print("\n" + "=" * 50)
    print("RLLM Policy Quantization")
    print("=" * 50)
    
    quantize_policy(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
    )
    
    print("=" * 50 + "\n")
    return 0


if __name__ == "__main__":
    exit(main())
