"""
JIT-compiled CUDA sampling operations for Rengine inference.

This module provides a drop-in replacement for PyTorch sampling operations
using custom CUDA kernels for improved performance.
"""

import os
import torch
from pathlib import Path
from torch.utils.cpp_extension import load

_kernels = None

def _load_kernels():
    """Lazy load the CUDA kernels via JIT compilation."""
    global _kernels
    if _kernels is not None:
        return _kernels
    
    kernel_dir = Path(__file__).parent
    sources = [
        str(kernel_dir / 'sample_binding.cu'),
        str(kernel_dir / 'sample.cu'),
    ]
    
    # Get CUDA arch from environment or detect
    cuda_arch = os.environ.get('CUDA_ARCH', 'sm_86')
    
    print("Compiling CUDA sampling kernels (this may take a minute)...")
    _kernels = load(
        name='rengine_sample_ops',
        sources=sources,
        extra_cuda_cflags=[
            '-O3',
            f'-arch={cuda_arch}',
            '--use_fast_math',
            '-lineinfo',
        ],
        verbose=True,
    )
    print("✓ CUDA kernels loaded successfully")
    return _kernels


def greedy_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Greedy sampling using custom CUDA kernel.
    
    Args:
        logits: [batch, vocab_size] tensor of logits
        temperature: sampling temperature (default: 1.0)
    
    Returns:
        token_ids: [batch] tensor of sampled token indices
    
    Example:
        >>> logits = torch.randn(32, 50257, device='cuda')
        >>> tokens = greedy_sample(logits, temperature=1.0)
        >>> tokens.shape
        torch.Size([32])
    """
    if not logits.is_cuda:
        raise ValueError("logits must be on CUDA device")
    
    if logits.dtype != torch.float32:
        logits = logits.float()
    
    kernels = _load_kernels()
    return kernels.greedy_sample(logits, temperature)


def sample_with_cuda_fallback(
    logits: torch.Tensor,
    temperature: float = 1.0,
    use_cuda_kernel: bool = True,
) -> torch.Tensor:
    """
    Sample with automatic fallback to PyTorch if CUDA kernel fails.
    
    Args:
        logits: [batch, vocab_size] tensor
        temperature: sampling temperature
        use_cuda_kernel: if True, try CUDA kernel first
    
    Returns:
        token_ids: [batch] tensor
    """
    if use_cuda_kernel and logits.is_cuda:
        try:
            return greedy_sample(logits, temperature)
        except Exception as e:
            print(f"CUDA kernel failed, falling back to PyTorch: {e}")
    
    # PyTorch fallback
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    
    logits = logits / temperature
    return torch.argmax(logits, dim=-1)
