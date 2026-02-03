
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
from ..kernals.cuda_sampler import sample_with_cuda_fallback


@dataclass
class ModelConfig:
    vocab_size: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    intermediate_dim: int
    max_seq_len: int = 2048
    head_dim: int = -1
    tie_weights: bool = True

    def __post_init__(self):
        if self.head_dim == -1:
            self.head_dim = self.hidden_dim // self.num_heads


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Precompute RoPE complex-exponential frequencies.

    Returns:
        freqs_cis: [max_seq_len, head_dim // 2]  (complex64)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    start_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K.

    Args:
        xq: [B, num_heads, S, head_dim]
        xk: [B, num_heads, S, head_dim]
        freqs_cis: [max_seq_len, head_dim // 2]
        start_pos: position offset for KV-cache decode steps
    """
    S = xq.shape[2]
    freqs = freqs_cis[start_pos : start_pos + S]  # [S, head_dim//2]

    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    freqs = freqs.unsqueeze(0).unsqueeze(0)

    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class KVCache:
    """Pre-allocated KV cache that persists across decoding steps.

    Layout per layer:
        K_cache[layer]: [batch, num_heads, max_seq_len, head_dim]
        V_cache[layer]: [batch, num_heads, max_seq_len, head_dim]
    """

    def __init__(
        self,
        num_layers: int,
        batch: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.K = [
            torch.zeros(batch, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.V = [
            torch.zeros(batch, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.seq_len = 0

    def reset(self):
        self.seq_len = 0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class Attention(nn.Module):
    """Multi-head attention with RoPE, SDPA, and KV-cache support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        dim = config.hidden_dim
        self.Wq = nn.Linear(dim, config.num_heads * config.head_dim, bias=False)
        self.Wk = nn.Linear(dim, config.num_heads * config.head_dim, bias=False)
        self.Wv = nn.Linear(dim, config.num_heads * config.head_dim, bias=False)
        self.Wo = nn.Linear(config.num_heads * config.head_dim, dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = h.shape  # S = 1 during decode, prompt_len during prefill

        # Project Q, K, V  →  [B, num_heads, S, head_dim]
        Q = self.Wq(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        pos = kv_cache.seq_len
        Q, K = apply_rope(Q, K, freqs_cis, pos)

        # Append to cache
        kv_cache.K[layer_idx][:, :, pos : pos + S, :] = K
        kv_cache.V[layer_idx][:, :, pos : pos + S, :] = V

        # Read full cached K, V up to current position
        K_all = kv_cache.K[layer_idx][:, :, : pos + S, :]
        V_all = kv_cache.V[layer_idx][:, :, : pos + S, :]

        # Use PyTorch SDPA — automatically selects Flash Attention / memory-efficient
        # kernel on CUDA, and handles causal masking internally.
        out = F.scaled_dot_product_attention(
            Q, K_all, V_all,
            is_causal=(S > 1),  # causal mask only needed during prefill
        )

        # Merge heads  →  [B, S, hidden_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.Wo(out)


class FeedForward(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.w2 = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: norm → attn → residual → norm → ffn → residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.ffn_norm = RMSNorm(config.hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        h = h + self.attention(self.attn_norm(h), kv_cache, layer_idx, freqs_cis)
        h = h + self.feed_forward(self.ffn_norm(h))
        return h


class Transformer(nn.Module):
    """Decoder-only transformer for inference."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.embed.weight

        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(config.head_dim, config.max_seq_len),
            persistent=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len]
            kv_cache:  KVCache instance

        Returns:
            logits:       [batch, seq_len, vocab_size]
            hidden_state: [batch, seq_len, hidden_dim]
        """
        h = self.embed(token_ids)

        for i, layer in enumerate(self.layers):
            h = layer(h, kv_cache, i, self.freqs_cis)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, h


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out everything outside the top-k logits."""
    if k <= 0:
        return logits
    top_values, top_indices = torch.topk(logits, k, dim=-1)
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(1, top_indices, top_values)
    return mask


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    sorted_mask = cumulative_probs > p
    sorted_mask = torch.cat(
        [sorted_mask.new_zeros(sorted_mask.shape[:-1] + (1,)), sorted_mask[..., :-1]],
        dim=-1,
    )

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    output = torch.full_like(logits, float("-inf"))
    output.scatter_(1, sorted_indices, sorted_logits)
    return output



def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    use_cuda_kernel: bool = True,
) -> torch.Tensor:
    """Sample next token from logits with temperature, top-k, and top-p.

    Args:
        logits: [batch, vocab_size]
        temperature: 0 = greedy
        top_k: 0 = disabled
        top_p: 1.0 = disabled
        use_cuda_kernel:True

    Returns:
        token_ids: [batch]
    """
    if (use_cuda_kernel and 
        top_k == 0 and 
        top_p >= 1.0 and
        logits.is_cuda):
        return sample_with_cuda_fallback(logits, temperature, use_cuda_kernel=True)
    

    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class InferenceEngine:
    """Stateless engine that wraps a Transformer for autoregressive generation.

    Usage:
        config = ModelConfig(...)
        model  = Transformer(config).cuda().half()
        engine = InferenceEngine(model, device=torch.device("cuda"))

        tokens, logprobs = engine.generate(prompt_ids, max_new_tokens=128)
    """

    def __init__(self, model: Transformer, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.config = model.config


    def create_cache(self, batch_size: int, max_seq_len: int) -> KVCache:
        dtype = next(self.model.parameters()).dtype
        return KVCache(
            self.config.num_layers,
            batch_size,
            max_seq_len,
            self.config.num_heads,
            self.config.head_dim,
            self.device,
            dtype,
        )


    @torch.no_grad()
    def forward_step(
        self,
        input_token_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single forward step through the transformer.

        Args:
            input_token_ids: [batch] (decode) or [batch, seq_len] (prefill)
            kv_cache: KVCache — updated **in-place**

        Returns:
            logits:       [batch, vocab_size]  (last position)
            hidden_state: [batch, hidden_dim]  (last position)
        """
        if input_token_ids.dim() == 1:
            input_token_ids = input_token_ids.unsqueeze(1)

        logits, h = self.model(input_token_ids, kv_cache)

        # Return only the last position
        return logits[:, -1, :], h[:, -1, :]

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive decode loop.

        Args:
            prompt_tokens:  [batch, prompt_len]  (token ids on self.device)
            max_new_tokens: number of new tokens to generate
            temperature:    sampling temperature (0 = greedy)
            top_k:          top-k filtering (0 = disabled)
            top_p:          nucleus sampling threshold (1.0 = disabled)
            eos_token_id:   stop generation when every sequence hits this id

        Returns:
            generated_tokens: [batch, num_generated]
            token_logprobs:   [batch, num_generated]
        """
        batch_size, prompt_len = prompt_tokens.shape
        max_seq_len = prompt_len + max_new_tokens

        kv_cache = self.create_cache(batch_size, max_seq_len)

        # Prefill: process entire prompt in one forward pass
        logits, _ = self.forward_step(prompt_tokens, kv_cache)
        kv_cache.seq_len += prompt_len

        tokens: List[torch.Tensor] = []
        logprobs: List[torch.Tensor] = []

        for step in range(max_new_tokens):
            # On step 0 reuse prefill logits; afterwards run a decode step
            if step > 0:
                logits, _ = self.forward_step(current_token, kv_cache)
                kv_cache.seq_len += 1

            logp = torch.log_softmax(logits, dim=-1)
            next_token = sample(logits, temperature, top_k, top_p)

            tokens.append(next_token)
            logprobs.append(logp.gather(1, next_token.unsqueeze(1)).squeeze(1))

            current_token = next_token

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return torch.stack(tokens, dim=1), torch.stack(logprobs, dim=1)
