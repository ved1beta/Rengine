"""Run inference benchmarks for trained RLLM policies."""

import argparse
import torch
import time
from pathlib import Path

from RLLM.inference.load_model import load_policy
from RLLM.envs.cartpole import CartPoleEnv


def validate_numeric(value: float, name: str) -> float:
    """Validate that a numeric value is finite and non-negative."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value)}")
    if not (value >= 0 and value < float('inf')):
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def run_inference(
    checkpoint_path: str,
    num_episodes: int = 10,
    warmup_steps: int = 100,
    seed: int = 123,
    verbose: bool = True,
) -> dict:
    """Run inference and collect performance metrics.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        num_episodes: Number of episodes to run for evaluation.
        warmup_steps: Number of warmup inference steps.
        seed: Random seed for reproducibility.
        verbose: Whether to print results.
        
    Returns:
        Dictionary containing metrics:
        - latency_ms: Average inference latency in milliseconds.
        - mean_reward: Mean episode reward.
        - mean_length: Mean episode length.
        - episodes: Number of episodes run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = CartPoleEnv(device=device, seed=seed)
    
    model = load_policy(
        checkpoint_path,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    
    obs = env.reset()
    
    # Warm-up to stabilize GPU/CPU performance
    for _ in range(warmup_steps):
        with torch.no_grad():
            model(obs)
    
    # Benchmark inference latency
    latency_samples = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            action, _, _ = model(obs)
        end = time.perf_counter()
        latency_samples.append((end - start) * 1000)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    latency_ms = sum(latency_samples) / len(latency_samples)
    latency_ms = validate_numeric(latency_ms, "latency_ms")
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    obs = env.reset()
    
    for _ in range(num_episodes):
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action, _, _ = model(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward.item()
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        obs = env.reset()
    
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    mean_length = sum(episode_lengths) / len(episode_lengths)
    
    mean_reward = validate_numeric(mean_reward, "mean_reward")
    mean_length = validate_numeric(mean_length, "mean_length")
    
    results = {
        "latency_ms": latency_ms,
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "episodes": num_episodes,
        "device": str(device),
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"RLLM Inference Results")
        print(f"{'='*50}")
        print(f"Device:              {device}")
        print(f"Checkpoint:          {checkpoint_path}")
        print(f"Episodes:            {num_episodes}")
        print(f"{'='*50}")
        print(f"Latency (avg):       {latency_ms:.3f} ms")
        print(f"Mean Reward:         {mean_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        print(f"{'='*50}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RLLM inference benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ckpt_000300.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed",
    )
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    run_inference(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        warmup_steps=args.warmup,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    exit(main())



