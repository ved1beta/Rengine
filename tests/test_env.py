import torch
import tempfile
from pathlib import Path

from RLLM.envs.cartpole import CartPoleEnv
from RLLM.models.policy import ActorCritic
from RLLM.rollout import RolloutBuffer
from RLLM.inference import load_policy, validate_numeric
from RLLM.inference.quant import quantize_policy, load_quantized_policy


def test_cartpole_env():
    device = torch.device("cpu")
    env = CartPoleEnv(seed=42, device=device)

    obs = env.reset()
    assert obs.shape[0] == env.obs_dim

    done = False
    steps = 0

    while not done and steps < 500:
        action = torch.randint(0, env.action_dim, (1,))
        obs, reward, done, info = env.step(action)

        assert obs.shape[0] == env.obs_dim
        assert reward.shape == ()
        steps += 1



def test_policy_forward():
    obs_dim = 4
    action_dim = 2

    model = ActorCritic(obs_dim, action_dim)

    obs = torch.randn(obs_dim)

    action, logprob, value = model(obs)

    assert action.dtype == torch.int64
    assert logprob.ndim == 0
    assert value.ndim == 0

def test_rollout_buffer():
    device = torch.device("cpu")
    buf = RolloutBuffer(4, obs_dim=3, device=device)

    for i in range(4):
        buf.add(
            obs=torch.randn(3),
            action=torch.tensor(1),
            reward=torch.tensor(1.0),
            done=False,
            logprob=torch.tensor(-0.5),
            value=torch.tensor(0.2),
        )

    data = buf.get()
    buf.check_finite()

    assert data["obs"].shape == (4, 3)
    assert data["actions"].shape == (4,)


def test_validate_numeric():
    """Test numeric validation function."""
    # Valid values
    assert validate_numeric(1.0, "test") == 1.0
    assert validate_numeric(0.0, "test") == 0.0
    assert validate_numeric(100, "test") == 100
    
    # Invalid values
    try:
        validate_numeric(float('inf'), "test")
        assert False, "Should raise ValueError for infinity"
    except ValueError:
        pass
    
    try:
        validate_numeric(-1.0, "test")
        assert False, "Should raise ValueError for negative"
    except ValueError:
        pass
    
    try:
        validate_numeric("not a number", "test")
        assert False, "Should raise ValueError for non-numeric"
    except ValueError:
        pass


def test_load_policy():
    """Test loading a policy from checkpoint."""
    obs_dim = 4
    action_dim = 2
    device = torch.device("cpu")
    
    # Create a temporary checkpoint
    model = ActorCritic(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": 10,
        }
        torch.save(checkpoint, ckpt_path)
        
        # Load the policy
        loaded_model = load_policy(
            str(ckpt_path),
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        
        # Verify it's in eval mode
        assert not loaded_model.training
        
        # Verify forward pass works
        obs = torch.randn(obs_dim)
        action, logprob, value = loaded_model(obs)
        
        assert action.dtype == torch.int64
        assert logprob.ndim == 0
        assert value.ndim == 0


def test_quantize_policy():
    """Test quantization of a policy."""
    obs_dim = 4
    action_dim = 2
    
    # Create a temporary checkpoint
    model = ActorCritic(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        quant_path = Path(tmpdir) / "test_quant.pt"
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": 10,
        }
        torch.save(checkpoint, ckpt_path)
        
        # Quantize the policy
        results = quantize_policy(
            checkpoint_path=str(ckpt_path),
            output_path=str(quant_path),
            obs_dim=obs_dim,
            action_dim=action_dim,
            verbose=False,
        )
        
        # Verify results are valid
        assert results["original_size_mb"] > 0
        assert results["quantized_size_mb"] > 0
        assert results["compression_ratio"] > 0
        assert quant_path.exists()


def test_load_quantized_policy():
    """Test loading a quantized policy."""
    obs_dim = 4
    action_dim = 2
    
    # Create a temporary checkpoint
    model = ActorCritic(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        quant_path = Path(tmpdir) / "test_quant.pt"
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": 10,
        }
        torch.save(checkpoint, ckpt_path)
        
        # Quantize the policy
        quantize_policy(
            checkpoint_path=str(ckpt_path),
            output_path=str(quant_path),
            obs_dim=obs_dim,
            action_dim=action_dim,
            verbose=False,
        )
        
        # Load quantized model
        quant_model = load_quantized_policy(
            str(quant_path),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        
        # Verify forward pass works (quantized linear requires batch dimension)
        obs = torch.randn(1, obs_dim)
        action, logprob, value = quant_model(obs)
        
        assert action.dtype == torch.int64
        assert logprob.ndim == 1
        assert value.ndim == 1

