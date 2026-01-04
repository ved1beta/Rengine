import torch
from src.RLLM.envs.cartpole import CartPoleEnv


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
