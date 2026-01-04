import torch
from RLLM.envs.cartpole import CartPoleEnv
from RLLM.models.policy import ActorCritic

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
