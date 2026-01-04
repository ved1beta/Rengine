import gymnasium as gym
import torch

from .base import BaseEnv


class CartPoleEnv(BaseEnv):
    def __init__(self, device: torch.device):
        self.env = gym.make("CartPole-v1")
        self.device = device

        self.env.reset(seed=42)
        self.env.action_space.seed(42)


        self._obs_dim = self.env.observation_space.shape[0]
        self._action_dim = self.env.action_space.n

    def reset(self):
        obs, _ = self.env.reset()
        return self._to_tensor(obs)

    def step(self, action):
        action = int(action.item())

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        return (
            self._to_tensor(obs),
            torch.tensor(reward, dtype=torch.float32, device=self.device),
            done,
            info,
        )

    def _to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def action_dim(self):
        return self._action_dim
