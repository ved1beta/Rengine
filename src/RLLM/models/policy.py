import torch
import torch.nn as nn
from torch.distributions import Categorical

class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, output_dim: int):
        super(MLPEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class PolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.encoder = MLPEncoder(obs_dim, 64, 128)
        self.policy_head = PolicyHead(128, action_dim)
        self.value_head = ValueHead(128)

    def forward(self, obs: torch.Tensor):
        features = self.encoder(obs)

        logits = self.policy_head(features)
        value = self.value_head(features)

        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob, value

    def get_entropy(self, obs: torch.Tensor):
        features = self.encoder(obs)
        logits = self.policy_head(features)
        dist = Categorical(logits=logits)
        return dist.entropy()
