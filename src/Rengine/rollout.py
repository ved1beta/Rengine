import torch

class RolloutBuffer:
    def __init__(self, rollout_len: int, obs_dim: int, device: torch.device):
        self.rollout_len = rollout_len
        self.device = device
        self.ptr = 0
        self.full = False

        self.obs = torch.zeros((rollout_len, obs_dim), device=device)
        self.actions = torch.zeros(rollout_len, device=device)
        self.rewards = torch.zeros(rollout_len, device=device)
        self.dones = torch.zeros(rollout_len, device=device)
        self.logprobs = torch.zeros(rollout_len, device=device)
        self.values = torch.zeros(rollout_len, device=device)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: bool,
        logprob: torch.Tensor,
        value: torch.Tensor,
    ):
        if self.full:
            raise RuntimeError("RolloutBuffer is full")

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value

        self.ptr += 1
        if self.ptr == self.rollout_len:
            self.full = True
    
    def get(self):
        if not self.full:
            raise RuntimeError("RolloutBuffer not full yet")

        return {
            "obs": self.obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "logprobs": self.logprobs,
            "values": self.values,
        }
    
    def reset(self):
        self.ptr = 0
        self.full = False

    def check_finite(self):
        for name, tensor in self.get().items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite values in rollout: {name}")
