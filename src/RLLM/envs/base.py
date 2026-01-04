from abc import ABC, abstractmethod
import torch


class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @property
    @abstractmethod
    def obs_dim(self):
        pass

    @property
    @abstractmethod
    def action_dim(self):
        pass
