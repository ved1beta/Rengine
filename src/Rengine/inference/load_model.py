import torch
from Rengine.models.policy import ActorCritic

def load_policy(chkpt_path, obs_dim, action_dim, device):
    model = ActorCritic(obs_dim, action_dim).to(device)
    checkpoint = torch.load(chkpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

