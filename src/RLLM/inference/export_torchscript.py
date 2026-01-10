import torch
from RLLM.models.policy import ActorCritic

device = torch.device("cpu")

model = ActorCritic(obs_dim=4, action_dim=2)
ckpt = torch.load("checkpoints/ckpt_000300.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

example_input = torch.randn(4)
scripted = torch.jit.trace(model, example_input)
scripted.save("artifacts/policy_ts.pt")