
import torch
from RLLM.models.policy import ActorCritic

model = ActorCritic(obs_dim=4, action_dim=2)
ckpt = torch.load("checkpoints/ckpt_000300.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

torch.save(quantized.state_dict(), "artifacts/policy_int8.pt")
