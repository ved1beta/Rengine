import torch 
import time
from RLLM.inference.load_model import load_policy
from RLLM.envs.cartpole import CartPoleEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CartPoleEnv(device=device, seed=123)

model = load_policy(
    "checkpoints/ckpt_000300.pt",
    obs_dim=env.obs_dim,
    action_dim=env.action_dim,
    device=device,
)

obs = env.reset() 

# warm-up
for _ in range(100):
  
  with torch.no_grad():
    model(obs)

N = 1 
start = time.pref_counter()

for _ in range(N):
  with torch.no_grad():
    action, _, _ = model(obs)
    obs, _, done, _ = env.step(action)
    if done:
        obs = env.reset()
end = time.perf_counter()

latency_ms = (end - start) * 1000 / N
print(f"Inference latency (avg): {latency_ms:.3f} ms")



