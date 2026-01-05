import torch
import time
from RLLM.envs.cartpole import CartPoleEnv
from RLLM.models.policy import ActorCritic
from RLLM.rollout import RolloutBuffer
from RLLM.losses import policy_loss, value_loss, total_loss, compute_gae

# Set seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Hyperparameters
ROLLOUT_LEN = 2048
NUM_UPDATES = 50
LOG_INTERVAL = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = CartPoleEnv(device=device, seed=42)

model = ActorCritic(obs_dim=env.obs_dim, action_dim=env.action_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

buffer = RolloutBuffer(rollout_len=ROLLOUT_LEN, obs_dim=env.obs_dim, device=device)

obs = env.reset()

for update in range(NUM_UPDATES):
    start_time = time.time()
    

    buffer.reset()
    episode_rewards = []
    current_episode_reward = 0.0
    
    while not buffer.full:
        with torch.no_grad():
            action, logprob, value = model(obs)

        next_obs, reward, done, _ = env.step(action)
        current_episode_reward += reward.item()

        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            logprob=logprob.detach(),
            value=value.detach(),
        )

        obs = next_obs
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs = env.reset()

    data = buffer.get()

    advantages, returns = compute_gae(
        data["rewards"],
        data["values"],
        data["dones"],
    )

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    

    action_new, logprob_new, value_new = model(data["obs"])
    entropy = model.get_entropy(data["obs"])

    p_loss = policy_loss(
        logprob_new,
        data["logprobs"],
        advantages,
    )
    
    v_loss = value_loss(value_new, returns)
    ent = entropy.mean()
    
    loss = total_loss(p_loss, v_loss, ent)

    if not torch.isfinite(loss):
        print(f"ERROR: Non-finite loss detected at update {update}")
        break

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    if update % LOG_INTERVAL == 0:
        elapsed_time = time.time() - start_time
        steps_per_sec = ROLLOUT_LEN / elapsed_time
        
        mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        
        print(f"Update {update:4d} | "
              f"Mean Reward: {mean_reward:7.2f} | "
              f"Policy Loss: {p_loss.item():7.4f} | "
              f"Value Loss: {v_loss.item():7.4f} | "
              f"Entropy: {ent.item():6.4f} | "
              f"Steps/sec: {steps_per_sec:6.1f}")

        try:
            buffer.check_finite()
        except RuntimeError as e:
            print(f"ERROR: {e}")
            break

print("\nTraining completed!")

